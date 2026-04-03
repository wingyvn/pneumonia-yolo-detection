"""
运行时向 Ultralytics 注入 CBAM 模块与自定义 parse_model。
===========================================================

目的:
    - 不修改 site-packages
    - 仅在改进实验脚本运行时启用 CBAM
"""

from __future__ import annotations

import ast
import contextlib

import torch
from ultralytics.nn import tasks
from ultralytics.utils.ops import make_divisible

from cbam_modules import CBAM


def apply_cbam_patch():
    """在运行时补丁 Ultralytics，使其支持 YAML 中的 CBAM 模块。"""
    tasks.CBAM = CBAM
    tasks.parse_model = custom_parse_model


def custom_parse_model(d, ch, verbose=True):
    """
    基于当前 Ultralytics parse_model 改写，新增 CBAM 支持。

    这里只扩展当前实验需要的模块分支，尽量保持和原版行为一致。
    """
    legacy = True
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            tasks.LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        tasks.Conv.default_act = eval(act)
        if verbose:
            tasks.LOGGER.info(f"{tasks.colorstr('activation:')} {act}")

    if verbose:
        tasks.LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    base_modules = frozenset(
        {
            tasks.Classify,
            tasks.Conv,
            tasks.ConvTranspose,
            tasks.GhostConv,
            tasks.Bottleneck,
            tasks.GhostBottleneck,
            tasks.SPP,
            tasks.SPPF,
            tasks.C2fPSA,
            tasks.C2PSA,
            tasks.DWConv,
            tasks.Focus,
            tasks.BottleneckCSP,
            tasks.C1,
            tasks.C2,
            tasks.C2f,
            tasks.C3k2,
            tasks.RepNCSPELAN4,
            tasks.ELAN1,
            tasks.ADown,
            tasks.AConv,
            tasks.SPPELAN,
            tasks.C2fAttn,
            tasks.C3,
            tasks.C3TR,
            tasks.C3Ghost,
            torch.nn.ConvTranspose2d,
            tasks.DWConvTranspose2d,
            tasks.C3x,
            tasks.RepC3,
            tasks.PSA,
            tasks.SCDown,
            tasks.C2fCIB,
            tasks.A2C2f,
        }
    )
    repeat_modules = frozenset(
        {
            tasks.BottleneckCSP,
            tasks.C1,
            tasks.C2,
            tasks.C2f,
            tasks.C3k2,
            tasks.C2fAttn,
            tasks.C3,
            tasks.C3TR,
            tasks.C3Ghost,
            tasks.C3x,
            tasks.RepC3,
            tasks.C2fPSA,
            tasks.C2fCIB,
            tasks.C2PSA,
            tasks.A2C2f,
        }
    )

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n

        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is tasks.C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m is tasks.C3k2:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is tasks.A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is tasks.C2fCIB:
                legacy = False
        elif m is CBAM:
            c1 = ch[f]
            c2 = c1
            args = [c1, *args]
        elif m is tasks.AIFI:
            args = [ch[f], *args]
        elif m in frozenset({tasks.HGStem, tasks.HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is tasks.HGBlock:
                args.insert(4, n)
                n = 1
        elif m is tasks.ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is tasks.Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {tasks.Detect, tasks.WorldDetect, tasks.YOLOEDetect, tasks.Segment, tasks.YOLOESegment, tasks.Pose, tasks.OBB, tasks.ImagePoolingAttn, tasks.v10Detect}
        ):
            args.append([ch[x] for x in f])
            if m is tasks.Segment or m is tasks.YOLOESegment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {tasks.Detect, tasks.YOLOEDetect, tasks.Segment, tasks.YOLOESegment, tasks.Pose, tasks.OBB}:
                m.legacy = legacy
        elif m is tasks.RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is tasks.CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is tasks.CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({tasks.TorchVision, tasks.Index}):
            c2 = args[0]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            tasks.LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return torch.nn.Sequential(*layers), sorted(save)
