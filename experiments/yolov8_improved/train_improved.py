"""
改进型 YOLOv8 训练入口
=====================

用途:
    - 将改进实验的训练与现有基线分离
    - 输出到 experiments/yolov8_improved/runs
    - 后续接入 CBAM / DCNv2 时无需改 src/train.py

使用示例:
    python experiments/yolov8_improved/train_improved.py --model yolov8n.yaml
    python experiments/yolov8_improved/train_improved.py --model experiments/yolov8_improved/model_configs/yolov8_cbam.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from experiments.experiment_defaults import FAIR_TRAINING_DEFAULTS
from experiments.yolov8_improved.patch_ultralytics_cbam import apply_cbam_patch


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = Path(__file__).resolve().parent / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train improved YOLOv8 experiments in isolation")
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).resolve().parent / "model_configs" / "yolov8_cbam.yaml")
    )
    parser.add_argument("--data", type=str, default=FAIR_TRAINING_DEFAULTS["data"])
    parser.add_argument("--epochs", type=int, default=FAIR_TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--imgsz", type=int, default=FAIR_TRAINING_DEFAULTS["imgsz"])
    parser.add_argument("--batch", type=int, default=FAIR_TRAINING_DEFAULTS["batch"])
    parser.add_argument("--device", type=str, default=FAIR_TRAINING_DEFAULTS["device"])
    parser.add_argument("--workers", type=int, default=FAIR_TRAINING_DEFAULTS["workers"])
    parser.add_argument("--patience", type=int, default=FAIR_TRAINING_DEFAULTS["patience"])
    parser.add_argument("--name", type=str, default="pneumonia_cbam_exp1")
    parser.add_argument("--pretrained", type=str, default=FAIR_TRAINING_DEFAULTS["model"])
    return parser.parse_args()


def main():
    args = parse_args()
    DEFAULT_PROJECT.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if "cbam" in model_path.name.lower():
        apply_cbam_patch()

    model = YOLO(args.model)
    if args.pretrained:
        model = model.load(args.pretrained)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(DEFAULT_PROJECT),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
