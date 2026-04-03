"""
Faster R-CNN 对比实验评估脚本
============================

用途:
    - 在 val/test 集上评估 Faster R-CNN
    - 输出 mAP@0.5、mAP@0.5:0.95、每类 AP50、平均推理时延和 FPS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    CLASS_NAMES,
    DEFAULT_OUTPUT_ROOT,
    build_model,
    create_dataloader,
    evaluate_model,
    measure_inference_speed,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN baseline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT / "checkpoints" / "best_map50.pth"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, data_loader = create_dataloader(args.split, batch_size=1, shuffle=False)

    model = build_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model(model, data_loader, device, score_threshold=args.score_threshold)
    speed = measure_inference_speed(model, dataset, device)

    report = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "mAP50": metrics["mAP50"],
        "mAP50_95": metrics["mAP50_95"],
        "per_class_AP50": metrics["per_class_AP50"],
        "avg_latency_sec": speed["avg_latency_sec"],
        "fps": speed["fps"],
        "num_measured_images": speed["num_measured_images"],
        "class_names": CLASS_NAMES,
    }

    report_path = DEFAULT_OUTPUT_ROOT / "reports" / f"eval_{args.split}.json"
    save_json(report, report_path)

    print(f"split={args.split}")
    print(f"mAP50={metrics['mAP50']:.4f}")
    print(f"mAP50_95={metrics['mAP50_95']:.4f}")
    print(f"avg_latency_sec={speed['avg_latency_sec']:.4f}")
    print(f"fps={speed['fps']:.2f}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
