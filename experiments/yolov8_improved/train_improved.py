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


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_DATA = WORKSPACE / "data.yaml"
DEFAULT_PROJECT = Path(__file__).resolve().parent / "runs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train improved YOLOv8 experiments in isolation")
    parser.add_argument("--model", type=str, default="yolov8n.yaml")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="improved_exp")
    return parser.parse_args()


def main():
    args = parse_args()
    DEFAULT_PROJECT.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(DEFAULT_PROJECT),
        name=args.name,
    )


if __name__ == "__main__":
    main()
