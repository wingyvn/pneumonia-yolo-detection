"""
共享实验默认配置
================

统一复用 backup/train.py 中的 YOLOv8 基线训练参数，保证后续对比实验口径一致。
"""

from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_YAML = WORKSPACE / "data.yaml"

FAIR_TRAINING_DEFAULTS = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "device": "0",
    "workers": 2,
    "patience": 10,
    "model": "yolov8n.pt",
    "data": str(DATA_YAML),
}
