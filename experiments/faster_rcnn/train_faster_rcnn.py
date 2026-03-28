"""
Faster R-CNN 对比实验训练脚本
============================

用途:
    - 复用当前项目的 YOLO 标注数据
    - 训练 torchvision Faster R-CNN 基线
    - 输出到 experiments/faster_rcnn/outputs，不影响现有 YOLOv8

使用示例:
    python experiments/faster_rcnn/train_faster_rcnn.py --epochs 20 --batch-size 2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


WORKSPACE = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
CLASS_NAMES = ['Pneumonia Bacteria', 'Pneumonia Virus', 'Sick', 'healthy', 'tuberculosis']
NUM_CLASSES = len(CLASS_NAMES) + 1  # background + 5 classes


class YoloDetectionDataset(Dataset):
    """读取现有 YOLO txt 标签并转换为 Faster R-CNN 训练格式。"""

    def __init__(self, split: str):
        self.images_dir = DATA_ROOT / "images" / split
        self.labels_dir = DATA_ROOT / "labels" / split
        self.samples = []

        if not self.images_dir.exists():
            raise FileNotFoundError(f"找不到图像目录: {self.images_dir}")

        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            self.samples.append((image_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])

                    x1 = (cx - bw / 2) * width
                    y1 = (cy - bh / 2) * height
                    x2 = (cx + bw / 2) * width
                    y2 = (cy + bh / 2) * height

                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id + 1)  # 0 号保留给 background

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }

        if len(boxes) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / max(len(data_loader), 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--split", type=str, default="train")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "checkpoints").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YoloDetectionDataset(args.split)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = build_model(NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, data_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f}")

        checkpoint_path = OUTPUT_ROOT / "checkpoints" / f"epoch_{epoch + 1:03d}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_loss,
                "class_names": CLASS_NAMES,
            },
            checkpoint_path,
        )

    print(f"训练完成，权重保存在: {OUTPUT_ROOT / 'checkpoints'}")


if __name__ == "__main__":
    main()
