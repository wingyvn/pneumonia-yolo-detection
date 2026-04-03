"""
Faster R-CNN 实验公共工具
=========================

提供:
    - YOLO 标签到 torchvision 检测格式的数据集转换
    - Faster R-CNN 模型构建
    - mAP@0.5 / mAP@0.5:0.95 评估
    - 推理速度测试
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


WORKSPACE = Path(__file__).resolve().parents[2]
DATA_ROOT = WORKSPACE
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
CLASS_NAMES = ['Pneumonia Bacteria', 'Pneumonia Virus', 'Sick', 'healthy', 'tuberculosis']
NUM_CLASSES = len(CLASS_NAMES) + 1  # background + 5 classes


class YoloDetectionDataset(Dataset):
    """读取现有 YOLO txt 标签并转换为 Faster R-CNN 训练格式。"""

    def __init__(self, split: str, imgsz: int = 640):
        self.images_dir = DATA_ROOT / "images" / split
        self.labels_dir = DATA_ROOT / "labels" / split
        self.samples = []
        self.imgsz = imgsz

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
        orig_width, orig_height = image.size
        if self.imgsz:
            image = image.resize((self.imgsz, self.imgsz))
            width, height = self.imgsz, self.imgsz
        else:
            width, height = orig_width, orig_height

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
                    labels.append(class_id + 1)  # background reserved for 0

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index]),
            "image_path": str(image_path),
        }

        if len(boxes) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        return F.to_tensor(image), target


def collate_fn(batch):
    return tuple(zip(*batch))


def create_dataloader(split: str, batch_size: int, shuffle: bool, imgsz: int = 640, num_workers: int = 2):
    dataset = YoloDetectionDataset(split, imgsz=imgsz)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataset, loader


def build_model(num_classes: int = NUM_CLASSES):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, data_loader, optimizer, device, accum_steps: int = 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step_idx, (images, targets) in enumerate(data_loader, start=1):
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items() if key != "image_path"}
            for target in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        scaled_losses = losses / max(accum_steps, 1)

        scaled_losses.backward()

        if step_idx % max(accum_steps, 1) == 0 or step_idx == len(data_loader):
            optimizer.step()
            optimizer.zero_grad()

        total_loss += losses.item()

    return total_loss / max(len(data_loader), 1)


def inference_dataset(model, data_loader, device, score_threshold: float = 0.05):
    model.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                image_path = target["image_path"]
                gt_boxes = target["boxes"].cpu().numpy().tolist()
                gt_labels = target["labels"].cpu().numpy().tolist()

                preds = []
                boxes = output["boxes"].detach().cpu().numpy().tolist()
                labels = output["labels"].detach().cpu().numpy().tolist()
                scores = output["scores"].detach().cpu().numpy().tolist()
                for box, label, score in zip(boxes, labels, scores):
                    if score < score_threshold:
                        continue
                    preds.append({
                        "image_path": image_path,
                        "class_id": int(label) - 1,
                        "score": float(score),
                        "bbox": box,
                    })

                predictions.extend(preds)

                for box, label in zip(gt_boxes, gt_labels):
                    ground_truths.append({
                        "image_path": image_path,
                        "class_id": int(label) - 1,
                        "bbox": box,
                    })

    return predictions, ground_truths


def compute_iou(box_a, box_b) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))


def evaluate_predictions(predictions: list, ground_truths: list, iou_thresholds: list[float] | None = None):
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]

    ap_by_threshold = defaultdict(list)
    per_class_ap50 = {}

    for iou_threshold in iou_thresholds:
        for class_id in range(len(CLASS_NAMES)):
            class_predictions = [pred for pred in predictions if pred["class_id"] == class_id]
            class_predictions.sort(key=lambda item: item["score"], reverse=True)
            class_gts = [gt for gt in ground_truths if gt["class_id"] == class_id]

            gt_by_image = defaultdict(list)
            for gt in class_gts:
                gt_by_image[gt["image_path"]].append({"bbox": gt["bbox"], "matched": False})

            tp = np.zeros(len(class_predictions))
            fp = np.zeros(len(class_predictions))

            for idx, pred in enumerate(class_predictions):
                candidates = gt_by_image[pred["image_path"]]
                best_iou = 0.0
                best_gt = None

                for gt in candidates:
                    if gt["matched"]:
                        continue
                    iou = compute_iou(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt

                if best_gt is not None and best_iou >= iou_threshold:
                    best_gt["matched"] = True
                    tp[idx] = 1
                else:
                    fp[idx] = 1

            total_gt = len(class_gts)
            if total_gt == 0:
                ap = 0.0
            else:
                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(fp)
                recalls = cum_tp / max(total_gt, 1)
                precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
                ap = compute_ap(recalls, precisions)

            ap_by_threshold[iou_threshold].append(ap)
            if iou_threshold == 0.50:
                per_class_ap50[CLASS_NAMES[class_id]] = ap

    map50 = float(np.mean(ap_by_threshold[0.50])) if ap_by_threshold[0.50] else 0.0
    map50_95 = float(np.mean([np.mean(values) for _, values in sorted(ap_by_threshold.items())])) if ap_by_threshold else 0.0

    return {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "per_class_AP50": per_class_ap50,
        "iou_thresholds": iou_thresholds,
    }


def evaluate_model(model, data_loader, device, score_threshold: float = 0.05):
    predictions, ground_truths = inference_dataset(model, data_loader, device, score_threshold=score_threshold)
    return evaluate_predictions(predictions, ground_truths)


def measure_inference_speed(model, dataset, device, warmup: int = 5, max_images: int = 100):
    model.eval()
    latencies = []
    sample_count = min(len(dataset), max_images)

    with torch.no_grad():
        for idx in range(sample_count):
            image, _ = dataset[idx]
            image = image.to(device)
            if idx < warmup:
                _ = model([image])
                continue

            start = time.perf_counter()
            _ = model([image])
            latencies.append(time.perf_counter() - start)

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = 1.0 / avg_latency if avg_latency > 0 else 0.0
    return {
        "avg_latency_sec": avg_latency,
        "fps": fps,
        "num_measured_images": len(latencies),
    }


def save_json(data: dict, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
