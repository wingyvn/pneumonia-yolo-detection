"""
基于验证集的分类别阈值推荐脚本
================================

用途:
    1. 在验证集上运行模型预测
    2. 为每个类别绘制 Precision-Recall 曲线
    3. 依据预设策略推荐分类别阈值
    4. 输出 JSON / TXT / PNG，供 PyQt 应用模式手动回填

使用示例:
    python src/recommend_thresholds.py
    python src/recommend_thresholds.py --model runs/detect/pneumonia_exp1/weights/best.pt --recall-target 0.90
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO


CLASS_NAMES = ['Pneumonia Bacteria', 'Pneumonia Virus', 'Sick', 'healthy', 'tuberculosis']
CLASS_NAMES_CN = ['细菌性肺炎', '病毒性肺炎', '患病', '健康', '肺结核']
CLASS_COLORS = ['#E53935', '#1E88E5', '#FDD835', '#43A047', '#8E24AA']

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = WORKSPACE / 'runs' / 'detect' / 'pneumonia_exp1' / 'weights' / 'best.pt'
DEFAULT_DATA = WORKSPACE / 'data.yaml'
OUTPUT_DIR = WORKSPACE / 'outputs' / 'threshold_recommendation'


def parse_args():
    parser = argparse.ArgumentParser(description='Recommend class-wise thresholds from validation PR curves')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL))
    parser.add_argument('--data', type=str, default=str(DEFAULT_DATA))
    parser.add_argument('--iou-match', type=float, default=0.5, help='IoU threshold used for TP/FP matching')
    parser.add_argument('--recall-target', type=float, default=0.90, help='Target recall for tuberculosis threshold selection')
    return parser.parse_args()


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data_yaml(data_yaml_path: str) -> dict:
    with open(data_yaml_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (WORKSPACE / path).resolve()
    return path


def infer_label_dir_from_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if 'images' in parts:
        idx = parts.index('images')
        parts[idx] = 'labels'
        return Path(*parts)
    return WORKSPACE / 'labels' / image_dir.name


def yolo_to_xyxy(box, width: int, height: int):
    class_id, cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    return int(class_id), [x1, y1, x2, y2]


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


def load_ground_truths(val_images_dir: Path, val_labels_dir: Path):
    dataset = []
    for image_path in sorted(val_images_dir.iterdir()):
        if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        label_path = val_labels_dir / f'{image_path.stem}.txt'
        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes = []
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, bbox = yolo_to_xyxy(tuple(map(float, parts)), width, height)
                    gt_boxes.append({'class_id': class_id, 'bbox': bbox})
        dataset.append({
            'image_path': image_path,
            'width': width,
            'height': height,
            'gt_boxes': gt_boxes
        })
    return dataset


def run_predictions(model: YOLO, dataset: list):
    results = model.predict(
        source=[str(item['image_path']) for item in dataset],
        conf=0.01,
        iou=0.70,
        verbose=False,
        save=False
    )

    predictions_by_image = {}
    for item, result in zip(dataset, results):
        preds = []
        for box in result.boxes:
            preds.append({
                'class_id': int(box.cls[0]),
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
        predictions_by_image[str(item['image_path'])] = preds
    return predictions_by_image


def evaluate_class_at_threshold(dataset: list, predictions_by_image: dict, class_id: int, threshold: float, iou_match: float):
    tp = fp = fn = 0

    for item in dataset:
        image_key = str(item['image_path'])
        gt_boxes = [gt['bbox'] for gt in item['gt_boxes'] if gt['class_id'] == class_id]
        pred_boxes = [
            pred for pred in predictions_by_image.get(image_key, [])
            if pred['class_id'] == class_id and pred['confidence'] >= threshold
        ]
        pred_boxes.sort(key=lambda entry: entry['confidence'], reverse=True)

        matched = set()
        for pred in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gt_bbox in enumerate(gt_boxes):
                if idx in matched:
                    continue
                iou = compute_iou(pred['bbox'], gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= iou_match:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1

        fn += max(0, len(gt_boxes) - len(matched))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def recommend_threshold(class_name: str, metrics: list, recall_target: float):
    if class_name == 'tuberculosis':
        candidates = [item for item in metrics if item['recall'] >= recall_target]
        if candidates:
            return max(candidates, key=lambda item: (item['precision'], item['recall'], -item['threshold']))
        return max(metrics, key=lambda item: (item['recall'], item['precision']))

    if class_name == 'healthy':
        return max(metrics, key=lambda item: (item['precision'], item['recall'], item['threshold']))

    return max(metrics, key=lambda item: (item['f1'], item['precision'], item['recall']))


def plot_pr_curves(metrics_by_class: dict):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, class_name in enumerate(CLASS_NAMES):
        ax = axes[idx]
        metrics = metrics_by_class[class_name]
        recalls = [item['recall'] for item in metrics]
        precisions = [item['precision'] for item in metrics]
        ax.plot(recalls, precisions, color=CLASS_COLORS[idx], linewidth=2)
        ax.set_title(f'{CLASS_NAMES_CN[idx]} PR 曲线', fontsize=12, fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)

    axes[-1].axis('off')
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'pr_curves.png'
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    return save_path


def plot_threshold_metrics(metrics_by_class: dict, recommendations: dict):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, class_name in enumerate(CLASS_NAMES):
        ax = axes[idx]
        metrics = metrics_by_class[class_name]
        thresholds = [item['threshold'] for item in metrics]
        precisions = [item['precision'] for item in metrics]
        recalls = [item['recall'] for item in metrics]
        f1_scores = [item['f1'] for item in metrics]

        ax.plot(thresholds, precisions, label='Precision', color='#1E88E5')
        ax.plot(thresholds, recalls, label='Recall', color='#43A047')
        ax.plot(thresholds, f1_scores, label='F1', color='#E53935')

        rec = recommendations[class_name]
        ax.axvline(rec['threshold'], color='#6A1B9A', linestyle='--', linewidth=1.5)
        ax.set_title(f'{CLASS_NAMES_CN[idx]} 阈值曲线', fontsize=12, fontweight='bold')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metric')
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].axis('off')
    plt.tight_layout()
    save_path = OUTPUT_DIR / 'threshold_metric_curves.png'
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    return save_path


def save_recommendations(recommendations: dict, metrics_by_class: dict):
    json_path = OUTPUT_DIR / 'recommended_thresholds.json'
    txt_path = OUTPUT_DIR / 'recommended_thresholds.txt'

    json_data = {
        class_name: {
            'class_name_cn': CLASS_NAMES_CN[idx],
            'recommended_threshold': round(info['threshold'], 2),
            'precision': round(info['precision'], 4),
            'recall': round(info['recall'], 4),
            'f1': round(info['f1'], 4),
        }
        for idx, (class_name, info) in enumerate(recommendations.items())
    }
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=2)

    lines = [
        "=======================================",
        "      验证集分类别阈值推荐报告",
        "=======================================",
        ""
    ]
    for idx, class_name in enumerate(CLASS_NAMES):
        info = recommendations[class_name]
        lines.extend([
            f"[{CLASS_NAMES_CN[idx]} / {class_name}]",
            f"推荐阈值: {info['threshold']:.2f}",
            f"Precision: {info['precision']:.4f}",
            f"Recall: {info['recall']:.4f}",
            f"F1: {info['f1']:.4f}",
            ""
        ])

    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(lines))

    return json_path, txt_path


def main():
    args = parse_args()
    ensure_output_dir()

    model_path = resolve_path(args.model)
    data_yaml = load_data_yaml(args.data)
    val_images_dir = resolve_path(data_yaml['val'])
    val_labels_dir = infer_label_dir_from_image_dir(val_images_dir)

    if not model_path.exists():
        raise FileNotFoundError(f'模型文件不存在: {model_path}')
    if not val_images_dir.exists():
        raise FileNotFoundError(f'验证集图像目录不存在: {val_images_dir}')
    if not val_labels_dir.exists():
        raise FileNotFoundError(f'验证集标签目录不存在: {val_labels_dir}')

    print(f'加载模型: {model_path}')
    model = YOLO(str(model_path))
    dataset = load_ground_truths(val_images_dir, val_labels_dir)
    predictions_by_image = run_predictions(model, dataset)

    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    metrics_by_class = {}
    recommendations = {}

    for class_id, class_name in enumerate(CLASS_NAMES):
        class_metrics = [
            evaluate_class_at_threshold(dataset, predictions_by_image, class_id, float(threshold), args.iou_match)
            for threshold in thresholds
        ]
        metrics_by_class[class_name] = class_metrics
        recommendations[class_name] = recommend_threshold(class_name, class_metrics, args.recall_target)

    pr_curve_path = plot_pr_curves(metrics_by_class)
    metric_curve_path = plot_threshold_metrics(metrics_by_class, recommendations)
    json_path, txt_path = save_recommendations(recommendations, metrics_by_class)

    print('推荐阈值如下:')
    for class_name in CLASS_NAMES:
        info = recommendations[class_name]
        print(
            f"- {class_name}: threshold={info['threshold']:.2f}, "
            f"P={info['precision']:.4f}, R={info['recall']:.4f}, F1={info['f1']:.4f}"
        )

    print(f'PR 曲线: {pr_curve_path}')
    print(f'阈值曲线: {metric_curve_path}')
    print(f'JSON 报告: {json_path}')
    print(f'TXT 报告: {txt_path}')


if __name__ == '__main__':
    main()
