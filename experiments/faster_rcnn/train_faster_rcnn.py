"""
Faster R-CNN 对比实验训练脚本
============================

用途:
    - 复用当前项目的 YOLO 标注数据
    - 训练 torchvision Faster R-CNN 基线
    - 每轮在验证集上评估 mAP@0.5 / mAP@0.5:0.95
    - 输出到 experiments/faster_rcnn/outputs，不影响现有 YOLOv8

使用示例:
    python experiments/faster_rcnn/train_faster_rcnn.py --epochs 20 --batch-size 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..experiment_defaults import FAIR_TRAINING_DEFAULTS
from .common import (
    CLASS_NAMES,
    DEFAULT_OUTPUT_ROOT,
    build_model,
    create_dataloader,#
    evaluate_model,
    save_json,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN baseline")
    parser.add_argument("--epochs", type=int, default=FAIR_TRAINING_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=4, help="单步微批大小；默认结合梯度累积实现等效 batch=16")
    parser.add_argument("--effective-batch", type=int, default=FAIR_TRAINING_DEFAULTS["batch"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=FAIR_TRAINING_DEFAULTS["imgsz"])
    parser.add_argument("--workers", type=int, default=FAIR_TRAINING_DEFAULTS["workers"])
    parser.add_argument("--device", type=str, default=FAIR_TRAINING_DEFAULTS["device"])
    parser.add_argument("--patience", type=int, default=FAIR_TRAINING_DEFAULTS["patience"])
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = DEFAULT_OUTPUT_ROOT
    checkpoints_dir = output_root / "checkpoints"
    reports_dir = output_root / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and args.device not in {"cpu", "-1"}
    device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    _, train_loader = create_dataloader(
        args.train_split,
        batch_size=args.batch_size,
        shuffle=True,
        imgsz=args.imgsz,
        num_workers=args.workers,
    )
    _, val_loader = create_dataloader(
        args.val_split,
        batch_size=1,
        shuffle=False,
        imgsz=args.imgsz,
        num_workers=args.workers,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    accum_steps = max(1, args.effective_batch // max(1, args.batch_size))

    history = []
    best_map50 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint_path = checkpoints_dir / "best_map50.pth"

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device, accum_steps=accum_steps)
        metrics = evaluate_model(model, val_loader, device, score_threshold=args.score_threshold)

        record = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "mAP50": metrics["mAP50"],
            "mAP50_95": metrics["mAP50_95"],
            "per_class_AP50": metrics["per_class_AP50"],
        }
        history.append(record)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss:.4f} | mAP50={metrics['mAP50']:.4f} | mAP50_95={metrics['mAP50_95']:.4f}"
        )

        checkpoint_payload = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "class_names": CLASS_NAMES,
            "train_args": vars(args),
        }

        latest_checkpoint = checkpoints_dir / f"epoch_{epoch + 1:03d}.pth"
        torch.save(checkpoint_payload, latest_checkpoint)

        if metrics["mAP50"] > best_map50:
            best_map50 = metrics["mAP50"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(checkpoint_payload, best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch + 1}, best epoch was {best_epoch}.")
            break

    save_json(
        {
            "history": history,
            "best_mAP50": best_map50,
            "best_epoch": best_epoch,
            "effective_batch": args.effective_batch,
            "micro_batch": args.batch_size,
            "accum_steps": accum_steps,
            "fair_reference": FAIR_TRAINING_DEFAULTS,
        },
        reports_dir / "train_history.json"
    )
    print(f"训练完成，最佳权重保存在: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
