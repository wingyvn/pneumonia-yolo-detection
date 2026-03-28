# Faster R-CNN Baseline

这个目录用于构建两阶段检测对比实验，不影响现有 YOLOv8 工程。

当前提供：

- `train_faster_rcnn.py`
  - 读取现有 `images/`、`labels/` 的 YOLO 标注
  - 转换为 `torchvision` 所需的目标检测数据格式
  - 训练 Faster R-CNN ResNet50 FPN 基线

建议输出：

- 模型权重：`experiments/faster_rcnn/outputs/checkpoints/`
- 日志：`experiments/faster_rcnn/outputs/logs/`

建议实验指标：

- mAP@0.5
- mAP@0.5:0.95
- Precision / Recall
- 单张平均推理时延
- FPS

注意：

- Faster R-CNN 是深度学习对比基线，不属于传统图像处理方法。
- 如果导师强制要求“传统方法”，建议再补一组 `CLAHE + 纹理特征 + SVM` 作为轻量传统基线。
