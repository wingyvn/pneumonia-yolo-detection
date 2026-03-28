# Experiments Workspace

这个目录用于放置对比实验与改进实验，避免影响当前 `src/` 下的基线 YOLOv8 系统。

建议约定：

- `experiments/faster_rcnn/`
  - 两阶段检测对比基线
  - 输出目录建议写到 `experiments/faster_rcnn/outputs/`
- `experiments/yolov8_improved/`
  - 改进型 YOLOv8 实验
  - 输出目录建议写到 `experiments/yolov8_improved/runs/`

论文中的三组实验建议统一遵循：

1. 基线一：Faster R-CNN
2. 基线二：YOLOv8
3. 改进组：YOLOv8 + CBAM，DCNv2 作为进一步消融项

这样可以把“检测框架对比”和“同框架改进效果”分开叙述，结构更清晰。
