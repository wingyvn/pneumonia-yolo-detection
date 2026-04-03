# Improved YOLOv8 Experiments

这个目录用于放置改进型 YOLOv8 实验，不影响当前基线训练和推理流程。

建议推进顺序：

1. 先做 `YOLOv8 + CBAM`
2. 在此基础上再尝试引入 `DCNv2`
3. 最后做消融：
   - YOLOv8
   - YOLOv8 + CBAM
   - YOLOv8 + CBAM + DCNv2

原因：

- CBAM 集成简单，论文表述清晰，先做更稳。
- DCNv2 需要处理自定义模块与环境兼容，建议第二阶段再上。

目录建议：

- `model_configs/`
  - 放改进模型 yaml
- `runs/`
  - 放改进实验输出
- `train_improved.py`
  - 单独的训练入口，不改现有 `src/train.py`

当前已提供：

- `cbam_modules.py`
  - CBAM 模块实现
- `patch_ultralytics_cbam.py`
  - 运行时为 Ultralytics 注入 CBAM，不修改 site-packages
- `model_configs/yolov8_cbam.yaml`
  - YOLOv8 + CBAM 的实验配置

运行示例：

- YOLOv8 + CBAM
  - `python experiments/yolov8_improved/train_improved.py --model experiments/yolov8_improved/model_configs/yolov8_cbam.yaml --pretrained yolov8n.pt --name yolov8_cbam_exp`
