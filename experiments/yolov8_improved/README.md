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
