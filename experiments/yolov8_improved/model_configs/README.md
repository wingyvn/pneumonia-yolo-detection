# Model Config Notes

这里放改进模型的 yaml 配置文件。

建议文件命名：

- `yolov8_cbam.yaml`
- `yolov8_cbam_dcnv2.yaml`

建议做法：

1. 先复制一份与你基线规模一致的 YOLOv8 yaml
2. 先插入 CBAM 模块并完成训练
3. 确认训练链路稳定后，再加入 DCNv2

不要直接修改 `.venv` 或 `ultralytics` 源目录中的默认模型配置，避免污染基线环境。
