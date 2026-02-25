# 肺炎检测系统

这是一个基于YOLOv8的肺炎X光片检测系统。该系统可以自动检测X光片中的肺炎病灶。

## 环境要求

- Python 3.8+
- CUDA支持（推荐，用于GPU加速）

## 安装

1. 克隆项目：

```bash
git clone <repository_url>
cd ai_work
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集结构

```
pneumonia_testing/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── ai_work/
    ├── src/
    │   ├── train.py
    │   ├── app.py
    │   └── data_check.py
    ├── data.yaml
    └── requirements.txt
```

## 使用方法

1. 数据验证：

```bash
python src/data_check.py
```

2. 模型训练：

```bash
python src/train.py
```

3. 启动Web界面：

```bash
streamlit run src/app.py
```

## 功能特点

- 数据集完整性检查
- 自动化训练流程
- 用户友好的Web界面
- 实时预测结果可视化

## 注意事项

- 确保数据集格式正确（YOLO格式）
- 训练前请验证数据集完整性
- 建议使用GPU进行训练

## 许可证

[添加许可证信息]