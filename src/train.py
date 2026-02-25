from ultralytics import YOLO
import yaml
import os

def train_model(data_yaml_path: str, epochs: int = 100):
    """
    训练YOLO模型
    
    Args:
        data_yaml_path (str): 数据配置文件路径
        epochs (int): 训练轮数
    """
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        patience=30,
        save=True
    )
    
    return results

def evaluate_model(model_path: str, data_yaml_path: str):
    """
    评估模型性能
    
    Args:
        model_path (str): 训练好的模型路径
        data_yaml_path (str): 数据配置文件路径
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml_path)
    return results

if __name__ == "__main__":
    data_yaml_path = "data.yaml"
    
    # 训练模型
    results = train_model(data_yaml_path)
    
    # 评估模型
    model_path = "runs/detect/train/weights/best.pt"  # 默认保存路径
    eval_results = evaluate_model(model_path, data_yaml_path)