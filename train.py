from ultralytics import YOLO

def main():
    # 1. 加载模型
    # 毕设建议：先用 yolov8n.pt (Nano版) 跑通流程，速度快。
    # 如果跑通后觉得精度不够，再改用 yolov8s.pt (Small版)。
    # 第一次运行会自动下载权重文件，不用担心。
    model = YOLO('yolov8n.pt') 

    # 2. 开始训练
    print("🚀 开始训练...")
    model.train(
        data='data.yaml',   # 指定配置文件
        epochs=50,          # 训练轮数：毕设建议 50-100，先跑 50 试试
        imgsz=640,          # 图片大小：标准是 640
        batch=16,           # 批次大小：3060 6G显存建议设 16。如果报错 OOM (显存不足)，改成 8 或 4
        device=0,           # device=0 表示强制使用第一张显卡 (RTX 3060)
        workers=2,          # Windows下多线程加载数据容易报错，建议设为 2 或 0
        name='pneumonia_exp1', # 结果保存的文件夹名字
        patience=10,        # 如果 10 轮都没有提升，提前停止训练（省时间）
        exist_ok=True       # 如果文件夹已存在，允许覆盖
    )

    # 3. 简单的验证
    print("✅ 训练结束，正在验证...")
    metrics = model.val()
    print(f"验证集 mAP50: {metrics.box.map50}")

if __name__ == '__main__':
    # Windows 系统必须把代码放在 if __name__ == '__main__': 下面运行
    main()