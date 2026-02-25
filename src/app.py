import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os

def load_model(model_path: str):
    """
    加载训练好的模型
    """
    return YOLO(model_path)

def process_image(image, model):
    """
    处理图像并进行预测
    """
    results = model.predict(image)
    return results

def main():
    st.title("肺炎X光片检测系统")
    
    # 加载模型
    model_path = "runs/detect/train/weights/best.pt"  # 根据实际保存路径调整
    if not os.path.exists(model_path):
        st.error("请先训练模型！")
        return
        
    model = load_model(model_path)
    
    # 文件上传
    uploaded_file = st.file_uploader("上传X光片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 显示原始图片
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_column_width=True)
        
        # 进行预测
        if st.button("开始检测"):
            results = process_image(image, model)
            
            # 显示结果
            for r in results:
                im_array = r.plot()  # 绘制结果
                st.image(im_array, caption="检测结果", use_column_width=True)
                
                # 显示置信度
                for box in r.boxes:
                    conf = float(box.conf[0])
                    st.write(f"置信度: {conf:.2%}")

if __name__ == "__main__":
    main()