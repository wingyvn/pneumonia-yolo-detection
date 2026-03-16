"""
增强版肺炎 X 光片检测 Web 界面
================================

功能说明:
    基于 Streamlit 构建的完整医学影像检测系统界面，支持:
    1. 侧边栏参数调节（置信度阈值、IoU 阈值）
    2. 单张 / 批量图片上传检测
    3. 检测结果详情表格（类别、置信度、边界框坐标）
    4. 检测统计面板（各类别检出数量饼图 & 柱状图）
    5. 标注图片 + JSON 结果一键下载
    6. 会话内检测历史记录

使用方式:
    cd workspace
    streamlit run src/app.py
"""

import streamlit as st
import numpy as np
import json
import io
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

# ============================================================
# 类别名称（与 data.yaml 保持一致）
# ============================================================
CLASS_NAMES = ['Pneumonia Bacteria', 'Pneumonia Virus', 'Sick', 'healthy', 'tuberculosis']

# 为每个类别定义颜色（用于统计图表）
CLASS_COLORS = ['#E53935', '#1E88E5', '#FDD835', '#43A047', '#8E24AA']


# ============================================================
# 页面基础配置
# ============================================================
st.set_page_config(
    page_title="肺炎 X 光片检测系统",
    page_icon="🫁",
    layout="wide",           # 使用宽屏布局，充分利用屏幕空间
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_path: str):
    """
    加载 YOLOv8 模型（使用 st.cache_resource 缓存）。
    
    st.cache_resource 装饰器的作用:
        模型只在第一次调用时加载，后续调用直接复用缓存，
        避免每次交互都重新加载模型（加载一次约需几秒）。
    
    Args:
        model_path: 训练好的模型权重文件路径 (.pt)
    
    Returns:
        YOLO 模型对象，若文件不存在则返回 None
    """
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)


def detect_image(model, image: Image.Image, conf_threshold: float, iou_threshold: float):
    """
    对单张图片执行目标检测。
    
    Args:
        model: 加载好的 YOLO 模型
        image: PIL Image 对象
        conf_threshold: 置信度阈值，低于此值的检测框将被过滤
        iou_threshold: IoU 阈值，用于非极大值抑制（NMS），
                       值越小则过滤越激进（重叠框更少）
    
    Returns:
        results: YOLO 推理结果列表
    """
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False       # 不在控制台打印推理信息
    )
    return results


def extract_detection_details(results) -> list:
    """
    从 YOLO 推理结果中提取每个检测框的详细信息。
    
    YOLO 推理结果结构说明:
        results[0].boxes 包含所有检测框:
            - boxes.cls:  类别索引 (tensor)
            - boxes.conf: 置信度 (tensor)
            - boxes.xyxy: 边界框坐标 [x1, y1, x2, y2] (tensor)
    
    Args:
        results: model.predict() 返回的结果
    
    Returns:
        detections: 字典列表，每个字典包含一个检测框的完整信息
    """
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
            conf = float(box.conf[0])
            # xyxy 格式: [左上角x, 左上角y, 右下角x, 右下角y]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                '类别': cls_name,
                '置信度': round(conf, 4),
                'x1': round(x1, 1),
                'y1': round(y1, 1),
                'x2': round(x2, 1),
                'y2': round(y2, 1),
                '宽度': round(x2 - x1, 1),
                '高度': round(y2 - y1, 1)
            })
    
    return detections


def render_sidebar():
    """
    渲染侧边栏控件，包括模型选择和检测参数调节。
    
    Returns:
        config: 字典，包含用户选择的所有配置参数
    """
    st.sidebar.title("⚙️ 检测参数设置")
    st.sidebar.markdown("---")
    
    # ---------- 模型选择 ----------
    st.sidebar.subheader("🧠 模型选择")
    
    # 自动扫描 runs/detect 下的所有模型权重文件
    model_options = {}
    runs_dir = './runs/detect'
    if os.path.isdir(runs_dir):
        for exp_name in os.listdir(runs_dir):
            weights_dir = os.path.join(runs_dir, exp_name, 'weights')
            if os.path.isdir(weights_dir):
                for weight_file in ['best.pt', 'last.pt']:
                    weight_path = os.path.join(weights_dir, weight_file)
                    if os.path.exists(weight_path):
                        display_name = f"{exp_name}/{weight_file}"
                        model_options[display_name] = weight_path
    
    # 如果没有找到任何模型，显示提示
    if not model_options:
        st.sidebar.warning("⚠️ 未找到训练好的模型，请先运行训练脚本")
        return None
    
    selected_model = st.sidebar.selectbox(
        "选择模型权重",
        options=list(model_options.keys()),
        help="选择要使用的模型权重文件"
    )
    model_path = model_options[selected_model]
    
    st.sidebar.markdown("---")
    
    # ---------- 检测参数 ----------
    st.sidebar.subheader("🎯 检测参数")
    
    # 置信度阈值滑块
    conf_threshold = st.sidebar.slider(
        "置信度阈值",
        min_value=0.05,
        max_value=0.95,
        value=0.25,          # 默认值
        step=0.05,
        help="低于此阈值的检测结果将被过滤。值越高，结果越严格（减少误检）；"
             "值越低，结果越宽松（减少漏检）。"
    )
    
    # IoU 阈值滑块
    iou_threshold = st.sidebar.slider(
        "IoU 阈值 (NMS)",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="非极大值抑制（NMS）的 IoU 阈值。"
             "值越小，重叠检测框过滤越激进；值越大，保留更多重叠框。"
    )
    
    st.sidebar.markdown("---")
    
    # ---------- 系统信息 ----------
    st.sidebar.subheader("ℹ️ 系统信息")
    st.sidebar.info(
        f"**当前模型**: {selected_model}\n\n"
        f"**类别数**: {len(CLASS_NAMES)}\n\n"
        f"**类别**: {', '.join(CLASS_NAMES)}"
    )
    
    return {
        'model_path': model_path,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold
    }


def render_detection_results(image: Image.Image, results, detections: list, image_name: str):
    """
    渲染单张图片的检测结果，包括标注图、详情表格和下载按钮。
    
    Args:
        image: 原始图片
        results: YOLO 推理结果
        detections: 检测详情列表
        image_name: 图片文件名
    """
    # 使用两列布局: 左侧原图 | 右侧检测结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📷 原始图片**")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("**🔍 检测结果**")
        # 绘制检测框到图片上
        for r in results:
            annotated = r.plot()  # 返回 BGR numpy 数组
            # BGR 转 RGB 用于 Streamlit 显示
            st.image(annotated[:, :, ::-1], use_container_width=True)
    
    # ---------- 检测详情表格 ----------
    if detections:
        st.markdown(f"**📋 检测详情 — 共检出 {len(detections)} 个目标**")
        df = pd.DataFrame(detections)
        
        # 使用 Streamlit 的原生表格，支持排序
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                '置信度': st.column_config.ProgressColumn(
                    '置信度',
                    min_value=0,
                    max_value=1,
                    format='%.4f'    # 显示4位小数
                )
            }
        )
        
        # ---------- 下载按钮 ----------
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            # 下载标注后的图片
            for r in results:
                annotated_img = r.plot()[:, :, ::-1]
                pil_img = Image.fromarray(annotated_img)
                buf = io.BytesIO()
                pil_img.save(buf, format='JPEG', quality=95)
                st.download_button(
                    label="📥 下载标注图片",
                    data=buf.getvalue(),
                    file_name=f"detected_{image_name}",
                    mime="image/jpeg"
                )
        
        with download_col2:
            # 下载 JSON 格式的检测结果
            json_result = {
                'image': image_name,
                'detections': detections,
                'total_count': len(detections),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            st.download_button(
                label="📥 下载 JSON 结果",
                data=json.dumps(json_result, ensure_ascii=False, indent=2),
                file_name=f"result_{image_name}.json",
                mime="application/json"
            )
    else:
        st.info("✅ 未检测到异常区域")


def render_statistics_panel(all_detections: list):
    """
    渲染检测统计面板：汇总当前会话中所有检测结果的统计信息。
    
    包括:
        - 各类别检出数量（饼图 + 柱状图）
        - 平均置信度统计
    
    Args:
        all_detections: 所有图片的检测结果合并列表
    """
    if not all_detections:
        return
    
    st.markdown("---")
    st.subheader("📊 检测统计面板")
    
    # 统计各类别数量
    df_all = pd.DataFrame(all_detections)
    class_counts = df_all['类别'].value_counts()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # 总体统计数字
        st.metric("检测总数", len(all_detections))
    with col2:
        st.metric("平均置信度", f"{df_all['置信度'].mean():.4f}")
    with col3:
        st.metric("涉及类别数", len(class_counts))
    
    # 饼图与柱状图并排
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**📊 类别分布饼图**")
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(6, 6))
        # 计算每个类别的颜色
        colors = []
        for name in class_counts.index:
            if name in CLASS_NAMES:
                colors.append(CLASS_COLORS[CLASS_NAMES.index(name)])
            else:
                colors.append('#999999')
        
        wedges, texts, autotexts = ax.pie(
            class_counts.values,
            labels=class_counts.index,
            colors=colors,
            autopct='%1.1f%%',   # 显示百分比
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax.set_title('各类别检出占比', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with chart_col2:
        st.markdown("**📊 类别数量柱状图**")
        # 使用 Streamlit 原生柱状图
        st.bar_chart(class_counts, color='#2196F3')
    
    # 每类别平均置信度
    st.markdown("**📊 各类别平均置信度**")
    avg_conf = df_all.groupby('类别')['置信度'].mean().sort_values(ascending=False)
    conf_df = pd.DataFrame({
        '类别': avg_conf.index,
        '平均置信度': avg_conf.values
    })
    st.dataframe(conf_df, use_container_width=True, hide_index=True)


def main():
    """
    应用主入口。
    
    整体流程:
        1. 渲染页面标题和说明
        2. 渲染侧边栏（参数设置）
        3. 文件上传区域（支持多文件）
        4. 对每张图片执行检测并展示结果
        5. 汇总统计面板
        6. 维护检测历史记录
    """
    
    # ==================== 页面标题 ====================
    st.title("🫁 肺炎 X 光片智能检测系统")
    st.markdown(
        "基于 **YOLOv8** 深度学习模型，自动检测 X 光片中的肺炎病灶区域。"
        "支持细菌性肺炎、病毒性肺炎、患病、健康、肺结核五分类检测。"
    )
    st.markdown("---")
    
    # ==================== 侧边栏 ====================
    config = render_sidebar()
    if config is None:
        st.stop()    # 没有可用模型，停止执行
    
    # 加载模型
    model = load_model(config['model_path'])
    if model is None:
        st.error("❌ 模型加载失败，请检查权重文件路径。")
        st.stop()
    
    # ==================== 初始化会话状态 ====================
    # st.session_state 在整个浏览器会话期间持久化
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []    # 检测历史记录
    if 'all_detections' not in st.session_state:
        st.session_state.all_detections = []       # 所有检测结果（用于统计）
    
    # ==================== 文件上传 ====================
    st.subheader("📤 上传 X 光片")
    uploaded_files = st.file_uploader(
        "选择一张或多张 X 光片进行检测",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,    # 支持批量上传
        help="支持 JPG、PNG、BMP 格式，可同时选择多张图片"
    )
    
    # ==================== 开始检测 ====================
    if uploaded_files:
        st.markdown("---")
        
        # 检测按钮
        if st.button("🚀 开始检测", type="primary", use_container_width=True):
            
            # 用于收集本次批量检测的所有结果
            batch_detections = []
            
            # 使用进度条显示批量检测进度
            progress_bar = st.progress(0, text="正在检测...")
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # 更新进度条
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress, text=f"正在检测: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
                
                # 打开图片
                image = Image.open(uploaded_file)
                
                # 执行检测
                results = detect_image(
                    model, image,
                    conf_threshold=config['conf_threshold'],
                    iou_threshold=config['iou_threshold']
                )
                
                # 提取检测详情
                detections = extract_detection_details(results)
                batch_detections.extend(detections)
                
                # 渲染单张结果
                st.markdown(f"### 📸 {uploaded_file.name}")
                render_detection_results(image, results, detections, uploaded_file.name)
                st.markdown("---")
                
                # 记录到检测历史
                st.session_state.detection_history.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'filename': uploaded_file.name,
                    'count': len(detections),
                    'conf_threshold': config['conf_threshold']
                })
            
            # 完成进度条
            progress_bar.progress(1.0, text="✅ 检测完成！")
            
            # 累积到全局统计
            st.session_state.all_detections.extend(batch_detections)
            
            # 显示统计面板
            render_statistics_panel(st.session_state.all_detections)
    
    # ==================== 检测历史记录 ====================
    if st.session_state.detection_history:
        st.markdown("---")
        st.subheader("📜 检测历史记录")
        
        history_df = pd.DataFrame(st.session_state.detection_history)
        history_df.columns = ['时间', '文件名', '检出数量', '置信度阈值']
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # 清空历史按钮
        if st.button("🗑️ 清空历史记录"):
            st.session_state.detection_history = []
            st.session_state.all_detections = []
            st.rerun()    # 刷新页面


if __name__ == '__main__':
    main()