"""
评估指标深度分析模块
======================

功能说明:
    对训练好的 YOLOv8 模型进行多维度评估分析，生成论文级别的图表。
    基于已有训练结果(results.csv)和 best.pt 模型进行分析。

生成内容 (输出到 outputs/eval/):
    1. training_curves.png      — 训练损失与指标收敛曲线
    2. per_class_metrics.png    — 每类别 Precision/Recall/F1 柱状对比图
    3. confusion_matrix.png     — 自定义混淆矩阵热力图
    4. confidence_analysis.png  — 各类别置信度分布箱线图
    5. error_analysis/          — 漏检与误检样本可视化
    6. eval_report.txt          — 评估报告文本

使用方式:
    cd workspace
    python src/evaluate.py
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ultralytics import YOLO
from collections import defaultdict

# ============================================================
# 全局配置
# ============================================================

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置（相对于 workspace 目录运行）
MODEL_PATH = './runs/detect/pneumonia_exp1/weights/best.pt'
RESULTS_CSV = './runs/detect/pneumonia_exp1/results.csv'
DATA_YAML = './data.yaml'
OUTPUT_DIR = './outputs/eval'

# 类别名称（与 data.yaml 保持一致）
CLASS_NAMES = ['Pneumonia Bacteria', 'Pneumonia Virus', 'Sick', 'healthy', 'tuberculosis']


def ensure_output_dir():
    """确保输出目录和子目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'error_analysis'), exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(OUTPUT_DIR)}")


def parse_results_csv(csv_path: str) -> dict:
    """
    解析 YOLO 训练产出的 results.csv 文件。
    
    该文件由 ultralytics 在训练过程中自动生成，每行对应一个 epoch 的指标。
    字段包括:
        - epoch: 当前epoch编号
        - train/box_loss, train/cls_loss, train/dfl_loss: 训练损失（三部分）
        - metrics/precision(B), metrics/recall(B): 精确率和召回率
        - metrics/mAP50(B), metrics/mAP50-95(B): mAP 指标
        - val/box_loss, val/cls_loss, val/dfl_loss: 验证损失
    
    Args:
        csv_path: results.csv 文件路径
    
    Returns:
        data: 字典，键为列名（去除空格），值为该列数值列表
    """
    data = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                # CSV 列名可能有前导/尾随空格，统一去除
                clean_key = key.strip()
                try:
                    data[clean_key].append(float(value))
                except (ValueError, TypeError):
                    data[clean_key].append(None)
    
    return dict(data)


def plot_training_curves(csv_data: dict):
    """
    绘制训练过程中的损失与指标收敛曲线。
    
    上排 3 张图: 训练损失（box_loss, cls_loss, dfl_loss）
    下排 3 张图: mAP50, mAP50-95, Precision & Recall
    
    这些曲线直观展示模型是否收敛，是否存在过拟合等问题。
    
    Args:
        csv_data: parse_results_csv() 返回的数据字典
    """
    print("📊 绘制训练收敛曲线...")
    
    epochs = csv_data.get('epoch', list(range(1, len(csv_data.get('train/box_loss', [])) + 1)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ---------- 上排: 损失曲线 ----------
    
    # 1. Box Loss（边界框定位损失）
    ax = axes[0, 0]
    train_box = csv_data.get('train/box_loss', [])
    val_box = csv_data.get('val/box_loss', [])
    if train_box:
        ax.plot(epochs, train_box, 'b-', linewidth=1.5, label='Train', alpha=0.8)
    if val_box:
        ax.plot(epochs, val_box, 'r--', linewidth=1.5, label='Val', alpha=0.8)
    ax.set_title('Box Loss (边界框损失)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Cls Loss（分类损失）
    ax = axes[0, 1]
    train_cls = csv_data.get('train/cls_loss', [])
    val_cls = csv_data.get('val/cls_loss', [])
    if train_cls:
        ax.plot(epochs, train_cls, 'b-', linewidth=1.5, label='Train', alpha=0.8)
    if val_cls:
        ax.plot(epochs, val_cls, 'r--', linewidth=1.5, label='Val', alpha=0.8)
    ax.set_title('Cls Loss (分类损失)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. DFL Loss（Distribution Focal Loss，YOLOv8 特有的边界框回归损失）
    ax = axes[0, 2]
    train_dfl = csv_data.get('train/dfl_loss', [])
    val_dfl = csv_data.get('val/dfl_loss', [])
    if train_dfl:
        ax.plot(epochs, train_dfl, 'b-', linewidth=1.5, label='Train', alpha=0.8)
    if val_dfl:
        ax.plot(epochs, val_dfl, 'r--', linewidth=1.5, label='Val', alpha=0.8)
    ax.set_title('DFL Loss (分布焦点损失)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ---------- 下排: 指标曲线 ----------
    
    # 4. mAP50
    ax = axes[1, 0]
    map50 = csv_data.get('metrics/mAP50(B)', [])
    if map50:
        ax.plot(epochs, map50, 'g-', linewidth=2, marker='', alpha=0.8)
        # 标注最大值点
        best_idx = np.argmax(map50)
        ax.annotate(f'最优: {map50[best_idx]:.4f}',
                    xy=(epochs[best_idx], map50[best_idx]),
                    xytext=(epochs[best_idx] + 2, map50[best_idx] - 0.02),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red', fontweight='bold')
    ax.set_title('mAP@0.5', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    
    # 5. mAP50-95
    ax = axes[1, 1]
    map50_95 = csv_data.get('metrics/mAP50-95(B)', [])
    if map50_95:
        ax.plot(epochs, map50_95, '#FF6F00', linewidth=2, alpha=0.8)
        best_idx = np.argmax(map50_95)
        ax.annotate(f'最优: {map50_95[best_idx]:.4f}',
                    xy=(epochs[best_idx], map50_95[best_idx]),
                    xytext=(epochs[best_idx] + 2, map50_95[best_idx] - 0.02),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red', fontweight='bold')
    ax.set_title('mAP@0.5:0.95', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP50-95')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    
    # 6. Precision & Recall（精确率与召回率）
    ax = axes[1, 2]
    precision = csv_data.get('metrics/precision(B)', [])
    recall = csv_data.get('metrics/recall(B)', [])
    if precision:
        ax.plot(epochs, precision, 'b-', linewidth=1.5, label='Precision', alpha=0.8)
    if recall:
        ax.plot(epochs, recall, 'r-', linewidth=1.5, label='Recall', alpha=0.8)
    ax.set_title('Precision & Recall', fontsize=11, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('值')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle('模型训练过程分析', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def run_evaluation(model_path: str, data_yaml: str):
    """
    使用 best.pt 在测试集/验证集上运行评估，获取详细指标。
    
    ultralytics 的 model.val() 方法会返回包含以下信息的结果对象:
        - box.maps: 每类的 mAP50-95
        - box.map50: 整体 mAP50
        - box.mp: 整体 Precision
        - box.mr: 整体 Recall
        - confusion_matrix: 混淆矩阵对象
    
    Args:
        model_path: 模型权重路径
        data_yaml: 数据配置文件路径
    
    Returns:
        results: YOLO 评估结果对象
    """
    print("🔍 加载模型并运行评估...")
    model = YOLO(model_path)
    
    # split='test' 在测试集上评估；如无测试集会自动回退到验证集
    results = model.val(data=data_yaml, split='test', verbose=False)
    
    return model, results


def plot_per_class_metrics(results):
    """
    绘制每个类别的 Precision / Recall / F1 柱状对比图。
    
    这张图可以直观看出模型在哪些类别上表现好/差，
    例如 "healthy" 类可能因为样本多而表现好，
    "tuberculosis" 可能因为样本少而表现差。
    
    Args:
        results: model.val() 返回的结果对象
    """
    print("📊 绘制每类别指标对比图...")
    
    # 获取每个类别的 AP50 值
    # results.box.ap50 是一个数组，每个元素对应一个类别的 AP@0.5
    ap50_per_class = results.box.ap50  # shape: (num_classes,)
    
    # 从结果中获取每类的 precision 和 recall
    # results.box.p 和 results.box.r 分别是每类的值
    p_per_class = results.box.p    # 每类 precision
    r_per_class = results.box.r    # 每类 recall
    
    # 计算 F1 = 2 * P * R / (P + R)
    f1_per_class = np.where(
        (p_per_class + r_per_class) > 0,
        2 * p_per_class * r_per_class / (p_per_class + r_per_class),
        0
    )
    
    # 绘制分组柱状图
    x = np.arange(len(CLASS_NAMES))
    bar_width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - bar_width * 1.5, p_per_class, bar_width, label='Precision', color='#2196F3')
    ax.bar(x - bar_width * 0.5, r_per_class, bar_width, label='Recall', color='#FF9800')
    ax.bar(x + bar_width * 0.5, f1_per_class, bar_width, label='F1-Score', color='#4CAF50')
    ax.bar(x + bar_width * 1.5, ap50_per_class, bar_width, label='AP@0.5', color='#9C27B0')
    
    # 在柱子顶部标注数值
    for bars in ax.containers:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('各类别检测性能指标对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'per_class_metrics.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 已保存: {save_path}")
    
    return p_per_class, r_per_class, f1_per_class, ap50_per_class


def plot_custom_confusion_matrix(results):
    """
    绘制自定义归一化混淆矩阵热力图。
    
    虽然 YOLO 训练时会自动生成混淆矩阵，但这里提供:
    1. 更清晰的中文标注
    2. 归一化百分比显示
    3. 适配论文的配色方案
    
    混淆矩阵的行表示真实类别，列表示预测类别。
    对角线上的值越大越好（表示正确分类）。
    
    Args:
        results: model.val() 返回的结果对象
    """
    print("📊 绘制自定义混淆矩阵...")
    
    # 获取混淆矩阵数据
    # results.confusion_matrix.matrix 是一个 (nc+1) x (nc+1) 的 numpy 数组
    # 最后一行/列表示 "background"（未检测到/误检）
    cm = results.confusion_matrix.matrix
    
    # 只取前 nc 个类别（不含 background），使图表更清晰
    nc = len(CLASS_NAMES)
    cm_classes = cm[:nc, :nc]
    
    # 归一化：按行（真实类别）归一化，使每行之和为 1
    row_sums = cm_classes.sum(axis=1, keepdims=True)
    # 避免除零
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_normalized = cm_classes / row_sums
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用 Blues 配色（蓝色系，适合学术论文）
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, label='归一化比例')
    
    # 在每个格子中显示数值
    for i in range(nc):
        for j in range(nc):
            # 颜色自适应（深色背景用白字，浅色背景用黑字）
            text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            # 同时显示归一化百分比和原始计数
            text = f'{cm_normalized[i, j]:.1%}\n({int(cm_classes[i, j])})'
            ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=9)
    
    ax.set_xticks(range(nc))
    ax.set_yticks(range(nc))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel('预测类别', fontsize=12, fontweight='bold')
    ax.set_ylabel('真实类别', fontsize=12, fontweight='bold')
    ax.set_title('归一化混淆矩阵', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_confidence_distribution(model, data_yaml: str):
    """
    绘制各类别检测结果的置信度分布箱线图。
    
    箱线图可以展示:
    - 中位数（箱内线）: 模型对该类别的平均确信程度
    - 四分位距（箱体范围）: 置信度的集中趋势
    - 异常值（散点）: 极端不确定的检测结果
    
    Args:
        model: 加载好的 YOLO 模型
        data_yaml: 数据配置文件路径
    """
    print("📊 绘制置信度分布箱线图...")
    
    # 在测试集上做推理，收集所有检测结果的置信度
    results = model.predict(
        source='./images/test',
        conf=0.1,        # 使用较低阈值以获取更完整的分布
        verbose=False,
        save=False
    )
    
    # 按类别收集置信度
    class_confidences = defaultdict(list)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id < len(CLASS_NAMES):
                class_confidences[cls_id].append(conf)
    
    # 准备箱线图数据
    box_data = []
    labels = []
    for cid in range(len(CLASS_NAMES)):
        confs = class_confidences.get(cid, [])
        if confs:
            box_data.append(confs)
            labels.append(f'{CLASS_NAMES[cid]}\n(n={len(confs)})')
        else:
            box_data.append([0])
            labels.append(f'{CLASS_NAMES[cid]}\n(n=0)')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制箱线图
    colors = ['#E53935', '#1E88E5', '#FDD835', '#43A047', '#8E24AA']
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    
    # 设置每个箱体的颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('置信度', fontsize=12)
    ax.set_title('各类别检测置信度分布', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'confidence_analysis.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def save_error_analysis(model, max_samples: int = 10):
    """
    自动筛选并保存错误案例的可视化图片。
    
    "错误案例" 包括两类:
        1. 低置信度检测: 模型虽然检出了但不太确定（conf < 0.5）
        2. 密集检测: 单张图片检出过多目标，可能存在误检
    
    这些案例有助于分析模型的薄弱环节，写论文时讨论模型的局限性。
    
    Args:
        model: 加载好的 YOLO 模型
        max_samples: 保存的最大样本数
    """
    print("📊 分析错误案例...")
    
    # 在测试集上推理
    results = model.predict(
        source='./images/test',
        conf=0.25,
        verbose=False,
        save=False
    )
    
    error_dir = os.path.join(OUTPUT_DIR, 'error_analysis')
    saved_count = 0
    
    # 收集低置信度和可能误检的样本
    for i, r in enumerate(results):
        if saved_count >= max_samples:
            break
        
        # 检查是否有低置信度检测（阈值 0.5 以下）
        has_low_conf = any(float(box.conf[0]) < 0.5 for box in r.boxes)
        # 检查是否检出过多目标（可能误检）
        has_many_boxes = len(r.boxes) > 3
        
        if has_low_conf or has_many_boxes:
            # 绘制检测结果并保存
            im_array = r.plot()   # 返回带标注的 numpy 数组（BGR 格式）
            
            save_path = os.path.join(error_dir, f'error_case_{saved_count + 1}.jpg')
            
            # 使用 matplotlib 保存（自动处理 BGR->RGB 转换）
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(im_array[:, :, ::-1])  # BGR 转 RGB
            
            # 构建标题信息
            box_info = []
            for box in r.boxes:
                cls_name = CLASS_NAMES[int(box.cls[0])] if int(box.cls[0]) < len(CLASS_NAMES) else 'unknown'
                conf = float(box.conf[0])
                box_info.append(f'{cls_name}: {conf:.2f}')
            title = '\n'.join(box_info[:5])  # 最多显示 5 个
            if len(box_info) > 5:
                title += f'\n... 还有 {len(box_info) - 5} 个检测结果'
            
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            saved_count += 1
    
    print(f"  ✅ 已保存 {saved_count} 个错误案例到: {error_dir}")


def generate_eval_report(csv_data: dict, results, p_per_class, r_per_class, f1_per_class, ap50_per_class):
    """
    生成评估报告文本文件。
    
    汇总所有评估指标，格式化为可直接用于论文的文本。
    
    Args:
        csv_data: 训练 CSV 数据
        results: 评估结果对象
        p_per_class: 每类 precision
        r_per_class: 每类 recall
        f1_per_class: 每类 F1 值
        ap50_per_class: 每类 AP50 值
    """
    print("📝 生成评估报告...")
    
    lines = []
    lines.append("=" * 65)
    lines.append("           YOLOv8 肺炎检测模型评估报告")
    lines.append("=" * 65)
    
    # --- 训练概况 ---
    map50 = csv_data.get('metrics/mAP50(B)', [])
    map50_95 = csv_data.get('metrics/mAP50-95(B)', [])
    
    lines.append(f"\n--- 训练概况 ---")
    lines.append(f"  训练总轮数: {len(map50)}")
    lines.append(f"  最优 mAP@0.5:     {max(map50):.4f} (Epoch {np.argmax(map50) + 1})")
    if map50_95:
        lines.append(f"  最优 mAP@0.5:0.95: {max(map50_95):.4f} (Epoch {np.argmax(map50_95) + 1})")
    
    # --- 整体指标 ---
    lines.append(f"\n--- 模型整体性能 ---")
    lines.append(f"  Precision:    {results.box.mp:.4f}")
    lines.append(f"  Recall:       {results.box.mr:.4f}")
    lines.append(f"  mAP@0.5:      {results.box.map50:.4f}")
    lines.append(f"  mAP@0.5:0.95: {results.box.map:.4f}")
    
    # --- 每类指标表格 ---
    lines.append(f"\n--- 各类别详细指标 ---")
    lines.append(f"  {'类别':<22s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'AP@0.5':>10s}")
    lines.append(f"  {'-'*62}")
    
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {name:<22s} {p_per_class[i]:>10.4f} {r_per_class[i]:>10.4f} "
                      f"{f1_per_class[i]:>10.4f} {ap50_per_class[i]:>10.4f}")
    
    # 均值行
    lines.append(f"  {'-'*62}")
    lines.append(f"  {'平均':<22s} {np.mean(p_per_class):>10.4f} {np.mean(r_per_class):>10.4f} "
                  f"{np.mean(f1_per_class):>10.4f} {np.mean(ap50_per_class):>10.4f}")
    
    lines.append("\n" + "=" * 65)
    
    # 写入文件
    report_text = '\n'.join(lines)
    save_path = os.path.join(OUTPUT_DIR, 'eval_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n  ✅ 报告已保存: {save_path}")


def main():
    """
    评估分析主流程:
        1. 解析训练日志 (results.csv)
        2. 绘制训练收敛曲线
        3. 加载模型运行评估
        4. 绘制每类指标、混淆矩阵、置信度分布
        5. 错误案例分析
        6. 生成评估报告
    """
    print("=" * 50)
    print("📋 开始模型评估分析")
    print("=" * 50)
    
    # 1. 检查必要文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("   请先运行 train.py 完成训练！")
        return
    
    if not os.path.exists(RESULTS_CSV):
        print(f"❌ 训练日志不存在: {RESULTS_CSV}")
        return
    
    ensure_output_dir()
    
    # 2. 解析训练日志并绘制收敛曲线
    print("\n📂 解析训练日志...")
    csv_data = parse_results_csv(RESULTS_CSV)
    print(f"  共 {len(csv_data.get('epoch', []))} 个 epoch 的记录")
    
    plot_training_curves(csv_data)
    
    # 3. 加载模型并运行评估
    model, results = run_evaluation(MODEL_PATH, DATA_YAML)
    
    # 4. 绘制详细指标图表
    p, r, f1, ap50 = plot_per_class_metrics(results)
    plot_custom_confusion_matrix(results)
    plot_confidence_distribution(model, DATA_YAML)
    
    # 5. 错误案例分析
    save_error_analysis(model, max_samples=10)
    
    # 6. 生成报告
    print()
    generate_eval_report(csv_data, results, p, r, f1, ap50)
    
    print(f"\n🎉 评估分析完成！所有结果已保存到: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()
