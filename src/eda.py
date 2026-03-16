"""
数据集探索性分析 (EDA) 模块
============================

功能说明:
    对肺炎检测数据集进行全面的统计分析与可视化，帮助理解数据分布特征。
    
生成内容 (输出到 outputs/eda/):
    1. class_distribution.png  — 各类别样本数量分布柱状图
    2. bbox_size_scatter.png   — 标注框宽高分布散点图
    3. bbox_ratio_hist.png     — 标注框宽高比分布直方图
    4. bbox_heatmap.png        — 病灶位置热力图（所有bbox中心叠加）
    5. image_size_dist.png     — 图像分辨率分布直方图
    6. eda_summary.txt         — 数据集统计摘要文本

使用方式:
    cd workspace
    python src/eda.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from collections import defaultdict, Counter

# ============================================================
# 全局配置
# ============================================================

# 使用非交互式后端，避免弹窗（服务器/无显示器环境也能运行）
matplotlib.use('Agg')

# 设置中文字体支持（Windows 下使用微软雅黑）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 数据集根目录（相对于 workspace 目录运行）
IMAGES_DIR = './images'
LABELS_DIR = './labels'
OUTPUT_DIR = './outputs/eda'

# 类别名称映射（与 data.yaml 中的定义保持一致）
CLASS_NAMES = {
    0: 'Pneumonia Bacteria',   # 细菌性肺炎
    1: 'Pneumonia Virus',      # 病毒性肺炎
    2: 'Sick',                 # 患病（其他）
    3: 'healthy',              # 健康
    4: 'tuberculosis'          # 肺结核
}


def ensure_output_dir():
    """
    确保输出目录存在，不存在则自动创建。
    使用 exist_ok=True 避免目录已存在时报错。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 输出目录: {os.path.abspath(OUTPUT_DIR)}")


def parse_all_labels(split: str):
    """
    解析指定数据子集（train/val/test）中的所有标签文件。
    
    YOLO 标签格式说明:
        每行一个目标: <class_id> <center_x> <center_y> <width> <height>
        其中坐标均为归一化值（0~1），相对于图像宽高。
    
    Args:
        split: 数据子集名称，可选 'train', 'val', 'test'
    
    Returns:
        labels: 列表，每个元素为字典:
            {'class_id': int, 'cx': float, 'cy': float, 'w': float, 'h': float}
    """
    labels_dir = os.path.join(LABELS_DIR, split)
    labels = []  # 存储解析后的所有标注信息
    
    # 如果目录不存在，打印警告并返回空列表
    if not os.path.isdir(labels_dir):
        print(f"  ⚠️ 标签目录不存在: {labels_dir}")
        return labels
    
    # 遍历所有 .txt 标签文件
    for txt_file in glob.glob(os.path.join(labels_dir, '*.txt')):
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                # YOLO 格式每行恰好有 5 个数值
                if len(parts) == 5:
                    try:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        labels.append({
                            'class_id': class_id,
                            'cx': cx,   # 中心点 x（归一化）
                            'cy': cy,   # 中心点 y（归一化）
                            'w': w,     # 宽度（归一化）
                            'h': h      # 高度（归一化）
                        })
                    except ValueError:
                        # 跳过格式异常的行
                        continue
    
    return labels


def get_image_sizes(split: str):
    """
    获取指定数据子集中所有图像的宽和高（像素值）。
    
    Args:
        split: 数据子集名称
    
    Returns:
        sizes: 列表，每个元素为 (width, height) 元组
    """
    images_dir = os.path.join(IMAGES_DIR, split)
    sizes = []
    
    if not os.path.isdir(images_dir):
        print(f"  ⚠️ 图像目录不存在: {images_dir}")
        return sizes
    
    # 支持常见的图像格式
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    for img_path in image_files:
        try:
            # 使用 PIL 读取图像尺寸（不加载像素数据，速度快）
            with Image.open(img_path) as img:
                sizes.append(img.size)  # (width, height)
        except Exception:
            continue
    
    return sizes


def plot_class_distribution(all_labels_by_split: dict):
    """
    绘制各类别在 train/val/test 中的样本数量分布柱状图。
    
    使用分组柱状图，每组对应一个类别，每根柱子对应一个数据子集，
    方便直观比较各子集的类别分布是否均衡。
    
    Args:
        all_labels_by_split: 字典，键为 split 名称，值为该 split 的标签列表
    """
    print("📊 绘制类别分布图...")
    
    # 统计每个 split 中各类别的计数
    split_class_counts = {}
    for split, labels in all_labels_by_split.items():
        counter = Counter([lb['class_id'] for lb in labels])
        split_class_counts[split] = counter
    
    # 准备绘图数据
    class_ids = sorted(CLASS_NAMES.keys())
    class_labels = [CLASS_NAMES[cid] for cid in class_ids]
    splits = list(all_labels_by_split.keys())
    
    # 柱状图位置计算
    x = np.arange(len(class_ids))          # 类别位置
    bar_width = 0.25                        # 每根柱子的宽度
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 为每个 split 绘制一组柱子
    colors = ['#2196F3', '#FF9800', '#4CAF50']  # 蓝/橙/绿 配色
    for i, split in enumerate(splits):
        counts = [split_class_counts[split].get(cid, 0) for cid in class_ids]
        offset = (i - len(splits) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, counts, bar_width, label=split, color=colors[i % 3])
        
        # 在柱子顶部标注具体数值
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(count), ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('标注框数量', fontsize=12)
    ax.set_title('各类别标注数量分布 (按数据子集)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=15, ha='right')
    ax.legend(title='数据子集')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_bbox_size_scatter(all_labels: list):
    """
    绘制标注框宽高分布散点图。
    
    每个点代表一个标注框，x 轴为宽度，y 轴为高度（均为归一化值 0~1）。
    不同类别用不同颜色区分，可以观察各类病灶的大小分布特征。
    
    Args:
        all_labels: 所有 split 合并后的标签列表
    """
    print("📊 绘制标注框尺寸散点图...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 为每个类别分别绘制散点
    colors = ['#E53935', '#1E88E5', '#FDD835', '#43A047', '#8E24AA']
    for cid in sorted(CLASS_NAMES.keys()):
        # 筛选当前类别的所有标注框
        ws = [lb['w'] for lb in all_labels if lb['class_id'] == cid]
        hs = [lb['h'] for lb in all_labels if lb['class_id'] == cid]
        if ws:
            ax.scatter(ws, hs, alpha=0.4, s=15, label=CLASS_NAMES[cid],
                       color=colors[cid % len(colors)])
    
    ax.set_xlabel('标注框宽度 (归一化)', fontsize=12)
    ax.set_ylabel('标注框高度 (归一化)', fontsize=12)
    ax.set_title('标注框宽高分布散点图', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'bbox_size_scatter.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_bbox_ratio_hist(all_labels: list):
    """
    绘制标注框宽高比分布直方图。
    
    宽高比 = width / height，反映标注框的形状特征。
    宽高比 > 1 表示横向矩形，< 1 表示纵向矩形，= 1 表示正方形。
    
    Args:
        all_labels: 所有 split 合并后的标签列表
    """
    print("📊 绘制宽高比分布图...")
    
    # 计算宽高比（避免除零：高度为 0 时跳过）
    ratios = [lb['w'] / lb['h'] for lb in all_labels if lb['h'] > 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(ratios, bins=50, color='#42A5F5', edgecolor='white', alpha=0.8)
    
    # 标注均值线
    mean_ratio = np.mean(ratios)
    ax.axvline(mean_ratio, color='#E53935', linestyle='--', linewidth=2,
               label=f'均值 = {mean_ratio:.2f}')
    
    ax.set_xlabel('宽高比 (width / height)', fontsize=12)
    ax.set_ylabel('标注框数量', fontsize=12)
    ax.set_title('标注框宽高比分布', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'bbox_ratio_hist.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_bbox_heatmap(all_labels: list):
    """
    绘制病灶位置热力图。
    
    将所有标注框的中心点坐标（归一化到 0~1）映射到一张 100x100 的网格上，
    统计每个格子的落点数量，用颜色深浅表示密度。
    通过热力图可以观察病灶在 X 光片中的位置偏好（如是否集中在肺部中央）。
    
    Args:
        all_labels: 所有 split 合并后的标签列表
    """
    print("📊 绘制病灶位置热力图...")
    
    # 创建 100x100 的密度网格
    grid_size = 100
    heatmap = np.zeros((grid_size, grid_size))
    
    for lb in all_labels:
        # 将归一化坐标映射到网格索引（clamp 防止越界）
        gx = min(int(lb['cx'] * grid_size), grid_size - 1)
        gy = min(int(lb['cy'] * grid_size), grid_size - 1)
        heatmap[gy, gx] += 1  # 注意: y 轴对应行，x 轴对应列
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 使用 'hot' 色彩映射：黑→红→黄→白，密度越高颜色越亮
    im = ax.imshow(heatmap, cmap='hot', interpolation='gaussian')
    plt.colorbar(im, ax=ax, label='标注框密度')
    
    ax.set_xlabel('X 方向（归一化）', fontsize=12)
    ax.set_ylabel('Y 方向（归一化）', fontsize=12)
    ax.set_title('标注框中心位置热力图', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'bbox_heatmap.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def plot_image_size_distribution(all_sizes: list):
    """
    绘制图像分辨率分布直方图。
    
    分别展示图像宽度和高度的分布情况。如果所有图像尺寸一致，
    说明数据集已经过预处理（统一 resize）。
    
    Args:
        all_sizes: 列表，每个元素为 (width, height) 元组
    """
    print("📊 绘制图像尺寸分布图...")
    
    widths = [s[0] for s in all_sizes]
    heights = [s[1] for s in all_sizes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 宽度分布
    ax1.hist(widths, bins=30, color='#42A5F5', edgecolor='white', alpha=0.8)
    ax1.set_xlabel('图像宽度 (像素)', fontsize=12)
    ax1.set_ylabel('图像数量', fontsize=12)
    ax1.set_title('图像宽度分布', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 高度分布
    ax2.hist(heights, bins=30, color='#FF7043', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('图像高度 (像素)', fontsize=12)
    ax2.set_ylabel('图像数量', fontsize=12)
    ax2.set_title('图像高度分布', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('图像分辨率分布统计', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'image_size_dist.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 已保存: {save_path}")


def generate_summary(all_labels_by_split: dict, all_sizes: list):
    """
    生成数据集统计摘要文本文件。
    
    包含总体数据量、各子集各类别的标注数、图像尺寸统计等信息，
    可直接复制到毕设论文中使用。
    
    Args:
        all_labels_by_split: 按 split 分组的标签数据
        all_sizes: 所有图像的尺寸列表
    """
    print("📝 生成统计摘要...")
    
    lines = []
    lines.append("=" * 60)
    lines.append("         肺炎检测数据集 EDA 统计摘要")
    lines.append("=" * 60)
    
    # --- 各子集统计 ---
    total_images = 0
    total_annotations = 0
    
    for split in ['train', 'val', 'test']:
        labels = all_labels_by_split.get(split, [])
        # 统计图像数（通过标签文件夹中文件数量近似）
        label_dir = os.path.join(LABELS_DIR, split)
        if os.path.isdir(label_dir):
            n_files = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        else:
            n_files = 0
        
        total_images += n_files
        total_annotations += len(labels)
        
        lines.append(f"\n--- {split} 子集 ---")
        lines.append(f"  图像数量: {n_files}")
        lines.append(f"  标注框总数: {len(labels)}")
        
        # 按类别统计
        counter = Counter([lb['class_id'] for lb in labels])
        for cid in sorted(CLASS_NAMES.keys()):
            count = counter.get(cid, 0)
            lines.append(f"    {CLASS_NAMES[cid]}: {count}")
    
    lines.append(f"\n--- 总计 ---")
    lines.append(f"  总图像数: {total_images}")
    lines.append(f"  总标注框数: {total_annotations}")
    
    # --- 标注框尺寸统计 ---
    all_labels = []
    for labels in all_labels_by_split.values():
        all_labels.extend(labels)
    
    if all_labels:
        ws = [lb['w'] for lb in all_labels]
        hs = [lb['h'] for lb in all_labels]
        ratios = [lb['w'] / lb['h'] for lb in all_labels if lb['h'] > 0]
        
        lines.append(f"\n--- 标注框统计 ---")
        lines.append(f"  宽度: 均值={np.mean(ws):.4f}, 中位数={np.median(ws):.4f}, "
                      f"最小={np.min(ws):.4f}, 最大={np.max(ws):.4f}")
        lines.append(f"  高度: 均值={np.mean(hs):.4f}, 中位数={np.median(hs):.4f}, "
                      f"最小={np.min(hs):.4f}, 最大={np.max(hs):.4f}")
        lines.append(f"  宽高比: 均值={np.mean(ratios):.4f}, 中位数={np.median(ratios):.4f}")
    
    # --- 图像尺寸统计 ---
    if all_sizes:
        widths = [s[0] for s in all_sizes]
        heights = [s[1] for s in all_sizes]
        
        lines.append(f"\n--- 图像尺寸统计 ---")
        lines.append(f"  宽度: 均值={np.mean(widths):.0f}, "
                      f"最小={min(widths)}, 最大={max(widths)}")
        lines.append(f"  高度: 均值={np.mean(heights):.0f}, "
                      f"最小={min(heights)}, 最大={max(heights)}")
        
        # 检查是否所有图像尺寸相同
        unique_sizes = set(all_sizes)
        if len(unique_sizes) == 1:
            w, h = unique_sizes.pop()
            lines.append(f"  所有图像尺寸统一: {w} x {h}")
        else:
            lines.append(f"  存在 {len(unique_sizes)} 种不同分辨率")
    
    lines.append("\n" + "=" * 60)
    
    # 写入文件
    summary_text = '\n'.join(lines)
    save_path = os.path.join(OUTPUT_DIR, 'eda_summary.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    # 同时打印到控制台
    print(summary_text)
    print(f"\n  ✅ 摘要已保存: {save_path}")


def main():
    """
    EDA 主流程:
        1. 创建输出目录
        2. 解析所有标签文件
        3. 获取图像尺寸信息
        4. 依次生成各项可视化图表
        5. 输出统计摘要
    """
    print("=" * 50)
    print("🔍 开始数据集探索性分析 (EDA)")
    print("=" * 50)
    
    # 1. 创建输出目录
    ensure_output_dir()
    
    # 2. 解析所有标签
    print("\n📂 解析标签文件...")
    all_labels_by_split = {}
    for split in ['train', 'val', 'test']:
        labels = parse_all_labels(split)
        all_labels_by_split[split] = labels
        print(f"  {split}: {len(labels)} 个标注框")
    
    # 合并所有标签（用于全局统计）
    all_labels = []
    for labels in all_labels_by_split.values():
        all_labels.extend(labels)
    print(f"  总计: {len(all_labels)} 个标注框")
    
    # 3. 获取图像尺寸（采样 train 子集，避免全量读取太慢）
    print("\n📐 获取图像尺寸信息...")
    all_sizes = []
    for split in ['train', 'val', 'test']:
        sizes = get_image_sizes(split)
        all_sizes.extend(sizes)
        print(f"  {split}: {len(sizes)} 张图像")
    
    # 4. 生成可视化图表
    print("\n🎨 生成可视化图表...")
    plot_class_distribution(all_labels_by_split)    # 类别分布
    plot_bbox_size_scatter(all_labels)               # 标注框尺寸
    plot_bbox_ratio_hist(all_labels)                 # 宽高比
    plot_bbox_heatmap(all_labels)                    # 位置热力图
    plot_image_size_distribution(all_sizes)           # 图像尺寸
    
    # 5. 统计摘要
    print()
    generate_summary(all_labels_by_split, all_sizes)
    
    print("\n🎉 EDA 分析完成！所有图表已保存到:", os.path.abspath(OUTPUT_DIR))


if __name__ == '__main__':
    main()
