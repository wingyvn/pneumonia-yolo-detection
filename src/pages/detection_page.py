"""
智能检测页
===========

功能:
    - 拖拽/按钮上传 X 光片
    - 左右分栏: 原图 ↔ 检测标注图
    - 检测结果详情面板（类别、置信度进度条、bbox 坐标）
    - 批量检测（选择文件夹）
    - 检测历史列表
    - 单次检测可视化（热力图、面积占比、检测能力柱状图）+ 紫色导出按钮
"""

import os
import json
import glob
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QScrollArea, QFrame, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QGroupBox, QSizePolicy, QMessageBox, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from utils.detector import PneumoniaDetector, find_available_models, CLASS_NAMES, CLASS_NAMES_CN, CLASS_COLORS
from utils.db_manager import save_detection_record, get_user_settings


def _fig_to_qpixmap(fig) -> QPixmap:
    """将 matplotlib Figure 转换为 QPixmap（用于在 PyQt 中显示图表）"""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    # 获取 RGBA 缓冲区
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    qimg = QImage(buf, w, h, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


class DetectionPage(QWidget):
    """
    智能检测页面。

    信号:
        detection_completed: 检测完成时发射，通知可视化页面刷新
    """

    detection_completed = pyqtSignal()

    def __init__(self, username: str):
        super().__init__()
        self.username = username
        self.detector = PneumoniaDetector()
        self.current_detections = []        # 当前检测结果
        self.current_image_path = None      # 当前图片路径

        self._load_model()
        self._init_ui()

    def _load_model(self):
        """加载模型"""
        # 优先使用用户设置中的模型路径
        settings = get_user_settings(self.username)
        model_path = settings.get('model_path', '')

        if not model_path or not os.path.exists(model_path):
            # 自动查找可用模型
            models = find_available_models()
            if models:
                model_path = list(models.values())[0]

        if model_path:
            self.detector.load_model(model_path)

    def _init_ui(self):
        """构建检测页面布局"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)

        # ========== 顶部工具栏 ==========
        toolbar = QHBoxLayout()

        self.btn_select = QPushButton("📁 选择图片")
        self.btn_select.setObjectName("toolButton")
        self.btn_select.clicked.connect(self._select_image)
        toolbar.addWidget(self.btn_select)

        self.btn_batch = QPushButton("📂 批量检测")
        self.btn_batch.setObjectName("toolButton")
        self.btn_batch.clicked.connect(self._batch_detect)
        toolbar.addWidget(self.btn_batch)

        toolbar.addStretch()

        # 模型状态标签
        model_status = "✅ 模型已加载" if self.detector.is_loaded() else "❌ 模型未加载"
        self.model_label = QLabel(model_status)
        self.model_label.setStyleSheet("color: #666; font-size: 12px;")
        toolbar.addWidget(self.model_label)

        layout.addLayout(toolbar)

        # ========== 主内容区 (上下分割) ==========
        splitter = QSplitter(Qt.Vertical)

        # --- 上半部: 图像对比区 ---
        image_area = QWidget()
        image_layout = QHBoxLayout(image_area)
        image_layout.setSpacing(15)

        # 左: 原图
        left_group = QGroupBox("📷 原始 X 光片")
        left_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        left_inner = QVBoxLayout(left_group)
        self.label_original = QLabel("请上传 X 光片图像")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setMinimumSize(400, 350)
        self.label_original.setStyleSheet(
            "background: #FAFAFA; border: 2px dashed #CCC; border-radius: 10px; "
            "color: #999; font-size: 14px;"
        )
        left_inner.addWidget(self.label_original)
        image_layout.addWidget(left_group)

        # 右: 检测结果图
        right_group = QGroupBox("🔍 检测结果")
        right_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        right_inner = QVBoxLayout(right_group)
        self.label_result = QLabel("检测结果将显示在这里")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setMinimumSize(400, 350)
        self.label_result.setStyleSheet(
            "background: #FAFAFA; border: 2px dashed #CCC; border-radius: 10px; "
            "color: #999; font-size: 14px;"
        )
        right_inner.addWidget(self.label_result)
        image_layout.addWidget(right_group)

        splitter.addWidget(image_area)

        # --- 下半部: 结果详情 + 单次检测可视化 ---
        bottom_tabs = QTabWidget()
        bottom_tabs.setStyleSheet("QTabWidget { font-size: 12px; }")

        # Tab 1: 检测详情表格
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        self.result_table = QTableWidget(0, 6)
        self.result_table.setHorizontalHeaderLabels(
            ['类别', '中文名', '置信度', 'Bbox (x1,y1,x2,y2)', '面积占比', '置信度条']
        )
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setStyleSheet(
            "QTableWidget { background: white; gridline-color: #E0E0E0; }"
            "QHeaderView::section { background: #E3F2FD; font-weight: bold; padding: 6px; }"
        )
        detail_layout.addWidget(self.result_table)
        bottom_tabs.addTab(detail_widget, "📋 检测详情")

        # Tab 2: 单次检测可视化
        vis_widget = QWidget()
        vis_layout = QVBoxLayout(vis_widget)

        # 三张图表水平排列
        charts_layout = QHBoxLayout()
        self.chart_heatmap = QLabel("待检测")
        self.chart_heatmap.setAlignment(Qt.AlignCenter)
        self.chart_heatmap.setMinimumHeight(220)
        self.chart_heatmap.setStyleSheet("background: white; border: 1px solid #E0E0E0; border-radius: 6px;")
        charts_layout.addWidget(self.chart_heatmap)

        self.chart_area = QLabel("待检测")
        self.chart_area.setAlignment(Qt.AlignCenter)
        self.chart_area.setMinimumHeight(220)
        self.chart_area.setStyleSheet("background: white; border: 1px solid #E0E0E0; border-radius: 6px;")
        charts_layout.addWidget(self.chart_area)

        self.chart_capability = QLabel("待检测")
        self.chart_capability.setAlignment(Qt.AlignCenter)
        self.chart_capability.setMinimumHeight(220)
        self.chart_capability.setStyleSheet("background: white; border: 1px solid #E0E0E0; border-radius: 6px;")
        charts_layout.addWidget(self.chart_capability)

        vis_layout.addLayout(charts_layout)

        # 紫色导出按钮
        export_layout = QHBoxLayout()
        export_layout.addStretch()

        self.btn_export_heatmap = QPushButton("导出热力图")
        self.btn_export_heatmap.setObjectName("exportButton")
        self.btn_export_heatmap.clicked.connect(lambda: self._export_chart('heatmap'))
        export_layout.addWidget(self.btn_export_heatmap)

        self.btn_export_area = QPushButton("导出面积占比")
        self.btn_export_area.setObjectName("exportButton")
        self.btn_export_area.clicked.connect(lambda: self._export_chart('area'))
        export_layout.addWidget(self.btn_export_area)

        self.btn_export_capability = QPushButton("导出检测能力")
        self.btn_export_capability.setObjectName("exportButton")
        self.btn_export_capability.clicked.connect(lambda: self._export_chart('capability'))
        export_layout.addWidget(self.btn_export_capability)

        export_layout.addStretch()
        vis_layout.addLayout(export_layout)

        bottom_tabs.addTab(vis_widget, "📊 检测可视化")

        # Tab 3: 对照验证（真实标注 vs 模型预测）
        compare_widget = QWidget()
        compare_layout = QVBoxLayout(compare_widget)

        compare_title = QLabel("📝 自动对照验证: 模型预测 vs 真实标注")
        compare_title.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1A237E; "
            "font-family: 'Microsoft YaHei'; margin-bottom: 5px;"
        )
        compare_layout.addWidget(compare_title)

        compare_desc = QLabel(
            "系统自动查找上传图片对应的真实标注文件（labels/test 或 labels/val），"
            "将医学专家标注与模型预测结果进行自动比对。"
        )
        compare_desc.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        compare_desc.setWordWrap(True)
        compare_layout.addWidget(compare_desc)

        # 对照结果显示区域
        self.compare_result = QLabel("请先执行检测")
        self.compare_result.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.compare_result.setWordWrap(True)
        self.compare_result.setMinimumHeight(180)
        self.compare_result.setStyleSheet(
            "background: white; border: 1px solid #E0E0E0; border-radius: 8px; "
            "padding: 15px; font-size: 13px; font-family: 'Microsoft YaHei'; line-height: 1.8;"
        )
        compare_layout.addWidget(self.compare_result)

        bottom_tabs.addTab(compare_widget, "✅ 对照验证")

        splitter.addWidget(bottom_tabs)
        splitter.setSizes([450, 300])

        layout.addWidget(splitter)

        # 应用页面样式
        self.setStyleSheet(self._get_stylesheet())

    # ============================================================
    # 图片选择与检测
    # ============================================================

    def _select_image(self):
        """打开文件对话框选择单张图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 X 光片",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if file_path:
            self._run_detection(file_path)

    def _batch_detect(self):
        """批量检测: 选择文件夹并处理所有图片"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return

        # 收集所有图片文件
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(extensions)
        ]

        if not image_files:
            QMessageBox.information(self, "提示", "所选文件夹中没有图片文件")
            return

        # 逐张处理
        for i, file_path in enumerate(image_files):
            self._run_detection(file_path)

        QMessageBox.information(
            self, "批量检测完成",
            f"共处理 {len(image_files)} 张图片，结果已保存到数据库"
        )

    def _run_detection(self, image_path: str):
        """
        对单张图片执行检测并更新界面。

        Args:
            image_path: 图片文件路径
        """
        if not self.detector.is_loaded():
            QMessageBox.warning(self, "错误", "模型未加载，请在设置页指定模型路径")
            return

        self.current_image_path = image_path

        # 获取用户设置的阈值
        settings = get_user_settings(self.username)
        conf = settings.get('conf_threshold', 0.25)
        iou = settings.get('iou_threshold', 0.45)

        from PIL import Image
        pil_image = Image.open(image_path)

        # 执行检测
        detections, annotated = self.detector.detect(pil_image, conf=conf, iou=iou)
        self.current_detections = detections

        # ---------- 更新原图显示 ----------
        pixmap_orig = QPixmap(image_path)
        scaled_orig = pixmap_orig.scaled(
            self.label_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label_original.setPixmap(scaled_orig)
        self.label_original.setStyleSheet("background: #FAFAFA; border-radius: 10px;")

        # ---------- 更新检测结果图 ----------
        if annotated is not None:
            h, w, ch = annotated.shape
            qimg = QImage(annotated.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap_result = QPixmap.fromImage(qimg)
            scaled_result = pixmap_result.scaled(
                self.label_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.label_result.setPixmap(scaled_result)
            self.label_result.setStyleSheet("background: #FAFAFA; border-radius: 10px;")
        else:
            self.label_result.setText("✅ 未检测到异常")

        # ---------- 更新详情表格 ----------
        self._update_result_table(detections)

        # ---------- 更新单次检测可视化图表 ----------
        self._update_detection_charts(detections)

        # ---------- 更新对照验证 ----------
        self._update_comparison(image_path, detections)

        # ---------- 保存检测记录到数据库 ----------
        save_detection_record(
            username=self.username,
            image_name=os.path.basename(image_path),
            image_path=image_path,
            detections=detections,
            conf_threshold=conf,
            iou_threshold=iou
        )

        # 通知可视化页面刷新
        self.detection_completed.emit()

    # ============================================================
    # 界面更新方法
    # ============================================================

    def _update_result_table(self, detections: list):
        """更新检测详情表格"""
        self.result_table.setRowCount(len(detections))

        for row, det in enumerate(detections):
            # 类别（英文）
            self.result_table.setItem(row, 0, QTableWidgetItem(det['class_name']))

            # 类别（中文）
            self.result_table.setItem(row, 1, QTableWidgetItem(det['class_name_cn']))

            # 置信度数值
            conf_item = QTableWidgetItem(f"{det['confidence']:.4f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(row, 2, conf_item)

            # Bbox 坐标
            bbox = det['bbox']
            bbox_str = f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
            self.result_table.setItem(row, 3, QTableWidgetItem(bbox_str))

            # 面积占比
            area_item = QTableWidgetItem(f"{det['area_ratio']:.2%}")
            area_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(row, 4, area_item)

            # 置信度进度条
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(det['confidence'] * 100))
            bar.setFormat(f"{det['confidence']:.1%}")
            # 根据置信度设置颜色
            if det['confidence'] >= 0.8:
                bar.setStyleSheet("QProgressBar::chunk { background: #43A047; }")
            elif det['confidence'] >= 0.5:
                bar.setStyleSheet("QProgressBar::chunk { background: #FF9800; }")
            else:
                bar.setStyleSheet("QProgressBar::chunk { background: #E53935; }")
            self.result_table.setCellWidget(row, 5, bar)

    def _update_detection_charts(self, detections: list):
        """
        为当前检测结果生成三张可视化图表。
        """
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # --- 1. 目标位置热力图 ---
        self._current_fig_heatmap = self._create_heatmap_chart(detections)
        self.chart_heatmap.setPixmap(
            _fig_to_qpixmap(self._current_fig_heatmap).scaled(
                self.chart_heatmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        plt.close(self._current_fig_heatmap)

        # --- 2. 目标面积占比 ---
        self._current_fig_area = self._create_area_chart(detections)
        self.chart_area.setPixmap(
            _fig_to_qpixmap(self._current_fig_area).scaled(
                self.chart_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        plt.close(self._current_fig_area)

        # --- 3. 检测能力柱状图 ---
        self._current_fig_capability = self._create_capability_chart(detections)
        self.chart_capability.setPixmap(
            _fig_to_qpixmap(self._current_fig_capability).scaled(
                self.chart_capability.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        plt.close(self._current_fig_capability)

    # ============================================================
    # 图表生成方法
    # ============================================================

    def _create_heatmap_chart(self, detections: list):
        """
        目标位置热力图: 将检测到的 bbox 中心映射到图像坐标系，
        用热力图展示病灶分布位置。
        """
        fig, ax = plt.subplots(figsize=(4, 3.2), dpi=100)

        if detections:
            # 提取归一化中心坐标
            xs = [d['bbox_norm'][0] for d in detections]
            ys = [d['bbox_norm'][1] for d in detections]

            # 创建热力网格
            grid_size = 50
            heatmap = np.zeros((grid_size, grid_size))
            for x, y in zip(xs, ys):
                gx = min(int(x * grid_size), grid_size - 1)
                gy = min(int(y * grid_size), grid_size - 1)
                heatmap[gy, gx] += 1

            # 高斯模糊使热力图更平滑
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=3)

            ax.imshow(heatmap, cmap='hot', interpolation='bilinear', aspect='auto')
            ax.set_title('目标位置热力图', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center',
                    fontsize=12, color='#999', transform=ax.transAxes)
            ax.set_title('目标位置热力图', fontsize=11, fontweight='bold')

        ax.set_xlabel('X 方向')
        ax.set_ylabel('Y 方向')
        plt.tight_layout()
        return fig

    def _create_area_chart(self, detections: list):
        """
        目标面积占比: 饼图展示各检测目标的面积在图像中的占比。
        """
        fig, ax = plt.subplots(figsize=(4, 3.2), dpi=100)

        if detections:
            labels = [f"{d['class_name_cn']}" for d in detections]
            areas = [d['area_ratio'] * 100 for d in detections]
            # 剩余区域（无检测）
            remaining = max(0, 100 - sum(areas))
            labels.append('无病灶区域')
            areas.append(remaining)

            colors = []
            for d in detections:
                cid = d['class_id']
                c = CLASS_COLORS[cid] if cid < len(CLASS_COLORS) else (150, 150, 150)
                colors.append(f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}')
            colors.append('#E0E0E0')  # 灰色表示无病灶区域

            wedges, texts, autotexts = ax.pie(
                areas, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 8}
            )
            ax.set_title('目标面积占比', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center',
                    fontsize=12, color='#999', transform=ax.transAxes)
            ax.set_title('目标面积占比', fontsize=11, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_capability_chart(self, detections: list):
        """
        检测能力柱状图: 展示每个类别的检出数量和平均置信度。
        """
        fig, ax = plt.subplots(figsize=(4, 3.2), dpi=100)

        if detections:
            # 按类别统计
            from collections import Counter, defaultdict
            class_counts = Counter()
            class_conf_sum = defaultdict(float)

            for d in detections:
                name = d['class_name_cn']
                class_counts[name] += 1
                class_conf_sum[name] += d['confidence']

            names = list(class_counts.keys())
            counts = [class_counts[n] for n in names]
            avg_confs = [class_conf_sum[n] / class_counts[n] for n in names]

            x = np.arange(len(names))
            bar_width = 0.35

            bars1 = ax.bar(x - bar_width / 2, counts, bar_width,
                           label='检出数量', color='#42A5F5')
            bars2 = ax.bar(x + bar_width / 2, avg_confs, bar_width,
                           label='平均置信度', color='#66BB6A')

            # 标注数值
            for bar in bars1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f'{int(bar.get_height())}', ha='center', fontsize=8)
            for bar in bars2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{bar.get_height():.2f}', ha='center', fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=9)
            ax.legend(fontsize=8)
            ax.set_title('检测能力柱状图', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center',
                    fontsize=12, color='#999', transform=ax.transAxes)
            ax.set_title('检测能力柱状图', fontsize=11, fontweight='bold')

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return fig

    # ============================================================
    # 对照验证（自动查找真实标注并比对）
    # ============================================================

    def _find_label_file(self, image_path: str) -> str:
        """
        根据图片路径自动查找对应的标签文件。

        搜索策略:
            1. 根据图片路径判断属于 train/val/test 哪个集
            2. 在 labels 目录下查找同名 .txt 文件
            3. 如果在 images/test 下找不到，尝试遍历所有 labels 子目录

        Returns:
            标签文件路径，未找到返回 ''
        """
        # 获取图片文件名（不含扩展名）
        img_stem = os.path.splitext(os.path.basename(image_path))[0]

        # 尝试从图片路径推断数据集位置
        img_dir = os.path.dirname(image_path)
        parent_dir = os.path.dirname(img_dir)  # images 的父目录
        dir_name = os.path.basename(img_dir)     # test / val / train

        # 方案 1: images/test -> labels/test
        label_path = os.path.join(parent_dir.replace('images', 'labels'), dir_name, img_stem + '.txt')
        if os.path.exists(label_path):
            return label_path

        # 方案 2: 遍历 workspace/labels 下所有子目录
        workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        labels_root = os.path.join(workspace_dir, 'labels')
        if os.path.isdir(labels_root):
            for split in ['test', 'val', 'train']:
                candidate = os.path.join(labels_root, split, img_stem + '.txt')
                if os.path.exists(candidate):
                    return candidate

        return ''

    def _parse_label_file(self, label_path: str) -> list:
        """
        解析 YOLO 格式标签文件。

        Returns:
            列表，每个元素为 {'class_id': int, 'class_name': str, 'class_name_cn': str}
        """
        ground_truths = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        ground_truths.append({
                            'class_id': cls_id,
                            'class_name': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}',
                            'class_name_cn': CLASS_NAMES_CN[cls_id] if cls_id < len(CLASS_NAMES_CN) else f'类别_{cls_id}'
                        })
        except Exception:
            pass
        return ground_truths

    def _update_comparison(self, image_path: str, detections: list):
        """
        对照验证: 自动查找标签文件，将真实标注与模型预测结果进行比对。
        """
        label_path = self._find_label_file(image_path)

        if not label_path:
            self.compare_result.setText(
                '<p style="color:#FF9800; font-size:14px;">⚠️ <b>未找到对应的标签文件</b></p>'
                '<p>请使用 images/test 或 images/val 目录中的图片进行对照验证。</p>'
                '<p>系统会自动在 labels/ 目录下查找同名的 .txt 标注文件。</p>'
            )
            return

        # 解析真实标注
        ground_truths = self._parse_label_file(label_path)

        # 林列比对结果
        from collections import Counter
        gt_classes = Counter(gt['class_name_cn'] for gt in ground_truths)
        pred_classes = Counter(d['class_name_cn'] for d in detections)

        html = '<table style="width:100%; border-collapse:collapse; font-size:13px;">'
        html += '<tr style="background:#E3F2FD; font-weight:bold;">'
        html += '<td style="padding:8px; border:1px solid #E0E0E0;">项目</td>'
        html += '<td style="padding:8px; border:1px solid #E0E0E0;">真实标注 (Ground Truth)</td>'
        html += '<td style="padding:8px; border:1px solid #E0E0E0;">模型预测 (Prediction)</td>'
        html += '<td style="padding:8px; border:1px solid #E0E0E0;">结果</td>'
        html += '</tr>'

        # 标注来源
        html += f'<tr><td style="padding:8px; border:1px solid #E0E0E0;">标注文件</td>'
        html += f'<td style="padding:8px; border:1px solid #E0E0E0; color:#666;" colspan="3">{os.path.basename(label_path)}</td></tr>'

        # 检测数戮
        html += f'<tr><td style="padding:8px; border:1px solid #E0E0E0;">目标数量</td>'
        html += f'<td style="padding:8px; border:1px solid #E0E0E0;">{len(ground_truths)} 个</td>'
        html += f'<td style="padding:8px; border:1px solid #E0E0E0;">{len(detections)} 个</td>'
        count_match = len(ground_truths) == len(detections)
        html += f'<td style="padding:8px; border:1px solid #E0E0E0;">'
        html += '✅ 一致' if count_match else '⚠️ 不一致'
        html += '</td></tr>'

        # 按类别对比
        all_classes = set(list(gt_classes.keys()) + list(pred_classes.keys()))
        correct_count = 0
        total_classes = len(all_classes) if all_classes else 1

        for cls_name in sorted(all_classes):
            gt_count = gt_classes.get(cls_name, 0)
            pred_count = pred_classes.get(cls_name, 0)
            match = gt_count == pred_count and gt_count > 0
            if match:
                correct_count += 1

            html += f'<tr>'
            html += f'<td style="padding:8px; border:1px solid #E0E0E0;">{cls_name}</td>'
            html += f'<td style="padding:8px; border:1px solid #E0E0E0;">{gt_count} 个</td>'
            html += f'<td style="padding:8px; border:1px solid #E0E0E0;">{pred_count} 个</td>'
            html += f'<td style="padding:8px; border:1px solid #E0E0E0;">'
            if match:
                html += '✅ 正确'
            elif gt_count > 0 and pred_count > 0:
                html += '⚠️ 数量偏差'
            elif gt_count == 0:
                html += '❌ 误检'
            else:
                html += '❌ 漏检'
            html += '</td></tr>'

        html += '</table>'

        # 总结
        accuracy = correct_count / total_classes * 100 if all_classes else 0
        if accuracy == 100:
            summary = '<p style="margin-top:10px; font-size:15px; color:#2E7D32;">'\
                      '✅ <b>完全一致</b> — 模型预测与医学标注完全吻合，证明模型检测能力与专业标注水平相当。</p>'
        elif accuracy >= 50:
            summary = f'<p style="margin-top:10px; font-size:15px; color:#E65100;">'\
                      f'⚠️ <b>部分一致 ({accuracy:.0f}%)</b> — 模型对部分类别判断正确。</p>'
        else:
            summary = f'<p style="margin-top:10px; font-size:15px; color:#C62828;">'\
                      f'❌ <b>偏差较大 ({accuracy:.0f}%)</b> — 建议检查置信度阈值设置。</p>'

        html += summary
        self.compare_result.setText(html)

    # ============================================================
    # 图表导出
    # ============================================================

    def _export_chart(self, chart_type: str):
        """
        导出可视化图表为 PNG 文件。

        Args:
            chart_type: 'heatmap', 'area', 或 'capability'
        """
        if not self.current_detections:
            QMessageBox.information(self, "提示", "请先执行检测再导出图表")
            return

        # 文件保存对话框
        default_name = f"{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出图表", default_name, "PNG 图像 (*.png)"
        )
        if not file_path:
            return

        # 重新生成高清图表并保存
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if chart_type == 'heatmap':
            fig = self._create_heatmap_chart(self.current_detections)
        elif chart_type == 'area':
            fig = self._create_area_chart(self.current_detections)
        else:
            fig = self._create_capability_chart(self.current_detections)

        fig.savefig(file_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        QMessageBox.information(self, "导出成功", f"图表已保存到:\n{file_path}")

    # ============================================================
    # 样式
    # ============================================================

    def _get_stylesheet(self) -> str:
        return """
            #toolButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 18px;
                font-size: 14px;
                font-family: "Microsoft YaHei";
            }
            #toolButton:hover { background-color: #1565C0; }

            #exportButton {
                background-color: #7B1FA2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-family: "Microsoft YaHei";
            }
            #exportButton:hover { background-color: #6A1B9A; }

            QGroupBox {
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
                background: white;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 8px;
            }
        """
