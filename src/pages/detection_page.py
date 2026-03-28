"""
智能检测页
===========

功能:
    - 单张/批量上传 X 光片
    - QThread 异步检测（防止界面卡顿）
    - 批量检测进度条 + 逐张结果展示
    - 左侧结果列表 + 右侧图像对比区
    - 检测详情表格、可视化图表、对照验证
    - 紫色导出按钮
"""

import os
import json
import glob
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QScrollArea, QFrame, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QGroupBox, QSizePolicy, QMessageBox, QTabWidget,
    QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QIcon

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from utils.detector import PneumoniaDetector, find_available_models, CLASS_NAMES, CLASS_NAMES_CN, CLASS_COLORS
from utils.db_manager import save_detection_record, get_user_settings


def _fig_to_qpixmap(fig) -> QPixmap:
    """将 matplotlib Figure 转换为 QPixmap"""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    qimg = QImage(buf, w, h, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


# ============================================================
# 检测工作线程 — 在后台执行 YOLO 推理，避免阻塞 UI
# ============================================================

class DetectionWorker(QThread):
    """
    后台检测线程。

    信号:
        progress(int, int, str): 当前进度(已完成数, 总数, 文件名)
        result_ready(str, list): 单张检测完成(图片路径, 检测列表)
        all_done: 全部检测完成
    """
    progress = pyqtSignal(int, int, str)      # (当前, 总数, 文件名)
    result_ready = pyqtSignal(str, list)  # (路径, 检测结果)
    all_done = pyqtSignal()

    def __init__(self, detector, username, image_paths, settings):
        super().__init__()
        self.detector = detector
        self.username = username
        self.image_paths = image_paths
        self.settings = settings
        self.conf = settings.get('conf_threshold', 0.25)
        self.iou = settings.get('iou_threshold', 0.45)

    def run(self):
        """在子线程中逐张执行检测"""
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            try:
                from PIL import Image
                pil_image = Image.open(path)
                detections, _ = self.detector.detect(pil_image, self.conf, self.iou)
                detections = self.detector.apply_inference_policy(detections, self.settings)

                # 在后台线程中写入数据库，避免主线程在批量检测时频繁阻塞。
                save_detection_record(
                    username=self.username,
                    image_name=os.path.basename(path),
                    image_path=path,
                    detections=detections,
                    conf_threshold=self.conf,
                    iou_threshold=self.iou,
                    inference_mode=self.settings.get('inference_mode', 'standard')
                )
                self.result_ready.emit(path, detections)
            except Exception:
                self.result_ready.emit(path, [])
            self.progress.emit(i + 1, total, os.path.basename(path))
        self.all_done.emit()


# ============================================================
# 检测页面主体
# ============================================================

class DetectionPage(QWidget):
    """智能检测页面"""

    detection_completed = pyqtSignal()

    def __init__(self, username: str):
        super().__init__()
        self.username = username
        self.detector = PneumoniaDetector()
        self.current_detections = []
        self.current_image_path = None
        # 批量检测结果存储: [{path, detections, annotated}, ...]
        self.batch_results = []
        self.worker = None
        self.is_batch_mode = False
        self.current_run_settings = get_user_settings(self.username)

        self._load_model()
        self._init_ui()

    def _load_model(self):
        """加载模型"""
        settings = get_user_settings(self.username)
        model_path = settings.get('model_path', '')
        if not model_path or not os.path.exists(model_path):
            models = find_available_models()
            if models:
                model_path = list(models.values())[0]
        if model_path:
            self.detector.load_model(model_path)

    def _init_ui(self):
        """构建检测页面布局"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)

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

        # 进度条（默认隐藏）
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(300)
        self.progress_bar.setFixedHeight(26)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #D0D5DD;
                border-radius: 6px;
                background: #F5F5F5;
                text-align: center;
                font-size: 16px;
            }
            QProgressBar::chunk {
                background: #42A5F5;
                border-radius: 5px;
            }
        """)
        toolbar.addWidget(self.progress_bar)

        # 进度文字
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("font-size: 16px; color: #666;")
        self.progress_label.setVisible(False)
        toolbar.addWidget(self.progress_label)

        self.mode_label = QLabel("")
        self.mode_label.setStyleSheet(
            "color: white; background: #455A64; border-radius: 10px; "
            "padding: 4px 12px; font-size: 14px; font-weight: bold;"
        )
        toolbar.addWidget(self.mode_label)

        toolbar.addStretch()

        model_status = "✅ 模型已加载" if self.detector.is_loaded() else "❌ 模型未加载"
        self.model_label = QLabel(model_status)
        self.model_label.setStyleSheet("color: #666; font-size: 18px;")
        toolbar.addWidget(self.model_label)

        self._apply_mode_label(self.current_run_settings)

        layout.addLayout(toolbar)

        # ========== 主内容区 (左右 + 上下分割) ==========
        main_splitter = QSplitter(Qt.Horizontal)

        # --- 左侧: 结果列表（批量检测时显示每张图的缩略） ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)

        list_title = QLabel("📋 检测结果列表")
        list_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1A237E;")
        left_layout.addWidget(list_title)

        self.result_list = QListWidget()
        self.result_list.setStyleSheet("""
            QListWidget {
                background: white;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                font-size: 17px;
                font-family: "Microsoft YaHei";
            }
            QListWidget::item {
                padding: 8px 10px;
                border-bottom: 1px solid #F0F0F0;
            }
            QListWidget::item:selected {
                background: #E3F2FD;
                color: #1565C0;
            }
        """)
        self.result_list.currentRowChanged.connect(self._on_result_selected)
        left_layout.addWidget(self.result_list)

        left_panel.setFixedWidth(260)
        main_splitter.addWidget(left_panel)

        # --- 右侧: 图像对比 + 详情 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 0, 0, 0)

        right_splitter = QSplitter(Qt.Vertical)

        # -- 图像对比区 --
        image_area = QWidget()
        image_layout = QHBoxLayout(image_area)
        image_layout.setSpacing(12)

        left_group = QGroupBox("📷 原始 X 光片")
        left_inner = QVBoxLayout(left_group)
        self.label_original = QLabel("请上传 X 光片图像")
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setMinimumSize(350, 300)
        self.label_original.setStyleSheet(
            "background: #FAFAFA; border: 2px dashed #CCC; border-radius: 10px; "
            "color: #999; font-size: 20px;"
        )
        left_inner.addWidget(self.label_original)
        image_layout.addWidget(left_group)

        right_group = QGroupBox("🔍 检测结果")
        right_inner = QVBoxLayout(right_group)
        self.label_result = QLabel("检测结果将显示在这里")
        self.label_result.setAlignment(Qt.AlignCenter)
        self.label_result.setMinimumSize(350, 300)
        self.label_result.setStyleSheet(
            "background: #FAFAFA; border: 2px dashed #CCC; border-radius: 10px; "
            "color: #999; font-size: 20px;"
        )
        right_inner.addWidget(self.label_result)
        image_layout.addWidget(right_group)

        right_splitter.addWidget(image_area)

        # -- 下半部: Tab 页 --
        self.bottom_tabs = QTabWidget()

        # Tab 1: 检测详情
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        self.result_table = QTableWidget(0, 6)
        self.result_table.setHorizontalHeaderLabels(
            ['类别', '中文名', '置信度', 'Bbox (x1,y1,x2,y2)', '面积占比', '置信度条']
        )
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setStyleSheet("""
            QTableWidget { background: white; gridline-color: #E0E0E0; font-size: 17px; }
            QHeaderView::section { background: #E3F2FD; font-weight: bold; padding: 8px; font-size: 17px; }
        """)
        detail_layout.addWidget(self.result_table)
        self.bottom_tabs.addTab(detail_widget, "📋 检测详情")

        # Tab 2: 可视化图表
        vis_widget = QWidget()
        vis_layout = QVBoxLayout(vis_widget)
        charts_layout = QHBoxLayout()

        self.chart_heatmap = QLabel("待检测")
        self.chart_heatmap.setAlignment(Qt.AlignCenter)
        self.chart_heatmap.setMinimumHeight(200)
        self.chart_heatmap.setStyleSheet("background: white; border: 1px solid #E0E0E0; border-radius: 6px;")
        charts_layout.addWidget(self.chart_heatmap)

        self.chart_area = QLabel("待检测")
        self.chart_area.setAlignment(Qt.AlignCenter)
        self.chart_area.setMinimumHeight(200)
        self.chart_area.setStyleSheet("background: white; border: 1px solid #E0E0E0; border-radius: 6px;")
        charts_layout.addWidget(self.chart_area)

        self.chart_capability = QLabel("待检测")
        self.chart_capability.setAlignment(Qt.AlignCenter)
        self.chart_capability.setMinimumHeight(200)
        self.chart_capability.setStyleSheet("background: white; border: 1px solid #E0E0E0; border-radius: 6px;")
        charts_layout.addWidget(self.chart_capability)

        vis_layout.addLayout(charts_layout)

        export_layout = QHBoxLayout()
        export_layout.addStretch()
        for label, key in [("导出热力图", "heatmap"), ("导出面积占比", "area"), ("导出检测能力", "capability")]:
            btn = QPushButton(label)
            btn.setObjectName("exportButton")
            btn.clicked.connect(lambda checked, k=key: self._export_chart(k))
            export_layout.addWidget(btn)
        export_layout.addStretch()
        vis_layout.addLayout(export_layout)
        self.bottom_tabs.addTab(vis_widget, "📊 检测可视化")

        # Tab 3: 对照验证
        compare_widget = QWidget()
        compare_layout = QVBoxLayout(compare_widget)

        compare_title = QLabel("📝 自动对照验证: 模型预测 vs 真实标注")
        compare_title.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #1A237E; margin-bottom: 5px;"
        )
        compare_layout.addWidget(compare_title)

        compare_desc = QLabel(
            "系统自动查找上传图片对应的真实标注文件（labels/test 或 labels/val），"
            "将医学专家标注与模型预测结果进行自动比对。"
        )
        compare_desc.setStyleSheet("color: #666; font-size: 17px; margin-bottom: 10px;")
        compare_desc.setWordWrap(True)
        compare_layout.addWidget(compare_desc)

        self.compare_result = QLabel("请先执行检测")
        self.compare_result.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.compare_result.setWordWrap(True)
        self.compare_result.setMinimumHeight(160)
        self.compare_result.setStyleSheet(
            "background: white; border: 1px solid #E0E0E0; border-radius: 8px; "
            "padding: 15px; font-size: 18px; font-family: 'Microsoft YaHei';"
        )
        compare_layout.addWidget(self.compare_result)
        self.bottom_tabs.addTab(compare_widget, "✅ 对照验证")

        # Tab 4: 批量汇总
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)

        self.summary_overview = QLabel("请先执行检测")
        self.summary_overview.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.summary_overview.setWordWrap(True)
        self.summary_overview.setMinimumHeight(120)
        self.summary_overview.setStyleSheet(
            "background: white; border: 1px solid #E0E0E0; border-radius: 8px; "
            "padding: 15px; font-size: 18px; font-family: 'Microsoft YaHei';"
        )
        summary_layout.addWidget(self.summary_overview)

        summary_chart_layout = QHBoxLayout()
        self.summary_status_chart = QLabel("待检测")
        self.summary_status_chart.setAlignment(Qt.AlignCenter)
        self.summary_status_chart.setMinimumHeight(220)
        self.summary_status_chart.setStyleSheet("background: white; border: 1px solid #E0E0E0; border-radius: 6px;")
        summary_chart_layout.addWidget(self.summary_status_chart)

        self.summary_class_chart = QLabel("待检测")
        self.summary_class_chart.setAlignment(Qt.AlignCenter)
        self.summary_class_chart.setMinimumHeight(220)
        self.summary_class_chart.setStyleSheet("background: white; border: 1px solid #E0E0E0; border-radius: 6px;")
        summary_chart_layout.addWidget(self.summary_class_chart)
        summary_layout.addLayout(summary_chart_layout)

        self.summary_risk_table = QTableWidget(0, 4)
        self.summary_risk_table.setHorizontalHeaderLabels(['图片', '风险等级', '主要结果', '最高置信度'])
        self.summary_risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_risk_table.setAlternatingRowColors(True)
        self.summary_risk_table.setStyleSheet("""
            QTableWidget { background: white; gridline-color: #E0E0E0; font-size: 16px; }
            QHeaderView::section { background: #E8F5E9; font-weight: bold; padding: 8px; font-size: 16px; }
        """)
        summary_layout.addWidget(self.summary_risk_table)
        self.bottom_tabs.addTab(summary_widget, "🧾 批量汇总")

        right_splitter.addWidget(self.bottom_tabs)
        right_splitter.setSizes([400, 300])

        right_layout.addWidget(right_splitter)
        main_splitter.addWidget(right_panel)

        layout.addWidget(main_splitter)
        self.setStyleSheet(self._get_stylesheet())

    # ============================================================
    # 图片选择与检测入口
    # ============================================================

    def _select_image(self):
        """选择单张图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 X 光片", "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if file_path:
            self._start_detection([file_path])

    def _batch_detect(self):
        """批量检测: 选择文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(extensions)
        ])
        if not image_files:
            QMessageBox.information(self, "提示", "所选文件夹中没有图片文件")
            return
        self._start_detection(image_files)

    def _start_detection(self, image_paths: list):
        """
        启动异步检测（QThread）。

        Args:
            image_paths: 图片路径列表
        """
        if not self.detector.is_loaded():
            QMessageBox.warning(self, "错误", "模型未加载，请在设置页指定模型路径")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "检测仍在进行中，请等待完成")
            return

        # 重置批量结果
        self.is_batch_mode = len(image_paths) > 1
        self.batch_results = []
        self.result_list.clear()
        self.result_list.setEnabled(not self.is_batch_mode)
        self.current_detections = []
        self.current_image_path = None
        self.current_run_settings = get_user_settings(self.username)
        self._apply_mode_label(self.current_run_settings)
        self._reset_batch_summary()

        if self.is_batch_mode:
            self._set_busy_preview()

        # 显示进度条
        self.progress_bar.setMaximum(len(image_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_label.setText(f"检测中 0/{len(image_paths)}...")
        self.btn_select.setEnabled(False)
        self.btn_batch.setEnabled(False)

        # 创建并启动工作线程
        self.worker = DetectionWorker(self.detector, self.username, image_paths, self.current_run_settings)
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_single_result)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.start()

    # ============================================================
    # 工作线程回调
    # ============================================================

    def _on_progress(self, current, total, filename):
        """更新进度条"""
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"检测中 {current}/{total} — {filename}")

    def _on_single_result(self, image_path, detections):
        """
        每检测完一张图片时调用:
            - 存入 batch_results
            - 添加到左侧列表
            - 单张检测时自动刷新右侧显示
        """
        result = {
            'path': image_path,
            'detections': detections,
            'annotated': None
        }
        self.batch_results.append(result)

        # 添加到列表
        name = os.path.basename(image_path)
        n = len(detections)
        status = f"✅ {n} 个目标" if n > 0 else "✔ 正常"
        mode_tag = self._mode_text(self.current_run_settings)
        item = QListWidgetItem(f"{name}\n   {mode_tag}  {status}")
        self.result_list.addItem(item)

        # 单张检测时直接展示结果；批量时仅更新左侧列表和进度。
        if not self.is_batch_mode:
            self.result_list.setCurrentRow(self.result_list.count() - 1)
            self._update_batch_summary()

    def _on_all_done(self):
        """全部检测完成"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.btn_select.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.result_list.setEnabled(True)
        self.detection_completed.emit()

        total = len(self.batch_results)
        self._update_batch_summary()
        if self.is_batch_mode and total > 0 and self.result_list.currentRow() < 0:
            self.result_list.setCurrentRow(0)
            self.bottom_tabs.setCurrentIndex(3)

        if total > 1:
            QMessageBox.information(
                self, "批量检测完成",
                f"共处理 {total} 张图片，点击左侧列表可查看每张结果"
            )

        self.is_batch_mode = False

    # ============================================================
    # 列表项点击 — 切换显示
    # ============================================================

    def _on_result_selected(self, row):
        """用户在左侧列表点击某一项时，更新右侧所有内容"""
        if row < 0 or row >= len(self.batch_results):
            return

        result = self.batch_results[row]
        image_path = result['path']
        detections = result['detections']

        self.current_image_path = image_path
        self.current_detections = detections

        # 更新原图
        pixmap_orig = QPixmap(image_path)
        self.label_original.setPixmap(pixmap_orig.scaled(
            self.label_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self.label_original.setStyleSheet("background: #FAFAFA; border-radius: 10px;")

        # 更新检测结果图
        annotated = result.get('annotated')
        if annotated is None:
            annotated = self.detector.annotate_image(image_path, detections)
            result['annotated'] = annotated

        if annotated is not None:
            h, w, ch = annotated.shape
            qimg = QImage(annotated.data, w, h, ch * w, QImage.Format_RGB888)
            self.label_result.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.label_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.label_result.setStyleSheet("background: #FAFAFA; border-radius: 10px;")
        else:
            self.label_result.clear()
            self.label_result.setText("✅ 未检测到异常")
            self.label_result.setStyleSheet("background: #FAFAFA; border-radius: 10px;")

        # 更新详情表格
        self._update_result_table(detections)
        # 更新可视化图表
        self._update_detection_charts(detections)
        # 更新对照验证
        self._update_comparison(image_path, detections)

    # ============================================================
    # 界面更新
    # ============================================================

    def _update_result_table(self, detections: list):
        """更新检测详情表格"""
        self.result_table.setRowCount(len(detections))
        for row, det in enumerate(detections):
            self.result_table.setItem(row, 0, QTableWidgetItem(det['class_name']))
            self.result_table.setItem(row, 1, QTableWidgetItem(det['class_name_cn']))

            conf_item = QTableWidgetItem(f"{det['confidence']:.4f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(row, 2, conf_item)

            bbox = det['bbox']
            self.result_table.setItem(row, 3, QTableWidgetItem(
                f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
            ))

            area_item = QTableWidgetItem(f"{det['area_ratio']:.2%}")
            area_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(row, 4, area_item)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(det['confidence'] * 100))
            bar.setFormat(f"{det['confidence']:.1%}")
            bar.setFixedHeight(22)
            if det['confidence'] >= 0.8:
                bar.setStyleSheet("QProgressBar::chunk { background: #43A047; } QProgressBar { font-size: 15px; }")
            elif det['confidence'] >= 0.5:
                bar.setStyleSheet("QProgressBar::chunk { background: #FF9800; } QProgressBar { font-size: 15px; }")
            else:
                bar.setStyleSheet("QProgressBar::chunk { background: #E53935; } QProgressBar { font-size: 15px; }")
            self.result_table.setCellWidget(row, 5, bar)

    def _set_busy_preview(self):
        """批量检测时冻结右侧详情区域，避免主线程频繁重绘。"""
        self.label_original.clear()
        self.label_original.setText("批量检测进行中\n完成后可查看原图")
        self.label_original.setStyleSheet(
            "background: #FAFAFA; border: 2px dashed #CCC; border-radius: 10px; "
            "color: #999; font-size: 20px;"
        )

        self.label_result.clear()
        self.label_result.setText("批量检测进行中\n右侧详情已暂停刷新")
        self.label_result.setStyleSheet(
            "background: #FAFAFA; border: 2px dashed #CCC; border-radius: 10px; "
            "color: #999; font-size: 20px;"
        )

        self.result_table.setRowCount(0)
        for chart in [self.chart_heatmap, self.chart_area, self.chart_capability]:
            chart.clear()
            chart.setText("批量检测中")
        self.compare_result.setText("批量检测进行中，完成后选择左侧图片查看对照验证。")
        self.summary_overview.setText("批量检测进行中，系统会在完成后自动生成当前批次的检测汇总。")
        self.summary_status_chart.clear()
        self.summary_status_chart.setText("批量检测中")
        self.summary_class_chart.clear()
        self.summary_class_chart.setText("批量检测中")
        self.summary_risk_table.setRowCount(0)

    def _apply_mode_label(self, settings: dict):
        """更新顶部当前模式提示。"""
        mode_text = self._mode_text(settings)
        if settings.get('inference_mode', 'standard') == 'application':
            self.mode_label.setStyleSheet(
                "color: white; background: #8E24AA; border-radius: 10px; "
                "padding: 4px 12px; font-size: 14px; font-weight: bold;"
            )
        else:
            self.mode_label.setStyleSheet(
                "color: white; background: #455A64; border-radius: 10px; "
                "padding: 4px 12px; font-size: 14px; font-weight: bold;"
            )
        self.mode_label.setText(f"当前模式：{mode_text}")

    @staticmethod
    def _mode_text(settings: dict) -> str:
        return "应用模式" if settings.get('inference_mode', 'standard') == 'application' else "标准模式"

    def _reset_batch_summary(self):
        """重置批量汇总区域。"""
        self.summary_overview.setText("请先执行检测")
        self.summary_status_chart.clear()
        self.summary_status_chart.setText("待检测")
        self.summary_class_chart.clear()
        self.summary_class_chart.setText("待检测")
        self.summary_risk_table.setRowCount(0)

    def _update_batch_summary(self):
        """根据当前批次结果生成汇总信息。"""
        total = len(self.batch_results)
        if total == 0:
            self._reset_batch_summary()
            return

        image_level_class_counter = Counter()
        total_detection_counter = Counter()
        abnormal_images = 0
        healthy_images = 0
        no_detection_images = 0
        avg_conf_values = []
        risk_rows = []

        for result in self.batch_results:
            detections = result['detections']
            unique_classes = {det['class_name_cn'] for det in detections}
            for class_name_cn in unique_classes:
                image_level_class_counter[class_name_cn] += 1
            for det in detections:
                total_detection_counter[det['class_name_cn']] += 1
                avg_conf_values.append(det['confidence'])

            risk_level, major_result, risk_score = self._evaluate_result_risk(detections)
            if risk_level == '异常':
                abnormal_images += 1
            elif risk_level == '健康':
                healthy_images += 1
            else:
                no_detection_images += 1

            risk_rows.append({
                'image_name': os.path.basename(result['path']),
                'risk_level': risk_level,
                'major_result': major_result,
                'risk_score': risk_score
            })

        avg_conf = float(np.mean(avg_conf_values)) if avg_conf_values else 0.0
        mode_text = self._mode_text(self.current_run_settings)

        summary_lines = [
            f"<b>当前批次模式：</b>{mode_text}",
            f"<b>总图片数：</b>{total} 张",
            f"<b>异常图片：</b>{abnormal_images} 张",
            f"<b>健康图片：</b>{healthy_images} 张",
            f"<b>未检出图片：</b>{no_detection_images} 张",
            f"<b>涉及类别数：</b>{len(image_level_class_counter)} 类",
            f"<b>平均置信度：</b>{avg_conf:.4f}",
        ]
        self.summary_overview.setText("<br>".join(summary_lines))

        status_fig = self._create_batch_status_chart(abnormal_images, healthy_images, no_detection_images)
        self.summary_status_chart.setPixmap(_fig_to_qpixmap(status_fig).scaled(
            self.summary_status_chart.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        plt.close(status_fig)

        class_fig = self._create_batch_class_chart(image_level_class_counter)
        self.summary_class_chart.setPixmap(_fig_to_qpixmap(class_fig).scaled(
            self.summary_class_chart.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        plt.close(class_fig)

        risk_rows.sort(key=lambda item: item['risk_score'], reverse=True)
        top_rows = risk_rows[:5]
        self.summary_risk_table.setRowCount(len(top_rows))
        for row, item in enumerate(top_rows):
            self.summary_risk_table.setItem(row, 0, QTableWidgetItem(item['image_name']))
            self.summary_risk_table.setItem(row, 1, QTableWidgetItem(item['risk_level']))
            self.summary_risk_table.setItem(row, 2, QTableWidgetItem(item['major_result']))
            self.summary_risk_table.setItem(row, 3, QTableWidgetItem(f"{item['risk_score']:.4f}"))

    @staticmethod
    def _evaluate_result_risk(detections: list) -> tuple:
        """返回 (风险等级, 主要结果, 风险分数)。"""
        if not detections:
            return '未检出', '无异常检出', 0.0

        sorted_detections = sorted(detections, key=lambda det: det.get('confidence', 0.0), reverse=True)
        best = sorted_detections[0]
        class_name = best.get('class_name', '')
        risk_score = float(best.get('confidence', 0.0))
        major_result = best.get('class_name_cn', class_name)

        if class_name == 'healthy':
            return '健康', major_result, risk_score
        return '异常', major_result, risk_score

    def _update_detection_charts(self, detections: list):
        """为当前检测结果生成三张可视化图表"""
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig1 = self._create_heatmap_chart(detections)
        self.chart_heatmap.setPixmap(_fig_to_qpixmap(fig1).scaled(
            self.chart_heatmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        plt.close(fig1)

        fig2 = self._create_area_chart(detections)
        self.chart_area.setPixmap(_fig_to_qpixmap(fig2).scaled(
            self.chart_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        plt.close(fig2)

        fig3 = self._create_capability_chart(detections)
        self.chart_capability.setPixmap(_fig_to_qpixmap(fig3).scaled(
            self.chart_capability.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        plt.close(fig3)

    # ============================================================
    # 图表生成
    # ============================================================

    def _create_heatmap_chart(self, detections):
        fig, ax = plt.subplots(figsize=(4, 3.2), dpi=100)
        if detections:
            xs = [d['bbox_norm'][0] for d in detections]
            ys = [d['bbox_norm'][1] for d in detections]
            grid_size = 50
            heatmap = np.zeros((grid_size, grid_size))
            for x, y in zip(xs, ys):
                gx = min(int(x * grid_size), grid_size - 1)
                gy = min(int(y * grid_size), grid_size - 1)
                heatmap[gy, gx] += 1
            from scipy.ndimage import gaussian_filter
            heatmap = gaussian_filter(heatmap, sigma=3)
            ax.imshow(heatmap, cmap='hot', interpolation='bilinear', aspect='auto')
            ax.set_title('目标位置热力图', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center', fontsize=13, color='#999', transform=ax.transAxes)
            ax.set_title('目标位置热力图', fontsize=12, fontweight='bold')
        ax.set_xlabel('X 方向', fontsize=10)
        ax.set_ylabel('Y 方向', fontsize=10)
        plt.tight_layout()
        return fig

    def _create_area_chart(self, detections):
        fig, ax = plt.subplots(figsize=(4, 3.2), dpi=100)
        if detections:
            labels = [d['class_name_cn'] for d in detections]
            areas = [d['area_ratio'] * 100 for d in detections]
            remaining = max(0, 100 - sum(areas))
            labels.append('无病灶区域')
            areas.append(remaining)
            colors = []
            for d in detections:
                cid = d['class_id']
                c = CLASS_COLORS[cid] if cid < len(CLASS_COLORS) else (150, 150, 150)
                colors.append(f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}')
            colors.append('#E0E0E0')
            ax.pie(areas, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
            ax.set_title('目标面积占比', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center', fontsize=13, color='#999', transform=ax.transAxes)
            ax.set_title('目标面积占比', fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    def _create_capability_chart(self, detections):
        fig, ax = plt.subplots(figsize=(4, 3.2), dpi=100)
        if detections:
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
            w = 0.35
            ax.bar(x - w/2, counts, w, label='检出数量', color='#42A5F5')
            ax.bar(x + w/2, avg_confs, w, label='平均置信度', color='#66BB6A')
            for i, (c, a) in enumerate(zip(counts, avg_confs)):
                ax.text(i - w/2, c + 0.05, str(c), ha='center', fontsize=9)
                ax.text(i + w/2, a + 0.02, f'{a:.2f}', ha='center', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=10)
            ax.legend(fontsize=9)
            ax.set_title('检测能力柱状图', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center', fontsize=13, color='#999', transform=ax.transAxes)
            ax.set_title('检测能力柱状图', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return fig

    def _create_batch_status_chart(self, abnormal_images: int, healthy_images: int, no_detection_images: int):
        fig, ax = plt.subplots(figsize=(4, 3.2), dpi=100)
        values = [abnormal_images, healthy_images, no_detection_images]
        labels = ['异常', '健康', '未检出']
        colors = ['#E53935', '#43A047', '#90A4AE']

        if sum(values) > 0:
            ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
            ax.set_title('批量结果状态占比', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center', fontsize=13, color='#999', transform=ax.transAxes)
            ax.set_title('批量结果状态占比', fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    def _create_batch_class_chart(self, image_level_class_counter: Counter):
        fig, ax = plt.subplots(figsize=(4, 3.2), dpi=100)
        if image_level_class_counter:
            names = list(image_level_class_counter.keys())
            counts = [image_level_class_counter[name] for name in names]
            colors = ['#42A5F5'] * len(names)
            x = np.arange(len(names))
            ax.bar(x, counts, color=colors)
            for idx, count in enumerate(counts):
                ax.text(idx, count + 0.05, str(count), ha='center', fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
            ax.set_title('各类别命中图片数', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, '暂无类别命中', ha='center', va='center', fontsize=13, color='#999', transform=ax.transAxes)
            ax.set_title('各类别命中图片数', fontsize=12, fontweight='bold')
        plt.tight_layout()
        return fig

    # ============================================================
    # 对照验证
    # ============================================================

    def _find_label_file(self, image_path: str) -> str:
        """根据图片路径自动查找对应的 YOLO 标签文件"""
        img_stem = os.path.splitext(os.path.basename(image_path))[0]
        img_dir = os.path.dirname(image_path)
        parent_dir = os.path.dirname(img_dir)
        dir_name = os.path.basename(img_dir)

        label_path = os.path.join(parent_dir.replace('images', 'labels'), dir_name, img_stem + '.txt')
        if os.path.exists(label_path):
            return label_path

        workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        labels_root = os.path.join(workspace_dir, 'labels')
        if os.path.isdir(labels_root):
            for split in ['test', 'val', 'train']:
                candidate = os.path.join(labels_root, split, img_stem + '.txt')
                if os.path.exists(candidate):
                    return candidate
        return ''

    def _parse_label_file(self, label_path: str) -> list:
        """解析 YOLO 格式标签文件"""
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
        """对照验证: 自动查找标签文件并与预测结果比对"""
        label_path = self._find_label_file(image_path)

        if not label_path:
            self.compare_result.setText(
                '<p style="color:#FF9800; font-size:19px;">⚠️ <b>未找到对应的标签文件</b></p>'
                '<p style="font-size:17px;">请使用 images/test 或 images/val 目录中的图片进行对照验证。</p>'
            )
            return

        ground_truths = self._parse_label_file(label_path)
        gt_classes = Counter(gt['class_name_cn'] for gt in ground_truths)
        pred_classes = Counter(d['class_name_cn'] for d in detections)

        html = '<table style="width:100%; border-collapse:collapse; font-size:17px;">'
        html += '<tr style="background:#E3F2FD; font-weight:bold;">'
        html += '<td style="padding:10px; border:1px solid #E0E0E0;">项目</td>'
        html += '<td style="padding:10px; border:1px solid #E0E0E0;">真实标注 (GT)</td>'
        html += '<td style="padding:10px; border:1px solid #E0E0E0;">模型预测</td>'
        html += '<td style="padding:10px; border:1px solid #E0E0E0;">结果</td>'
        html += '</tr>'

        html += f'<tr><td style="padding:10px; border:1px solid #E0E0E0;">标注文件</td>'
        html += f'<td style="padding:10px; border:1px solid #E0E0E0; color:#666;" colspan="3">{os.path.basename(label_path)}</td></tr>'

        html += f'<tr><td style="padding:10px; border:1px solid #E0E0E0;">目标数量</td>'
        html += f'<td style="padding:10px; border:1px solid #E0E0E0;">{len(ground_truths)} 个</td>'
        html += f'<td style="padding:10px; border:1px solid #E0E0E0;">{len(detections)} 个</td>'
        count_match = len(ground_truths) == len(detections)
        html += f'<td style="padding:10px; border:1px solid #E0E0E0;">{"✅ 一致" if count_match else "⚠️ 不一致"}</td></tr>'

        all_classes = set(list(gt_classes.keys()) + list(pred_classes.keys()))
        correct_count = 0
        total_classes = len(all_classes) if all_classes else 1

        for cls_name in sorted(all_classes):
            gt_count = gt_classes.get(cls_name, 0)
            pred_count = pred_classes.get(cls_name, 0)
            match = gt_count == pred_count and gt_count > 0
            if match:
                correct_count += 1
            status = '✅ 正确' if match else ('⚠️ 数量偏差' if gt_count > 0 and pred_count > 0 else ('❌ 误检' if gt_count == 0 else '❌ 漏检'))
            html += f'<tr><td style="padding:10px; border:1px solid #E0E0E0;">{cls_name}</td>'
            html += f'<td style="padding:10px; border:1px solid #E0E0E0;">{gt_count} 个</td>'
            html += f'<td style="padding:10px; border:1px solid #E0E0E0;">{pred_count} 个</td>'
            html += f'<td style="padding:10px; border:1px solid #E0E0E0;">{status}</td></tr>'

        html += '</table>'

        accuracy = correct_count / total_classes * 100 if all_classes else 0
        if accuracy == 100:
            html += '<p style="margin-top:12px; font-size:19px; color:#2E7D32;">✅ <b>完全一致</b> — 模型预测与医学标注完全吻合。</p>'
        elif accuracy >= 50:
            html += f'<p style="margin-top:12px; font-size:19px; color:#E65100;">⚠️ <b>部分一致 ({accuracy:.0f}%)</b></p>'
        else:
            html += f'<p style="margin-top:12px; font-size:19px; color:#C62828;">❌ <b>偏差较大 ({accuracy:.0f}%)</b></p>'

        self.compare_result.setText(html)

    # ============================================================
    # 图表导出
    # ============================================================

    def _export_chart(self, chart_type: str):
        """导出可视化图表为 PNG"""
        if not self.current_detections:
            QMessageBox.information(self, "提示", "请先执行检测再导出图表")
            return
        default_name = f"{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path, _ = QFileDialog.getSaveFileName(self, "导出图表", default_name, "PNG 图像 (*.png)")
        if not file_path:
            return
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
                padding: 10px 22px;
                font-size: 20px;
                font-family: "Microsoft YaHei";
            }
            #toolButton:hover { background-color: #1565C0; }
            #toolButton:disabled { background-color: #90CAF9; }

            #exportButton {
                background-color: #7B1FA2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 18px;
                font-family: "Microsoft YaHei";
            }
            #exportButton:hover { background-color: #6A1B9A; }

            QGroupBox {
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 20px;
                background: white;
                font-size: 20px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 8px;
            }

            QTabWidget::pane {
                border: 1px solid #E0E0E0;
                border-radius: 6px;
                background: white;
            }
            QTabBar::tab {
                padding: 10px 20px;
                font-size: 18px;
            }
            QTabBar::tab:selected {
                color: #1976D2;
                border-bottom: 2px solid #1976D2;
            }
        """
