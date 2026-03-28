"""
系统设置页
===========

功能:
    - 模型参数调节: 置信度阈值滑块、IoU 阈值滑块
    - 模型管理: 显示/切换模型文件
    - 界面设置: 背景图片选择与预览
    - 所有设置持久化到 SQLite
"""

import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QFileDialog, QGroupBox,
    QComboBox, QScrollArea, QMessageBox, QFrame,
    QDoubleSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap

from utils.db_manager import get_user_settings, save_user_settings
from utils.detector import find_available_models, CLASS_NAMES, CLASS_NAMES_CN

DEFAULT_APPLICATION_THRESHOLDS = {
    'Pneumonia Bacteria': 0.25,
    'Pneumonia Virus': 0.25,
    'Sick': 0.35,
    'healthy': 0.60,
    'tuberculosis': 0.20
}


class SettingsPage(QWidget):
    """
    系统设置页面。

    信号:
        settings_changed: 设置变更时发射，通知主窗口更新（如背景图片）
    """

    settings_changed = pyqtSignal()

    def __init__(self, username: str):
        super().__init__()
        self.username = username
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        """构建设置页面布局"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(20)

        # ========== 页面标题 ==========
        title = QLabel("⚙️ 系统设置")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #1A237E; font-family: 'Microsoft YaHei';"
        )
        layout.addWidget(title)

        # ========== 1. 模型参数 ==========
        param_group = QGroupBox("🎯 检测参数")
        param_group.setStyleSheet(self._group_style())
        param_layout = QVBoxLayout(param_group)

        # 置信度阈值
        conf_label_layout = QHBoxLayout()
        conf_label_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_value_label = QLabel("0.25")
        self.conf_value_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        conf_label_layout.addWidget(self.conf_value_label)
        conf_label_layout.addStretch()
        param_layout.addLayout(conf_label_layout)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)     # 0.05 ~ 0.95
        self.conf_slider.setValue(25)         # 默认 0.25
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        param_layout.addWidget(self.conf_slider)

        desc1 = QLabel("低于此阈值的检测框将被过滤。值越高结果越严格，值越低结果越宽松。")
        desc1.setStyleSheet("color: #888; font-size: 11px;")
        param_layout.addWidget(desc1)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #E0E0E0;")
        param_layout.addWidget(line)

        # IoU 阈值
        iou_label_layout = QHBoxLayout()
        iou_label_layout.addWidget(QLabel("IoU 阈值 (NMS):"))
        self.iou_value_label = QLabel("0.45")
        self.iou_value_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        iou_label_layout.addWidget(self.iou_value_label)
        iou_label_layout.addStretch()
        param_layout.addLayout(iou_label_layout)

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(10, 90)     # 0.10 ~ 0.90
        self.iou_slider.setValue(45)          # 默认 0.45
        self.iou_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_slider.setTickInterval(10)
        self.iou_slider.valueChanged.connect(self._on_iou_changed)
        param_layout.addWidget(self.iou_slider)

        desc2 = QLabel("非极大值抑制阈值。值越小，重叠检测框过滤越激进。")
        desc2.setStyleSheet("color: #888; font-size: 11px;")
        param_layout.addWidget(desc2)

        layout.addWidget(param_group)

        # ========== 2. 应用判读模式 ==========
        mode_group = QGroupBox("🩺 应用判读模式")
        mode_group.setStyleSheet(self._group_style())
        mode_layout = QVBoxLayout(mode_group)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("推理模式:"))
        self.inference_mode_combo = QComboBox()
        self.inference_mode_combo.addItem("标准模式（与原始 YOLO 输出一致）", "standard")
        self.inference_mode_combo.addItem("应用模式（分类别阈值 + 规则后处理）", "application")
        self.inference_mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.inference_mode_combo)
        mode_row.addStretch()
        mode_layout.addLayout(mode_row)

        mode_desc = QLabel(
            "标准模式用于论文复现实验；应用模式用于系统部署判读，支持具体病种优先、healthy 严格判定。"
        )
        mode_desc.setWordWrap(True)
        mode_desc.setStyleSheet("color: #666; font-size: 11px;")
        mode_layout.addWidget(mode_desc)

        margin_row = QHBoxLayout()
        margin_row.addWidget(QLabel("healthy 覆盖 margin:"))
        self.healthy_margin_spin = QDoubleSpinBox()
        self.healthy_margin_spin.setRange(0.00, 0.50)
        self.healthy_margin_spin.setSingleStep(0.01)
        self.healthy_margin_spin.setDecimals(2)
        self.healthy_margin_spin.setValue(0.15)
        self.healthy_margin_spin.setSuffix(" 分")
        margin_row.addWidget(self.healthy_margin_spin)
        margin_row.addStretch()
        mode_layout.addLayout(margin_row)

        threshold_desc = QLabel("应用模式下可为每个类别设置独立阈值，建议后续依据验证集 PR 曲线再回填。")
        threshold_desc.setWordWrap(True)
        threshold_desc.setStyleSheet("color: #888; font-size: 11px;")
        mode_layout.addWidget(threshold_desc)

        self.threshold_form = QFormLayout()
        self.threshold_form.setLabelAlignment(Qt.AlignRight)
        self.class_threshold_spins = {}
        for class_name, class_name_cn in zip(CLASS_NAMES, CLASS_NAMES_CN):
            spin = QDoubleSpinBox()
            spin.setRange(0.05, 0.95)
            spin.setSingleStep(0.01)
            spin.setDecimals(2)
            spin.setValue(DEFAULT_APPLICATION_THRESHOLDS.get(class_name, 0.25))
            self.class_threshold_spins[class_name] = spin
            self.threshold_form.addRow(f"{class_name_cn} ({class_name})", spin)
        mode_layout.addLayout(self.threshold_form)

        layout.addWidget(mode_group)

        # ========== 3. 模型管理 ==========
        model_group = QGroupBox("🧠 模型管理")
        model_group.setStyleSheet(self._group_style())
        model_layout = QVBoxLayout(model_group)

        model_layout.addWidget(QLabel("当前模型:"))
        model_select_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(350)
        self.model_combo.setStyleSheet(
            "QComboBox { padding: 8px; border: 1px solid #D0D5DD; border-radius: 6px; background: white; }"
        )
        model_select_layout.addWidget(self.model_combo)

        btn_browse = QPushButton("浏览...")
        btn_browse.setObjectName("secondaryButton")
        btn_browse.clicked.connect(self._browse_model)
        model_select_layout.addWidget(btn_browse)

        model_layout.addLayout(model_select_layout)

        # 刷新模型列表
        self._refresh_model_list()

        layout.addWidget(model_group)

        # ========== 4. 界面设置 ==========
        ui_group = QGroupBox("🎨 界面设置")
        ui_group.setStyleSheet(self._group_style())
        ui_layout = QVBoxLayout(ui_group)

        bg_label_layout = QHBoxLayout()
        bg_label_layout.addWidget(QLabel("背景图片:"))
        self.bg_path_label = QLabel("未设置")
        self.bg_path_label.setStyleSheet("color: #666; font-size: 12px;")
        bg_label_layout.addWidget(self.bg_path_label)
        bg_label_layout.addStretch()
        ui_layout.addLayout(bg_label_layout)

        bg_btn_layout = QHBoxLayout()

        btn_select_bg = QPushButton("选择背景图片")
        btn_select_bg.setObjectName("secondaryButton")
        btn_select_bg.clicked.connect(self._select_background)
        bg_btn_layout.addWidget(btn_select_bg)

        btn_clear_bg = QPushButton("恢复默认")
        btn_clear_bg.setObjectName("dangerButton")
        btn_clear_bg.clicked.connect(self._clear_background)
        bg_btn_layout.addWidget(btn_clear_bg)

        bg_btn_layout.addStretch()
        ui_layout.addLayout(bg_btn_layout)

        # 背景预览
        self.bg_preview = QLabel("背景预览")
        self.bg_preview.setAlignment(Qt.AlignCenter)
        self.bg_preview.setFixedHeight(150)
        self.bg_preview.setStyleSheet(
            "border: 1px dashed #CCC; border-radius: 6px; background: #FAFAFA; color: #999;"
        )
        ui_layout.addWidget(self.bg_preview)

        layout.addWidget(ui_group)

        # ========== 保存按钮 ==========
        btn_save = QPushButton("💾 保存设置")
        btn_save.setObjectName("primaryButton")
        btn_save.clicked.connect(self._save_settings)
        layout.addWidget(btn_save, alignment=Qt.AlignCenter)

        layout.addStretch()

        scroll.setWidget(container)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

        self.setStyleSheet(self._get_stylesheet())

    # ============================================================
    # 设置加载与保存
    # ============================================================

    def _load_settings(self):
        """从数据库加载用户设置"""
        settings = get_user_settings(self.username)

        conf = settings.get('conf_threshold', 0.25)
        iou = settings.get('iou_threshold', 0.45)
        model_path = settings.get('model_path', '')
        bg_image = settings.get('background_image', '')
        inference_mode = settings.get('inference_mode', 'standard')
        healthy_margin = settings.get('healthy_margin', 0.15)
        per_class_thresholds = settings.get('per_class_thresholds', {})

        # 应用到控件
        self.conf_slider.setValue(int(conf * 100))
        self.iou_slider.setValue(int(iou * 100))
        self.healthy_margin_spin.setValue(float(healthy_margin))

        mode_index = self.inference_mode_combo.findData(inference_mode)
        if mode_index >= 0:
            self.inference_mode_combo.setCurrentIndex(mode_index)
        self._on_mode_changed()

        for class_name, spin in self.class_threshold_spins.items():
            spin.setValue(float(per_class_thresholds.get(class_name, DEFAULT_APPLICATION_THRESHOLDS.get(class_name, conf))))

        # 模型路径
        if model_path:
            idx = self.model_combo.findData(model_path)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

        # 背景图片
        if bg_image and os.path.exists(bg_image):
            self.bg_path_label.setText(os.path.basename(bg_image))
            self.bg_path_label.setProperty('full_path', bg_image)  # 关键：保存完整路径
            pixmap = QPixmap(bg_image).scaled(
                self.bg_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.bg_preview.setPixmap(pixmap)

    def _save_settings(self):
        """保存当前设置到数据库"""
        conf = self.conf_slider.value() / 100.0
        iou = self.iou_slider.value() / 100.0
        model_path = self.model_combo.currentData() or ''
        bg_image = self.bg_path_label.property('full_path') or ''
        inference_mode = self.inference_mode_combo.currentData() or 'standard'
        per_class_thresholds = {
            class_name: spin.value()
            for class_name, spin in self.class_threshold_spins.items()
        }

        save_user_settings(
            self.username,
            conf_threshold=conf,
            iou_threshold=iou,
            model_path=model_path,
            background_image=bg_image,
            inference_mode=inference_mode,
            per_class_thresholds=per_class_thresholds,
            healthy_margin=self.healthy_margin_spin.value()
        )

        # 发射信号通知主窗口
        self.settings_changed.emit()

        QMessageBox.information(self, "保存成功", "设置已保存")

    # ============================================================
    # 事件处理
    # ============================================================

    def _on_conf_changed(self, value):
        """置信度滑块值变化时更新显示"""
        self.conf_value_label.setText(f"{value / 100:.2f}")

    def _on_iou_changed(self, value):
        """IoU 滑块值变化时更新显示"""
        self.iou_value_label.setText(f"{value / 100:.2f}")

    def _on_mode_changed(self):
        """切换推理模式时更新应用模式参数区的可编辑状态。"""
        is_application = (self.inference_mode_combo.currentData() == 'application')
        self.healthy_margin_spin.setEnabled(is_application)
        for spin in self.class_threshold_spins.values():
            spin.setEnabled(is_application)

    def _refresh_model_list(self):
        """刷新可用模型列表"""
        self.model_combo.clear()
        models = find_available_models()
        for display_name, path in models.items():
            self.model_combo.addItem(display_name, path)

    def _browse_model(self):
        """手动浏览选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt);;所有文件 (*)"
        )
        if file_path:
            display = os.path.basename(file_path)
            self.model_combo.addItem(display, file_path)
            self.model_combo.setCurrentText(display)

    def _select_background(self):
        """选择背景图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择背景图片", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.bg_path_label.setText(os.path.basename(file_path))
            self.bg_path_label.setProperty('full_path', file_path)
            pixmap = QPixmap(file_path).scaled(
                self.bg_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.bg_preview.setPixmap(pixmap)

    def _clear_background(self):
        """清除背景图片"""
        self.bg_path_label.setText("未设置")
        self.bg_path_label.setProperty('full_path', '')
        self.bg_preview.clear()
        self.bg_preview.setText("背景预览")

    # ============================================================
    # 样式
    # ============================================================

    def _group_style(self) -> str:
        return """
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                background: white;
                margin-top: 14px;
                padding: 20px 15px 15px 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 8px;
            }
        """

    def _get_stylesheet(self) -> str:
        return """
            QLabel { font-family: "Microsoft YaHei"; font-size: 13px; }

            #primaryButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 40px;
                font-size: 14px;
                font-weight: bold;
                font-family: "Microsoft YaHei";
                min-width: 160px;
            }
            #primaryButton:hover { background-color: #1565C0; }

            #secondaryButton {
                background-color: #E3F2FD;
                color: #1976D2;
                border: 1px solid #90CAF9;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-family: "Microsoft YaHei";
            }
            #secondaryButton:hover { background-color: #BBDEFB; }

            #dangerButton {
                background-color: #FFEBEE;
                color: #C62828;
                border: 1px solid #EF9A9A;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-family: "Microsoft YaHei";
            }
            #dangerButton:hover { background-color: #FFCDD2; }

            QSlider::groove:horizontal {
                height: 6px;
                background: #E0E0E0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #1976D2;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #42A5F5;
                border-radius: 3px;
            }
        """
