"""
数据可视化页
=============

功能:
    - 从 SQLite 检测记录中实时生成图表
    - 三种可视化: 热力图、面积占比、检测能力柱状图
    - 历史检测统计: 按日期的检测趋势折线图
    - 紫色导出按钮: 导出当前图表为 PNG
"""

import os
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QComboBox, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from utils.db_manager import get_detection_records
from utils.detector import CLASS_NAMES, CLASS_NAMES_CN, CLASS_COLORS


def _fig_to_qpixmap(fig) -> QPixmap:
    """将 matplotlib Figure 转换为 QPixmap"""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    qimg = QImage(buf, w, h, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


class VisualizationPage(QWidget):
    """数据可视化页面: 基于历史检测记录生成统计图表"""

    def __init__(self, username: str):
        super().__init__()
        self.username = username
        self._current_figs = {}   # 缓存当前图表的 Figure 对象（供导出使用）
        self._init_ui()

    def _init_ui(self):
        """构建可视化页面布局"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)

        # ========== 页面标题 ==========
        title = QLabel("📊 数据可视化分析")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1A237E; font-family: 'Microsoft YaHei';")
        layout.addWidget(title)

        subtitle = QLabel("基于历史检测记录实时生成，执行检测后自动更新")
        subtitle.setStyleSheet("font-size: 12px; color: #666; margin-bottom: 10px;")
        layout.addWidget(subtitle)

        # ========== 上排: 三张核心图表 ==========
        top_charts = QHBoxLayout()
        top_charts.setSpacing(15)

        # 1. 目标位置热力图
        group1 = QGroupBox("🔥 目标位置热力图")
        group1.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12px; border: 1px solid #E0E0E0; border-radius: 8px; background: white; margin-top: 12px; padding-top: 18px; } QGroupBox::title { subcontrol-origin: margin; padding: 0 8px; }")
        g1_layout = QVBoxLayout(group1)
        self.chart_heatmap = QLabel("暂无数据")
        self.chart_heatmap.setAlignment(Qt.AlignCenter)
        self.chart_heatmap.setMinimumHeight(280)
        g1_layout.addWidget(self.chart_heatmap)
        btn1 = QPushButton("导出热力图")
        btn1.setObjectName("exportButton")
        btn1.clicked.connect(lambda: self._export('heatmap'))
        g1_layout.addWidget(btn1, alignment=Qt.AlignCenter)
        top_charts.addWidget(group1)

        # 2. 目标面积占比
        group2 = QGroupBox("📐 目标面积占比")
        group2.setStyleSheet(group1.styleSheet())
        g2_layout = QVBoxLayout(group2)
        self.chart_area = QLabel("暂无数据")
        self.chart_area.setAlignment(Qt.AlignCenter)
        self.chart_area.setMinimumHeight(280)
        g2_layout.addWidget(self.chart_area)
        btn2 = QPushButton("导出面积占比")
        btn2.setObjectName("exportButton")
        btn2.clicked.connect(lambda: self._export('area'))
        g2_layout.addWidget(btn2, alignment=Qt.AlignCenter)
        top_charts.addWidget(group2)

        # 3. 检测能力柱状图
        group3 = QGroupBox("📊 检测能力柱状图")
        group3.setStyleSheet(group1.styleSheet())
        g3_layout = QVBoxLayout(group3)
        self.chart_capability = QLabel("暂无数据")
        self.chart_capability.setAlignment(Qt.AlignCenter)
        self.chart_capability.setMinimumHeight(280)
        g3_layout.addWidget(self.chart_capability)
        btn3 = QPushButton("导出检测能力")
        btn3.setObjectName("exportButton")
        btn3.clicked.connect(lambda: self._export('capability'))
        g3_layout.addWidget(btn3, alignment=Qt.AlignCenter)
        top_charts.addWidget(group3)

        layout.addLayout(top_charts)

        # ========== 下排: 历史趋势图 ==========
        group4 = QGroupBox("📈 历史检测趋势")
        group4.setStyleSheet(group1.styleSheet())
        g4_layout = QVBoxLayout(group4)
        self.chart_trend = QLabel("暂无数据")
        self.chart_trend.setAlignment(Qt.AlignCenter)
        self.chart_trend.setMinimumHeight(300)
        g4_layout.addWidget(self.chart_trend)
        btn4 = QPushButton("导出趋势图")
        btn4.setObjectName("exportButton")
        btn4.clicked.connect(lambda: self._export('trend'))
        g4_layout.addWidget(btn4, alignment=Qt.AlignCenter)
        layout.addWidget(group4)

        layout.addStretch()

        scroll.setWidget(container)

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

        self.setStyleSheet(self._get_stylesheet())

    def refresh_data(self):
        """
        从数据库拉取最新记录，重新生成所有图表。
        每次切换到可视化页面或检测完成时自动调用。
        """
        records = get_detection_records(self.username, limit=500)

        # 聚合所有检测结果
        all_detections = []
        for rec in records:
            all_detections.extend(rec.get('detections', []))

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # --- 1. 热力图 ---
        fig1 = self._gen_heatmap(all_detections)
        self._current_figs['heatmap'] = fig1
        self.chart_heatmap.setPixmap(
            _fig_to_qpixmap(fig1).scaled(
                self.chart_heatmap.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        plt.close(fig1)

        # --- 2. 面积占比 ---
        fig2 = self._gen_area(all_detections)
        self._current_figs['area'] = fig2
        self.chart_area.setPixmap(
            _fig_to_qpixmap(fig2).scaled(
                self.chart_area.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        plt.close(fig2)

        # --- 3. 检测能力柱状图 ---
        fig3 = self._gen_capability(all_detections)
        self._current_figs['capability'] = fig3
        self.chart_capability.setPixmap(
            _fig_to_qpixmap(fig3).scaled(
                self.chart_capability.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        plt.close(fig3)

        # --- 4. 历史趋势 ---
        fig4 = self._gen_trend(records)
        self._current_figs['trend'] = fig4
        self.chart_trend.setPixmap(
            _fig_to_qpixmap(fig4).scaled(
                self.chart_trend.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        plt.close(fig4)

    # ============================================================
    # 图表生成
    # ============================================================

    def _gen_heatmap(self, detections: list):
        """目标位置热力图（全部历史检测汇总）"""
        fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=100)

        if detections:
            grid_size = 50
            heatmap = np.zeros((grid_size, grid_size))
            for d in detections:
                norm = d.get('bbox_norm', [0.5, 0.5, 0, 0])
                gx = min(int(norm[0] * grid_size), grid_size - 1)
                gy = min(int(norm[1] * grid_size), grid_size - 1)
                heatmap[gy, gx] += 1

            try:
                from scipy.ndimage import gaussian_filter
                heatmap = gaussian_filter(heatmap, sigma=3)
            except ImportError:
                pass

            ax.imshow(heatmap, cmap='hot', interpolation='bilinear', aspect='auto')
            ax.set_title(f'目标位置热力图 (n={len(detections)})', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center',
                    fontsize=14, color='#999', transform=ax.transAxes)
            ax.set_title('目标位置热力图', fontsize=11, fontweight='bold')

        plt.tight_layout()
        return fig

    def _gen_area(self, detections: list):
        """目标面积占比饼图"""
        fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=100)

        if detections:
            # 按类别汇总面积
            area_by_class = defaultdict(float)
            count_by_class = Counter()
            for d in detections:
                cn = d.get('class_name_cn', d.get('class_name', '未知'))
                area_by_class[cn] += d.get('area_ratio', 0)
                count_by_class[cn] += 1

            # 取平均面积
            labels = list(area_by_class.keys())
            avg_areas = [area_by_class[l] / count_by_class[l] * 100 for l in labels]

            colors = []
            for l in labels:
                idx = CLASS_NAMES_CN.index(l) if l in CLASS_NAMES_CN else -1
                if 0 <= idx < len(CLASS_COLORS):
                    c = CLASS_COLORS[idx]
                    colors.append(f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}')
                else:
                    colors.append('#999999')

            ax.pie(avg_areas, labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
            ax.set_title('各类别平均面积占比', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center',
                    fontsize=14, color='#999', transform=ax.transAxes)
            ax.set_title('目标面积占比', fontsize=11, fontweight='bold')

        plt.tight_layout()
        return fig

    def _gen_capability(self, detections: list):
        """检测能力柱状图: 每类检出数量 + 平均置信度"""
        fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=100)

        if detections:
            class_counts = Counter()
            class_conf_sum = defaultdict(float)

            for d in detections:
                cn = d.get('class_name_cn', d.get('class_name', '未知'))
                class_counts[cn] += 1
                class_conf_sum[cn] += d.get('confidence', 0)

            names = list(class_counts.keys())
            counts = [class_counts[n] for n in names]
            avg_confs = [class_conf_sum[n] / class_counts[n] for n in names]

            x = np.arange(len(names))
            w = 0.35

            bars1 = ax.bar(x - w/2, counts, w, label='检出数量', color='#42A5F5')
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + w/2, avg_confs, w, label='平均置信度', color='#66BB6A')

            ax.set_ylabel('检出数量', fontsize=10)
            ax2.set_ylabel('平均置信度', fontsize=10)
            ax2.set_ylim(0, 1.1)

            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=8, rotation=15)

            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

            ax.set_title('检测能力柱状图', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测数据', ha='center', va='center',
                    fontsize=14, color='#999', transform=ax.transAxes)
            ax.set_title('检测能力柱状图', fontsize=11, fontweight='bold')

        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return fig

    def _gen_trend(self, records: list):
        """历史检测趋势折线图: 按日期统计检测次数"""
        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=100)

        if records:
            # 按日期聚合
            date_counts = Counter()
            date_detections = Counter()
            for rec in records:
                date_str = rec.get('created_at', '')[:10]  # 取日期部分
                if date_str:
                    date_counts[date_str] += 1
                    date_detections[date_str] += rec.get('num_detections', 0)

            dates = sorted(date_counts.keys())
            counts = [date_counts[d] for d in dates]
            det_counts = [date_detections[d] for d in dates]

            ax.plot(dates, counts, 'o-', color='#1976D2', linewidth=2,
                    markersize=6, label='检测次数')
            ax.plot(dates, det_counts, 's--', color='#E53935', linewidth=2,
                    markersize=6, label='检出目标数')

            ax.set_xlabel('日期', fontsize=10)
            ax.set_ylabel('数量', fontsize=10)
            ax.legend(fontsize=9)

            # 日期标签旋转避免重叠
            if len(dates) > 5:
                plt.xticks(rotation=30, ha='right')

            ax.set_title('历史检测趋势', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '暂无检测记录', ha='center', va='center',
                    fontsize=14, color='#999', transform=ax.transAxes)
            ax.set_title('历史检测趋势', fontsize=11, fontweight='bold')

        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    # ============================================================
    # 图表导出
    # ============================================================

    def _export(self, chart_type: str):
        """导出指定图表为 PNG 文件"""
        records = get_detection_records(self.username, limit=500)
        all_detections = []
        for rec in records:
            all_detections.extend(rec.get('detections', []))

        if not all_detections and chart_type != 'trend':
            QMessageBox.information(self, "提示", "暂无检测数据，请先执行检测")
            return

        default_name = f"vis_{chart_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出图表", default_name, "PNG 图像 (*.png)"
        )
        if not file_path:
            return

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 重新生成高清图表
        if chart_type == 'heatmap':
            fig = self._gen_heatmap(all_detections)
        elif chart_type == 'area':
            fig = self._gen_area(all_detections)
        elif chart_type == 'capability':
            fig = self._gen_capability(all_detections)
        else:
            fig = self._gen_trend(records)

        fig.savefig(file_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        QMessageBox.information(self, "导出成功", f"图表已保存到:\n{file_path}")

    def _get_stylesheet(self) -> str:
        return """
            #exportButton {
                background-color: #7B1FA2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 12px;
                font-family: "Microsoft YaHei";
                min-width: 100px;
            }
            #exportButton:hover { background-color: #6A1B9A; }
        """
