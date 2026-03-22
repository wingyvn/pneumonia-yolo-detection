"""
主窗口
=======

左侧垂直导航栏 + 右侧 QStackedWidget 页面区域。
导航栏包含三个按钮: 智能检测 / 数据可视化 / 系统设置。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget, QLabel, QSpacerItem, QSizePolicy,
    QMenu, QAction
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPalette, QBrush

from pages.detection_page import DetectionPage
from pages.visualization_page import VisualizationPage
from pages.settings_page import SettingsPage
from utils.db_manager import get_user_settings


class MainWindow(QMainWindow):
    """
    主窗口：左侧导航栏 + 右侧内容页。
    """

    # 退出登录信号
    logout_requested = pyqtSignal()

    def __init__(self, username: str):
        """
        Args:
            username: 当前登录的用户名
        """
        super().__init__()
        self.username = username
        self.setWindowTitle(f"肺炎检测系统 — {username}")
        self.setMinimumSize(1200, 750)
        self.resize(1300, 800)

        # 存储导航按钮列表，方便管理选中状态
        self.nav_buttons = []

        self._init_ui()
        self._apply_background()

    def _init_ui(self):
        """构建主界面布局"""
        # 中心容器
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ========== 左侧导航栏 ==========
        nav_widget = QWidget()
        nav_widget.setObjectName("navBar")
        nav_widget.setFixedWidth(200)
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        # 系统 Logo / 标题
        logo_label = QLabel("🫁 肺炎检测系统")
        logo_label.setObjectName("logoLabel")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedHeight(70)
        nav_layout.addWidget(logo_label)

        # 用户信息（可点击，弹出退出登录菜单）
        self.user_button = QPushButton(f"👤 {self.username}")
        self.user_button.setObjectName("userButton")
        self.user_button.setFixedHeight(40)
        self.user_button.setCursor(Qt.PointingHandCursor)
        self.user_button.clicked.connect(self._show_user_menu)
        nav_layout.addWidget(self.user_button)

        # 分隔线
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: rgba(255,255,255,0.15);")
        nav_layout.addWidget(separator)

        # 导航按钮
        nav_items = [
            ("🔍  智能检测", 0),
            ("📊  数据可视化", 1),
            ("⚙️  系统设置", 2),
        ]
        for text, index in nav_items:
            btn = QPushButton(text)
            btn.setObjectName("navButton")
            btn.setFixedHeight(50)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setCheckable(True)          # 可切换选中状态
            btn.clicked.connect(lambda checked, idx=index: self._switch_page(idx))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        # 弹簧把按钮推到上方
        nav_layout.addStretch()

        # 底部版本信息
        version_label = QLabel("v1.0  YOLOv8")
        version_label.setObjectName("versionLabel")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setFixedHeight(40)
        nav_layout.addWidget(version_label)

        main_layout.addWidget(nav_widget)

        # ========== 右侧页面区域 ==========
        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")

        # 创建三个子页面，传入用户名以便数据库操作
        self.detection_page = DetectionPage(self.username)
        self.visualization_page = VisualizationPage(self.username)
        self.settings_page = SettingsPage(self.username)

        # 当设置变更时刷新主窗口背景
        self.settings_page.settings_changed.connect(self._apply_background)
        # 当检测完成时通知可视化页面刷新数据
        self.detection_page.detection_completed.connect(
            self.visualization_page.refresh_data
        )

        self.page_stack.addWidget(self.detection_page)      # index 0
        self.page_stack.addWidget(self.visualization_page)   # index 1
        self.page_stack.addWidget(self.settings_page)        # index 2

        main_layout.addWidget(self.page_stack)

        # 默认选中第一个页面
        self._switch_page(0)

        # 应用全局样式
        self.setStyleSheet(self._get_stylesheet())

    def _switch_page(self, index: int):
        """
        切换页面并更新导航按钮选中状态。

        Args:
            index: 页面索引 (0=检测, 1=可视化, 2=设置)
        """
        self.page_stack.setCurrentIndex(index)

        # 更新按钮选中状态
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

        # 切换到可视化页面时自动刷新数据
        if index == 1:
            self.visualization_page.refresh_data()

    def _show_user_menu(self):
        """
        点击用户名时弹出菜单，包含退出登录选项。
        """
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: white;
                border: 1px solid #D0D5DD;
                border-radius: 6px;
                padding: 5px;
                font-family: "Microsoft YaHei";
                font-size: 13px;
            }
            QMenu::item {
                padding: 8px 25px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #FFEBEE;
                color: #C62828;
            }
        """)

        logout_action = QAction("🚪 退出登录", self)
        logout_action.triggered.connect(self._on_logout)
        menu.addAction(logout_action)

        # 在用户按钮下方弹出菜单
        pos = self.user_button.mapToGlobal(self.user_button.rect().bottomLeft())
        menu.exec_(pos)

    def _on_logout(self):
        """处理退出登录"""
        self.logout_requested.emit()
        self.close()

    def _apply_background(self):
        """
        根据用户设置应用背景图片。
        使用 stylesheet 方式设置背景（兼容性更好）。
        """
        settings = get_user_settings(self.username)
        bg_path = settings.get('background_image', '')

        if bg_path and os.path.exists(bg_path):
            # 将路径中的反斜杠转为正斜杠（CSS 路径要求）
            css_path = bg_path.replace('\\', '/')
            self.page_stack.setStyleSheet(f"""
                #pageStack {{
                    background-image: url("{css_path}");
                    background-repeat: no-repeat;
                    background-position: center;
                }}
            """)
        else:
            self.page_stack.setStyleSheet("#pageStack { background-color: #F0F2F5; }")

    def _get_stylesheet(self) -> str:
        """主窗口全局样式表"""
        return """
            /* ========== 导航栏 ========== */
            #navBar {
                background-color: #1A237E;
            }
            #logoLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
                font-family: "Microsoft YaHei";
                padding-top: 10px;
            }
            #userButton {
                background: transparent;
                color: rgba(255,255,255,0.8);
                border: none;
                font-size: 13px;
                font-family: "Microsoft YaHei";
            }
            #userButton:hover {
                color: white;
                background-color: rgba(255,255,255,0.08);
            }
            #navButton {
                background: transparent;
                color: rgba(255,255,255,0.75);
                border: none;
                text-align: left;
                padding-left: 25px;
                font-size: 15px;
                font-family: "Microsoft YaHei";
            }
            #navButton:hover {
                background-color: rgba(255,255,255,0.08);
                color: white;
            }
            #navButton:checked {
                background-color: rgba(255,255,255,0.15);
                color: white;
                border-left: 3px solid #42A5F5;
                font-weight: bold;
            }
            #versionLabel {
                color: rgba(255,255,255,0.4);
                font-size: 11px;
            }

            /* ========== 页面区域 ========== */
            #pageStack {
                background-color: #F0F2F5;
            }
        """
