"""
登录 / 注册窗口
================

功能:
    - 用户登录与注册（两个 Tab 切换）
    - 输入校验（用户名非空、密码≥6位、确认密码一致）
    - "记住密码" 复选框
    - 登录成功后发射信号，由 main.py 打开主窗口
"""

import sys
import os

# 将 src 目录加入路径，以便导入 utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QCheckBox, QMessageBox,
    QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from utils.db_manager import (
    init_database, register_user, verify_user,
    save_remembered_login, get_remembered_login, clear_remembered_login,
    _hash_password
)


class LoginWindow(QWidget):
    """
    登录/注册窗口。

    信号:
        login_success(str): 登录成功时发射，参数为用户名
    """

    # 自定义信号：登录成功时发射，携带用户名
    login_success = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("肺炎检测系统 — 用户登录")
        self.setFixedSize(420, 500)
        self.setStyleSheet(self._get_stylesheet())
        self._init_ui()
        self._load_remembered_login()

    def _init_ui(self):
        """构建界面布局"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)

        # ========== 标题 ==========
        title = QLabel("🫁 肺炎检测系统")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Microsoft YaHei", 18, QFont.Bold))
        title.setStyleSheet("color: #1565C0; margin-bottom: 10px;")
        layout.addWidget(title)

        subtitle = QLabel("基于 YOLOv8 的智能医学影像分析平台")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 20px;")
        layout.addWidget(subtitle)

        # ========== Tab 切换: 登录 / 注册 ==========
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Microsoft YaHei", 10))

        # --- 登录 Tab ---
        login_tab = QWidget()
        login_layout = QVBoxLayout(login_tab)
        login_layout.setSpacing(12)

        login_layout.addWidget(QLabel("用户名"))
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("请输入用户名")
        login_layout.addWidget(self.login_username)

        login_layout.addWidget(QLabel("密码"))
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("请输入密码")
        self.login_password.setEchoMode(QLineEdit.Password)  # 密码隐藏显示
        login_layout.addWidget(self.login_password)

        # 记住密码复选框
        self.remember_checkbox = QCheckBox("记住密码")
        self.remember_checkbox.setStyleSheet("color: #555;")
        login_layout.addWidget(self.remember_checkbox)

        # 登录按钮
        self.btn_login = QPushButton("登  录")
        self.btn_login.setObjectName("primaryButton")
        self.btn_login.clicked.connect(self._on_login)
        login_layout.addWidget(self.btn_login)

        login_layout.addStretch()
        self.tabs.addTab(login_tab, "登录")

        # --- 注册 Tab ---
        reg_tab = QWidget()
        reg_layout = QVBoxLayout(reg_tab)
        reg_layout.setSpacing(12)

        reg_layout.addWidget(QLabel("用户名"))
        self.reg_username = QLineEdit()
        self.reg_username.setPlaceholderText("请设置用户名")
        reg_layout.addWidget(self.reg_username)

        reg_layout.addWidget(QLabel("密码"))
        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("请设置密码（至少6位）")
        self.reg_password.setEchoMode(QLineEdit.Password)
        reg_layout.addWidget(self.reg_password)

        reg_layout.addWidget(QLabel("确认密码"))
        self.reg_confirm = QLineEdit()
        self.reg_confirm.setPlaceholderText("请再次输入密码")
        self.reg_confirm.setEchoMode(QLineEdit.Password)
        reg_layout.addWidget(self.reg_confirm)

        # 注册按钮
        self.btn_register = QPushButton("注  册")
        self.btn_register.setObjectName("primaryButton")
        self.btn_register.clicked.connect(self._on_register)
        reg_layout.addWidget(self.btn_register)

        reg_layout.addStretch()
        self.tabs.addTab(reg_tab, "注册")

        layout.addWidget(self.tabs)

        # 支持回车键登录
        self.login_password.returnPressed.connect(self._on_login)
        self.reg_confirm.returnPressed.connect(self._on_register)

    def _load_remembered_login(self):
        """加载记住的登录信息"""
        username, pwd_hash = get_remembered_login()
        if username:
            self.login_username.setText(username)
            # 用特殊标记表示密码来自记住功能（不能还原明文）
            self.login_password.setText("••••••••")
            self.login_password.setProperty("remembered_hash", pwd_hash)
            self.remember_checkbox.setChecked(True)

    def _on_login(self):
        """处理登录按钮点击"""
        username = self.login_username.text().strip()
        password = self.login_password.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "提示", "用户名和密码不能为空")
            return

        # 检查是否使用记住的密码
        remembered_hash = self.login_password.property("remembered_hash")
        if remembered_hash and password == "••••••••":
            # 使用记住的密码哈希直接比对
            from utils.db_manager import _get_connection
            conn = _get_connection()
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
            conn.close()

            if row and row['password_hash'] == remembered_hash:
                self.login_success.emit(username)
                self.close()
                return
            else:
                QMessageBox.warning(self, "错误", "记住的密码已失效，请重新输入")
                self.login_password.clear()
                self.login_password.setProperty("remembered_hash", None)
                return

        # 正常登录验证
        success, msg = verify_user(username, password)
        if success:
            # 处理记住密码
            if self.remember_checkbox.isChecked():
                save_remembered_login(username, password)
            else:
                clear_remembered_login()

            self.login_success.emit(username)
            self.close()
        else:
            QMessageBox.warning(self, "登录失败", msg)

    def _on_register(self):
        """处理注册按钮点击"""
        username = self.reg_username.text().strip()
        password = self.reg_password.text().strip()
        confirm = self.reg_confirm.text().strip()

        if not username or not password:
            QMessageBox.warning(self, "提示", "用户名和密码不能为空")
            return

        if password != confirm:
            QMessageBox.warning(self, "提示", "两次输入的密码不一致")
            return

        success, msg = register_user(username, password)
        if success:
            QMessageBox.information(self, "成功", "注册成功！请切换到登录页登录。")
            # 自动切换到登录 Tab 并填入用户名
            self.tabs.setCurrentIndex(0)
            self.login_username.setText(username)
            self.login_password.setFocus()
        else:
            QMessageBox.warning(self, "注册失败", msg)

    def _get_stylesheet(self) -> str:
        """返回窗口样式表"""
        return """
            QWidget {
                background-color: #F5F7FA;
                font-family: "Microsoft YaHei";
            }
            QLineEdit {
                padding: 10px 14px;
                border: 1px solid #D0D5DD;
                border-radius: 8px;
                background: white;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 2px solid #1976D2;
            }
            QPushButton#primaryButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#primaryButton:hover {
                background-color: #1565C0;
            }
            QPushButton#primaryButton:pressed {
                background-color: #0D47A1;
            }
            QTabWidget::pane {
                border: 1px solid #E0E0E0;
                border-radius: 8px;
                background: white;
            }
            QTabBar::tab {
                padding: 8px 30px;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                color: #1976D2;
                border-bottom: 2px solid #1976D2;
            }
            QLabel {
                font-size: 13px;
                color: #333;
            }
        """
