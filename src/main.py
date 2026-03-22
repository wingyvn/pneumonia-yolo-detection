"""
肺炎检测系统 — 程序入口
========================

启动流程:
    1. 初始化数据库
    2. 显示登录窗口
    3. 登录成功 → 关闭登录窗口，打开主窗口

使用方式:
    cd workspace
    python src/main.py
"""

import sys
import os

# 确保 src 目录在 Python 路径中
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 设置工作目录为 workspace（使数据路径一致）
workspace_dir = os.path.dirname(src_dir)
os.chdir(workspace_dir)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

from utils.db_manager import init_database
from login_window import LoginWindow
from main_window import MainWindow


class App:
    """
    应用程序管理器：协调登录窗口和主窗口的生命周期。
    支持退出登录后重新显示登录窗口。
    """

    def __init__(self, qt_app):
        self.qt_app = qt_app
        self.main_window = None
        self.login_window = None

    def show_login(self):
        """显示登录窗口"""
        self.login_window = LoginWindow()
        self.login_window.login_success.connect(self.on_login_success)
        self.login_window.show()

    def on_login_success(self, username: str):
        """
        登录成功回调: 创建并显示主窗口。

        Args:
            username: 登录成功的用户名
        """
        self.main_window = MainWindow(username)
        # 连接退出登录信号
        self.main_window.logout_requested.connect(self._on_logout)
        self.main_window.show()

    def _on_logout(self):
        """退出登录: 关闭主窗口，重新打开登录窗口"""
        self.main_window = None
        self.show_login()


def main():
    """程序入口"""
    # 1. 创建 QApplication（PyQt5 必须先创建 QApplication）
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 11))

    # 2. 初始化数据库（建表）
    init_database()

    # 3. 创建应用管理器并显示登录
    manager = App(app)
    manager.show_login()

    # 4. 进入事件循环
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
