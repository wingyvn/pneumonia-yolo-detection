"""
SQLite 数据库管理模块
=====================

管理三张表:
    1. users        — 用户注册/登录信息
    2. detection_records — 每次检测的记录（含检测结果 JSON）
    3. settings     — 用户个人设置（阈值、背景图等）

数据库文件自动创建在 workspace/data/app.db
"""

import os
import sqlite3
import json
import hashlib
from datetime import datetime


# 数据库文件路径
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'data')
DB_PATH = os.path.join(DB_DIR, 'app.db')


def _get_connection():
    """
    获取数据库连接。
    每次调用创建新连接，用完后需手动关闭。
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 使查询结果可以按列名访问
    return conn


def init_database():
    """
    初始化数据库，创建所有必要的表。
    使用 IF NOT EXISTS 确保重复调用不会出错。
    """
    conn = _get_connection()
    cursor = conn.cursor()

    # ========== 用户表 ==========
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,       -- 用户名（唯一）
            password_hash TEXT NOT NULL,         -- 密码的 SHA256 哈希值
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    ''')

    # ========== 检测记录表 ==========
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,              -- 操作用户
            image_name TEXT NOT NULL,            -- 原始图片文件名
            image_path TEXT,                     -- 图片存储路径
            detections_json TEXT,                -- 检测结果 JSON 字符串
            num_detections INTEGER DEFAULT 0,    -- 检出目标数量
            conf_threshold REAL,                 -- 使用的置信度阈值
            iou_threshold REAL,                  -- 使用的 IoU 阈值
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    ''')

    # ========== 设置表 ==========
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,       -- 用户名（每人一条设置）
            conf_threshold REAL DEFAULT 0.25,    -- 置信度阈值
            iou_threshold REAL DEFAULT 0.45,     -- IoU 阈值
            model_path TEXT DEFAULT '',          -- 当前选用的模型路径
            background_image TEXT DEFAULT '',    -- 背景图片路径
            remember_password INTEGER DEFAULT 0, -- 是否记住密码 (0/1)
            saved_username TEXT DEFAULT '',      -- 记住的用户名
            saved_password TEXT DEFAULT ''       -- 记住的密码（加密存储）
        )
    ''')

    conn.commit()
    conn.close()


# ============================================================
# 用户管理
# ============================================================

def _hash_password(password: str) -> str:
    """
    使用 SHA256 对密码进行哈希。
    实际生产环境建议用 bcrypt，但 SHA256 对毕设足够且无需额外依赖。
    """
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def register_user(username: str, password: str) -> tuple:
    """
    注册新用户。

    Args:
        username: 用户名
        password: 明文密码

    Returns:
        (success: bool, message: str)
    """
    if not username or not password:
        return False, "用户名和密码不能为空"
    if len(password) < 6:
        return False, "密码长度不能少于6位"

    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, _hash_password(password))
        )
        conn.commit()
        # 同时为新用户创建默认设置
        conn.execute(
            "INSERT OR IGNORE INTO settings (username) VALUES (?)",
            (username,)
        )
        conn.commit()
        return True, "注册成功"
    except sqlite3.IntegrityError:
        return False, "用户名已存在"
    finally:
        conn.close()


def verify_user(username: str, password: str) -> tuple:
    """
    验证用户登录。

    Returns:
        (success: bool, message: str)
    """
    if not username or not password:
        return False, "用户名和密码不能为空"

    conn = _get_connection()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()

    if row is None:
        return False, "用户名不存在"

    if row['password_hash'] == _hash_password(password):
        return True, "登录成功"
    else:
        return False, "密码错误"


# ============================================================
# 记住密码
# ============================================================

def save_remembered_login(username: str, password: str):
    """保存记住的登录信息"""
    conn = _get_connection()
    conn.execute('''
        INSERT OR REPLACE INTO settings (username, remember_password, saved_username, saved_password)
        VALUES (?, 1, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            remember_password = 1,
            saved_username = excluded.saved_username,
            saved_password = excluded.saved_password
    ''', ('__global__', username, _hash_password(password)))
    conn.commit()
    conn.close()


def get_remembered_login() -> tuple:
    """
    获取记住的登录信息。

    Returns:
        (username, password_hash) 或 (None, None)
    """
    conn = _get_connection()
    # 为 __global__ 创建设置行（如果不存在）
    conn.execute("INSERT OR IGNORE INTO settings (username) VALUES ('__global__')")
    conn.commit()

    row = conn.execute(
        "SELECT saved_username, saved_password, remember_password FROM settings WHERE username = '__global__'"
    ).fetchone()
    conn.close()

    if row and row['remember_password'] == 1 and row['saved_username']:
        return row['saved_username'], row['saved_password']
    return None, None


def clear_remembered_login():
    """清除记住的登录信息"""
    conn = _get_connection()
    conn.execute('''
        UPDATE settings SET remember_password = 0, saved_username = '', saved_password = ''
        WHERE username = '__global__'
    ''')
    conn.commit()
    conn.close()


# ============================================================
# 检测记录
# ============================================================

def save_detection_record(username: str, image_name: str, image_path: str,
                          detections: list, conf_threshold: float, iou_threshold: float):
    """
    保存一条检测记录。

    Args:
        username: 操作用户
        image_name: 图片文件名
        image_path: 图片路径
        detections: 检测结果列表（字典列表）
        conf_threshold: 使用的置信度阈值
        iou_threshold: 使用的 IoU 阈值
    """
    conn = _get_connection()
    conn.execute('''
        INSERT INTO detection_records
        (username, image_name, image_path, detections_json, num_detections, conf_threshold, iou_threshold)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        username, image_name, image_path,
        json.dumps(detections, ensure_ascii=False),
        len(detections),
        conf_threshold, iou_threshold
    ))
    conn.commit()
    conn.close()


def get_detection_records(username: str = None, limit: int = 100) -> list:
    """
    查询检测记录。

    Args:
        username: 过滤指定用户的记录，None 表示查询所有
        limit: 最大返回条数

    Returns:
        记录列表，每条记录为字典
    """
    conn = _get_connection()

    if username:
        rows = conn.execute(
            "SELECT * FROM detection_records WHERE username = ? ORDER BY created_at DESC LIMIT ?",
            (username, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM detection_records ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()

    conn.close()

    # 将 Row 对象转为字典列表
    records = []
    for row in rows:
        record = dict(row)
        # 解析 JSON 字段
        if record.get('detections_json'):
            record['detections'] = json.loads(record['detections_json'])
        else:
            record['detections'] = []
        records.append(record)

    return records


# ============================================================
# 用户设置
# ============================================================

def get_user_settings(username: str) -> dict:
    """获取用户的设置，不存在则返回默认值"""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM settings WHERE username = ?", (username,)
    ).fetchone()
    conn.close()

    if row:
        return dict(row)
    else:
        return {
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'model_path': '',
            'background_image': ''
        }


def save_user_settings(username: str, **kwargs):
    """
    保存/更新用户设置。

    用法示例:
        save_user_settings('admin', conf_threshold=0.3, background_image='bg1.jpg')
    """
    conn = _get_connection()

    # 确保设置行存在
    conn.execute("INSERT OR IGNORE INTO settings (username) VALUES (?)", (username,))
    conn.commit()

    # 动态构建 UPDATE 语句
    allowed_fields = {'conf_threshold', 'iou_threshold', 'model_path', 'background_image'}
    updates = []
    values = []
    for key, value in kwargs.items():
        if key in allowed_fields:
            updates.append(f"{key} = ?")
            values.append(value)

    if updates:
        values.append(username)
        conn.execute(
            f"UPDATE settings SET {', '.join(updates)} WHERE username = ?",
            values
        )
        conn.commit()

    conn.close()
