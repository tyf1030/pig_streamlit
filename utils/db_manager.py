# db_manager.py
import sqlite3
import bcrypt
import pandas as pd

DB_NAME = 'users.db'


def init_db():
    """初始化数据库表结构"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    # 预制管理员: admin / 123456
    try:
        c.execute("SELECT username FROM users WHERE username = 'admin'")
        if not c.fetchone():
            hashed_pw = bcrypt.hashpw('123456'.encode(), bcrypt.gensalt()).decode()
            c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                      ('admin', hashed_pw, 'admin'))
            conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def verify_login(username, password):
    """校验登录，成功返回用户信息字典，失败返回 None"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT username, password, role FROM users WHERE username = ?", (username,))
    data = c.fetchone()
    conn.close()

    if data and bcrypt.checkpw(password.encode(), data[1].encode()):
        return {"username": data[0], "role": data[2]}
    return None


def create_user(username, password, role='user'):
    """创建新用户"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  (username, hashed_pw, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def fetch_all_users():
    """获取所有用户数据（DataFrame格式）"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT username, role FROM users", conn)
    conn.close()
    return df