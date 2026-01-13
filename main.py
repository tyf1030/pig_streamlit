# main.py
import streamlit as st
import utils.db_manager as db
import logging
import os
from logging.handlers import RotatingFileHandler

class UserContextFilter(logging.Filter):
    """
    这个过滤器会自动把当前登录的用户名注入到日志记录中。
    在格式化字符串中可以使用 %(user)s 来引用。
    """
    def filter(self, record):
        try:
            # 尝试访问 Streamlit 的 session_state
            # 注意：如果是在后台线程(Worker)中，访问 session_state 可能会失败，
            # 这里用 try-except 兜底，失败时归类为 'System'
            if hasattr(st, 'session_state') and 'user_info' in st.session_state:
                # 获取用户名，默认为 Guest
                record.user = st.session_state.user_info.get('username', 'Guest')
            else:
                record.user = 'Guest'
        except Exception:
            # 如果完全脱离了 Streamlit 上下文（比如后台纯算法线程）
            record.user = 'System'
        return True

# === 2. 日志配置函数 ===
def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建 Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 防止 Streamlit 热重载导致重复添加 Handler
    if not logger.handlers:
        # 1. 创建 Handler：限制大小 5MB，保留 3 个备份
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "pig_app.log"),
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,             # 保留3个旧文件
            encoding='utf-8'
        )
        
        # 2. 设置格式：重点是加入了 [%(user)s]
        formatter = logging.Formatter(
            '%(asctime)s - [%(user)s] - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 3. 添加过滤器
        file_handler.addFilter(UserContextFilter())
        
        logger.addHandler(file_handler)
        
        # (可选) 如果你也想在控制台看到，可以解开下面两行
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(UserContextFilter())
        logger.addHandler(console_handler)

setup_logging()
logger = logging.getLogger("Main")

# --- 初始化 (保持不变) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_info = {}
    db.init_db()
    logger.info("系统启动，初始化数据库连接")

# --- 定义页面对象 (注意这里变了) ---

# 1. 登录页 (假设 login.py 也是传统脚本)
pg_login = st.Page("views/login.py", title="登录/注册", icon="🔒")

# # 2. 业务页 (直接指向文件路径)
# # 只要 views/dashboard.py 存在，Streamlit 就会去运行那个文件
# pg_dashboard = st.Page("views/new_app.py", title="上传视频", icon="📸", default=True)

# # 3. 管理页
# pg_admin = st.Page("views/web_cam.py", title="用户管理", icon="🛡️")


# 4. 登出 (因为这是一个动作，可以用简单的函数，也可以写个 logout.py)
def logout():
    logger.info("用户退出登录")
    st.session_state.logged_in = False
    st.session_state.user_info = {}
    st.rerun()


pg_logout = st.Page(logout, title="退出登录", icon="👋")

# --- 路由逻辑 (保持不变) ---
if not st.session_state.logged_in:
    pg = st.navigation([pg_login])
else:
    nav_structure = [
        st.Page("views/new_app.py", title="上传视频", icon="📸", default=True),
        st.Page("views/webcam_sqlite.py", title="SQLite", icon="🛡️"),
        st.Page("views/dashboard.py", title="Dashboard", icon="📊"),
        pg_logout,

    ]
    # if st.session_state.user_info.get('role') == 'admin':
    #     nav_structure = {"后台管理": [pg_admin], **nav_structure}

    pg = st.navigation(nav_structure)

pg.run()