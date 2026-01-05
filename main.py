# main.py
import streamlit as st
import db_manager as db

# --- 初始化 (保持不变) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_info = {}
    db.init_db()

# --- 定义页面对象 (注意这里变了) ---

# 1. 登录页 (假设 login.py 也是传统脚本)
pg_login = st.Page("views/login.py", title="登录/注册", icon="🔒")

# 2. 业务页 (直接指向文件路径)
# 只要 views/dashboard.py 存在，Streamlit 就会去运行那个文件
pg_dashboard = st.Page("views/dashboard.py", title="检测控制台", icon="📸", default=True)

# 3. 管理页
pg_admin = st.Page("views/admin.py", title="用户管理", icon="🛡️")


# 4. 登出 (因为这是一个动作，可以用简单的函数，也可以写个 logout.py)
def logout():
    st.session_state.logged_in = False
    st.session_state.user_info = {}
    st.rerun()


pg_logout = st.Page(logout, title="退出登录", icon="👋")

# --- 路由逻辑 (保持不变) ---
if not st.session_state.logged_in:
    pg = st.navigation([pg_login])
else:
    nav_structure = {
        "业务功能": [pg_dashboard],
        "账户设置": [pg_logout]
    }
    if st.session_state.user_info.get('role') == 'admin':
        nav_structure = {"后台管理": [pg_admin], **nav_structure}

    pg = st.navigation(nav_structure)

pg.run()