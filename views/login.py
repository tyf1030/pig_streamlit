# views/login.py
import streamlit as st
import time
import db_manager as db  # 导入同一目录下的 db_manager (如果在根目录运行main.py)



st.header("🔐 用户入口")

tab1, tab2 = st.tabs(["登录", "注册"])

# --- 登录逻辑 ---
with tab1:
    with st.form("login_form"):
        user = st.text_input("用户名")
        pwd = st.text_input("密码", type="password")
        if st.form_submit_button("立即登录"):
            user_data = db.verify_login(user, pwd)
            if user_data:
                st.success("登录成功，正在跳转...")
                st.session_state.logged_in = True
                st.session_state.user_info = user_data
                time.sleep(0.5)
                st.rerun()  # 触发 main.py 重新判断路由
            else:
                st.error("账号或密码错误")

# --- 注册逻辑 ---
with tab2:
    with st.form("register_form"):
        new_u = st.text_input("新用户名")
        new_p = st.text_input("设置密码", type="password")
        if st.form_submit_button("注册"):
            if db.create_user(new_u, new_p):
                st.success("注册成功！请切换到登录标签页。")
            else:
                st.error("用户名已存在")