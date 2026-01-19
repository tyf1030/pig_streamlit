import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import config
import os
import numpy as np

# st.set_page_config(page_title="数据分析看板", layout="wide", page_icon="📊")

st.title("📊 猪只行为数据可视化")

# ==========================================
# 1. 数据加载与预处理
# ==========================================
@st.cache_data(ttl=60)
def load_data():
    # 兼容配置读取
    db_path = getattr(config, 'VIDEO_RECOGNITION_DATABASE', "recognition_results.db")

    if not os.path.exists(db_path):
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        query = """
            SELECT 
                img_id, filename, user_name, category, 
                bbox_x, bbox_y, bbox_w, bbox_h, 
                height, width, confidence, timestamp 
            FROM recognition_results
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return df

        # --- 数据清洗与特征工程 ---

        # 
        df['user_name'] = df['user_name'].fillna("未知用户")
        df['user_name'] = df['user_name'].replace('', '未知用户')
        
        # 1. 来源区分
        # 摄像头数据通常包含 'webcam'，上传视频则是文件名
        df['source_type'] = df['filename'].apply(
            lambda x: '摄像头' if 'webcam' in str(x).lower() else '视频文件'
        )

        # 2. 时间转换 (兼容性处理)
        # 上传视频的 timestamp 可能是 float (相对秒数)，摄像头是 'YYYY-MM-DD...'
        # 我们尝试强制转换，无法解析的设为 NaT
        df['dt_record'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['dt_record'].dt.date
        
        # 3. 计算检测框面积
        df['box_area'] = df['bbox_w'] * df['bbox_h']
        
        # 4. 归一化坐标 (防止除以0)
        df['norm_x'] = df['bbox_x'] / df['width'].replace(0, 1)
        df['norm_y'] = df['bbox_y'] / df['height'].replace(0, 1)
        
        return df
    except Exception as e:
        st.error(f"读取数据库失败: {e}")
        return pd.DataFrame()

df_raw = load_data()

# ==========================================
# 2. 侧边栏筛选器
# ==========================================
with st.sidebar:
    st.header("🔍 数据筛选")
    
    if df_raw.empty:
        st.warning("数据库暂无数据")
        st.stop()

    # 1. 来源筛选
    all_sources = ["全部"] + list(df_raw['source_type'].unique())
    selected_source = st.selectbox("选择数据来源", all_sources)

    # [新增] 1.5 用户筛选
    all_users = ["全部"] + list(df_raw['user_name'].unique())
    selected_user = st.selectbox("选择操作用户", all_users)
    
    # 2. 类别筛选
    all_cats = ["全部"] + list(df_raw['category'].unique())
    selected_cat = st.selectbox("选择行为类别", all_cats)
    
    # 3. 置信度筛选
    min_conf = st.slider("最低置信度过滤", 0.0, 1.0, 0.25, 0.05)
    
    # 4. 时间筛选 (UI 保留，但功能暂时禁用)
    # st.text("选择时间范围")
    
    valid_dates = df_raw['date'].dropna().sort_values()
    if not valid_dates.empty:
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        default_val = [min_date, max_date]
    else:
        min_date = pd.Timestamp.now().date()
        default_val = [min_date, min_date]

    date_range = st.date_input(label="选择时间范围",value = default_val)


# --- 应用筛选逻辑 ---
df = df_raw.copy()

if selected_source != "全部":
    df = df[df['source_type'] == selected_source]

if selected_cat != "全部":
    df = df[df['category'] == selected_cat]


# [新增] 用户过滤逻辑
if selected_user != "全部":
    df = df[df['user_name'] == selected_user]

df = df[df['confidence'] >= min_conf]

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
    # 只选了一天的情况
    single_date = date_range[0]
    df = df[df['date'] == single_date]

# ==========================================
# 3. 统计展示 (保持不变)
# ==========================================
st.markdown("### 📈 关键指标")
m1, m2, m3, m4 = st.columns(4)
m1.metric("总识别次数", f"{len(df):,}")
m2.metric("涉及来源数量", f"{df['filename'].nunique()}")
top_cat = df['category'].value_counts().idxmax() if not df.empty else "无"
m3.metric("最高频行为", top_cat)
avg_conf = df['confidence'].mean() if not df.empty else 0
m4.metric("平均置信度", f"{avg_conf:.2%}")

st.markdown("---")

# ==========================================
# 4. 可视化图表
# ==========================================
# (代码结构与之前一致，仅确保 df 不为空)

if df.empty:
    st.info("当前筛选条件下无数据。")
else:
    # --- Row 1 ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. 行为类别占比")
        cat_counts = df['category'].value_counts().reset_index()
        cat_counts.columns = ['类别', '数量']
        fig_pie = px.pie(cat_counts, values='数量', names='类别', hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, width='stretch')

    with col2:
        st.subheader("2. 行为发生时间趋势")
        # 尝试使用 timestamp 进行重采样，如果全是 NaT (上传视频)，则回退到按索引/数量展示
        if not df.empty:
            timeline_df = df.set_index('dt_record').resample('H')['category'].count().reset_index()
            timeline_df.columns = ['时间', '识别数量']
            
            # 如果想看细分流的行为
            fig_line = px.area(timeline_df, x='时间', y='识别数量', 
                               title="每小时识别频次趋势", markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
            


    # --- Row 2: 空间分布 尺寸分布 ---
    
    # 1. 生成热力图
    fig_heat = px.scatter(
            df, 
            x='norm_x', 
            y='norm_y', 
            color='category', # 关键：按类别着色，生成图例
            render_mode='webgl',
            hover_data=['confidence', 'timestamp'] # 悬停显示更多信息
    )
    fig_heat.update_traces(
            marker=dict(
                size=5,       # 点的大小
                opacity=0.3,  # 关键：设置半透明，重叠处颜色会加深
            ),
            selector=dict(mode='markers')
    )
    fig_heat.update_layout(
            width=600,
            height=600,
            autosize=False,
            margin=dict(l=10, r=10, t=30, b=10),
            
            # 使用透明或深色背景，让彩色点更明显
            plot_bgcolor='rgba(20, 20, 20, 0.05)', 
            paper_bgcolor='rgba(0,0,0,0)',
            
            # 图例设置
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.5)" # 图例背景半透明，防止遮挡
            )
    )
    fig_heat.update_xaxes(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            title="水平位置",
            constrain='domain'
        )
        
    fig_heat.update_yaxes(
        range=[1, 0], # 翻转 Y 轴
        showgrid=False,
        zeroline=False,
        title="垂直位置",
        scaleanchor="x",
        scaleratio=1,
        constrain='domain'
    )



    fig_rects = go.Figure()
        
    categories = df['category'].unique()
    
    # 为了性能，限制最大绘制数量
    max_samples = 3000
    df_plot = df
    if len(df) > max_samples:
        st.caption(f"⚠️ 数据量较大，已随机采样 {max_samples} 条进行展示")
        df_plot = df.sample(max_samples)
    
    for cat in categories:
        cat_df = df_plot[df_plot['category'] == cat]
        
        w_half = cat_df['bbox_w'].values / 2
        h_half = cat_df['bbox_h'].values / 2
        
        n = len(cat_df)
        x_pts = np.empty((n, 6))
        y_pts = np.empty((n, 6))
        
        # 填充 X 坐标: [-w, w, w, -w, -w, None]
        x_pts[:, 0] = -w_half
        x_pts[:, 1] = w_half
        x_pts[:, 2] = w_half
        x_pts[:, 3] = -w_half
        x_pts[:, 4] = -w_half
        x_pts[:, 5] = np.nan 
        
        # 填充 Y 坐标: [h, h, -h, -h, h, None]
        y_pts[:, 0] = h_half
        y_pts[:, 1] = h_half
        y_pts[:, 2] = -h_half
        y_pts[:, 3] = -h_half
        y_pts[:, 4] = h_half
        y_pts[:, 5] = np.nan
        
        x_flat = x_pts.flatten()
        y_flat = y_pts.flatten()
        
        fig_rects.add_trace(go.Scattergl(
            x=x_flat,
            y=y_flat,
            mode='lines',
            name=cat,
            opacity=0.15,
            line=dict(width=1),
            hoverinfo='name'
        ))


    fig_rects.update_layout(
        xaxis_title="宽度 (px) - 相对于中心",
        yaxis_title="高度 (px) - 相对于中心",
        showlegend=True,
        height=500,
        xaxis=dict(showgrid=False, zeroline=True, zerolinewidth=2, zerolinecolor='grey'),
        # 合并后的 yaxis 配置
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False, 
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor='grey'
        )
    )

    # 4. 渲染
    col_heat, col_rects = st.columns([1, 1])
    with col_heat:
        st.subheader("行为空间分布")
        st.plotly_chart(fig_heat, use_container_width=False)
    with col_rects:
        st.subheader("检测框尺寸分布")
        st.plotly_chart(fig_rects, use_container_width=False)

    # --- Row 3: 质量分析 ---
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("4. 类别置信度分布")
        fig_box = px.box(df, x='confidence', y='category', color='category', orientation='h')
        st.plotly_chart(fig_box, width='stretch')
    
    with c2:
        st.subheader("5. 检测框大小分布")
        fig_hist = px.histogram(df, x='box_area', color='category', nbins=50, opacity=0.7)
        st.plotly_chart(fig_hist, width='stretch')
    
# --- 原始数据 ---
with st.expander("查看原始数据明细"):
    st.dataframe(
        df[['timestamp', 'filename', 'category', 'confidence', 'box_area']].head(100),
        width='stretch'
    )