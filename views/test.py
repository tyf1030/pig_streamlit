import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import threading
import time
import queue
import random
import os
import shutil
from collections import deque
import numpy as np
import copy
import traceback
import sqlite3
import datetime
import pandas as pd  # 新增：用于美化表格显示

# 引入项目配置
import config 
from utils.model_loader import load_ar_model_cached, load_od_model_cached
from backend.processors import filter_and_analyze_tracking_results, process_video_regions
from backend.inference import inference_recognizer_simplified

# ==========================================
# 🎨 0. 界面美化配置 (仅修改前端样式)
# ==========================================
st.set_page_config(
    page_title="智能监控驾驶舱",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全局字体与背景优化 */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 侧边栏美化：浅灰色背景，保证文字清晰 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    /* 标题样式 */
    h1 {
        color: #1f2937;
        font-size: 1.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* 卡片容器样式 */
    .css-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        border: 1px solid #e5e7eb;
    }
    
    /* 指标文字 */
    .metric-label { font-size: 0.8rem; color: #6b7280; }
    .metric-value { font-size: 1.2rem; font-weight: bold; color: #111827; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 辅助工具函数 (保持原始逻辑不变)
# ==========================================
def extract_yolo_data_to_cpu(yolo_results: list) -> list:
    """
    将 YOLO 结果转换为 CPU 上的 numpy 字典列表，便于跨线程传递。
    """
    cpu_data = []
    for res in yolo_results:
        n_boxes = len(res.boxes)
        if res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy()
        else:
            if n_boxes > 0:
                ids = np.full((n_boxes,), -1.0) 
            else:
                ids = np.array([])

        frame_data = {
            "boxes": res.boxes.xyxy.cpu().numpy(),
            "conf": res.boxes.conf.cpu().numpy(),
            "cls": res.boxes.cls.cpu().numpy(),
            "id": ids 
        }
        cpu_data.append(frame_data)
    return cpu_data

def save_uploaded_od_model(uploaded_file):
    if uploaded_file is None: return
    os.makedirs(config.OD_MODEL_DIR, exist_ok=True)
    save_path = os.path.join(config.OD_MODEL_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.toast(f"✅ OD 模型已保存: {uploaded_file.name}")

def save_uploaded_ar_model(uploaded_files):
    if not uploaded_files: return
    py_file = next((f for f in uploaded_files if f.name.endswith('.py')), None)
    pth_file = next((f for f in uploaded_files if f.name.endswith('.pth')), None)
    
    if not py_file or not pth_file:
        st.error("❌ 必须要同时上传 .py 和 .pth 文件")
        return

    py_name = os.path.splitext(py_file.name)[0]
    pth_name = os.path.splitext(pth_file.name)[0]
    if py_name != pth_name:
        st.error(f"❌ 文件名不一致: {py_name}.py vs {pth_name}.pth")
        return

    model_dir = os.path.join(config.AR_MODEL_DIR, py_name)
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, py_file.name), "wb") as f:
        f.write(py_file.getbuffer())
    with open(os.path.join(model_dir, pth_file.name), "wb") as f:
        f.write(pth_file.getbuffer())
        
    st.toast(f"✅ AR 模型已保存至: {model_dir}")

# === OnlineVideoData (保持原始逻辑不变) ===
class OnlineVideoData:
    def __init__(self, frames:list, timestamps:list):
        self.frames = frames 
        self.timestamps = timestamps
        
        self.boxes = []
        self.conf = []
        self.cls = []
        self.id = []
        
        self.ar_box = [] 
        self.ar_id = []  
        self.ar_conf = [] 
        self.ar_cls = []
    
    def load_cpu_data(self, cpu_results: list):
        for result in cpu_results:
            self.boxes.append(result["boxes"])
            self.conf.append(result["conf"])
            self.cls.append(result["cls"])
            self.id.append(result["id"])
    
    def from_mmaction_result(self, mm_res:list):
        for result in mm_res:
            scores = result.pred_score.cpu().numpy()
            self.ar_conf.append(np.max(scores))
            self.ar_cls.append(np.argmax(scores))
    
    def get_unified_db_data(self, action_classes):
        db_rows = []
        img_h, img_w = 0, 0
        if self.frames:
            img_h, img_w = self.frames[0].shape[:2]

        for i in range(len(self.frames)):
            frame_boxes = self.boxes[i]
            frame_confs = self.conf[i]
            frame_clss = self.cls[i]
            ts_str = self.timestamps[i].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            for j in range(len(frame_boxes)):
                box = frame_boxes[j]
                conf = float(frame_confs[j])
                cls_id = int(frame_clss[j])
                category = f"OD_Raw:{cls_id}" 

                db_rows.append((
                    "webcam_stream", img_h, img_w, category,
                    float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                    conf, ts_str
                ))

        for idx in range(len(self.ar_id)):
            cls_idx = self.ar_cls[idx]
            if cls_idx < len(action_classes):
                action_name = action_classes[cls_idx]
            else:
                action_name = f"Action_{cls_idx}"
            
            confidence = float(self.ar_conf[idx])
            four_boxes = self.ar_box[idx] 
            
            for segment_idx in range(4):
                box = four_boxes[segment_idx]
                if box is None or np.isnan(box).any():
                    continue

                start_frame = segment_idx * 4
                end_frame = start_frame + 4
                
                for i in range(start_frame, end_frame):
                    if i >= len(self.timestamps): break
                    ts_str = self.timestamps[i].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    db_rows.append((
                        "webcam_stream", img_h, img_w, action_name,
                        float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                        confidence, ts_str
                    ))
                    
        return db_rows

# ==========================================
# 2. 定义全局共享资源类
# ==========================================
class GlobalContext:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=100)
        self.action_queue = queue.Queue(maxsize=10)
        self.db_queue = queue.Queue(maxsize=500)
        self.lock = threading.Lock()
        self.last_sample_time = 0
        self.results = {
            "action": "等待数据...",
            "confidence": 0.0,
            "history": deque(maxlen=50), # 增加点长度方便表格显示
            "last_update": time.time(),
            "status": "normal",
            "error_msg": ""
        }
        self.worker_running = False
        self.db_worker_running = False

@st.cache_resource
def get_context():
    return GlobalContext()

ctx = get_context()

# ==========================================
# 3. 侧边栏逻辑 (保持原始逻辑，仅样式微调)
# ==========================================
defaults = {
    'od_model_name': None,
    'ar_model_name': None,
    'od_conf': 0.5,
    'od_iou': 0.7,
    'last_saved_od_model': None,
    'last_saved_ar_model': None,
    'is_queue_cleared': False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

os.makedirs(config.OD_MODEL_DIR, exist_ok=True)
os.makedirs(config.AR_MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.TEST_DATABASE), exist_ok=True)

with st.sidebar:
    st.header("⚙️ 系统控制台")
    
    with st.expander("👁️ 目标检测 (OD)", expanded=True):
        st.session_state.od_conf = st.slider("置信度阈值", 0.0, 1.0, st.session_state.od_conf, 0.05)
        st.session_state.od_iou = st.slider("IoU 阈值", 0.0, 1.0, st.session_state.od_iou, 0.05)
        od_files = [f for f in os.listdir(config.OD_MODEL_DIR) if f.endswith(('.pt', '.onnx'))]
        index_od = 0
        if st.session_state.od_model_name in od_files:
            index_od = od_files.index(st.session_state.od_model_name)
        elif od_files:
            st.session_state.od_model_name = od_files[0]
        st.session_state.od_model_name = st.selectbox("OD 权重", od_files if od_files else ["无可用模型"], index=index_od)
        
        uploaded_od = st.file_uploader("上传 OD 模型", type=["pt", "onnx"])
        if uploaded_od and uploaded_od.name != st.session_state.last_saved_od_model:
            save_uploaded_od_model(uploaded_od)
            st.session_state.last_saved_od_model = uploaded_od.name
            st.rerun()

    with st.expander("🧠 行为识别 (AR)", expanded=True):
        ar_dirs = [d for d in os.listdir(config.AR_MODEL_DIR) if os.path.isdir(os.path.join(config.AR_MODEL_DIR, d))]
        index_ar = 0
        if st.session_state.ar_model_name in ar_dirs:
            index_ar = ar_dirs.index(st.session_state.ar_model_name)
        elif ar_dirs:
            st.session_state.ar_model_name = ar_dirs[0]
        st.session_state.ar_model_name = st.selectbox("AR 套件", ar_dirs if ar_dirs else ["无可用模型"], index=index_ar)
        
        uploaded_ar = st.file_uploader("上传 AR 套件", type=["pth", "py"], accept_multiple_files=True)
        if uploaded_ar:
            if len(uploaded_ar) == 2:
                current_fp = "|".join(sorted([f.name for f in uploaded_ar]))
                if current_fp != st.session_state.last_saved_ar_model:
                    save_uploaded_ar_model(uploaded_ar)
                    st.session_state.last_saved_ar_model = current_fp
                    st.rerun()
                    
    st.markdown("---")
    if st.button("🧹 刷新模型缓存", use_container_width=True):
        load_od_model_cached.clear()
        load_ar_model_cached.clear()
        st.toast("缓存已清除")
    
    device = st.selectbox("推理设备", ["cuda:0", "cpu"], index=0)

# ==========================================
# 4. 动态加载模型
# ==========================================
od_model = None
pred_args = {}
ar_model = None
ar_pipeline = None

if st.session_state.od_model_name and st.session_state.od_model_name != "无可用模型":
    try:
        od_path = os.path.join(config.OD_MODEL_DIR, st.session_state.od_model_name)
        od_model, pred_args = load_od_model_cached(model_path=od_path, device=device, conf=st.session_state.od_conf, iou=st.session_state.od_iou)
    except Exception as e:
        st.error(f"OD 模型加载失败: {e}")

if st.session_state.ar_model_name and st.session_state.ar_model_name != "无可用模型":
    try:
        ar_base = os.path.join(config.AR_MODEL_DIR, st.session_state.ar_model_name)
        pth_path = os.path.join(ar_base, st.session_state.ar_model_name + ".pth")
        cfg_path = os.path.join(ar_base, st.session_state.ar_model_name + ".py")
        ar_model, ar_pipeline, _ = load_ar_model_cached(pth_path=pth_path, cfg_path=cfg_path, device=device)
    except Exception as e:
        st.error(f"AR 模型加载失败: {e}")

# ==========================================
# 5. Worker 线程 (保持原始逻辑不变)
# ==========================================
SAMPLE_INTERVAL = 0.2
BATCH_SIZE = 16
PLAYBACK_DELAY = 0.1
ACTION_CLASSES = ["正常行走", "正在跑步", "跌倒检测", "挥手求救", "静止站立", "非法入侵"]

def db_writer_worker():
    # print(">>> 💾 数据库写入线程已启动 <<<")
    global ctx
    conn = sqlite3.connect(config.TEST_DATABASE, check_same_thread=False)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT, height INTEGER, width INTEGER, category TEXT,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                confidence REAL, timestamp TEXT 
            )
        ''')
        conn.commit()
    except Exception as e:
        print(f"DB Init Error: {e}")

    while True:
        try:
            data_batch = ctx.db_queue.get()
            cursor.executemany('''
                INSERT INTO recognition_results (filename, height, width, category, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_batch)
            conn.commit()
        except Exception as e:
            print(f"DB Write Error: {e}")
            time.sleep(1)

if not ctx.db_worker_running:
    t_db = threading.Thread(target=db_writer_worker, daemon=True)
    t_db.start()
    ctx.db_worker_running = True

def complex_worker():
    # print(">>> 🟢 后台行为识别线程已启动 <<<")
    global ctx, ar_model, ar_pipeline 
    
    while True:
        try:
            track_result, buffer = ctx.action_queue.get()
            frames = [b[0] for b in buffer]
            timestamps = [b[1] for b in buffer]
            
            if ar_model is None or ar_pipeline is None:
                ctx.results["action"] = "等待 AR 模型..."
                continue

            online_video_data = OnlineVideoData(frames, timestamps)
            online_video_data.load_cpu_data(track_result)
            
            ar_box = filter_and_analyze_tracking_results(
                boxes_list=online_video_data.boxes,
                track_ids_list=online_video_data.id,
                class_ids_list=online_video_data.cls,
                non_target_odcls=[],
                id_num_threshold=8
            )
            for k,v in ar_box.items():
                online_video_data.ar_box.append(v)
                online_video_data.ar_id.append(k)
            
            if len(online_video_data.ar_box) > 0:
                video_roi = process_video_regions(
                    frames=online_video_data.frames, 
                    detections=online_video_data.ar_box
                )
                preds = inference_recognizer_simplified(ar_model, video_roi, ar_pipeline)
                online_video_data.from_mmaction_result(preds)
                action = online_video_data.ar_cls.__str__()
                conf_val = online_video_data.ar_conf.__str__()
                
                unified_db_data = online_video_data.get_unified_db_data(ACTION_CLASSES)
                if unified_db_data and not ctx.db_queue.full():
                    ctx.db_queue.put(unified_db_data)

            else:
                action = "无目标"
                conf_val = "0.0"

            timestamp = time.strftime("%H:%M:%S")
            ctx.results["action"] = action
            ctx.results["confidence"] = conf_val
            ctx.results["last_update"] = time.time()
            # 记录字典到历史，方便 DataFrame 显示
            ctx.results["history"].append({"Time": timestamp, "Event": action, "Conf": f"{float(conf_val):.2f}"})
            ctx.results["status"] = "normal"
            
        except Exception as e:
            traceback.print_exc() 
            ctx.results["status"] = "error"
            ctx.results["error_msg"] = str(e) 
            time.sleep(1)

if not ctx.worker_running:
    t = threading.Thread(target=complex_worker, daemon=True)
    t.start()
    ctx.worker_running = True

# ==========================================
# 6. WebRTC 与 模拟检测
# ==========================================
def video_frame_callback(frame):
    current_dt = datetime.datetime.now()
    img = frame.to_ndarray(format="bgr24")
    current_time_float = current_dt.timestamp() 
    
    with ctx.lock:
        if current_time_float - ctx.last_sample_time >= SAMPLE_INTERVAL:
            if not ctx.frame_queue.full():
                ctx.frame_queue.put((img, current_dt))
                ctx.last_sample_time = current_time_float
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def mock_detect(buffer_with_ts) -> tuple[list, list]:
    global od_model, pred_args
    frames = [item[0] for item in buffer_with_ts]
    
    if od_model is None:
        return frames, []
        
    result = od_model.track(frames, persist=True, **pred_args)
    processed_frames = []
    for res in result:
        processed_frames.append(res.plot())
    return processed_frames, result

# ==========================================
# 7. 主界面布局 (重点修改区域)
# ==========================================
st.title("🛡️ 智能监控驾驶舱")

# 使用 [0.5, 2, 2, 0.5] 比例，将左右两边留白，从而强制中间两个画面变小
# 这样可以减少视频的高度，让下方的历史记录不用滚动就能看到
c_spacer1, c1, c2, c_spacer2 = st.columns([0.5, 2, 2, 0.5])

with c1:
    st.markdown("###### 📹 实时画面")
    # 使用 container 包裹以应用样式
    with st.container():
        webrtc_ctx = webrtc_streamer(
            key="stable-stream", 
            video_frame_callback=video_frame_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    status = st.empty()

with c2:
    st.markdown("###### 🔍 分析结果")
    monitor_ph = st.empty()

# 状态指标区
st.divider()
k1, k2, k3 = st.columns([1, 1, 2])

with k1:
    st.markdown('<div class="css-card"><div class="metric-label">当前行为</div>', unsafe_allow_html=True)
    act_disp = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with k2:
    st.markdown('<div class="css-card"><div class="metric-label">置信度</div>', unsafe_allow_html=True)
    conf_disp = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with k3:
    st.markdown('<div class="css-card"><div class="metric-label">实时事件日志</div>', unsafe_allow_html=True)
    hist_ph = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

error_ph = st.empty()

# ==========================================
# 8. 主循环
# ==========================================
buffer = [] 

if ctx.worker_running: 
    if webrtc_ctx.state.playing:
        status.empty()
        
        if not st.session_state.is_queue_cleared:
            status.text("🧹 正在清理缓存...")
            with ctx.lock:
                while not ctx.frame_queue.empty():
                    try: ctx.frame_queue.get_nowait()
                    except: pass
                while not ctx.action_queue.empty():
                    try: ctx.action_queue.get_nowait()
                    except: pass
                while not ctx.db_queue.empty():
                    try: ctx.db_queue.get_nowait()
                    except: pass
            st.session_state.is_queue_cleared = True
    
        while True:
            if ctx.results.get("status") == "error":
                error_ph.error(f"❌ 系统错误: {ctx.results.get('error_msg', '未知错误')}")
                break

            try:
                item = ctx.frame_queue.get(timeout=1.0)
                buffer.append(item)
                status.caption(f"🚀 数据处理中: {len(buffer)}/{BATCH_SIZE}")
            except queue.Empty:
                if not webrtc_ctx.state.playing:
                    break
                continue
                
            if len(buffer) == BATCH_SIZE:
                processed, track_result = mock_detect(buffer)

                if od_model is None:
                    ctx.results["action"] = "⚠️ OD 模型未加载"
                elif not ctx.action_queue.full() and track_result:
                    clean_track_data = extract_yolo_data_to_cpu(track_result)
                    ctx.action_queue.put((clean_track_data, copy.deepcopy(buffer)))
                
                # 回放与界面更新
                for img in processed:
                    monitor_ph.image(img, channels="BGR", use_container_width=True)
                    curr = ctx.results
                    
                    # 更新卡片文字
                    act_disp.markdown(f'<div class="metric-value">{curr["action"]}</div>', unsafe_allow_html=True)
                    conf_disp.markdown(f'<div class="metric-value">{curr["confidence"]}</div>', unsafe_allow_html=True)
                    
                    # 使用表格显示历史，更加美观紧凑
                    if curr["history"]:
                        df = pd.DataFrame(list(curr["history"]))
                        # 倒序显示，最新的在最上面
                        hist_ph.dataframe(
                            df.iloc[::-1], 
                            height=120, 
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        hist_ph.info("暂无事件")

                    time.sleep(PLAYBACK_DELAY)
                
                buffer = []
    else:
        status.info("👈 请点击 START 开启摄像头")
        # 初始占位显示
        monitor_ph.markdown("""
            <div style="height:300px;background:#f0f2f6;display:flex;align-items:center;justify-content:center;color:#aaa;border-radius:8px;">
            等待摄像头启动...
            </div>
        """, unsafe_allow_html=True)
        act_disp.markdown('<div class="metric-value">-</div>', unsafe_allow_html=True)
        conf_disp.markdown('<div class="metric-value">-</div>', unsafe_allow_html=True)
        hist_ph.info("等待数据...")
        st.session_state.is_queue_cleared = False