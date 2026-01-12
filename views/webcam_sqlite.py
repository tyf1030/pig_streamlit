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

# 引入项目配置
import config 
from utils.model_loader import load_ar_model_cached, load_od_model_cached
from backend.processors import filter_and_analyze_tracking_results, process_video_regions
from backend.inference import inference_recognizer_simplified
import logging
# ==========================================
# 0. 辅助工具函数
# ==========================================s

logger = logging.getLogger("Views.StreamAnalyzer")
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
    logger.info(f"✅ OD 模型已保存: {uploaded_file.name}")
    st.toast(f"✅ OD 模型已保存: {uploaded_file.name}")

def save_uploaded_ar_model(uploaded_files):
    if not uploaded_files: return
    py_file = next((f for f in uploaded_files if f.name.endswith('.py')), None)
    pth_file = next((f for f in uploaded_files if f.name.endswith('.pth')), None)
    
    if not py_file or not pth_file:
        logger.error("上传文件格式错误")
        st.error("❌ 必须要同时上传 .py 和 .pth 文件")
        return

    py_name = os.path.splitext(py_file.name)[0]
    pth_name = os.path.splitext(pth_file.name)[0]
    if py_name != pth_name:
        logger.error("文件名不一致: {py_name}.py vs {pth_name}.pth")
        st.error(f"❌ 文件名不一致: {py_name}.py vs {pth_name}.pth")
        return

    model_dir = os.path.join(config.AR_MODEL_DIR, py_name)
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, py_file.name), "wb") as f:
        f.write(py_file.getbuffer())
    with open(os.path.join(model_dir, pth_file.name), "wb") as f:
        f.write(pth_file.getbuffer())

    logger.info(f"✅ AR 模型已保存: {uploaded_file.name}")  
    st.toast(f"✅ AR 模型已保存至: {model_dir}")

# === 修改：OnlineVideoData 支持 16 帧插值与统一数据生成 ===
class OnlineVideoData:
    def __init__(self, frames:list, timestamps:list):
        self.frames = frames # list of np.array (16)
        self.timestamps = timestamps # list of datetime (16)
        
        # OD 原始数据 (Frame-by-Frame)
        self.boxes = []
        self.conf = []
        self.cls = []
        self.id = []
        
        # AR 结果数据
        self.ar_box = [] # 用于推理的 input box list
        self.ar_id = []  # 参与 AR 的 Track ID
        self.ar_conf = [] # AR 结果置信度
        self.ar_cls = []  # AR 结果类别索引
    
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
    
    def _interpolate_bbox(self, bbox_seq):
        """简单的线性插值，填充 None 的 bbox"""
        # bbox_seq: list of [bbox or None] with length 16
        seq_len = len(bbox_seq)
        
        # 1. 找到所有非空的索引
        valid_indices = [i for i, b in enumerate(bbox_seq) if b is not None]
        
        if not valid_indices:
            # 如果全是空，返回全 0
            return [np.zeros(4) for _ in range(seq_len)]
            
        # 2. 前向填充 (Fill Forward)
        for i in range(valid_indices[0]):
            bbox_seq[i] = bbox_seq[valid_indices[0]]
            
        # 3. 后向填充 (Fill Backward)
        for i in range(valid_indices[-1] + 1, seq_len):
            bbox_seq[i] = bbox_seq[valid_indices[-1]]
            
        # 4. 中间插值
        for k in range(len(valid_indices) - 1):
            start_idx = valid_indices[k]
            end_idx = valid_indices[k+1]
            steps = end_idx - start_idx
            
            start_box = bbox_seq[start_idx]
            end_box = bbox_seq[end_idx]
            
            for step in range(1, steps):
                alpha = step / steps
                interpolated_box = start_box * (1 - alpha) + end_box * alpha
                bbox_seq[start_idx + step] = interpolated_box
                
        return bbox_seq

    def get_unified_db_data(self, action_classes, username):
        """
        [重构版] 生成统一的数据库写入数据。
        策略：
        1. 包含所有原始 OD 检测结果 (每一帧的每个框都写入)
        2. 包含 AR 结果 (使用简单复制策略，将 4 个并集框扩展为 16 帧数据)
        """
        db_rows = []
        
        # 获取图像尺寸
        img_h, img_w = 0, 0
        if self.frames:
            img_h, img_w = self.frames[0].shape[:2]

        # ==========================================
        # 部分 1: 写入所有 OD (目标检测) 原始结果
        # ==========================================
        # 这一步不管是否被 AR 选中，只要 YOLO 看到了，就记录下来
        for i in range(len(self.frames)):
            frame_boxes = self.boxes[i]
            frame_confs = self.conf[i]
            frame_clss = self.cls[i]
            
            # 获取当前帧的时间戳
            ts_str = self.timestamps[i].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            for j in range(len(frame_boxes)):
                box = frame_boxes[j]
                conf = float(frame_confs[j])
                cls_id = int(frame_clss[j])
                
                # 标记为原始检测，保留原始类别ID
                # 格式示例: "OD_Raw:0" (0通常是Person)
                category = f"OD_Raw:{cls_id}" 

                db_rows.append((
                    username,
                    "webcam_stream", img_h, img_w, category,
                    float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                    conf, ts_str
                ))

        # ==========================================
        # 部分 2: 写入 AR (行为识别) 结果
        # ==========================================
        # 这一步针对识别出的行为，生成对应的 16 条轨迹记录
        for idx in range(len(self.ar_id)):
            # 1. 获取行为类别名称
            cls_idx = self.ar_cls[idx]
            if cls_idx < len(action_classes):
                action_name = action_classes[cls_idx]
            else:
                action_name = f"Action_{cls_idx}"
            
            confidence = float(self.ar_conf[idx])
            
            # 2. 获取该行为对应的 4 个时间段的框 (List of 4 arrays)
            # 注意：这是 filter_and_analyze_tracking_results 生成的并集框
            four_boxes = self.ar_box[idx] 
            
            # 3. 遍历 4 个时间段 (Segment)
            for segment_idx in range(4):
                # 获取当前段的框 (代表这 4 帧的并集范围)
                box = four_boxes[segment_idx]
                
                # 检查数据有效性：如果该段没有检测到目标 (可能是 NaN)，则跳过不写入
                # 这样数据库里就不会有垃圾数据
                if box is None or np.isnan(box).any():
                    continue

                # 4. 简单复制策略：将这 1 个框应用到该段的 4 帧上
                start_frame = segment_idx * 4
                end_frame = start_frame + 4
                
                for i in range(start_frame, end_frame):
                    # 保护：防止帧数越界
                    if i >= len(self.timestamps): break
                    
                    # 使用每一帧各自的真实时间戳
                    ts_str = self.timestamps[i].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    db_rows.append((
                        username ,"webcam_stream", img_h, img_w, action_name,
                        float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                        confidence, ts_str
                    ))
                    
        return db_rows

# ==========================================
# 1. 定义全局共享资源类
# ==========================================
class GlobalContext:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=100)
        self.action_queue = queue.Queue(maxsize=10)
        self.db_queue = queue.Queue(maxsize=500)
        self.lock = threading.Lock()
        self.last_sample_time = 0
        self.results = {
            "action": "系统初始化...",
            "confidence": 0.0,
            "history": deque(maxlen=10),
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
# 2. 页面配置与侧边栏逻辑
# ==========================================
st.set_page_config(layout="wide", page_title="实时监控加强版")

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
    st.write(f"当前用户: {st.session_state.user_info['username']}")
    st.header("⚙️ 模型设置面板")
    with st.expander("⚙️ 目标检测 (OD) 设置", expanded=True):
        st.session_state.od_conf = st.slider("置信度阈值", 0.0, 1.0, st.session_state.od_conf, 0.05)
        st.session_state.od_iou = st.slider("IoU 阈值", 0.0, 1.0, st.session_state.od_iou, 0.05)
        od_files = [f for f in os.listdir(config.OD_MODEL_DIR) if f.endswith(('.pt', '.onnx'))]
        index_od = 0
        if st.session_state.od_model_name in od_files:
            index_od = od_files.index(st.session_state.od_model_name)
        elif od_files:
            st.session_state.od_model_name = od_files[0]
        st.session_state.od_model_name = st.selectbox("选择 OD 权重文件", od_files if od_files else ["无可用模型"], index=index_od)
        uploaded_od = st.file_uploader("⬆️ 上传 OD 模型 (.pt)", type=["pt", "onnx"])
        if uploaded_od and uploaded_od.name != st.session_state.last_saved_od_model:
            save_uploaded_od_model(uploaded_od)
            st.session_state.last_saved_od_model = uploaded_od.name
            st.rerun()
        if st.button("🔄 刷新 OD 缓存"):
            load_od_model_cached.clear()
            st.toast("OD 缓存已清除")

    with st.expander("⚙️ 行为识别 (AR) 设置", expanded=True):
        ar_dirs = [d for d in os.listdir(config.AR_MODEL_DIR) if os.path.isdir(os.path.join(config.AR_MODEL_DIR, d))]
        index_ar = 0
        if st.session_state.ar_model_name in ar_dirs:
            index_ar = ar_dirs.index(st.session_state.ar_model_name)
        elif ar_dirs:
            st.session_state.ar_model_name = ar_dirs[0]
        st.session_state.ar_model_name = st.selectbox("选择 AR 模型套件", ar_dirs if ar_dirs else ["无可用模型"], index=index_ar)
        uploaded_ar = st.file_uploader("⬆️ 上传 AR 套件 (.py + .pth)", type=["pth", "py"], accept_multiple_files=True)
        if uploaded_ar:
            if len(uploaded_ar) == 2:
                current_fp = "|".join(sorted([f.name for f in uploaded_ar]))
                if current_fp != st.session_state.last_saved_ar_model:
                    save_uploaded_ar_model(uploaded_ar)
                    st.session_state.last_saved_ar_model = current_fp
                    st.rerun()
        if st.button("🔄 刷新 AR 缓存"):
            load_ar_model_cached.clear()
            st.toast("AR 缓存已清除")
    st.divider()
    device = st.selectbox("推理设备", ["cuda:0", "cpu"], index=0)

# ==========================================
# 3. 动态加载模型
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
# 4. 全局常量与 Worker
# ==========================================
SAMPLE_INTERVAL = 0.2
BATCH_SIZE = 16
PLAYBACK_DELAY = 0.1
ACTION_CLASSES = ["正常行走", "正在跑步", "跌倒检测", "挥手求救", "静止站立", "非法入侵"]

# === 新增：数据库写入线程 ===
def db_writer_worker():
    logger.info("数据库写入线程启动")
    print(">>> 💾 数据库写入线程已启动 <<<")
    global ctx
    
    conn = sqlite3.connect(config.TEST_DATABASE, check_same_thread=False)
    cursor = conn.cursor()
    
    try:
        # 统一结果表：包含 OD 和 AR 结果
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                filename TEXT,
                height INTEGER,
                width INTEGER,
                category TEXT,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                confidence REAL,
                timestamp TEXT 
            )
        ''')
        conn.commit()
    except Exception as e:
        logger.error(f"数据库初始化错误: {e}")
        print(f"DB Init Error: {e}")

    while True:
        try:
            # 1. 阻塞等待数据
            data_batch = ctx.db_queue.get()
            # 2. 批量插入
            cursor.executemany('''
                INSERT INTO recognition_results (user_name, filename, height, width, category, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', data_batch)
            conn.commit()
            
        except Exception as e:
            logger.error(f"数据库写入错误: {e}")
            print(f"DB Write Error: {e}")
            time.sleep(1)

if not ctx.db_worker_running:
    t_db = threading.Thread(target=db_writer_worker, daemon=True)
    t_db.start()
    ctx.db_worker_running = True

def complex_worker():
    logger.info("后台行为识别线程启动")
    print(">>> 🟢 后台行为识别线程已启动 <<<")
    global ctx, ar_model, ar_pipeline 
    
    while True:
        try:
            # buffer 包含 (frame, timestamp, username)
            track_result, buffer, username = ctx.action_queue.get()
            
            # 分离帧和时间戳
            frames = [b[0] for b in buffer]
            timestamps = [b[1] for b in buffer]
            
            if ar_model is None or ar_pipeline is None:
                ctx.results["action"] = "等待 AR 模型..."
                continue

            # 初始化 OVD，传入时间戳
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
                
                # === 生成统一的 DB 数据 (关键修改) ===
                unified_db_data = online_video_data.get_unified_db_data(ACTION_CLASSES, username)
                
                # === 推送至 DB 队列 ===
                if unified_db_data and not ctx.db_queue.full():
                    ctx.db_queue.put(unified_db_data)

            else:
                action = "无目标"
                conf_val = "0.0"

            timestamp = time.strftime("%H:%M:%S")
            ctx.results["action"] = action
            ctx.results["confidence"] = conf_val
            ctx.results["last_update"] = time.time()
            ctx.results["history"].append(f"{timestamp}: {action}")
            ctx.results["status"] = "normal"
            logger.info(f"后台完成分析: {action}")
            print(f"后台完成分析: {action}")
            
        except Exception as e:
            print("\n" + "="*50)
            logger.error(f"后台 Worker 线程发生异常: {e}")
            print(">>> ❌ 后台 Worker 线程发生异常！")
            traceback.print_exc() 
            print("="*50 + "\n")
            ctx.results["status"] = "error"
            ctx.results["error_msg"] = str(e) 
            time.sleep(1)

if not ctx.worker_running:
    t = threading.Thread(target=complex_worker, daemon=True)
    t.start()
    ctx.worker_running = True
    logger.info("后台行为识别线程已启动")
    print("--- 线程初始化完成 ---")

# ==========================================
# 5. WebRTC 与 模拟检测
# ==========================================
def video_frame_callback(frame):
    # 采集当前时间 (datetime对象)
    current_dt = datetime.datetime.now()
    img = frame.to_ndarray(format="bgr24")
    
    current_time_float = current_dt.timestamp() 
    
    with ctx.lock:
        if current_time_float - ctx.last_sample_time >= SAMPLE_INTERVAL:
            if not ctx.frame_queue.full():
                # 存入元组：(图片, 采集时间)
                ctx.frame_queue.put((img, current_dt))
                ctx.last_sample_time = current_time_float
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def mock_detect(buffer_with_ts) -> tuple[list, list]:
    global od_model, pred_args
    
    # 从 buffer 中提取仅图片部分用于 YOLO
    frames = [item[0] for item in buffer_with_ts]
    
    if od_model is None:
        return frames, []
        
    result = od_model.track(frames, persist=True, **pred_args)
    processed_frames = []
    for res in result:
        processed_frames.append(res.plot())
    return processed_frames, result

# ==========================================
# 6. 主 UI
# ==========================================
st.title("✅ 稳定修复版：统一数据库结果")

c1, c2 = st.columns(2)

with c1:
    st.subheader("实时输入画面")
    webrtc_ctx = webrtc_streamer(
        key="stable-stream", 
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    status = st.empty()

with c2:
    st.subheader("分析结果监控")
    monitor_ph = st.empty()
    st.divider()
    k1, k2 = st.columns(2)
    act_disp = k1.empty()
    conf_disp = k2.empty()

st.markdown("#### 📜 行为识别结果(实时更新)")
hist_ph = st.empty()
error_ph = st.empty()

# ==========================================
# 7. 主循环
# ==========================================
buffer = [] # 此时 buffer 存的是 (frame, timestamp)

if ctx.worker_running: 
    # status = st.empty()
    
    if webrtc_ctx.state.playing:
        if not st.session_state.get("_stream_logging_flag"):
            logger.info("开启摄像头")
            st.session_state._stream_logging_flag = True
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
                error_ph.error(f"❌ 后台服务发生严重错误: {ctx.results.get('error_msg', '未知错误')}")
                break

            try:
                item = ctx.frame_queue.get(timeout=1.0) # item 是 (img, ts)
                buffer.append(item)
                status.text(f"📷 正在缓冲数据: {len(buffer)}/{BATCH_SIZE}")
            except queue.Empty:
                if not webrtc_ctx.state.playing:
                    break
                continue
                
            if len(buffer) == BATCH_SIZE:
                status.text("⚡ 正在处理批次...")
                processed, track_result = mock_detect(buffer)

                if od_model is None:
                    ctx.results["action"] = "⚠️ OD 模型未加载"
                elif not ctx.action_queue.full() and track_result:
                    clean_track_data = extract_yolo_data_to_cpu(track_result)
                    # 传入 buffer (包含时间戳)
                    current_username = st.session_state.user_info["username"]
                    ctx.action_queue.put((clean_track_data, copy.deepcopy(buffer), current_username))
                
                for img in processed:
                    monitor_ph.image(img, width="stretch", caption="Analysis View", channels="BGR")
                    curr = ctx.results
                    act_disp.metric("当前行为", curr["action"])
                    conf_disp.metric("置信度", curr['confidence'])
                    
                    history_text = ""
                    for h in reversed(list(curr["history"])):
                        history_text += f"- {h}\n"
                    if history_text:
                        hist_ph.markdown(history_text)
                    time.sleep(PLAYBACK_DELAY)
                
                buffer = []
                status.text("🟢 等待下一批数据...")
    else:
        status.info("👋 系统就绪，请点击 START 开启摄像头进行分析")
        st.session_state.is_queue_cleared = False
        st.session_state._stream_logging_flag = False