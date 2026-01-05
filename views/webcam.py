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

# 引入项目配置
import config 
from utils.model_loader import load_ar_model_cached, load_od_model_cached
from backend.processors import filter_and_analyze_tracking_results, process_video_regions
from backend.inference import inference_recognizer_simplified

# ==========================================
# 0. 辅助工具函数
# ==========================================
def extract_yolo_data_to_cpu(yolo_results: list) -> list:
    """
    将 YOLO 结果转换为 CPU 上的 numpy 字典列表，便于跨线程传递。
    修复：确保 ids 长度与 boxes 数量一致，防止 "boolean index did not match" 崩溃。
    """
    cpu_data = []
    for res in yolo_results:
        # 1. 获取当前帧检测到的框的数量
        n_boxes = len(res.boxes)
        
        # 2. 处理 ID
        if res.boxes.id is not None:
            # 正常情况：有 ID，直接取用
            ids = res.boxes.id.cpu().numpy()
        else:
            # 异常情况：有框但无 ID (如刚开始检测时)，或者无框
            # 必须生成一个长度为 n_boxes 的数组，否则后续过滤时会报错
            if n_boxes > 0:
                # 生成全是 -1 的数组，表示暂无 ID
                ids = np.full((n_boxes,), -1.0) 
            else:
                # 没有框，ID 也是空的
                ids = np.array([])

        frame_data = {
            "boxes": res.boxes.xyxy.cpu().numpy(),
            "conf": res.boxes.conf.cpu().numpy(),
            "cls": res.boxes.cls.cpu().numpy(),
            "id": ids # 这里的长度现在严格等于 boxes 的长度
        }
        cpu_data.append(frame_data)
    return cpu_data

def save_uploaded_od_model(uploaded_file):
    """保存上传的 OD 模型"""
    if uploaded_file is None: return
    # 确保存储目录存在
    os.makedirs(config.OD_MODEL_DIR, exist_ok=True)
    save_path = os.path.join(config.OD_MODEL_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.toast(f"✅ OD 模型已保存: {uploaded_file.name}")

def save_uploaded_ar_model(uploaded_files):
    """保存上传的 AR 模型套件"""
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

class OnlineVideoData:
    def __init__(self, frames:list):
        self.frames = frames
        self.boxes = []
        self.conf = []
        self.cls = []
        self.id = []
        # ... 其他初始化 ...
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

# ==========================================
# 1. 定义全局共享资源类 (单例模式)
# ==========================================
class GlobalContext:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=100)
        self.action_queue = queue.Queue(maxsize=10)
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

@st.cache_resource
def get_context():
    return GlobalContext()

ctx = get_context()

# ==========================================
# 2. 页面配置与侧边栏逻辑 (移植自 new_app.py)
# ==========================================
st.set_page_config(layout="wide", page_title="实时监控加强版")

# --- 初始化 Session State ---
defaults = {
    'od_model_name': None,
    'ar_model_name': None,
    'od_conf': 0.5,
    'od_iou': 0.7,
    'last_saved_od_model': None,
    'last_saved_ar_model': None,
    'is_queue_cleared': False  # 新增：用于控制重启时的队列清理
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 确保目录存在
os.makedirs(config.OD_MODEL_DIR, exist_ok=True)
os.makedirs(config.AR_MODEL_DIR, exist_ok=True)

# --- 侧边栏 UI ---
with st.sidebar:
    st.header("⚙️ 模型设置面板")
    
    # 1. 目标检测参数
    with st.expander("⚙️ 目标检测 (OD) 设置", expanded=True):
        st.session_state.od_conf = st.slider("置信度阈值", 0.0, 1.0, st.session_state.od_conf, 0.05)
        st.session_state.od_iou = st.slider("IoU 阈值", 0.0, 1.0, st.session_state.od_iou, 0.05)
        
        # 扫描模型文件
        od_files = [f for f in os.listdir(config.OD_MODEL_DIR) if f.endswith(('.pt', '.onnx'))]
        
        # 自动选择逻辑
        index_od = 0
        if st.session_state.od_model_name in od_files:
            index_od = od_files.index(st.session_state.od_model_name)
        elif od_files:
            st.session_state.od_model_name = od_files[0]
            
        st.session_state.od_model_name = st.selectbox(
            "选择 OD 权重文件", 
            od_files if od_files else ["无可用模型"],
            index=index_od
        )
        
        # 上传
        uploaded_od = st.file_uploader("⬆️ 上传 OD 模型 (.pt)", type=["pt", "onnx"])
        if uploaded_od and uploaded_od.name != st.session_state.last_saved_od_model:
            save_uploaded_od_model(uploaded_od)
            st.session_state.last_saved_od_model = uploaded_od.name
            st.rerun()

        if st.button("🔄 刷新 OD 缓存"):
            load_od_model_cached.clear()
            st.toast("OD 缓存已清除")

    # 2. 行为识别参数
    with st.expander("⚙️ 行为识别 (AR) 设置", expanded=True):
        ar_dirs = [d for d in os.listdir(config.AR_MODEL_DIR) if os.path.isdir(os.path.join(config.AR_MODEL_DIR, d))]
        
        # 自动选择逻辑
        index_ar = 0
        if st.session_state.ar_model_name in ar_dirs:
            index_ar = ar_dirs.index(st.session_state.ar_model_name)
        elif ar_dirs:
            st.session_state.ar_model_name = ar_dirs[0]

        st.session_state.ar_model_name = st.selectbox(
            "选择 AR 模型套件", 
            ar_dirs if ar_dirs else ["无可用模型"],
            index=index_ar
        )
        
        # 上传
        uploaded_ar = st.file_uploader(
            "⬆️ 上传 AR 套件 (.py + .pth)", 
            type=["pth", "py"], 
            accept_multiple_files=True
        )
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

# 加载 OD 模型
if st.session_state.od_model_name and st.session_state.od_model_name != "无可用模型":
    try:
        od_path = os.path.join(config.OD_MODEL_DIR, st.session_state.od_model_name)
        od_model, pred_args = load_od_model_cached(
            model_path=od_path, 
            device=device, 
            conf=st.session_state.od_conf, 
            iou=st.session_state.od_iou
        )
    except Exception as e:
        st.error(f"OD 模型加载失败: {e}")

# 加载 AR 模型
if st.session_state.ar_model_name and st.session_state.ar_model_name != "无可用模型":
    try:
        ar_base = os.path.join(config.AR_MODEL_DIR, st.session_state.ar_model_name)
        # 假设文件名与文件夹名一致，这是 new_app.py 的逻辑
        pth_path = os.path.join(ar_base, st.session_state.ar_model_name + ".pth")
        cfg_path = os.path.join(ar_base, st.session_state.ar_model_name + ".py")
        ar_model, ar_pipeline, _ = load_ar_model_cached(
            pth_path=pth_path, 
            cfg_path=cfg_path, 
            device=device
        )
    except Exception as e:
        st.error(f"AR 模型加载失败: {e}")

# ==========================================
# 4. 全局常量与 Worker
# ==========================================
SAMPLE_INTERVAL = 0.2
BATCH_SIZE = 16
PLAYBACK_DELAY = 0.1

def complex_worker():
    print(">>> 🟢 后台行为识别线程已启动 <<<")
    # 引用全局变量，注意：当主线程更改模型时，这里下次循环会读取到新的全局对象
    global ctx, ar_model, ar_pipeline 
    
    while True:
        try:
            track_result, frames = ctx.action_queue.get()
            
            # 检查模型是否就绪
            if ar_model is None or ar_pipeline is None:
                # 模型未加载时跳过推理，避免崩溃
                # 更新状态但不报错，因为用户可能正在切换模型
                ctx.results["action"] = "等待 AR 模型..."
                continue

            online_video_data = OnlineVideoData(frames)
            online_video_data.load_cpu_data(track_result)
            
            # 后处理逻辑
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
            
            # 只有当有检测到目标时才进行 AR 推理
            if len(online_video_data.ar_box) > 0:
                video_roi = process_video_regions(
                    frames=online_video_data.frames, 
                    detections=online_video_data.ar_box
                )
                preds = inference_recognizer_simplified(ar_model, video_roi, ar_pipeline)
                online_video_data.from_mmaction_result(preds)
                action = online_video_data.ar_cls.__str__()
                conf_val = online_video_data.ar_conf.__str__()
            else:
                action = "无目标"
                conf_val = "0.0"

            timestamp = time.strftime("%H:%M:%S")
            ctx.results["action"] = action
            ctx.results["confidence"] = conf_val
            ctx.results["last_update"] = time.time()
            ctx.results["history"].append(f"{timestamp}: {action}")
            ctx.results["status"] = "normal" # 恢复正常状态
            
            print(f"后台完成分析: {action}")
            
        except Exception as e:
            # 完整的错误堆栈打印
            print("\n" + "="*50)
            print(">>> ❌ 后台 Worker 线程发生异常！")
            print(f">>> 错误类型: {type(e).__name__}")
            print(f">>> 错误详情: {e}")
            print("-" * 20 + " 完整堆栈 " + "-" * 20)
            traceback.print_exc() 
            print("="*50 + "\n")
            
            # 更新全局状态 (UI显示用)
            ctx.results["status"] = "error"
            ctx.results["error_msg"] = str(e) 
            
            time.sleep(1)

# 启动线程
if not ctx.worker_running:
    t = threading.Thread(target=complex_worker, daemon=True)
    t.start()
    ctx.worker_running = True
    print("--- 线程初始化完成 ---")

# ==========================================
# 5. WebRTC 与 模拟检测
# ==========================================
def video_frame_callback(frame):
    current_time = time.time()
    img = frame.to_ndarray(format="bgr24")
    
    with ctx.lock:
        if current_time - ctx.last_sample_time >= SAMPLE_INTERVAL:
            if not ctx.frame_queue.full():
                ctx.frame_queue.put(img)
                ctx.last_sample_time = current_time
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def mock_detect(frames) -> tuple[list, list]:
    global od_model, pred_args
    
    # 保护逻辑：如果没有模型，原样返回
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
st.title("✅ 稳定修复版：含侧边栏控制")

c1, c2 = st.columns(2)

with c1:
    st.subheader("摄像头输入")
    # 接收 webrtc_streamer 返回的上下文，用于判断播放状态
    webrtc_ctx = webrtc_streamer(
        key="stable-stream", 
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with c2:
    st.subheader("分析结果监控")
    monitor_ph = st.empty()
    st.divider()
    k1, k2 = st.columns(2)
    act_disp = k1.empty()
    conf_disp = k2.empty()

st.markdown("#### 📜 历史记录 (实时更新)")
hist_ph = st.empty()
error_ph = st.empty()

# ==========================================
# 7. 主循环
# ==========================================
buffer = []

if ctx.worker_running: 
    status = st.empty()
    
    # 只有当摄像头正在播放时，才进行处理
    if webrtc_ctx.state.playing:
        status.empty() # 播放时隐藏提示
        
        # === 启动清理逻辑 (防止重启时出现旧帧堆积) ===
        if not st.session_state.is_queue_cleared:
            status.text("🧹 正在清理缓存...")
            with ctx.lock:
                while not ctx.frame_queue.empty():
                    try: ctx.frame_queue.get_nowait()
                    except: pass
                while not ctx.action_queue.empty():
                    try: ctx.action_queue.get_nowait()
                    except: pass
            st.session_state.is_queue_cleared = True
            print(">>> 队列已清空，准备接收新画面")
        # ===========================================
    
        while True:
            # 错误检查
            if ctx.results.get("status") == "error":
                error_ph.error(f"❌ 后台服务发生严重错误: {ctx.results.get('error_msg', '未知错误')}")
                # break # 出错跳出

            # 尝试获取数据
            try:
                # 使用 timeout 避免死锁，同时配合 webrtc 状态退出
                f = ctx.frame_queue.get(timeout=1.0)
                buffer.append(f)
                status.text(f"📷 正在缓冲数据: {len(buffer)}/{BATCH_SIZE}")
            except queue.Empty:
                # 如果摄像头已停止，跳出循环
                if not webrtc_ctx.state.playing:
                    break
                continue
                
            # 攒够 Batch 处理
            if len(buffer) == BATCH_SIZE:
                status.text("⚡ 正在处理批次...")
                
                # A. 检测
                processed, track_result = mock_detect(buffer)

                # B. 发送给后台
                # 检查模型是否加载
                if od_model is None:
                    ctx.results["action"] = "⚠️ OD 模型未加载"
                # 正常发送
                elif not ctx.action_queue.full() and track_result:
                    clean_track_data = extract_yolo_data_to_cpu(track_result)
                    ctx.action_queue.put((clean_track_data, copy.deepcopy(buffer)))
                
                # C. 回放更新
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
        # 摄像头未开启时的显示
        status.info("👋 系统就绪，请点击 START 开启摄像头进行分析")
        # 重置清理标记，确保下次开启时再次清理
        st.session_state.is_queue_cleared = False