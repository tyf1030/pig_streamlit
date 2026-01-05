import streamlit as st
import os
import queue
import time
import tempfile
import shutil
import config  # 你的全局配置文件
import logging
# === 1. 导入工具库 ===
from utils.video_processor import convert_video_to_h264, check_ffmpeg_installed
from utils.model_loader import load_od_model_cached, load_ar_model_cached

# === 2. 导入核心后端 ===
try:
    from backend.structures import ODResult, ARResult, PlottedResult
    from backend.video_io import VideoReader
    from backend.processors import filter_and_analyze_tracking_results, process_video_regions
    from backend.inference import inference_recognizer_simplified
    from backend.utils.visualization import draw_detection_boxes_batch, process_image_sequence
    from backend.utils.exporters import get_res_to_sqlite, get_coco_annotations, get_annotated_images_zipfile
except ImportError as e:
    st.error(f"❌ 导入失败: {e}")
    st.stop()

# --- 页面配置 ---
st.set_page_config(
    page_title="猪只行为识别系统 Pro",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 状态初始化 ---
def init_session_state():
    defaults = {
        'video_path': None,           # 最终用于显示和推理的视频路径 (H.264)
        'raw_video_path': None,       # 原始上传文件路径 (用于去重)
        'processing_result': None,    # 存储结果对象 (PlottedResult)
        'output_video_path': None,    # 输出视频路径
        'result_dir': None,           # 结果输出目录
        'od_model_name': None,        # 当前 OD 模型
        'ar_model_name': None,        # 当前 AR 模型
        'od_conf': 0.25,
        'od_iou': 0.7,
        'last_saved_od_model': None,  # 记录上次保存的 OD 模型文件名
        'last_saved_ar_model': None,  # 记录上次保存的 AR 模型标识
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # 确保目录存在
    for d in [config.OD_MODEL_DIR, config.AR_MODEL_DIR, config.OUTPUT_DIR, "temp_uploads"]:
        os.makedirs(d, exist_ok=True)

init_session_state()

# --- 辅助函数：保存上传的模型 ---
def save_uploaded_od_model(uploaded_file):
    if uploaded_file is None: return
    save_path = os.path.join(config.OD_MODEL_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.toast(f"✅ OD 模型已保存: {uploaded_file.name}")

def save_uploaded_ar_model(uploaded_files):
    if not uploaded_files: return
    
    # 简单的逻辑：尝试找到 .py 和 .pth
    py_file = next((f for f in uploaded_files if f.name.endswith('.py')), None)
    pth_file = next((f for f in uploaded_files if f.name.endswith('.pth')), None)
    
    if not py_file or not pth_file:
        st.error("❌ 必须要同时上传 .py 和 .pth 文件")
        return

    # 检查文件名一致性 (例如 model.py 和 model.pth)
    py_name = os.path.splitext(py_file.name)[0]
    pth_name = os.path.splitext(pth_file.name)[0]
    
    if py_name != pth_name:
        st.error(f"❌ 文件名不一致: {py_name}.py vs {pth_name}.pth")
        return

    # 创建模型文件夹
    model_dir = os.path.join(config.AR_MODEL_DIR, py_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存文件
    with open(os.path.join(model_dir, py_file.name), "wb") as f:
        f.write(py_file.getbuffer())
    with open(os.path.join(model_dir, pth_file.name), "wb") as f:
        f.write(pth_file.getbuffer())
        
    st.toast(f"✅ AR 模型已保存至: {model_dir}")

# --- 核心任务逻辑 ---
def run_analysis_pipeline(conf, iou, device):
    """执行完整的视频分析流程"""
    status = st.empty()
    bar = st.progress(0)
    
    try:
        # 1. 准备模型
        status.text("⏳ 正在初始化 AI 引擎...")
        
        # 加载 OD 模型
        od_path = os.path.join(config.OD_MODEL_DIR, st.session_state.od_model_name)
        
        print("od_path:", od_path, flush=True)
        od_model, od_args = load_od_model_cached(od_path, device, conf, iou)
        
        # 加载 AR 模型
        ar_base = os.path.join(config.AR_MODEL_DIR, st.session_state.ar_model_name)
        pth_path = os.path.join(ar_base, st.session_state.ar_model_name + ".pth")
        cfg_path = os.path.join(ar_base, st.session_state.ar_model_name + ".py")
        ar_model, ar_pipeline, _ = load_ar_model_cached(pth_path, cfg_path, device)
        
        # 2. 视频流读取
        video_path = st.session_state.video_path
        status.text(f"📥 正在读取视频: {os.path.basename(video_path)}")
        
        data_queue = queue.Queue(maxsize=15)
        reader = VideoReader(video_path, data_queue)
        
        # 使用线程启动读取
        import threading
        read_thread = threading.Thread(target=reader.process_video, daemon=True)
        read_thread.start()
        
        video_name = os.path.basename(video_path)
        plotted_result = PlottedResult(video_name, reader.fps)
        
        # 3. 逐段处理
        segment_count = 0
        while True:
            try:
                video_data = data_queue.get(timeout=2)
            except queue.Empty:
                if not read_thread.is_alive(): break
                continue 
            
            if video_data is None: break
            
            status.text(f"⚙️ 正在分析第 {segment_count+1} 片段 ({len(video_data.frames_list)} 帧)...")
            
            # A. 目标检测
            tracks = od_model.track(video_data.frames_list, **od_args)
            od_res = ODResult()
            od_res.from_yolo_result(tracks)
            video_data.od_res = od_res
            
            # B. 轨迹过滤
            ar_boxes_map = filter_and_analyze_tracking_results(
                boxes_list=video_data.od_res.boxes,
                track_ids_list=video_data.od_res.id,
                class_ids_list=video_data.od_res.cls,
                non_target_odcls=[],
                id_num_threshold=8
            )
            
            ar_res = ARResult()
            for track_id, boxes in ar_boxes_map.items():
                ar_res.boxes.append(boxes)
                ar_res.id.append(track_id)
            video_data.act_res = ar_res
            
            # C. 行为识别
            if len(video_data.frames_list) >= 16 and len(ar_res.boxes) > 0:
                regions = process_video_regions(
                    frames=video_data.frames_list, 
                    detections=video_data.act_res.boxes
                )
                if ar_model:
                    preds = inference_recognizer_simplified(ar_model, regions, ar_pipeline)
                    video_data.act_res.from_mmaction_result(preds)
            
            # D. 绘图与合并
            video_data.merge_res_to_anno()
            plotted_imgs = draw_detection_boxes_batch(
                images=video_data.frames_list, 
                detections_list=video_data.anno
            )
            plotted_result.add_res(plotted_imgs, video_data.anno)
            
            segment_count += 1
            if reader.total_frames > 0:
                p = min(segment_count * 16 / reader.total_frames, 0.95)
                bar.progress(p)
                
        # 4. 合成
        status.text("🎬 正在渲染最终视频...")
        output_dir = os.path.join(config.OUTPUT_DIR, f"recognized_{video_name}")
        
        final_video, res_dir = process_image_sequence(
            images=plotted_result.images,
            output_dir=output_dir,
            output_type="video",
            fps=reader.fps
        )
        
        st.session_state.processing_result = plotted_result
        st.session_state.output_video_path = final_video
        st.session_state.result_dir = res_dir
        
        bar.progress(1.0)
        status.success("✅ 分析完成！")
        return True
        
    except Exception as e:
        status.error("❌ 处理中断")
        st.error(f"详细错误: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False

# --- 侧边栏：还原 Gradio 布局 ---
with st.sidebar:
    st.header("⚙️ 设置面板")
    
    # 1. 目标检测参数设置
    with st.expander("⚙️ 目标检测参数设置", expanded=False):
        # 参数
        conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05, key="od_conf_slider")
        iou = st.slider("IoU 阈值", 0.0, 1.0, 0.7, 0.05, key="od_iou_slider")
        
        # 模型选择
        od_files = [f for f in os.listdir(config.OD_MODEL_DIR) if f.endswith(('.pt', '.onnx'))] if os.path.exists(config.OD_MODEL_DIR) else []
        
        # 自动刷新选择：如果刚保存了新模型，强制选中它
        index_to_select = 0
        if st.session_state.last_saved_od_model and st.session_state.last_saved_od_model in od_files:
            index_to_select = od_files.index(st.session_state.last_saved_od_model)
        
        st.session_state.od_model_name = st.selectbox(
            "选择模型权重文件", 
            od_files if od_files else ["无可用模型"],
            index=0
        )
        
        
        # 上传新模型
        uploaded_od = st.file_uploader("⬆️ 上传新的目标检测模型权重", type=["pt", "onnx"])
        if uploaded_od:
            # 只有当上传的文件名与上次保存的不同时，才执行保存
            if st.session_state.last_saved_od_model != uploaded_od.name:
                with st.spinner(f"正在自动保存 {uploaded_od.name}..."):
                    save_uploaded_od_model(uploaded_od)
                    # 更新状态，防止循环保存
                    st.session_state.last_saved_od_model = uploaded_od.name
                    # 立即重新运行以刷新下拉列表
                    st.rerun()

        # 重新加载按钮 (Streamlit 中通常不需要，但为了还原界面)
        if st.button("🔄 重新加载 OD 模型"):
            load_od_model_cached.clear() # 清除缓存以强制重载
            st.toast("已清除缓存，下次运行时将重新加载模型")

    # 2. 行为识别参数设置
    with st.expander("⚙️ 行为识别参数设置", expanded=False):
        # 模型选择
        ar_dirs = [d for d in os.listdir(config.AR_MODEL_DIR) if os.path.isdir(os.path.join(config.AR_MODEL_DIR, d))] if os.path.exists(config.AR_MODEL_DIR) else []
        st.session_state.ar_model_name = st.selectbox(
            "选择行为识别模型权重", 
            ar_dirs if ar_dirs else ["无可用模型"],
            index=0
        )
        
        # 上传新模型
        uploaded_ar = st.file_uploader(
            "⬆️ 上传新的行为识别模型 (需同时上传 .pth 和 .py)", 
            type=["pth", "py"], 
            accept_multiple_files=True
        )
        
        if uploaded_ar:
            # 只有当上传了2个文件时才尝试保存
            if len(uploaded_ar) == 2:
                # 生成一个简单的指纹（将两个文件名排序后拼接），用于判断是否是新的一对文件
                current_ar_fingerprint = "|".join(sorted([f.name for f in uploaded_ar]))
                
                if st.session_state.last_saved_ar_model != current_ar_fingerprint:
                    with st.spinner("正在自动保存 AR 模型套件..."):
                        # save_uploaded_ar_model 内部已经处理了 toast 消息
                        # 但我们需要它返回一个布尔值来决定是否更新 last_saved_ar_model
                        # 这里我们假设原来的函数没有返回值，直接在这里调用
                        
                        # 为了安全，先检查一下是否真的是一堆匹配的文件，再保存
                        py_file = next((f for f in uploaded_ar if f.name.endswith('.py')), None)
                        pth_file = next((f for f in uploaded_ar if f.name.endswith('.pth')), None)
                        
                        if py_file and pth_file:
                             # 执行保存
                            save_uploaded_ar_model(uploaded_ar)
                            # 记录指纹，防止重复保存
                            st.session_state.last_saved_ar_model = current_ar_fingerprint
                            st.rerun()
                        else:
                            st.error("请确保上传的是一个 .py 和一个 .pth 文件")
            elif len(uploaded_ar) > 2:
                st.warning("⚠️ 请只上传 2 个文件（.py 和 .pth）")

        if st.button("🔄 重新加载 AR 模型"):
            load_ar_model_cached.clear()
            st.toast("已清除缓存，下次运行时将重新加载模型")
            
    # 3. 通用参数设置
    with st.expander("⚙️ 通用参数设置", expanded=False):
        device = st.selectbox("推理设备", ["cuda:0", "cpu"], index=0)
        save_db = st.checkbox("检测结果是否写入数据库", value=True)

# --- 主界面 ---
st.title("🐖 猪只行为智能分析系统")

col1, col2 = st.columns(2)

# === 左侧：上传与预览 ===
with col1:
    st.subheader("1. 视频输入")
    
    # 检查 FFmpeg
    if not check_ffmpeg_installed():
        st.error("🚨 未检测到 FFmpeg！")
        st.stop()
        
    uploaded_file = st.file_uploader("上传视频 (支持 MP4, AVI, MKV...)", type=['mp4', 'avi', 'mov', 'mkv', 'flv'])
    
    if uploaded_file:
        # 1. 生成当前上传文件的唯一指纹 (文件名_文件大小)
        # 这样即使你点击侧边栏，只要没换文件，这个指纹就不会变
        file_fingerprint = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # 2. 检查是否是新文件
        # 如果 session_state 中记录的指纹和当前不一样，才执行处理逻辑
        if st.session_state.get('current_file_fingerprint') != file_fingerprint:
            
            # 生成带时间戳的文件名 (只在真正处理新文件时生成一次)
            timestamp = int(time.time())
            raw_name = f"raw_{timestamp}_{uploaded_file.name}"
            raw_path = os.path.join("temp_uploads", raw_name)
            
            clean_name = f"clean_{timestamp}_{os.path.splitext(uploaded_file.name)[0]}.mp4"
            clean_path = os.path.join("temp_uploads", clean_name)
            
            # 开始处理流程
            with st.status("📦 检测到新视频，正在处理...", expanded=True) as status:
                st.write("1/2 正在保存原始文件...")
                with open(raw_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.write("2/2 正在进行 H.264 标准化转码 (可能需要几秒)...")
                success, msg = convert_video_to_h264(raw_path, clean_path)
                
                if success:
                    # 更新 Session State
                    st.session_state.video_path = clean_path
                    st.session_state.raw_video_path = raw_path
                    # 关键：记录当前文件的指纹
                    st.session_state.current_file_fingerprint = file_fingerprint
                    
                    # 清空旧结果
                    st.session_state.processing_result = None
                    st.session_state.output_video_path = None
                    st.session_state.result_dir = None
                    
                    status.update(label="✅ 视频处理完成", state="complete", expanded=False)
                    st.rerun() # 刷新页面以加载视频播放器
                else:
                    status.update(label="❌ 转码失败", state="error", expanded=True)
                    st.error(msg)
                    # 如果转码失败，清空指纹，允许用户重试
                    if 'current_file_fingerprint' in st.session_state:
                        del st.session_state.current_file_fingerprint

    # 3. 视频播放器 (完全依赖 Session State)
    # 这样即使 script 重新运行，因为 if 指纹判断不通过，不会重复转码，直接跳到这里播放
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        st.video(st.session_state.video_path, format="video/mp4")
        
        can_run = st.session_state.od_model_name != "无可用模型" and st.session_state.ar_model_name != "无可用模型"
        if not can_run:
            st.warning("⚠️ 请在左侧选择有效的模型")
            
        if st.button("🚀 开始识别", type="primary", disabled=not can_run):
            success = run_analysis_pipeline(conf, iou, device)
            if success and save_db:
                try:
                    get_res_to_sqlite(st.session_state.processing_result, config.VIDEO_RECOGNITION_DATABASE)
                    st.toast("💾 数据库已更新")
                except Exception as e:
                    st.error(f"数据库错误: {e}")
                st.rerun()

# === 右侧：结果展示 ===
with col2:
    st.subheader("2. 分析结果")
    
    if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
        st.video(st.session_state.output_video_path)
        
        st.divider()
        st.write("⬇️ **数据导出**")
        
        c1, c2, c3 = st.columns(3)
        
        with open(st.session_state.output_video_path, "rb") as f:
            c1.download_button("🎥 下载视频", f, file_name="result.mp4", mime="video/mp4")
            
        if st.session_state.result_dir:
            json_path = get_coco_annotations(st.session_state.processing_result, st.session_state.result_dir)
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    c2.download_button("📋 下载 JSON", f, file_name="annotations.json", mime="application/json")
        
        if st.button("🖼️ 打包关键帧 (ZIP)"):
            with st.spinner("正在打包..."):
                zip_path = get_annotated_images_zipfile(
                    images=st.session_state.processing_result.images,
                    output_dir=st.session_state.result_dir,
                    video_name=st.session_state.processing_result.video_name,
                    sample_step=1
                )
                with open(zip_path, "rb") as f:
                    st.download_button("📦 点击下载 ZIP", f, file_name="frames.zip", mime="application/zip")
                    
    else:
        st.info("👈 请先上传视频并点击开始识别")