import streamlit as st
import os
import queue
import time
import tempfile
import shutil
import config  # 你的全局配置文件
import logging

# === 1. 导入工具库 (保留原有逻辑) ===
try:
    from utils.video_processor import convert_video_to_h264, check_ffmpeg_installed
    from utils.model_loader import load_od_model_cached, load_ar_model_cached
except ImportError as e:
    # 为了防止代码因为缺少本地文件直接报错无法运行，这里做个提示，实际运行时请确保文件存在
    st.error(f"❌ 基础工具库导入失败: {e}")
    # 注意：如果是在没有 utils 文件夹的环境运行，这里会报错停止。
    # 为了展示完整 UI 代码，这里不 st.stop()，但在实际生产中建议开启。
    # st.stop()

# === 2. 导入核心后端 ===
try:
    from backend.structures import ODResult, ARResult, PlottedResult
    from backend.video_io import VideoReader
    from backend.processors import filter_and_analyze_tracking_results, process_video_regions
    from backend.inference import inference_recognizer_simplified
    from backend.utils.visualization import draw_detection_boxes_batch, process_image_sequence
    from backend.utils.exporters import get_res_to_sqlite, get_coco_annotations, get_annotated_images_zipfile
except ImportError as e:
    st.error(f"❌ 后端库导入失败: {e}")
    st.stop()

logger = logging.getLogger("Views.VideoAnalyzer")

# --- 页面配置 ---
st.set_page_config(
    page_title="猪只行为识别系统 Pro",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义 CSS (用于占位符对齐) ---
st.markdown("""
<style>
    /* 定义灰色占位符样式 */
    .placeholder-box {
        height: 450px; /* 固定高度，确保左右对齐 */
        background-color: #f0f2f6;
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #666;
        border: 2px dashed #ccc;
        margin-bottom: 1rem;
    }
    .placeholder-icon {
        font-size: 50px;
        margin-bottom: 15px;
    }
    .placeholder-text {
        font-size: 18px;
        font-weight: 600;
    }
    /* 微调按钮间距 */
    .stButton button {
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

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
        'current_file_fingerprint': None # 当前视频文件的指纹
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
    logger.info(f"OD 模型已保存: {uploaded_file.name}")
    st.toast(f"✅ OD 模型已保存: {uploaded_file.name}")

def save_uploaded_ar_model(uploaded_files):
    if not uploaded_files: return
    
    # 简单的逻辑：尝试找到 .py 和 .pth
    py_file = next((f for f in uploaded_files if f.name.endswith('.py')), None)
    pth_file = next((f for f in uploaded_files if f.name.endswith('.pth')), None)
    
    if not py_file or not pth_file:
        loffer.error("上传文件格式错误") 
        st.error("❌ 必须要同时上传 .py 和 .pth 文件")
        return

    # 检查文件名一致性
    py_name = os.path.splitext(py_file.name)[0]
    pth_name = os.path.splitext(pth_file.name)[0]
    
    if py_name != pth_name:
        logger.error(f"文件名不一致: {py_name}.py vs {pth_name}.pth")
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

    logger.info(f"AR 模型已保存: {py_file.name}")
    st.toast(f"✅ AR 模型已保存至: {model_dir}")

# --- 核心任务逻辑 ---
def run_analysis_pipeline(conf, iou, device):
    """执行完整的视频分析流程"""
    status = st.empty()
    bar = st.progress(0)
    video_name = os.path.basename(st.session_state.video_path)
    logger.info(f"启动分析任务: 视频={video_name}, OD模型={st.session_state.od_model_name}, AR模型={st.session_state.ar_model_name}, 设备={device}")
    
    try:
        # 1. 准备模型
        status.text("⏳ 正在初始化 AI 引擎...")
        
        # 加载 OD 模型
        od_path = os.path.join(config.OD_MODEL_DIR, st.session_state.od_model_name)
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
            
            if segment_count % 5 == 0:
                logger.info(f"已处理片段: {segment_count}")

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
        
        logger.info(f"分析任务成功完成: 共处理 {segment_count} 个片段，生成视频: {final_video}")
        bar.progress(1.0)
        status.success("✅ 分析完成！")
        return True
        
    except Exception as e:
        logger.error("分析流程发生严重错误", exc_info=True)
        status.error("❌ 处理中断")
        st.error(f"详细错误: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False

# --- 侧边栏：还原 Gradio 布局 ---
with st.sidebar:
    st.write(f"当前用户: {st.session_state.user_info['username']}")
    st.header("⚙️ 设置面板")
    
    # 1. 目标检测参数设置
    with st.expander("⚙️ 目标检测参数设置", expanded=False):
        # 参数
        conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05, key="od_conf_slider")
        iou = st.slider("IoU 阈值", 0.0, 1.0, 0.7, 0.05, key="od_iou_slider")
        
        # 模型选择
        od_files = [f for f in os.listdir(config.OD_MODEL_DIR) if f.endswith(('.pt', '.onnx'))] if os.path.exists(config.OD_MODEL_DIR) else []
        
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
            if st.session_state.last_saved_od_model != uploaded_od.name:
                with st.spinner(f"正在自动保存 {uploaded_od.name}..."):
                    save_uploaded_od_model(uploaded_od)
                    st.session_state.last_saved_od_model = uploaded_od.name
                    st.rerun()

        if st.button("🔄 重新加载 OD 模型"):
            load_od_model_cached.clear()
            logger.info("已重新加载 OD 模型")
            st.toast("已清除缓存")

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
            if len(uploaded_ar) == 2:
                current_ar_fingerprint = "|".join(sorted([f.name for f in uploaded_ar]))
                if st.session_state.last_saved_ar_model != current_ar_fingerprint:
                    with st.spinner("正在自动保存 AR 模型套件..."):
                        py_file = next((f for f in uploaded_ar if f.name.endswith('.py')), None)
                        pth_file = next((f for f in uploaded_ar if f.name.endswith('.pth')), None)
                        if py_file and pth_file:
                            save_uploaded_ar_model(uploaded_ar)
                            st.session_state.last_saved_ar_model = current_ar_fingerprint
                            st.rerun()
                        else:
                            st.error("请确保上传的是一个 .py 和一个 .pth 文件")
            elif len(uploaded_ar) > 2:
                st.warning("⚠️ 请只上传 2 个文件（.py 和 .pth）")

        if st.button("🔄 重新加载 AR 模型"):
            logger.info("已重新加载 AR 模型")
            load_ar_model_cached.clear()
            st.toast("已清除缓存")
            
    # 3. 通用参数设置
    with st.expander("⚙️ 通用参数设置", expanded=False):
        device = st.selectbox("推理设备", ["cuda:0", "cpu"], index=0)
        save_db = st.checkbox("检测结果是否写入数据库", value=True)

# --- 主界面 ---
st.title("🐖 猪只行为智能分析系统")

# 创建主布局列
col1, col2 = st.columns(2)

# ==========================================
# 左侧：视频输入 (Logic & UI)
# ==========================================
with col1:
    st.subheader("1. 视频输入")
    
    # 1. 定义布局容器：先占位，后填充
    video_display_container = st.empty()  # 视频播放区域
    control_container = st.container()    # 控制按钮区域
    st.divider()                          # 分割线
    upload_container = st.container()     # 上传区域 (放到底部)

    # 2. 上传逻辑处理 (放在底部容器渲染)
    with upload_container:
        # 检查 FFmpeg
        if not check_ffmpeg_installed():
            logger.error("未检测到 FFmpeg")
            st.error("🚨 未检测到 FFmpeg！")
            st.stop()
        
        uploaded_file = st.file_uploader(
            "⬇️ 点击上传视频 (支持 MP4, AVI, MKV...)", 
            type=['mp4', 'avi', 'mov', 'mkv', 'flv']
        )

        # 处理上传文件逻辑
        if uploaded_file:
            logger.info(f"上传新视频文件: {uploaded_file.name}, 大小: {uploaded_file.size/1024/1024:.2f}MB")
            file_fingerprint = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # 只有当指纹变化时才处理
            if st.session_state.get('current_file_fingerprint') != file_fingerprint:
                timestamp = int(time.time())
                raw_name = f"raw_{timestamp}_{uploaded_file.name}"
                raw_path = os.path.join("temp_uploads", raw_name)
                clean_name = f"clean_{timestamp}_{os.path.splitext(uploaded_file.name)[0]}.mp4"
                clean_path = os.path.join("temp_uploads", clean_name)
                
                # 进度提示
                progress_toast = st.toast("正在处理新视频...", icon="⏳")
                
                with open(raw_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                success, msg = convert_video_to_h264(raw_path, clean_path)
                
                if success:
                    st.session_state.video_path = clean_path
                    st.session_state.raw_video_path = raw_path
                    st.session_state.current_file_fingerprint = file_fingerprint
                    # 清空旧结果
                    st.session_state.processing_result = None
                    st.session_state.output_video_path = None
                    st.session_state.result_dir = None
                    
                    progress_toast.toast("✅ 视频预处理完成", icon="✅")
                    logger.info("已处理新视频")
                    st.rerun()
                else:
                    logger.error(f"转码失败: {msg}")
                    st.error(f"转码失败: {msg}")
                    # 允许重试
                    if 'current_file_fingerprint' in st.session_state:
                        del st.session_state.current_file_fingerprint

    # 3. 视频显示逻辑 (渲染在顶部的 video_display_container)
    with video_display_container.container():
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            st.video(st.session_state.video_path, format="video/mp4")
        else:
            # 显示 CSS 样式的占位符
            st.markdown("""
                <div class="placeholder-box">
                    <div class="placeholder-icon">📺</div>
                    <div class="placeholder-text">请在下方上传视频</div>
                </div>
            """, unsafe_allow_html=True)

    # 4. 控制按钮逻辑 (渲染在中间的 control_container)
    with control_container:
        st.write("") # 增加一点间距
        # 判断模型是否就绪
        model_ready = (st.session_state.od_model_name != "无可用模型" and 
                       st.session_state.ar_model_name != "无可用模型")
        video_ready = st.session_state.video_path is not None
        
        # 使用列来居中按钮或拉伸按钮
        b_col1, b_col2 = st.columns([1, 2])
        with b_col2:
             start_btn = st.button(
                "🚀 开始智能分析", 
                type="primary", 
                disabled=not (model_ready and video_ready),
                use_container_width=True
            )
        
        if not model_ready:
            st.caption("⚠️ 请在左侧侧边栏选择有效的 OD 和 AR 模型")

# ==========================================
# 右侧：结果展示 (Logic & UI)
# ==========================================
with col2:
    st.subheader("2. 分析结果")
    
    # 1. 定义布局
    result_display_container = st.empty() # 结果视频区域
    status_container = st.container()     # 进度条区域
    st.divider()
    download_container = st.container()   # 下载按钮区域

    # 2. 处理点击事件 (在 status_container 中显示进度)
    if start_btn:
        with status_container:
            logger.info("开始分析")
            success = run_analysis_pipeline(conf, iou, device)
            if success and save_db:
                try:
                    get_res_to_sqlite(st.session_state.processing_result, config.VIDEO_RECOGNITION_DATABASE)
                    st.toast("💾 数据库已更新")
                    logger.info(f"分析结果已存入数据库: {config.VIDEO_RECOGNITION_DATABASE}")
                except Exception as e:
                    logger.info(f"数据库错误: {e}")
                    st.error(f"数据库错误: {e}")
            if success:
                st.rerun() # 刷新以显示结果

    # 3. 结果显示逻辑 (渲染在顶部的 result_display_container)
    with result_display_container.container():
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            st.video(st.session_state.output_video_path)
        else:
            # 显示与左侧高度一致的占位符
            st.markdown("""
                <div class="placeholder-box">
                    <div class="placeholder-icon">⏳</div>
                    <div class="placeholder-text">等待分析结果...</div>
                </div>
            """, unsafe_allow_html=True)

    # 4. 下载按钮逻辑 (渲染在底部的 download_container)
    with download_container:
        st.write("⬇️ **数据导出**")
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            # 使用 3 列对齐按钮
            dc1, dc2, dc3 = st.columns(3)
            
            # 按钮 1: 视频
            with open(st.session_state.output_video_path, "rb") as f:
                dc1.download_button(
                    "🎥 下载视频", 
                    f, 
                    file_name="result.mp4", 
                    mime="video/mp4",
                    use_container_width=True,
                    on_click=lambda: logger.info(f"用户下载了分析结果视频: {st.session_state.output_video_path}")
                )
            
            # 按钮 2: JSON
            if st.session_state.result_dir:
                json_path = get_coco_annotations(st.session_state.processing_result, st.session_state.result_dir)
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        dc2.download_button(
                            "📋 下载 JSON", 
                            f, 
                            file_name="annotations.json", 
                            mime="application/json",
                            use_container_width=True,
                            on_click=lambda: logger.info(f"用户下载了分析结果JSON: {json_path}")
                        )
            
            # 按钮 3: ZIP
            # 注意：如果 ZIP 生成较慢，可以采用 if button -> generate -> show download 的逻辑
            # 这里为了布局对齐，直接使用 button 触发生成和下载
            if dc3.button("🖼️ 打包图片", use_container_width=True):
                logger.info("用户点击[打包图片]，开始生成ZIP文件...")
                with st.spinner("正在打包关键帧..."):
                    try:
                        zip_path = get_annotated_images_zipfile(
                            images=st.session_state.processing_result.images,
                            output_dir=st.session_state.result_dir,
                            video_name=st.session_state.processing_result.video_name,
                            sample_step=1
                        )
                        file_size = os.path.getsize(zip_path) / (1024 * 1024)
                        logger.info(f"ZIP打包成功: {zip_path} (大小: {file_size:.2f} MB)")
                        with open(zip_path, "rb") as f:
                            # 模拟点击下载
                            st.download_button(
                                "📦 确认下载 ZIP", 
                                f, 
                                file_name="frames.zip", 
                                mime="application/zip",
                                key="real_zip_download",
                                use_container_width=True,
                                on_click=lambda: logger.info(f"用户下载了关键帧ZIP: {zip_path}")
                            )
                    except Exception as e:
                        logger.info(f"打包失败: {e}", exc_info=True)
                        st.error(f"打包失败: {e}")
        else:
            # 如果没有结果，显示禁用的灰色按钮占位，保持布局美观
            dc1, dc2, dc3 = st.columns(3)
            dc1.button("🎥 下载视频", disabled=True, use_container_width=True)
            dc2.button("📋 下载 JSON", disabled=True, use_container_width=True)
            dc3.button("🖼️ 打包图片", disabled=True, use_container_width=True)