# backend/__init__.py

# 暴露核心数据结构
from .structures import VideoData, ODResult, ARResult, PlottedResult

# 暴露视频读取类
from .video_io import VideoReader

# 暴露模型构建和推理函数
from .inference import (
    inference_recognizer_simplified,
    build_od_model, build_ar_model
)

# 暴露纯算法处理函数
from .processors import (
    filter_and_analyze_tracking_results,
    process_video_regions,
    filter_and_analyze_tracking_results_rebust,
)

# 定义 import * 时导出的内容
__all__ = [
    "VideoData", "ODResult", "ARResult", "PlottedResult",
    "VideoReader",
    "build_od_model", "build_ar_model", "inference_recognizer_simplified",
    "filter_and_analyze_tracking_results", "process_video_regions","filter_and_analyze_tracking_results_rebust",
]