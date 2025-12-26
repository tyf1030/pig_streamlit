import shutil
import subprocess
import os
import cv2
import streamlit as st

def get_video_codec(file_path):
    """
    使用 ffprobe 获取视频流的编码格式 (如 'h264', 'hevc')
    """
    # 如果没装 ffprobe，直接返回 None，这就意味着无法判断，外层逻辑会走默认转码流程
    if not shutil.which("ffprobe"):
        return None
        
    try:
        # 构建 ffprobe 命令
        # -v error: 只显示错误
        # -select_streams v:0: 选择第一个视频流
        # -show_entries stream=codec_name: 只显示编码名称
        # -of default=...: 格式化输出，只返回纯文本
        command = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=codec_name", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            file_path
        ]
        
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        if result.returncode == 0:
            return result.stdout.strip().lower()
        else:
            return None
            
    except Exception as e:
        print(f"Codec check failed: {e}")
        return None

def check_ffmpeg_installed():
    """检查系统中是否安装了 FFmpeg"""
    return shutil.which("ffmpeg") is not None

def convert_video_to_h264(input_path, output_path):
    """
    使用 FFmpeg 将视频转码为 H.264/AAC MP4
    返回: (bool, str) -> (是否成功, 信息)
    """
    if not check_ffmpeg_installed():
        return False, "系统未安装 FFmpeg"

    try:
        command = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-y", output_path
        ]
        
        # 捕获输出，避免污染控制台
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        if result.returncode == 0:
            return True, "转码成功"
        else:
            return False, f"FFmpeg 错误: {result.stderr}"
    except Exception as e:
        return False, str(e)

def get_video_info(video_path):
    """获取视频的基本信息（时长、FPS等）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    
    return {
        "fps": fps,
        "frames": int(count),
        "duration": count / fps if fps > 0 else 0,
        "resolution": f"{int(width)}x{int(height)}"
    }