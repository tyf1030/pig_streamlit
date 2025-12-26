import cv2
import numpy as np
import os
from datetime import datetime
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import tempfile
import zipfile
from ..structures import PlottedResult

def draw_detection_boxes_single(
    image: np.ndarray,
    detections: List[Dict],
    box_thickness: int = 2,
    font_scale: float = 0.6,
    text_thickness: int = 2,
    color_map: Dict[str, Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    在单张图像上绘制检测框和标签信息
    
    Args:
        image: HWC BGR格式的numpy图像
        detections: 检测结果列表，每个元素包含bbox, id, cls, conf信息
        box_thickness: 框线粗细
        font_scale: 字体大小
        text_thickness: 文字粗细
        color_map: 类别到颜色的映射字典
    
    Returns:
        np.ndarray: 绘制后的图像
    """
    # 复制图像，避免修改原图
    img_with_boxes = image.copy()
    
    # 如果没有检测框，直接返回原图
    if not detections:
        return img_with_boxes
    
    # 默认颜色映射（如果未提供）
    if color_map is None:
        # 预定义一些颜色
        predefined_colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 深紫色
            (0, 128, 128),  # 橄榄色
        ]
        
        # 获取所有类别
        all_classes = list(set([det["cls"] for det in detections]))
        color_map = {}
        for i, cls in enumerate(all_classes):
            color_map[cls] = predefined_colors[i % len(predefined_colors)]
    
    # 设置字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for detection in detections:
        bbox = detection["bbox"]  # xyxy格式 [x1, y1, x2, y2]
        cls = detection["cls"]
        conf = detection["conf"]
        obj_id = detection.get("id", None)
        
        # 获取颜色
        color = color_map.get(cls, (0, 255, 0))
        
        # 转换为整数坐标
        x1, y1, x2, y2 = map(int, bbox)
        
        # 绘制检测框
        cv2.rectangle(
            img_with_boxes, 
            (x1, y1), (x2, y2), 
            color, 
            box_thickness
        )
        
        # 准备标签文本
        if obj_id is not None:
            label = f"{cls} {conf:.2f} ID:{obj_id}"
        else:
            label = f"{cls} {conf:.2f}"
        
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        
        # 绘制文本背景
        text_bg_y1 = max(y1 - text_height - baseline - 5, 0)  # 确保不超出图像上边界
        text_bg_y2 = y1
        
        cv2.rectangle(
            img_with_boxes,
            (x1, text_bg_y1),
            (x1 + text_width, text_bg_y2),
            color,
            -1  # 填充矩形
        )
        
        # 绘制文本
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),  # 白色文字
            text_thickness,
            cv2.LINE_AA
        )
    
    return img_with_boxes

def draw_detection_boxes_batch(
    images: List[np.ndarray],
    detections_list: List[List[Dict]],
    output_paths: Union[List[str], str, None] = None,
    box_thickness: int = 2,
    font_scale: float = 0.6,
    text_thickness: int = 2,
    color_map: Dict[str, Tuple[int, int, int]] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """
    在批量图像上绘制检测框和标签信息
    
    Args:
        images: HWC BGR格式的numpy图像列表
        detections_list: 检测结果列表的列表，每个元素对应一张图像的检测结果
        output_paths: 输出图像路径列表，可以是列表、目录路径或None
        box_thickness: 框线粗细
        font_scale: 字体大小
        text_thickness: 文字粗细
        color_map: 类别到颜色的映射字典
    
    Returns:
        Tuple[List[np.ndarray], List[str]]: 绘制后的图像列表和保存路径列表
    """
    # 验证输入
    # if len(images) != len(detections_list):
    #     raise ValueError("图像列表和检测结果列表的长度必须相同")
    
    # # 处理输出路径
    # if output_paths is None:
    #     # 自动生成输出目录和文件名
    #     output_dir = "output"
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_paths = [os.path.join(output_dir, f"detection_result_{i:04d}.jpg") 
    #                    for i in range(len(images))]
    # elif isinstance(output_paths, str):
    #     # 如果是目录路径，生成文件名
    #     output_dir = output_paths
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_paths = [os.path.join(output_dir, f"detection_result_{i:04d}.jpg") 
    #                    for i in range(len(images))]
    # else:
    #     # 已经是路径列表，直接使用
    #     if len(output_paths) != len(images):
    #         raise ValueError("输出路径列表的长度必须与图像列表相同")
    #     # 确保所有输出目录存在
    #     for path in output_paths:
    #         os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 构建全局颜色映射（基于所有检测结果）
    if color_map is None:
        all_classes = set()
        for detections in detections_list:
            for det in detections:
                all_classes.add(det["cls"])
        
        predefined_colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 深紫色
            (0, 128, 128),  # 橄榄色
        ]
        
        color_map = {}
        for i, cls in enumerate(all_classes):
            color_map[cls] = predefined_colors[i % len(predefined_colors)]
    
    # 处理每张图像
    result_images = []
    saved_paths = []
    
    for i, (image, detections) in enumerate(zip(images, detections_list)):
        # 绘制检测框
        result_image = draw_detection_boxes_single(
            image=image,
            detections=detections,
            box_thickness=box_thickness,
            font_scale=font_scale,
            text_thickness=text_thickness,
            color_map=color_map
        )
        
        # # 保存图像
        # output_path = output_paths[i]
        # cv2.imwrite(output_path, result_image)
        
        result_images.append(result_image)
        # saved_paths.append(output_path)
    
    return result_images


def process_image_sequence(
    images: List[np.ndarray],
    output_type: str = "both",  # "images", "video"
    output_prefix: Optional[str] = None,
    output_dir: str = "./output",
    video_output_path: Optional[str] = None,
    fps: float = 16/3,  # 16帧/3秒 ≈ 5.333 fps
    time_format: str = "seconds"  # "seconds" or "timestamp"
):
    """
    处理按顺序排序的图像序列
    
    参数:
        images: 按顺序排序的nparray列表，每个元素是HWC BGR格式的图像
        output_type: 输出类型 - "images", "video"
        output_prefix: 输出图片的前缀
        output_dir: 输出目录
        video_output_path: 视频输出路径
        fps: 视频帧率，默认16帧/3秒 ≈ 5.333 fps
        time_format: 时间戳格式 - "seconds"(秒数) 或 "timestamp"(时分秒)
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证输入
    if not images:
        raise ValueError("输入图像列表不能为空")
    
    # 获取图像尺寸
    height, width, channels = images[0].shape
    if channels != 3:
        raise ValueError("图像必须是BGR三通道格式")
    
    # 处理图片输出
    if output_type in ["images"]:
        if output_prefix is None:
            output_prefix = "frame"
        
        print(f"开始保存图片到目录: {output_dir}")
        for i, img in enumerate(images):
            # 计算时间戳（每16帧为3秒）
            time_seconds = (i * 3) / 16
            
            if time_format == "timestamp":
                # 转换为时分秒格式
                hours = int(time_seconds // 3600)
                minutes = int((time_seconds % 3600) // 60)
                seconds = time_seconds % 60
                timestamp_str = f"{hours:02d}{minutes:02d}{seconds:06.3f}"
            else:
                # 直接使用秒数
                timestamp_str = f"{time_seconds:.3f}"
            
            # 生成文件名
            filename = f"{output_prefix}_{timestamp_str}.png"
            filepath = os.path.join(output_dir, filename)
            
            # 保存图片
            success = cv2.imwrite(filepath, img)
            if success:
                print(f"已保存: {filename}")
            else:
                print(f"保存失败: {filename}")
    
    # 处理视频输出
    if output_type in ["video"]:
        if video_output_path is None:
            video_output_path = os.path.join(output_dir, "output_video.mp4")
    
    # 强制.mp4后缀
        video_output_path = os.path.splitext(video_output_path)[0] + '.mp4'
        print(f"目标视频文件: {video_output_path}")
        try:
            import subprocess
            import tempfile
            
            # 1. 先将所有帧保存为临时JPEG文件
            with tempfile.TemporaryDirectory() as temp_dir:
                frame_paths = []
                print(f"正在准备 {len(images)} 帧图像...")
                for i, img in enumerate(images):
                    # 确保图像为BGR格式，并保存为JPEG
                    if len(img.shape) == 3:
                        # 如果图像是RGB格式，转换为BGR（OpenCV标准）
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
                    else:
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    frame_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
                    success = cv2.imwrite(frame_path, img_bgr)
                    if not success:
                        print(f"警告: 无法保存帧 {i} 到临时文件")
                    frame_paths.append(frame_path)
                
                # 2. 使用FFmpeg将图像序列编码为H.264 MP4
                print("正在使用FFmpeg编码H.264视频...")
                ffmpeg_cmd = [
                    'ffmpeg', '-y',  # -y 覆盖输出文件
                    '-framerate', str(fps),  # 输入帧率
                    '-i', os.path.join(temp_dir, 'frame_%06d.jpg'),  # 输入图像序列
                    '-c:v', 'libx264',  # 使用libx264编码器
                    '-preset', 'medium',  # 编码速度与质量的平衡
                    '-crf', '23',  # 恒定质量系数（23是通用优质选择）
                    '-pix_fmt', 'yuv420p',  # 确保浏览器兼容的像素格式
                    '-movflags', '+faststart',  # 将元数据移至文件头，优化网页流式播放
                    video_output_path
                ]
                
                # 运行FFmpeg命令
                result = subprocess.run(
                    ffmpeg_cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=60  # 设置超时时间
                )
                
                if result.returncode == 0:
                    print(f"✅ 成功使用FFmpeg生成H.264视频: {video_output_path}")
                    if os.path.exists(video_output_path):
                        file_size = os.path.getsize(video_output_path) / (1024*1024)
                        print(f"   文件大小: {file_size:.2f} MB, 时长: {len(images)/fps:.2f}秒")
                        return (video_output_path, output_dir)
                    else:
                        print("❌ 错误：FFmpeg命令成功但文件未生成")
                else:
                    print(f"❌ FFmpeg编码失败，错误信息:\n{result.stderr[:500]}")
                    # FFmpeg失败，降级到方案2
                    raise RuntimeError("FFmpeg编码失败")
                
        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
            print(f"⚠️ FFmpeg方案不可用 ({e})，降级到OpenCV备用方案...")
            # 清理可能生成的不完整文件
            if os.path.exists(video_output_path):
                os.remove(video_output_path)

        # if video_output_path is None:
        #     video_output_path = os.path.join(output_dir, "output_video.mp4")
        
        # # 确保视频文件扩展名是.mp4
        # if not video_output_path.lower().endswith('.mp4'):
        #     video_output_path += '.mp4'
        
        # # 创建视频写入器
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')  
        # video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        # if not video_writer.isOpened():
        #     raise RuntimeError(f"无法创建视频文件: {video_output_path}")
        
        # print(f"开始生成视频: {video_output_path}")
        # for i, img in enumerate(images):
        #     video_writer.write(img)
        #     if (i + 1) % 50 == 0:  # 每50帧打印一次进度
        #         print(f"已处理 {i + 1}/{len(images)} 帧")
        
        # video_writer.release()
        # print(f"视频生成完成: {video_output_path}")
        # print(f"视频信息: {len(images)}帧, 帧率: {fps:.3f}fps, 时长: {len(images)/fps:.2f}秒")

        # return (video_output_path, output_dir)


def get_annotated_images_zipfile(
    images: List[np.ndarray],
    output_prefix: Optional[str] = None,
    output_dir: str = "./output",
    time_format: str = "seconds"  # "seconds" or "timestamp"  
):
    
    os.makedirs(output_dir, exist_ok=True)

    if not images:
        raise ValueError("输入图像列表不能为空")
    
    # 获取图像尺寸
    height, width, channels = images[0].shape
    if channels != 3:
        raise ValueError("图像必须是BGR三通道格式")
    
    zip_file_path = None

    if output_prefix is None:
        output_prefix = "frame"
    
    # 生成ZIP文件名
    zip_filename = f"{output_prefix}_annotated_images.zip"
    zip_file_path = os.path.join(output_dir, zip_filename)
    print(f"开始保存图片到ZIP文件: {zip_file_path}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # 首先将所有图片保存到临时目录
        temp_files = []
        for i, img in enumerate(images):
            # 计算时间戳（每16帧为3秒）
            time_seconds = (i * 3) / 16
            
            if time_format == "timestamp":
                # 转换为时分秒格式
                hours = int(time_seconds // 3600)
                minutes = int((time_seconds % 3600) // 60)
                seconds = time_seconds % 60
                timestamp_str = f"{hours:02d}{minutes:02d}{seconds:06.3f}"
            else:
                # 直接使用秒数
                timestamp_str = f"{time_seconds:.3f}"

            temp_filename = f"{output_prefix}_{timestamp_str}.jpg"  # 使用jpg减小文件大小
            temp_filepath = os.path.join(temp_dir, temp_filename)
            
            # 保存图片（使用JPEG格式，质量85%以平衡质量和文件大小）
            success = cv2.imwrite(temp_filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                temp_files.append((temp_filename, temp_filepath))
                if (i + 1) % 50 == 0:  # 每50帧打印一次进度
                    print(f"已准备 {i + 1}/{len(images)} 张图片")
            else:
                print(f"保存失败: {temp_filename}")
            
            # 创建ZIP文件
            print(f"正在创建ZIP文件...")
            with zipfile.ZipFile(zip_file_path, 'w', 
                               compression=zipfile.ZIP_DEFLATED,
                               compresslevel=6) as zipf:
                for filename, filepath in temp_files:
                    zipf.write(filepath, filename)
            
            print(f"ZIP文件创建完成: {zip_file_path}")
            print(f"包含 {len(temp_files)} 张图片")
        
    return zip_file_path


# 使用示例
if __name__ == "__main__":
    # 创建示例数据（3张图片）
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    ]
    
    # 每张图片对应的检测结果
    boxes_list = [
        np.array([[100, 100, 200, 200], [300, 150, 400, 250]], dtype=np.float32),
        np.array([[50, 50, 150, 150], [200, 100, 300, 200], [350, 300, 450, 400]], dtype=np.float32),
        np.array([[150, 200, 250, 300]], dtype=np.float32)
    ]
    
    confidences_list = [
        np.array([0.95, 0.72], dtype=np.float32),
        np.array([0.88, 0.65, 0.45], dtype=np.float32),
        np.array([0.91], dtype=np.float32)
    ]
    
    class_ids_list = [
        np.array([0, 1], dtype=np.int32),
        np.array([0, 2, 1], dtype=np.int32),
        np.array([2], dtype=np.int32)
    ]
    
    track_ids_list = [
        np.array([101, 102], dtype=np.int32),
        np.array([201, 202, 203], dtype=np.int32),
        np.array([301], dtype=np.int32)
    ]
    
    save_dir = "./batch_detection_results"
    
    # 批量处理
    result_images, save_paths = draw_detections_batch(
        images=images,
        boxes_list=boxes_list,
        confidences_list=confidences_list,
        class_ids_list=class_ids_list,
        track_ids_list=track_ids_list,
        save_dir=save_dir,
        return_rgb=False
    )
    
    # 输出结果统计
    print(f"\n批量处理完成摘要:")
    print(f"处理图片数量: {len(result_images)}")
    print(f"成功保存数量: {len([p for p in save_paths if p])}")
    print(f"结果图片形状: {[img.shape for img in result_images]}")
    print(f"保存路径: {save_paths}")