import json
import os
import datetime
import sqlite3
import cv2
import zipfile
import numpy as np
from typing import List
from ..structures import PlottedResult

def get_res_to_sqlite(
    result: PlottedResult,
    db_path: str,
    user_name: str = "unknown"
):

    if not os.path.exists(db_path):
        print("数据库文件不存在，创建新数据库")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    

        cursor.execute('''
                CREATE TABLE IF NOT EXISTS recognition_results (
                    img_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    user_name TEXT,
                    height INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    bbox_x REAL NOT NULL,
                    bbox_y REAL NOT NULL,
                    bbox_w REAL NOT NULL,
                    bbox_h REAL NOT NULL,
                    confidence REAL,
                    timestamp TEXT
                )
            ''')

    
        width, height = result.images[0].shape[1], result.images[0].shape[0]
        video_name = result.video_name
        data_to_insert = []
        current_time = datetime.datetime.now()
        for i in range(len(result.raw_anno)):
            # img_id += 1
            file_name = f"{video_name}_{i*3/16:.3f}.png"
            time_diff = datetime.timedelta(seconds=i*3/16)
            timestamp = (current_time + time_diff).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            for j in result.raw_anno[i]:
                bbox = j["bbox"]
                # print("============" + j["conf"] + "===============")
                bbox = [float((bbox[2]+bbox[0])*0.5), float((bbox[3]+bbox[1])*0.5), float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1])]
                data_to_insert.append((file_name, user_name, height, width, j["cls"], bbox[0], bbox[1], bbox[2], bbox[3], float(j["conf"]), timestamp))
        
        cursor.executemany('''
            INSERT INTO recognition_results (filename, user_name, height, width, category, bbox_x, bbox_y, bbox_w, bbox_h, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)

        conn.commit()
        print(f"成功插入 {len(data_to_insert)} 条标注记录到数据库 {db_path}")
    except sqlite3.Error as e:
        print(f"数据库操作失败: {e}")
        conn.rollback()
    finally:
        if conn:
            conn.close()    
    return 

def get_coco_annotations(
    result: PlottedResult,
    result_dir: str,
) -> str:


    json_path = os.path.join(result_dir, f"{result.video_name}_result.json")
    print(f"coco 格式识别结果将保存至 {json_path}")
    if os.path.exists(json_path):
        print(f"coco 格式识别结果已存在，跳过生成")
        return json_path

    info = {
    "description": f"action recognition results on video {result.video_name}",
    "year": datetime.datetime.now().year, # 年份
    "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 创建日期
    }

    categories = []
    images = []
    annotations = []
    cate_id = {}

    img_id = 0
    anno_id = 0
    cate_id_counter = 0
    width, height = result.images[0].shape[1], result.images[0].shape[0]
    for i in range(len(result.raw_anno)):
        img_id += 1
        file_name = f"frame_{i*3/16:.3f}.png"
        images.append({
            "id": img_id,
            "file_name": file_name,
            "height": height,
            "width": width,
        })
        for j in result.raw_anno[i]:
            anno_id += 1
            if j["cls"] in cate_id:
                category_id = cate_id[j["cls"]]
            else:
                cate_id_counter += 1
                category_id = cate_id_counter
                cate_id[j["cls"]] = category_id
                categories.append({
                    "id": category_id,
                    "name": j["cls"],
                    "supercategory": "none",
                })
            bbox = j["bbox"]
            bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]-bbox[0]), float(bbox[3]-bbox[1])]
            area = bbox[2] * bbox[3]
            iscrowd = 0
            annotations.append({
                "id": anno_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": iscrowd,
            }) 
    
    coco_data = {
    "info": info,
    "licenses": [],
    "categories": categories,
    "images": images,
    "annotations": annotations,
    }

    with open(json_path, "w") as f:
        json.dump(coco_data, f)
        print(f"coco 格式识别结果已保存至 {json_path}")

    return json_path

def get_annotated_images_zipfile(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str = "video",
    sample_step: int = 1,
    image_quality: int = 85
) -> str:
    """
    将图像列表打包为 ZIP 文件 (适配 Streamlit 和专用算法逻辑)。
    
    Args:
        images: 图像列表 (List[np.ndarray], BGR格式)
        output_dir: ZIP 文件保存的目录
        video_name: 视频名称（用于生成 ZIP 文件名）
        sample_step: 采样步长（例如 1 表示每帧都存，5 表示每 5 帧存一张）
        image_quality: JPEG 压缩质量 (1-100)，默认 85
        
    Returns:
        str: 生成的 ZIP 文件的绝对路径
    """
    
    if not images:
        raise ValueError("输入图像列表为空，无法生成 ZIP。")

    os.makedirs(output_dir, exist_ok=True)
    
    # 清理文件名
    clean_name = os.path.splitext(os.path.basename(video_name))[0]
    zip_filename = f"{clean_name}_frames.zip"
    zip_path = os.path.join(output_dir, zip_filename)
    
    print(f"开始打包图片到: {zip_path}")

    # 使用 ZIP_DEFLATED 压缩
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        
        for i, img in enumerate(images):
            # 1. 采样过滤
            if i % sample_step != 0:
                continue
                
            # 2. 核心逻辑修正：严格遵循算法采样率 (16帧对应3秒)
            # 这里的 i 是采样后的帧索引，不是原始视频帧索引
            time_seconds = (i * 3) / 16
            
            # 格式化时间戳：时:分:秒
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = time_seconds % 60
            timestamp_str = f"{hours:02d}{minutes:02d}{seconds:06.3f}"
            
            # 构造文件名: frame_0005_000012.345.jpg
            filename = f"frame_{i:04d}_{timestamp_str}.jpg"
            
            # 3. 内存直写优化
            # 将 numpy 数组编码为 jpg 字节流
            success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, image_quality])
            
            if success:
                # 直接写入 ZIP，无需创建磁盘临时文件
                zipf.writestr(filename, buffer.tobytes())
            else:
                print(f"⚠️ 警告: 第 {i} 帧编码失败，已跳过。")
                
    print(f"✅ ZIP 打包完成，包含 {len(images) // sample_step} 张图片")
    return zip_path