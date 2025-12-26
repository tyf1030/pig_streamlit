import numpy as np
import cv2
from typing import List, Dict

def filter_and_analyze_tracking_results(
    boxes_list: List[np.ndarray],
    track_ids_list: List[np.ndarray], 
    class_ids_list: List[np.ndarray],
    non_target_odcls: List[int],
    id_num_threshold: int = 3
) -> Dict[int, List[np.ndarray]]:
    """
    过滤跟踪结果并计算每个ID的活动范围
    
    参数:
        boxes_list: 16帧的检测框列表，每个为(n,4)的numpy数组，xyxy格式
        track_ids_list: 16帧的跟踪ID列表，每个为(n,)的numpy数组
        class_ids_list: 16帧的类别ID列表，每个为(n,)的numpy数组
        non_target_odcls: 需要过滤的非目标类别列表
        id_num_threshold: 跟踪ID出现次数的阈值，默认3次
        
    返回:
        Dict[int, np.ndarray]: 每个跟踪ID对应的合并后的边界框[x1, y1, x2, y2]
    """
    
    # 输入验证
    n_frames = len(boxes_list)
    if len(track_ids_list) != n_frames or len(class_ids_list) != n_frames:
        raise ValueError("输入列表长度必须一致")
    
    # 第一步：类别过滤
    filtered_boxes = []
    filtered_track_ids = []
    
    for i in range(n_frames):
        frame_boxes = boxes_list[i]
        frame_track_ids = track_ids_list[i] 
        frame_class_ids = class_ids_list[i]
        
        # 创建掩码：只保留不在non_target_odcls中的类别
        if len(frame_boxes) > 0:
            mask = ~np.isin(frame_class_ids, non_target_odcls)
            filtered_boxes.append(frame_boxes[mask])
            filtered_track_ids.append(frame_track_ids[mask])
        else:
            # 如果没有检测框，添加空数组
            filtered_boxes.append(np.empty((0, 4)))
            filtered_track_ids.append(np.empty((0,)))
    
    # 第二步：跟踪ID出现次数过滤
    # 统计每个track_id在所有帧中出现的总次数
    id_count = {}
    for frame_track_ids in filtered_track_ids:
        unique_ids, counts = np.unique(frame_track_ids, return_counts=True)
        for track_id, count in zip(unique_ids, counts):
            id_count[track_id] = id_count.get(track_id, 0) + count
    
    # 保留出现次数大于等于阈值的track_id
    valid_track_ids = {tid for tid, count in id_count.items() 
                      if count >= id_num_threshold}
    
    # 应用track_id过滤
    final_boxes = []
    final_track_ids = []
    
    for i in range(n_frames):
        frame_boxes = filtered_boxes[i]
        frame_track_ids = filtered_track_ids[i]
        
        if len(frame_boxes) > 0:
            mask = np.isin(frame_track_ids, list(valid_track_ids))
            final_boxes.append(frame_boxes[mask])
            final_track_ids.append(frame_track_ids[mask])
        else:
            final_boxes.append(np.empty((0, 4)))
            final_track_ids.append(np.empty((0,)))
    
    # 第三步：按track_id合并边界框，计算活动范围
    id_to_boxes = {}
    
    for i in range(n_frames):
        frame_boxes = final_boxes[i]
        frame_track_ids = final_track_ids[i]
        
        for j in range(len(frame_track_ids)):
            track_id = frame_track_ids[j]
            box = frame_boxes[j]
            
            if track_id not in id_to_boxes:
                id_to_boxes[track_id] = []
            id_to_boxes[track_id].append(box)
    
    # 计算每个track_id的边界框并集（活动范围）
    activity_ranges = {}
    for track_id, boxes in id_to_boxes.items():
        if boxes:  # 这是修改的关键部分
            boxes_array = np.vstack(boxes)  # 形状为(n, 4)
            n_boxes = len(boxes_array)
            
            # 计算如何分成4份，多余部分向前叠加
            base_size = n_boxes // 4
            remainder = n_boxes % 4
            
            period_boxes = []  # 存储4个时间段的并集框
            start_idx = 0
            
            # 将boxes_array分成4份
            for i in range(4):
                # 计算当前时间段的结束索引
                # 前remainder个时间段多分一个框
                end_idx = start_idx + base_size + (1 if i < remainder else 0)
                
                # 获取当前时间段的框
                period_box_set = boxes_array[start_idx:end_idx]
                
                if len(period_box_set) > 0:
                    # 计算该时间段的并集框
                    union_box = np.array([
                        np.min(period_box_set[:, 0]),  # min x1
                        np.min(period_box_set[:, 1]),  # min y1
                        np.max(period_box_set[:, 2]),  # max x2
                        np.max(period_box_set[:, 3])   # max y2
                    ])
                else:
                    # 如果时间段内没有框，使用NaN
                    union_box = np.array([np.nan, np.nan, np.nan, np.nan])
                
                period_boxes.append(union_box)
                start_idx = end_idx
            
            activity_ranges[track_id] = period_boxes
        else:
            # 如果没有框，创建4个NaN框
            activity_ranges[track_id] = [np.array([np.nan, np.nan, np.nan, np.nan])] * 4
    
    return activity_ranges

def process_video_regions(frames, detections):
    """
    处理视频区域，提取检测框内的图像并组合成CTHW格式
    
    Args:
        frames: 长度为16的列表，每个元素是一帧图像 (H, W, C) BGR格式
        detections: 长度为n的列表，每个元素是4个检测框的列表 [[x1,y1,x2,y2], ...]
    
    Returns:
        output_list: 长度为n的列表，每个元素是CTHW格式的numpy数组 (C, 16, H, W)
    """
    
    def letter_box(image, target_width, target_height):
        """
        使用letter box方法调整图像大小，保持宽高比
        
        Args:
            image: 输入图像
            target_width: 目标宽度
            target_height: 目标高度
            
        Returns:
            resized_image: 调整后的图像
        """
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整图像大小
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标图像并填充
        result = np.full((target_height, target_width, 3), 128, dtype=np.uint8)
        
        # 计算放置位置（居中）
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        
        # 放置调整后的图像
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return result
    
    output_list = []
    
    # 对每个跟踪结果进行处理
    for i, detection_boxes in enumerate(detections):
        # 检查检测框数量是否正确
        if len(detection_boxes) != 4:
            raise ValueError(f"检测框数量应为4，但得到{len(detection_boxes)}")
        
        # 计算每个跟踪结果的最大宽度和高度
        max_width = 0
        max_height = 0
        
        for box in detection_boxes:
            x1, y1, x2, y2 = box
            width = int(x2 - x1)
            height = int(y2 - y1)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        
        # 确保最小尺寸
        max_width = max(max_width, 1)
        max_height = max(max_height, 1)
        
        # 存储所有处理后的帧
        processed_frames = []
        
        # 每4帧对应一个检测框
        for segment_idx in range(4):
            # 获取当前段的检测框
            box = detection_boxes[segment_idx]
            x1, y1, x2, y2 = map(int, box)
            
            # 获取当前段的4帧
            start_frame = segment_idx * 4
            end_frame = start_frame + 4
            
            for frame_idx in range(start_frame, end_frame):
                frame = frames[frame_idx]
                
                # 提取检测框区域
                h, w = frame.shape[:2]
                
                # 确保坐标在图像范围内
                x1_clipped = max(0, min(x1, w - 1))
                y1_clipped = max(0, min(y1, h - 1))
                x2_clipped = max(x1_clipped + 1, min(x2, w))
                y2_clipped = max(y1_clipped + 1, min(y2, h))
                
                # 提取ROI
                roi = frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
                
                # 使用letter box调整大小
                resized_roi = letter_box(roi, max_width, max_height)
                
                # BGR转RGB
                rgb_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB)
                
                processed_frames.append(rgb_roi)
        
        # 转换为numpy数组并调整维度为CTHW
        frames_array = np.array(processed_frames)  # 形状: (16, H, W, C)
        frames_array = frames_array.transpose(3, 0, 1, 2)  # 转换为 (C, 16, H, W)
        
        output_list.append(frames_array)
    
    return output_list

def filter_and_analyze_tracking_results_rebust(
    boxes_list: list[np.ndarray],
    track_ids_list: list[np.ndarray], 
    class_ids_list: list[np.ndarray],
    non_target_odcls: list[int],
    id_num_threshold: int = 3
) -> dict[int, list[np.ndarray]]:
    """
    [健壮版] 过滤跟踪结果并计算每个ID的活动范围 (修复时间错位问题)
    """
    
    n_frames = len(boxes_list) # 应该是 16
    if len(track_ids_list) != n_frames or len(class_ids_list) != n_frames:
        raise ValueError("输入列表长度必须一致")
    
    # --- 第一步：整理数据，建立 {track_id: {frame_idx: box}} 的索引 ---
    # 这样我们就不丢失时间信息了
    
    # 统计 ID 出现次数
    id_counts = {}
    # 存储轨迹数据
    track_data = {} 
    
    for i in range(n_frames):
        # 过滤掉非目标类别
        if len(boxes_list[i]) == 0: continue
        
        valid_mask = ~np.isin(class_ids_list[i], non_target_odcls)
        valid_ids = track_ids_list[i][valid_mask]
        valid_boxes = boxes_list[i][valid_mask]
        
        for j, track_id in enumerate(valid_ids):
            # 统计次数
            id_counts[track_id] = id_counts.get(track_id, 0) + 1
            
            # 记录位置
            if track_id not in track_data:
                track_data[track_id] = {}
            track_data[track_id][i] = valid_boxes[j] # 记录第 i 帧该 ID 的框

    # --- 第二步：筛选有效 ID ---
    valid_track_ids = [tid for tid, count in id_counts.items() if count >= id_num_threshold]
    
    activity_ranges = {}
    
    # --- 第三步：按 4 个时间段计算活动范围 (关键修改) ---
    # 我们将 16 帧分为 4 段：0-3, 4-7, 8-11, 12-15
    segments = [(0, 4), (4, 8), (8, 12), (12, 16)]
    
    for tid in valid_track_ids:
        raw_boxes_map = track_data[tid]
        segment_boxes = [] # 存储 4 个时间段的框
        
        # 1. 遍历 4 个时间段，计算每个段的 Union Box
        for start_f, end_f in segments:
            # 找出该时间段内存在的所有框
            boxes_in_seg = []
            for f_idx in range(start_f, end_f):
                if f_idx in raw_boxes_map:
                    boxes_in_seg.append(raw_boxes_map[f_idx])
            
            if boxes_in_seg:
                # 如果该段有数据，计算并集框 (Union)
                boxes_arr = np.vstack(boxes_in_seg)
                union_box = np.array([
                    np.min(boxes_arr[:, 0]),
                    np.min(boxes_arr[:, 1]),
                    np.max(boxes_arr[:, 2]),
                    np.max(boxes_arr[:, 3])
                ])
                segment_boxes.append(union_box)
            else:
                # 如果该段完全没有数据 (比如前8帧)，先占位 None
                segment_boxes.append(None)
                
        # 2. 插值/外推补全 None 的段
        # 这一步保证：即使前8帧没检测到，也能根据后8帧生成一个合理的框
        filled_boxes = _fill_missing_segments(segment_boxes)
        
        activity_ranges[tid] = filled_boxes
        
    return activity_ranges

def _fill_missing_segments(boxes: list) -> list:
    """
    填充列表中为 None 的边界框。
    策略：最近邻填充 (Nearest Neighbor)。
    比线性插值更安全，因为 Union Box 代表的是区域，直接复制最近的有效区域通常足够。
    """
    # 1. 找到所有非空索引
    valid_indices = [i for i, b in enumerate(boxes) if b is not None]
    
    if not valid_indices:
        # 极端情况：全是 None (理论上会被阈值过滤掉，但防守一下)
        return [np.zeros(4) for _ in range(4)]
    
    # 2. 向前填充 (处理开头缺失，如 0-8 帧缺失)
    first_valid = valid_indices[0]
    for i in range(first_valid):
        boxes[i] = boxes[first_valid]
        
    # 3. 向后填充 (处理结尾缺失)
    last_valid = valid_indices[-1]
    for i in range(last_valid + 1, 4):
        boxes[i] = boxes[last_valid]
        
    # 4. 中间填充 (如有中间断档)
    for i in range(first_valid + 1, last_valid):
        if boxes[i] is None:
            # 这里简单取前一个有效值，或者你可以取前后的平均
            boxes[i] = boxes[i-1]
            
    return boxes