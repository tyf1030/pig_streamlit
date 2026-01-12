from ultralytics import YOLO
import argparse
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple

from mmengine import Config, DictAction

from mmaction.apis import init_recognizer
from mmaction.visualization import ActionVisualizer

import numpy as np

import os.path as osp
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.utils import track_iter_progress

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample
import cv2

class ODResult:
    """
    存储目标检测结果的类
    属性：
        boxes (list): 检测到的边界框列表，每个框为[x1, y1, x2, y2]numpy ndarray格式。
        conf (list): 每个边界框对应的置信度列表。
        cls (list): 每个边界框对应的类别ID列表。
        id (list): 每个边界框对应的对象ID列表（用于多目标跟踪）。
        id_map (dict): 用于跟踪的ID映射字典，键为检测到的对象ID，值为对应的类别ID。
        
    """
    def __init__(self,):
        self.boxes = []
        self.conf = []
        self.cls = []
        self.id = []
        self.id_map = []
    
    def from_yolo_result(self, yolo_result:list):
        """
        从YOLO检测结果初始化ODResult对象。
        
        参数:
            yolo_result: YOLO模型的检测结果对象，包含boxes, conf, cls等属性。
        """
        for result in yolo_result:
            self.boxes.append(result.boxes.xyxy.cpu().numpy())  # 边界框坐标
            self.conf.append(result.boxes.conf.cpu().numpy())  # 置信度
            self.cls.append(result.boxes.cls.cpu().numpy()) # 类别ID
            self.id.append(result.boxes.id.cpu().numpy())  # 对象ID
    
    def __str__(self) -> str:
        number = []
        for i in range(len(self.boxes)):
            number.append(self.od_res.boxes[i].shape[0])
        return (
            f'numbers of the boxes in each frame: {number}'
        )
               # todo: action recognize result class:

class ARResult:
    """
    储存行为识别结果的类
    属性：
        boxes:对应的边界框列表，长4，每个元素为(N,4)
        conf:每个边界框对应的置信度列表。长N
        cls:每个边界框对应的类别ID列表。长N

    """
    def __init__(self,):
        self.boxes = []
        self.conf = []
        self.cls = []
        self.id = []
    
    def __str__(self) -> str:
        return (
            f'numbers of action boxes: {len(self.boxes)}'
            f'class index of action boxes: {self.cls}'
            f'class index of action boxes: {self.conf}'
            f'id of action boxes: {self.id}'
        )
    
    def from_mmaction_result(self, mmaction_result:List[ActionDataSample]):
        """
        从MMACTION行为识别结果初始化ARResult对象。
        
        参数:
            mmaction_result: MMACTION模型的行为识别结果对象，包含pred_scores等属性。
        """
        for result in mmaction_result:
            scores = result.pred_score.cpu().numpy()
            self.conf.append(np.max(scores))
            self.cls.append(np.argmax(scores))
    
class VideoData:
    """
    存储一个视频段的数据容器。
    
    属性:
        frame_rate (float): 视频的原始帧率（FPS）。
        frames_list (list): 该视频段内采集的帧列表，每个帧为BGR顺序的HWC格式numpy数组。
        segment_start_time (float): 该视频段在原始视频中的开始时间（秒）。
        segment_end_time (float): 该视频段在原始视频中的结束时间（秒）。
    """
    def __init__(self, frame_rate, frames_list, segment_start_time, segment_end_time, frame_index, video_path):
        self.video_path = video_path
        self.video_name = video_path.split('/')[-1]
        self.frame_rate = frame_rate
        self.frames_list = frames_list
        self.segment_start_time = segment_start_time
        self.segment_end_time = segment_end_time
        self.frame_index = frame_index

        self.od_res = None  # 预留属性，用于存储目标检测结果
        self.act_res = None  # 预留属性，用于存储动作识别结果
        self.anno = None  # 预留属性，用于存储融合后的标注结果


    def __str__(self):
        return (f"VideoData(frame_rate={self.frame_rate}, "
                f"video_name={self.video_name}"
                f"num_frames={len(self.frames_list)}, "
                f"segment_start_time={self.segment_start_time}, "
                f"segment_end_time={self.segment_end_time}, "
                f"frame_index={self.frame_index})")
    
    def merge_res_to_anno(self,):

        if (len(self.frames_list) == 0) or (self.od_res is None):
            return

        anno_list = [[] for _ in range(len(self.frames_list))]

        for i in range(len(self.frames_list)):
            bboxes = self.od_res.boxes[i]
            track_ids = self.od_res.id[i]
            class_ids = self.od_res.cls[i]
            confidences = self.od_res.conf[i]
            for j in range(len(bboxes)):

                # print(temp_dict)
                anno_list[i].append(
                    {
                        "bbox": bboxes[j],
                        "id": track_ids[j],
                        "cls": "OD:" + str(int(class_ids[j])),
                        "conf": confidences[j],
                    }
                )
        
        if len(self.act_res.cls) > 0:
            for num in range(len(self.act_res.boxes)):
                act_id = "AR:" + str(int(self.act_res.cls[num]))
                conf = self.act_res.conf[num]
                id = self.act_res.id[num]
                for i in range(4):
                    bbox = self.act_res.boxes[num][i]
                    for j in range(4):
                        frame_index = i * 4 + j

                        anno_list[frame_index].append(
                            {
                                "bbox": bbox,
                                "id": id,
                                "cls": act_id,
                                "conf": conf,
                            }
                        )
        
        self.anno = anno_list
        return

class PlottedResult:
    """
    存储绘制结果的类
    属性：
        images (list): 绘制后的图像列表，每个图像为BGR顺序的HWC格式numpy数组。
        video_name (str): 视频名称。
        video_fps (float): 视频的帧率。
        raw_anno (list): 原始标注信息列表。
        
    """
    def __init__(self, video_name:str, 
                 video_fps:float, ):
        
        self.images = []
        self.video_name = video_name
        self.video_fps = video_fps
        self.raw_anno = []
    
    def __str__(self) -> str:
        return (
            f'numbers of plotted images: {len(self.images)}'
            f'video_name: {self.video_name}'
            f'video_fps: {self.video_fps}'
            f'number of raw annotations: {len(self.raw_anno)}'
        )
    
    def add_res(self, 
                images: List[np.ndarray],
                raw_anno: List,):
        self.images.extend(images)
        self.raw_anno.extend(raw_anno)
        return