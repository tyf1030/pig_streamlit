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

def build_od_model(model_path: str, device: str = 'cuda:0', **kwargs):
    """
    构建目标检测模型的函数。
    
    参数:
        model_name: str, 目标检测模型的名称（例如'yolov8n', 'yolov8s'等）。
        pretrained: bool, 是否加载预训练权重。
        device: str, 模型运行的设备（例如'cpu'或'cuda'）。
        
    返回:
        model: YOLO对象，构建好的目标检测模型。
        pred_args: dict, 检测的配置信息。
    """
    # pth_path = kwargs.pop('pth_path', "/media/pc3048/8e92fcae-9dcb-48b7-818b-82c5f8ed050e/aidata/Tyf/gradio/yolo11n.pt")
    pred_args = kwargs
    model = YOLO(model_path)
    return model, pred_args

def build_ar_model(**kwargs):
    '''
        kwargs: 各种参数，应包含pth_path, cfg_path, device等
    '''

    pth_path = kwargs.pop('pth_path', "/home/Tyf/mmaction2/faster+(l=MDM7-0.6-4)+CBFocal/best_acc_top1_epoch_164.pth")
    cfg_path = kwargs.pop('cfg_path', "/home/Tyf/mmaction2/faster+(l=MDM7-0.6-4)+CBFocal/Customfastertanet_t2_1xb8-dense-1x1x16-300e_pig5k-rgb-change_loss.py")
    
    test_pipeline = kwargs.pop('test_pipeline', None) # 一般不由用户传入
    device = kwargs.pop('device', 'cuda:0')
    
    cfg = Config.fromfile(cfg_path)
    init_default_scope(cfg.get('default_scope', 'mmaction'))

    # 初始化行为识别模型和数据处理流水线
    armodel = init_recognizer(cfg, pth_path, device=device)
    test_pipeline_cfg = [
    {'type': 'NumpyInitFromMemory'},
    {'clip_len': 16,
    'frame_interval': 1,
    'num_clips': 1,
    'test_mode': True,
    'num_sample_positions': 1,
    'type': 'DenseSampleFrames'},
    {'type': 'NumpyDecode'},
    {'scale': (-1, 256), 'type': 'Resize'},
    {'crop_size': 256, 'type': 'ThreeCrop'},
    {'input_format': 'NCHW', 'type': 'FormatShape'},
    {'type': 'PackActionInputs'}]

    test_pipeline = Compose(test_pipeline_cfg)
    
    return armodel, test_pipeline, kwargs

def inference_recognizer_simplified(model: nn.Module,
                         video: Union[np.array],
                         test_pipeline: Optional[Compose] = None
                         ) -> ActionDataSample:
    """Inference a video with the recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        video (Union[str, dict]): The video file path or the results
            dictionary (the input of pipeline).
        test_pipeline (:obj:`Compose`, optional): The test pipeline.
            If not specified, the test pipeline in the config will be
            used. Defaults to None.

    Returns:
        :obj:`ActionDataSample`: The inference results. Specifically, the
        predicted scores are saved at ``result.pred_score``.
    """


    if isinstance(video, np.ndarray):
        # 检查是否为CTHW格式的numpy数组
        if video.ndim == 4 and video.shape[0] in [1, 3]:  # C, T, H, W
            input_flag = 'numpy_array'
        else:
            raise ValueError(f"Expected numpy array in CTHW format, but got shape {video.shape}")
        data = test_pipeline(dict(video_array=video, label=-1, start_index=0, modality='RGB'))
        data = pseudo_collate([data])

    if isinstance(video, List):
        data = []
        for i in range(len(video)):
            
            data.append(
                test_pipeline(dict(video_array=video[i], label=-1, start_index=0, modality='RGB')
                )
                )

            
        data = pseudo_collate(data)
        # data.to("cuda:1")

    # Forward the model
    with torch.no_grad():
        result = model.test_step(data)

    return result