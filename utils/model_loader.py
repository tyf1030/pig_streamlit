# utils/model_loader.py
import streamlit as st
from backend import build_od_model, build_ar_model

@st.cache_resource
def load_od_model_cached(model_path, device, conf, iou):
    """带缓存的 OD 模型加载器"""
    return build_od_model(model_path=model_path, device=device, conf=conf, iou=iou)

@st.cache_resource
def load_ar_model_cached(pth_path, cfg_path, device):
    """带缓存的 AR 模型加载器"""
    return build_ar_model(pth_path=pth_path, cfg_path=cfg_path, device=device)