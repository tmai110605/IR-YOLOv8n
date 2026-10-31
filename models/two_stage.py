"""
two_stage.py
============
Implements classical Two-Stage Enhancement + Detector baseline:
[Input Image] -> [Enhancement Preprocessor (CLAHE / Contrast Normalization)] -> [YOLO Detector]
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np


class CLAHEPreprocessor:
    """Applies Contrast Limited Adaptive Histogram Equalization on Luminance channel."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance(self, bgr_img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def create_enhanced_dataset(src_dir: str = "test", dst_dir: str = "temp_splits/test_clahe", method: str = "clahe"):
    """
    Creates a temporary enhanced copy of the dataset (images and labels) for fair evaluation.
    Places enhanced images in {dst_dir}/images and mirrors labels in {dst_dir}/labels.
    """
    src_images = Path(src_dir) / "images" if (Path(src_dir) / "images").exists() else Path(src_dir)
    src_labels = Path(src_dir) / "labels" if (Path(src_dir) / "labels").exists() else Path(src_dir).parent / "labels"
    
    dst_images = Path(dst_dir) / "images"
    dst_labels = Path(dst_dir) / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    enhancer = CLAHEPreprocessor()

    # 1. Process and save enhanced images
    for fname in os.listdir(src_images):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        src_path = src_images / fname
        dst_path = dst_images / fname
        if not dst_path.exists():
            img = cv2.imread(str(src_path))
            if img is not None:
                enhanced = enhancer.enhance(img)
                cv2.imwrite(str(dst_path), enhanced)

    # 2. Mirror labels
    if src_labels.exists():
        for l_file in os.listdir(src_labels):
            if l_file.lower().endswith('.txt'):
                src_l = src_labels / l_file
                dst_l = dst_labels / l_file
                if not dst_l.exists():
                    shutil.copy2(src_l, dst_l)

    return str(dst_images)
