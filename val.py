#!/usr/bin/env python3
"""
=============================================================================
IR-YOLOv8n: Evaluation & Inference Script
=============================================================================
Evaluate a trained IR-YOLOv8n checkpoint on the validation or test split.

Usage:
    # Evaluate best weights on the test split
    python val.py --weights runs/detect/ir_yolov8n/weights/best.pt --data data.yaml --split test

    # Evaluate on GPU 0
    python val.py --weights runs/detect/ir_yolov8n/weights/best.pt --device 0
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# Register custom C2f-IGA attention module
from models.custom_modules import register_custom_modules
register_custom_modules()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IR-YOLOv8n checkpoint")
    parser.add_argument("--weights", type=str, default="runs/detect/ir_yolov8n/weights/best.pt", help="Path to trained .pt weights")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="CUDA device index or 'cpu'")
    return parser.parse_args()


def validate():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path.resolve()}")

    print("=" * 70)
    print("               EVALUATING IR-YOLOv8n CHECKPOINT")
    print("=" * 70)
    print(f"  Weights Path : {weights_path}")
    print(f"  Dataset Config: {args.data}")
    print(f"  Split        : {args.split}")
    print(f"  Device       : {args.device}")
    print("=" * 70)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=True
    )

    print("\n" + "=" * 60)
    print(f"       SUMMARY PERFORMANCE ON {args.split.upper()} SET")
    print("=" * 60)
    print(f"  mAP@50     : {float(metrics.box.map50) * 100:.2f}%")
    print(f"  mAP@50-95  : {float(metrics.box.map) * 100:.2f}%")
    print(f"  Precision  : {float(metrics.box.mp) * 100:.2f}%")
    print(f"  Recall     : {float(metrics.box.mr) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    validate()

