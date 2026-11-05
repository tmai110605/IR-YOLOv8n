#!/usr/bin/env python3
"""
=============================================================================
IR-YOLOv8n: Training Script for Banknote Recognition & Tear Detection
=============================================================================
Official PyTorch implementation for training the proposed IR-YOLOv8n detector
with Illumination-Gated Attention (C2f-IGA) under challenging illumination.

Usage:
    # Train with default hyperparameters (100 epochs, batch 16, GPU 0)
    python train.py --data data.yaml --device 0

    # Custom training parameters
    python train.py --epochs 100 --batch 16 --imgsz 640 --device 0 --name ir_yolov8n

    # Train on CPU
    python train.py --device cpu
"""

import os
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

# Register custom C2f-IGA attention module before building YOLO model
from models.custom_modules import register_custom_modules
register_custom_modules()


def parse_args():
    parser = argparse.ArgumentParser(description="Train IR-YOLOv8n on Banknote Adverse-Illumination Dataset")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="models/ir_yolov8n.yaml", help="Path to IR-YOLOv8n architecture config")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt", help="Pretrained COCO weights for tensor transfer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image resolution (default: 640)")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="CUDA device index or 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=43, help="Deterministic random seed")
    parser.add_argument("--project", type=str, default="runs/detect", help="Save directory")
    parser.add_argument("--name", type=str, default="ir_yolov8n", help="Experiment name")
    parser.add_argument("--val", action="store_true", default=True, help="Run validation/test evaluation after training")
    return parser.parse_args()


def train():
    args = parse_args()
    print("=" * 80)
    print("             TRAINING PROPOSED IR-YOLOv8n MODEL")
    print("=" * 80)
    print(f"  Architecture Config : {args.model}")
    print(f"  Pretrained Weights  : {args.pretrained}")
    print(f"  Dataset Config      : {args.data}")
    print(f"  Epochs / Batch Size : {args.epochs} / {args.batch}")
    print(f"  Image Size (imgsz)  : {args.imgsz}")
    print(f"  Compute Device      : {args.device}")
    print(f"  Output Directory    : {os.path.join(args.project, args.name)}")
    print("=" * 80)

    # 1. Instantiate model from architecture YAML and transfer official COCO pretrained weights
    print(f"\n[1/3] Building model from {args.model} and transferring weights from {args.pretrained}...")
    model = YOLO(args.model).load(args.pretrained)

    # 2. Train with the paper's standardized optimization recipe:
    #    - Cosine learning rate decay (cos_lr=True)
    #    - Automatic Mixed Precision (amp=True)
    #    - Seed 43 with deterministic training for exact reproducibility
    #    - Turn off mosaic for the final 10 epochs (close_mosaic=10)
    print("\n[2/3] Launching training...")
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        deterministic=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        exist_ok=True,
        verbose=True
    )

    # 3. Post-training evaluation on the designated test set
    if args.val:
        print("\n[3/3] Evaluating freshly trained optimal checkpoint on the test split...")
        best_pt = Path(args.project) / args.name / "weights" / "best.pt"
        if best_pt.exists():
            val_model = YOLO(str(best_pt))
            metrics = val_model.val(data=args.data, split="test", device=args.device, imgsz=args.imgsz)
            print("\n" + "=" * 60)
            print("                 EVALUATION METRICS (TEST SET)")
            print("=" * 60)
            print(f"  mAP@50     : {float(metrics.box.map50) * 100:.2f}%")
            print(f"  mAP@50-95  : {float(metrics.box.map) * 100:.2f}%")
            print(f"  Precision  : {float(metrics.box.mp) * 100:.2f}%")
            print(f"  Recall     : {float(metrics.box.mr) * 100:.2f}%")
            print("=" * 60)
            print(f"  Best weights saved at: {best_pt.resolve()}")
        else:
            print(f"[WARN] Checkpoint not found at {best_pt}")

    print("\n[DONE] Training pipeline completed successfully.")


if __name__ == "__main__":
    train()

