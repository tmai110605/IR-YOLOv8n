#!/usr/bin/env bash
# =============================================================================
# CashVision Automated Server Benchmark Runner (GPU / CUDA)
# =============================================================================
set -e

echo "================================================================================"
echo " Starting CashVision GPU Benchmark on Linux Server"
echo "================================================================================"

# Verify GPU
if command -v nvidia-smi &> /dev/null; then
    echo "[GPU] Active GPU status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
else
    echo "[WARN] nvidia-smi not found. Running on default PyTorch device."
fi

# Make sure output directories exist
mkdir -p results/tables results/figures

# Train & benchmark the proposed IR-YOLOv8n model (100 epochs, cosine LR, structural gradient energy attention):
# Automatically integrates with empirical baselines and regenerates all LaTeX tables & figures!
python run_pipeline.py --train --models ir_yolov8n --epochs 100 --batch 16 --workers 8 --device 0

# To evaluate all real checkpoints together without retraining (live verification):
# python run_pipeline.py --models yolov8n yolo11n yolov8s ir_yolov8n --device 0

echo "================================================================================"
echo " IR-YOLOv8n trained and evaluated. Artifacts updated in 'results/'"
echo "================================================================================"
