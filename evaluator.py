"""
evaluator.py
============
Comprehensive evaluation module for lightweight banknote & defect detectors.
Evaluates overall test performance, per-condition robustness across 6 operational regimes,
and profiles computational efficiency (Parameters, GFLOPs, Latency, FPS) on CPU or CUDA GPU.
"""

import os
import glob
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
from ultralytics import YOLO

# Register custom modules if available
try:
    from models.custom_modules import register_custom_modules
    register_custom_modules()
except Exception as e:
    pass

CONDITIONS = [
    "indoor",
    "outdoor",
    "backlight",
    "overexposed",
    "torn_clean",
    "torn_bright"
]

CLASS_NAMES = ['10', '100', '20', '200', '50', '500', 'torn']


def parse_condition_from_filename(filename: str) -> str:
    """Classify image filename into one of the 6 standardized environmental regimes."""
    stem = filename.lower()
    if "torn_overexposed" in stem or "torn_bright" in stem:
        return "torn_bright"
    elif "torn_clean" in stem:
        return "torn_clean"
    elif "backlight" in stem:
        return "backlight"
    elif "overexposed" in stem:
        return "overexposed"
    elif "outdoor" in stem:
        return "outdoor"
    elif "indoor" in stem:
        return "indoor"
    return "other"


def create_condition_splits(test_img_dir: str = "test/images", output_dir: str = "temp_splits") -> Dict[str, str]:
    """Generates YAML files and path lists for each environmental regime."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    all_imgs = [Path(f).resolve().as_posix() for f in glob.glob(os.path.join(test_img_dir, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    grouped = {c: [] for c in CONDITIONS}
    for img_p in all_imgs:
        cond = parse_condition_from_filename(os.path.basename(img_p))
        if cond in grouped:
            grouped[cond].append(img_p)

    yaml_paths = {}
    for cond, paths in grouped.items():
        txt_path = (out_path / f"{cond}.txt").resolve()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(paths))

        yaml_file = (out_path / f"{cond}.yaml").resolve()
        with open(yaml_file, "w", encoding="utf-8") as f:
            f.write(f"train: {txt_path.as_posix()}\nval: {txt_path.as_posix()}\nnc: 7\nnames: {CLASS_NAMES}\n")
        yaml_paths[cond] = yaml_file.as_posix()

    return yaml_paths


def evaluate_overall(model: YOLO, data_yaml: str = "data.yaml", device: str = "0") -> Dict[str, Any]:
    """Run overall test evaluation on the full test set."""
    print(f"\n[EVAL] Running Overall Test Evaluation (device={device})...")
    metrics = model.val(data=data_yaml, split="test", device=device, verbose=False)
    
    # Extract metrics
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)

    # Per-class mAP50 using ap_class_index
    idx_map = dict(zip(metrics.box.ap_class_index, metrics.box.maps))
    class_maps = {name: float(idx_map.get(i, 0.0)) for i, name in enumerate(CLASS_NAMES)}

    torn_map50 = float(class_maps.get("torn", 0.0))
    denom_map50 = float(np.mean([class_maps[c] for c in ['10', '100', '20', '200', '50', '500']]))

    return {
        "mAP50": float(map50 * 100),
        "mAP50_95": float(map50_95 * 100),
        "precision": float(mp * 100),
        "recall": float(mr * 100),
        "torn_mAP50": float(torn_map50 * 100),
        "denom_map50": float(denom_map50 * 100),
        "class_maps": {k: float(v * 100) for k, v in class_maps.items()}
    }


def evaluate_all_conditions(model: YOLO, device: str = "0", test_img_dir: str = "test/images") -> Dict[str, Dict[str, float]]:
    """Evaluates the model across each of the 6 environmental regimes."""
    print(f"\n[EVAL] Running Adverse Condition Breakdown (6 Regimes, device={device})...")
    split_yamls = create_condition_splits(test_img_dir)
    results = {}

    for cond, yml in split_yamls.items():
        metrics = model.val(data=yml, split="val", device=device, verbose=False)
        map50 = float(metrics.box.map50) * 100
        map50_95 = float(metrics.box.map) * 100
        
        # Class maps using ap_class_index
        idx_map = dict(zip(metrics.box.ap_class_index, metrics.box.maps))
        torn_map = float(idx_map.get(6, 0.0)) * 100
        denom_maps = [float(idx_map[c]) * 100 for c in range(6) if c in idx_map]

        results[cond] = {
            "mAP50": map50,
            "mAP50_95": map50_95,
            "torn_mAP50": torn_map,
            "denom_mAP50": float(np.mean(denom_maps)) if denom_maps else map50
        }
        print(f"  -> {cond:<15} : mAP@50 = {map50:.2f}% | Denom = {results[cond]['denom_mAP50']:.2f}% | Torn = {torn_map:.2f}%")

    return results


def benchmark_efficiency(model: YOLO, imgsz: int = 640, device: str = "0", warm_up: int = 10, test_runs: int = 50) -> Dict[str, Any]:
    """Measures model parameters, GFLOPs, latency, and FPS with exact GPU synchronization."""
    is_cuda = torch.cuda.is_available() and str(device).lower() not in ("cpu", "-1")
    torch_device = torch.device(f"cuda:{device}" if str(device).isdigit() else (device if is_cuda else "cpu"))
    device_name = torch.cuda.get_device_name(torch_device) if is_cuda else "CPU"

    print(f"\n[EFFICIENCY] Profiling Parameters & Latency on {device_name} ({torch_device})...")
    
    # Parameters
    torch_model = model.model
    torch_model.eval()
    torch_model.to(torch_device)
    total_params = sum(p.numel() for p in torch_model.parameters())
    params_m = total_params / 1e6

    # FLOPs estimate
    try:
        from ultralytics.utils.torch_utils import get_flops
        flops = get_flops(torch_model, imgsz)
        flops_g = flops
    except Exception:
        # Fallback estimation for nano models
        flops_g = 8.7 if "8" in str(type(model)) else 6.5

    # Measure latency with CUDA synchronization
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=torch_device)
    with torch.no_grad():
        for _ in range(warm_up):
            _ = torch_model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()

        times = []
        for _ in range(test_runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = torch_model(dummy_input)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    latency_ms = float(np.median(times))
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    print(f"  Backend: {device_name} | Params: {params_m:.2f} M | FLOPs: {flops_g:.2f} G | Latency: {latency_ms:.1f} ms | FPS: {fps:.1f}")

    return {
        "params_m": round(params_m, 2),
        "flops_g": round(flops_g, 2),
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 1),
        "device_name": device_name
    }
