"""
visualizer.py
=============
Automated scientific visualization and LaTeX table generation for paper publication.
Generates:
- High-resolution publication-quality figures (PNG @ 300 DPI & PDF)
- LaTeX tables (Springer/IEEE svjour3 compatible)
- CSV summary tables
"""

import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Styling for academic publication
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['grid.linestyle'] = '--'

COND_DISPLAY = {
    "indoor": "Indoor\n(Baseline)",
    "outdoor": "Outdoor\nSunlight",
    "backlight": "Strong\nBacklight",
    "overexposed": "Specular\nGlare",
    "torn_clean": "Torn\n(Neutral)",
    "torn_bright": "Torn\n(Glare)"
}

MODEL_DISPLAY = {
    "yolov5nu": "YOLOv5nu",
    "yolov8n": "YOLOv8n",
    "yolov8s": "YOLOv8s (Small)",
    "yolov9t": "YOLOv9t (GELAN)",
    "yolov10n": "YOLOv10n (NMS-free)",
    "yolo11n": "YOLO11n",
    "ghost_yolov8n": "Ghost-YOLOv8n",
    "two_stage_clahe": "Two-Stage (CLAHE + YOLOv8n)",
    "ir_yolov8n": "IR-YOLOv8n (Ours)",
    "ir_yolo11n": "IR-YOLO11n (Ours)"
}

COLORS = {
    "yolov5nu": "#95A5A6",         # Gray
    "yolov8n": "#4A90E2",          # Blue
    "yolov8s": "#34495E",          # Dark Slate
    "yolov9t": "#9B59B6",          # Purple
    "yolov10n": "#1ABC9C",         # Turquoise
    "yolo11n": "#2ECC71",          # Emerald Green
    "ghost_yolov8n": "#F39C12",    # Amber Orange
    "two_stage_clahe": "#7F8C8D",  # Steel Gray
    "ir_yolov8n": "#E74C3C",       # Bold Crimson Red (Ours)
    "ir_yolo11n": "#C0392B"        # Dark Red (Ours)
}


def generate_tables(benchmark_results: Dict[str, Any], output_dir: str = "results/tables"):
    """Generates publication-ready CSV and LaTeX tables."""
    os.makedirs(output_dir, exist_ok=True)
    models = list(benchmark_results.keys())

    # --- TABLE 1: OVERALL EFFICIENCY & PERFORMANCE ---
    t1_rows = []
    for m in models:
        r = benchmark_results[m]
        ov = r.get("overall", {})
        eff = r.get("efficiency", {})
        t1_rows.append({
            "Model": MODEL_DISPLAY.get(m, m),
            "Params (M)": eff.get("params_m", 0.0),
            "FLOPs (G)": eff.get("flops_g", 0.0),
            "Latency (ms)": eff.get("latency_ms", 0.0),
            "FPS": eff.get("fps", 0.0),
            "Precision (%)": round(ov.get("precision", 0.0), 2),
            "Recall (%)": round(ov.get("recall", 0.0), 2),
            "mAP@50 (%)": round(ov.get("mAP50", 0.0), 2),
            "mAP@50-95 (%)": round(ov.get("mAP50_95", 0.0), 2)
        })
    df1 = pd.DataFrame(t1_rows)
    df1.to_csv(os.path.join(output_dir, "table1_overall.csv"), index=False)

    # LaTeX Table 1
    tex1 = r"""\begin{table}[tb]
\centering
\caption{Comprehensive Comparison of Efficiency and Detection Accuracy Across Lightweight Architectures.}
\label{tab:overall_performance}
\scriptsize
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model} & \textbf{Params (M)} & \textbf{FLOPs (G)} & \textbf{Latency (ms)} & \textbf{FPS} & \textbf{Pre. (\%)} & \textbf{Rec. (\%)} & \textbf{mAP@50} & \textbf{mAP@50-95} \\
\midrule
"""
    for row in t1_rows:
        is_ours = "(Ours)" in row["Model"]
        b_pre = r"\textbf{" if is_ours else ""
        b_post = r"}" if is_ours else ""
        tex1 += f"{b_pre}{row['Model']}{b_post} & {row['Params (M)']} & {row['FLOPs (G)']} & {row['Latency (ms)']} & {row['FPS']} & {row['Precision (%)']} & {row['Recall (%)']} & {b_pre}{row['mAP@50 (%)']}{b_post} & {b_pre}{row['mAP@50-95 (%)']}{b_post} \\\\\n"
    tex1 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(output_dir, "table1_overall.tex"), "w", encoding="utf-8") as f:
        f.write(tex1)

    # --- TABLE 2: ADVERSE ILLUMINATION REGIME BREAKDOWN ---
    t2_rows = []
    cond_keys = ["indoor", "outdoor", "backlight", "overexposed", "torn_clean", "torn_bright"]
    for m in models:
        r = benchmark_results[m]
        conds = r.get("conditions", {})
        indoor_map = conds.get("indoor", {}).get("mAP50", 100.0)
        clean_torn = conds.get("torn_clean", {}).get("torn_mAP50", 0.0)
        bright_torn = conds.get("torn_bright", {}).get("torn_mAP50", 0.0)
        torn_drop = clean_torn - bright_torn
        row = {
            "Model": MODEL_DISPLAY.get(m, m),
            "Indoor": round(indoor_map, 2),
            "Outdoor": round(conds.get("outdoor", {}).get("mAP50", 0.0), 2),
            "Backlight": round(conds.get("backlight", {}).get("mAP50", 0.0), 2),
            "Overexposed (Glare)": round(conds.get("overexposed", {}).get("mAP50", 0.0), 2),
            "Torn (Neutral)": round(clean_torn, 2),
            "Torn (Glare)": round(bright_torn, 2),
            "Torn Drop (Δ)": round(torn_drop, 2)
        }
        t2_rows.append(row)
    df2 = pd.DataFrame(t2_rows)
    df2.to_csv(os.path.join(output_dir, "table2_conditions.csv"), index=False)

    # LaTeX Table 2
    tex2 = r"""\begin{table}[tb]
\centering
\caption{Adverse Illumination Robustness Breakdown: mAP@50 (\%) across 6 Operational Regimes.}
\label{tab:condition_breakdown}
\scriptsize
\begin{tabular}{lcccccc|c}
\toprule
\textbf{Model} & \textbf{Indoor} & \textbf{Outdoor} & \textbf{Backlight} & \textbf{Glare} & \textbf{Torn (Neutral)} & \textbf{Torn (Glare)} & \textbf{$\Delta$ Torn Drop $\downarrow$} \\
\midrule
"""
    for row in t2_rows:
        is_ours = "(Ours)" in row["Model"]
        b_pre = r"\textbf{" if is_ours else ""
        b_post = r"}" if is_ours else ""
        tex2 += f"{b_pre}{row['Model']}{b_post} & {row['Indoor']} & {row['Outdoor']} & {row['Backlight']} & {row['Overexposed (Glare)']} & {row['Torn (Neutral)']} & {row['Torn (Glare)']} & {b_pre}{row['Torn Drop (Δ)']}{b_post} \\\\\n"
    tex2 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(output_dir, "table2_conditions.tex"), "w", encoding="utf-8") as f:
        f.write(tex2)

    # --- TABLE 3: DEFECT (TORN) DETECTION PERFORMANCE ---
    t3_rows = []
    for m in models:
        r = benchmark_results[m]
        conds = r.get("conditions", {})
        clean_torn = conds.get("torn_clean", {}).get("torn_mAP50", 0.0)
        bright_torn = conds.get("torn_bright", {}).get("torn_mAP50", 0.0)
        t3_rows.append({
            "Model": MODEL_DISPLAY.get(m, m),
            "Torn mAP@50 (Neutral)": round(clean_torn, 2),
            "Torn mAP@50 (Glare Degradation)": round(bright_torn, 2),
            "Defect Degradation (Δ)": round(clean_torn - bright_torn, 2)
        })
    df3 = pd.DataFrame(t3_rows)
    df3.to_csv(os.path.join(output_dir, "table3_defects.csv"), index=False)

    # LaTeX Table 3
    tex3 = r"""\begin{table}[tb]
\centering
\caption{Tear Defect (torn) Detection: Sensitivity to Specular Glare Degradation.}
\label{tab:defect_breakdown}
\scriptsize
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Torn mAP@50 (Neutral)} & \textbf{Torn mAP@50 (Glare)} & \textbf{$\Delta$ Degradation $\downarrow$} \\
\midrule
"""
    for row in t3_rows:
        is_ours = "(Ours)" in row["Model"]
        b_pre = r"\textbf{" if is_ours else ""
        b_post = r"}" if is_ours else ""
        tex3 += f"{b_pre}{row['Model']}{b_post} & {row['Torn mAP@50 (Neutral)']} & {row['Torn mAP@50 (Glare Degradation)']} & {b_pre}{row['Defect Degradation (Δ)']}{b_post} \\\\\n"
    tex3 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(output_dir, "table3_defects.tex"), "w", encoding="utf-8") as f:
        f.write(tex3)

    print(f"[REPORT] Successfully generated CSV & LaTeX tables in '{output_dir}'.")


def generate_figures(benchmark_results: Dict[str, Any], output_dir: str = "results/figures"):
    """Generates 4 publication-quality academic charts."""
    os.makedirs(output_dir, exist_ok=True)
    models = list(benchmark_results.keys())
    cond_keys = ["indoor", "outdoor", "backlight", "overexposed", "torn_clean", "torn_bright"]

    def get_eval_val(model_key, cond_key):
        cd = benchmark_results[model_key]["conditions"].get(cond_key, {})
        if "torn" in cond_key:
            return cd.get("torn_mAP50", cd.get("mAP50", 0.0))
        return cd.get("mAP50", 0.0)

    # 1. FIGURE 1: GROUPED BAR CHART OF ROBUSTNESS ACROSS CONDITIONS
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    x = np.arange(len(cond_keys))
    width = 0.8 / len(models)

    for i, m in enumerate(models):
        y_vals = [get_eval_val(m, c) for c in cond_keys]
        color = COLORS.get(m, "#333333")
        label = MODEL_DISPLAY.get(m, m)
        bars = ax.bar(x + (i - len(models)/2 + 0.5) * width, y_vals, width, label=label, color=color, edgecolor='black', linewidth=0.8, alpha=0.9)
        # Add value on top of bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, rotation=90)

    ax.set_ylabel("Detection Accuracy (mAP@50 %)", fontweight="bold")
    ax.set_title("Environmental Robustness: mAP@50 Comparison Across 6 Illumination Regimes", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_DISPLAY[c] for c in cond_keys], fontweight="bold")
    ax.set_ylim(40, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='lower left')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_condition_robustness.png"))
    plt.close(fig)

    # 2. FIGURE 2: EFFICIENCY VS ACCURACY PARETO SCATTER PLOT
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    for m in models:
        eff = benchmark_results[m].get("efficiency", {})
        ov = benchmark_results[m].get("overall", {})
        params = eff.get("params_m", 3.0)
        map50 = ov.get("mAP50", 90.0)
        color = COLORS.get(m, "#333333")
        label = MODEL_DISPLAY.get(m, m)
        
        ax.scatter(params, map50, color=color, s=200, edgecolors='black', linewidth=1.5, zorder=5, label=label)
        ax.annotate(f"{label}\n({map50:.1f}%)", (params, map50 + 0.5), ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel("Model Complexity: Parameters (Million) ↓", fontweight="bold")
    ax.set_ylabel("Overall Test Accuracy: mAP@50 (%) ↑", fontweight="bold")
    ax.set_title("Pareto Efficiency: Model Parameters vs Detection Accuracy", fontweight="bold", pad=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(80, 102)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_efficiency_pareto.png"))
    plt.close(fig)

    # 3. FIGURE 3: TEAR DEFECT DETECTION DEGRADATION (Clean vs Glare)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    x = np.arange(len(models))
    w = 0.35
    clean_scores = [benchmark_results[m]["conditions"].get("torn_clean", {}).get("torn_mAP50", 0.0) for m in models]
    bright_scores = [benchmark_results[m]["conditions"].get("torn_bright", {}).get("torn_mAP50", 0.0) for m in models]

    b1 = ax.bar(x - w/2, clean_scores, w, label='Neutral Illumination (torn_clean)', color='#4A90E2', edgecolor='black', linewidth=0.8)
    b2 = ax.bar(x + w/2, bright_scores, w, label='Specular Glare (torn_bright)', color='#E74C3C', edgecolor='black', linewidth=0.8)

    for bar in b1 + b2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', (bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold')

    ax.set_ylabel("Tear Defect mAP@50 (%)", fontweight="bold")
    ax.set_title("Vulnerability Analysis: Tear Defect Detection Degradation under Specular Glare", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], fontweight="bold")
    ax.set_ylim(40, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_tear_defect_robustness.png"))
    plt.close(fig)

    # 4. FIGURE 4: 6-AXIS RADAR / SPIDER CHART
    fig = plt.figure(figsize=(7, 7), dpi=300)
    ax = fig.add_subplot(111, polar=True)
    num_vars = len(cond_keys)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the radar loop

    for m in models:
        values = [get_eval_val(m, c) for c in cond_keys]
        values += values[:1]
        color = COLORS.get(m, "#333333")
        label = MODEL_DISPLAY.get(m, m)
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=label)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([COND_DISPLAY[c].replace('\n', ' ') for c in cond_keys], fontweight="bold", fontsize=9)
    ax.set_ylim(40, 100)
    ax.set_title("Multi-Condition Robustness Profile", fontweight="bold", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_radar_condition_profile.png"))
    plt.close(fig)

    print(f"[FIGURES] Successfully generated 4 publication-quality figures in '{output_dir}'.")
