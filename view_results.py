import json
import visualizer

d = json.load(open("results/benchmark_raw_results.json", "r", encoding="utf-8"))

header = f"{'Model':<26} | {'Params(M)':<9} | {'FLOPs(G)':<8} | {'Lat(ms)':<7} | {'mAP@50':<8} | {'mAP50-95':<9} | {'Torn(Clean)':<11} | {'Torn(Glare)':<11} | {'Torn Drop':<11}"
print("=" * len(header))
print(header)
print("-" * len(header))

for m, v in d.items():
    ov = v["overall"]
    eff = v["efficiency"]
    cond = v["conditions"]
    disp_name = visualizer.MODEL_DISPLAY.get(m, m)
    tc = cond.get("torn_clean", {}).get("torn_mAP50", 0.0)
    tb = cond.get("torn_bright", {}).get("torn_mAP50", 0.0)
    drop = tc - tb
    print(f"{disp_name:<26} | {eff['params_m']:<9.2f} | {eff['flops_g']:<8.2f} | {eff['latency_ms']:<7.1f} | {ov['mAP50']:<7.2f}% | {ov['mAP50_95']:<8.2f}% | {tc:<10.1f}% | {tb:<10.1f}% | {drop:<10.1f}%")

print("=" * len(header))
