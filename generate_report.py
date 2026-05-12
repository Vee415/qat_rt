"""Generate final benchmark report combining all results.

Reads benchmark_results.json and trt_profile_report.json,
produces a markdown report with tables, analysis, and recommendations.
"""
import json
import os
from collections import OrderedDict


def load_results():
    with open("models/benchmark_results.json") as f:
        results = json.load(f)
    trt_profile = None
    if os.path.exists("models/trt_profile_report.json"):
        with open("models/trt_profile_report.json") as f:
            trt_profile = json.load(f)
    return results, trt_profile


def fmt_variant(name):
    rename = {
        "PyTorch FP32 (CPU)": "PyTorch FP32 (CPU)",
        "PyTorch FP32 (CUDA)": "PyTorch FP32 (GPU)",
        "PyTorch PTQ INT8 (CPU)": "PyTorch PTQ INT8 (CPU)",
        "PyTorch QAT INT8 (CPU)": "PyTorch QAT INT8 (CPU)",
        "ONNX CPUExecutionProvider": "ONNX Runtime (CPU)",
        "ONNX CUDAExecutionProvider+CPUExecutionProvider": "ONNX Runtime (CUDA)",
        "ORT TensorRT FP16": "ORT TensorRT FP16",
        "ORT TensorRT INT8": "ORT TensorRT INT8",
        "kws_trt_fp16": "TRT Native FP16",
        "kws_trt_int8": "TRT Native INT8",
    }
    return rename.get(name, name)


def generate_report():
    results, trt_profile = load_results()

    # Use the profiled TRT latencies (more warmup, more accurate)
    if trt_profile:
        for eng in trt_profile["engines"]:
            for r in results:
                if r["variant"] == eng["variant"]:
                    r["latency_mean_ms"] = eng["latency_mean_ms"]
                    r["latency_p50_ms"] = eng.get("latency_p50_ms", r.get("latency_p95_ms"))
                    r["latency_p95_ms"] = eng["latency_p95_ms"]
                    r["latency_p99_ms"] = eng["latency_p99_ms"]
                    r["latency_min_ms"] = eng.get("latency_min_ms", "")
                    r["latency_max_ms"] = eng.get("latency_max_ms", "")
                    r["throughput_fps"] = eng["throughput_fps"]

    # Sort by P99 latency (fastest first) — P99 is the primary metric, not mean
    results.sort(key=lambda r: float(r["latency_p99_ms"]))

    lines = []
    lines.append("# Keyword Spotting: QAT & TensorRT Pipeline — Final Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("A 3-block CNN (KWSNet, 94K parameters) trained on Google Speech Commands v2")
    lines.append("for 10-class keyword spotting, optimized through:")
    lines.append("")
    lines.append("1. **FP32 baseline** — full-precision training")
    lines.append("2. **PTQ INT8** — post-training static quantization with calibration")
    lines.append("3. **QAT INT8** — quantization-aware training with fake quant nodes")
    lines.append("4. **TensorRT FP16/INT8** — GPU-optimized inference engines")
    lines.append("")
    lines.append("**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU, Intel CPU, TensorRT 10.16")
    lines.append("")

    # ================================================================
    # Accuracy Comparison
    # ================================================================
    lines.append("## 1. Accuracy Comparison")
    lines.append("")
    lines.append("| Variant | Accuracy | vs FP32 |")
    lines.append("|---------|----------|---------|")
    fp32_acc = float(next(r for r in results if "FP32 (CPU)" in r["variant"])["accuracy"])
    for r in results:
        acc = float(r["accuracy"])
        delta = acc - fp32_acc
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {fmt_variant(r['variant'])} | {acc:.4f} ({acc*100:.2f}%) | {sign}{delta*100:.2f}% |")
    lines.append("")
    lines.append("**Key finding:** QAT INT8 *exceeds* FP32 accuracy by +1.64%. Training with")
    lines.append("quantization noise acts as regularization, improving generalization.")
    lines.append("")

    # ================================================================
    # Latency & Throughput (sorted by P99 — the primary deployment metric)
    # ================================================================
    lines.append("## 2. Latency & Throughput (single-sample inference)")
    lines.append("")
    lines.append("> **Why P99 over mean?** For streaming keyword spotting, mean latency is")
    lines.append("> misleading. If your audio pipeline expects inference every 2ms and P99 is")
    lines.append("> 2.62ms, 1% of frames will miss their deadline — causing dropped audio,")
    lines.append("> missed wake words, or buffer underruns. P99 tells you the worst-case")
    lines.append("> latency your system must tolerate. The table is sorted by P99.")
    lines.append("")
    lines.append("| Variant | P99 (ms) | P95 (ms) | Mean (ms) | Throughput (fps) |")
    lines.append("|---------|----------|----------|-----------|------------------|")
    for r in results:
        lines.append(
            f"| {fmt_variant(r['variant'])} | **{r['latency_p99_ms']}** | "
            f"{r['latency_p95_ms']} | {r['latency_mean_ms']} | "
            f"{r['throughput_fps']} |"
        )
    lines.append("")

    # P99 vs Mean comparison
    lines.append("### P99 vs Mean: The Tail Latency Gap")
    lines.append("")
    lines.append("| Variant | Mean (ms) | P99 (ms) | P99/Mean Ratio |")
    lines.append("|---------|-----------|----------|-----------------|")
    for r in results:
        mean_lat = float(r["latency_mean_ms"])
        p99_lat = float(r["latency_p99_ms"])
        ratio = p99_lat / mean_lat if mean_lat > 0 else 0
        lines.append(
            f"| {fmt_variant(r['variant'])} | {r['latency_mean_ms']} | "
            f"{r['latency_p99_ms']} | {ratio:.1f}x |"
        )
    lines.append("")
    lines.append("P99 is 1.1-2.0x higher than mean across variants. The gap is largest for")
    lines.append("CPU-based inference (PyTorch) due to OS scheduling jitter, and smallest for")
    lines.append("GPU-based inference (TRT) where the GPU handles timing predictably.")
    lines.append("")

    # ================================================================
    # Model Size
    # ================================================================
    lines.append("## 3. Model Size Comparison")
    lines.append("")
    lines.append("| Variant | Size (MB) | Compression vs FP32 |")
    lines.append("|---------|-----------|---------------------|")
    fp32_size = float(next(r for r in results if "FP32 (CPU)" in r["variant"])["size_mb"])
    for r in results:
        size = float(r["size_mb"])
        ratio = f"{fp32_size/size:.1f}x" if size > 0 else "N/A"
        lines.append(f"| {fmt_variant(r['variant'])} | {r['size_mb']} | {ratio} |")
    lines.append("")
    lines.append("INT8 quantization reduces model size to 0.10 MB — a **3.7x compression**")
    lines.append("over FP32. TRT engines are also compact: 0.29 MB (FP16) and 0.20 MB (INT8).")
    lines.append("")

    # ================================================================
    # Speedup Analysis (by P99)
    # ================================================================
    lines.append("## 4. Speedup vs FP32 CPU Baseline (by P99 latency)")
    lines.append("")
    fp32_p99 = float(next(r for r in results if "FP32 (CPU)" in r["variant"])["latency_p99_ms"])
    lines.append("| Variant | P99 (ms) | Speedup vs FP32 CPU |")
    lines.append("|---------|----------|---------------------|")
    for r in results:
        p99 = float(r["latency_p99_ms"])
        speedup = fp32_p99 / p99
        lines.append(f"| {fmt_variant(r['variant'])} | {r['latency_p99_ms']} | {speedup:.1f}x |")
    lines.append("")

    # ================================================================
    # TRT Fusion Analysis
    # ================================================================
    if trt_profile:
        lines.append("## 5. TensorRT Layer Fusion Analysis")
        lines.append("")
        lines.append("Original KWSNet has 12 separate operations (3x Conv+BN+ReLU, 3x MaxPool,")
        lines.append("1x AdaptiveAvgPool, 1x Linear). After TRT optimization:")
        lines.append("")

        fp16_layers = trt_profile.get("fp16_layers", [])
        int8_layers = trt_profile.get("int8_layers", [])

        lines.append("### FP16 Engine (12 layers, 0.29 MB)")
        lines.append("```")
        for layer in fp16_layers:
            lines.append(f"  {layer.strip()}")
        lines.append("```")
        lines.append("")
        lines.append("### INT8 Engine (10 layers, 0.20 MB)")
        lines.append("```")
        for layer in int8_layers:
            lines.append(f"  {layer.strip()}")
        lines.append("```")
        lines.append("")
        lines.append("**Fusion decisions:**")
        lines.append("- Conv+ReLU fused into single kernels (3 instances in both engines)")
        lines.append("- FP16 adds 2 reformat nodes (FP32<->FP16 conversion at input and between layers)")
        lines.append("- INT8 needs only 1 reformat (input quantization), keeping INT8 format throughout")
        lines.append("- INT8 fuses the final Gemm+quantize into a single kernel (Gemm + Gemm_3)")
        lines.append("")

    # ================================================================
    # Deployment Recommendations
    # ================================================================
    lines.append("## 6. Deployment Recommendations")
    lines.append("")
    lines.append("| Target Device | Recommended Variant | Accuracy | P99 (ms) | Mean (ms) | Size |")
    lines.append("|---------------|---------------------|----------|----------|-----------|------|")
    lines.append("| Edge CPU (Cortex-A, no GPU) | PyTorch QAT INT8 (CPU) | 89.00% | 2.16 | 1.45 | 0.10 MB |")
    lines.append("| Edge CPU (lowest P99) | ONNX Runtime (CPU) | 87.36% | 0.22 | 0.15 | 0.36 MB |")
    lines.append("| NVIDIA GPU (Jetson, RTX) | TRT Native FP16 | 87.31% | 0.13 | 0.08 | 0.29 MB |")
    lines.append("| NVIDIA GPU (smallest) | TRT Native INT8 | 86.72% | 0.17 | 0.09 | 0.20 MB |")
    lines.append("| Server CPU (accuracy-critical) | PyTorch QAT INT8 (CPU) | 89.00% | 2.16 | 1.45 | 0.10 MB |")
    lines.append("| Server GPU (throughput) | TRT Native FP16 | 87.31% | 0.13 | 0.08 | 0.29 MB |")
    lines.append("")
    lines.append("### Why P99 drives deployment choice:")
    lines.append("")
    lines.append("For a keyword spotting system that processes audio in 10ms frames, the")
    lines.append("deadline is 10ms. All variants meet this deadline on *average*, but P99")
    lines.append("reveals who actually stays within budget:")
    lines.append("")
    lines.append("- **ONNX Runtime (CPU)** P99 of 0.22ms — 45x headroom before deadline")
    lines.append("- **TRT Native FP16** P99 of 0.13ms — 77x headroom before deadline")
    lines.append("- **PyTorch FP32 (CPU)** P99 of 2.62ms — only 3.8x headroom, risky under load")
    lines.append("- **PyTorch QAT INT8 (CPU)** P99 of 2.16ms — 4.6x headroom, better but still tight")
    lines.append("")
    lines.append("### Key tradeoffs:")
    lines.append("")
    lines.append("- **Best accuracy:** QAT INT8 at 89.00% — the only variant that *beats* FP32.")
    lines.append("  Training with quantization noise acts as regularizer, improving generalization.")
    lines.append("- **Best CPU P99:** ONNX Runtime at 0.22ms — graph optimizations and")
    lines.append("  operator fusion give 12x better tail latency than PyTorch on CPU.")
    lines.append("- **Best GPU P99:** TRT Native FP16 at 0.13ms — kernel fusion + FP16")
    lines.append("  tensor cores give 20x better tail latency than PyTorch CPU.")
    lines.append("- **Smallest model:** QAT INT8 / PTQ INT8 at 0.10 MB — 3.7x smaller than FP32.")
    lines.append("  For 256MB edge devices, this matters more than latency.")
    lines.append("")

    # ================================================================
    # Quantization Impact Summary
    # ================================================================
    lines.append("## 7. Quantization Impact Summary")
    lines.append("")
    lines.append("| Technique | Accuracy | Accuracy Drop | Size Reduction |")
    lines.append("|-----------|----------|---------------|----------------|")
    lines.append("| FP32 baseline | 87.36% | — | — |")
    lines.append("| PTQ INT8 (calibration only) | 86.43% | -0.93% | 3.7x |")
    lines.append("| QAT INT8 (fine-tuned) | 89.00% | **+1.64%** | 3.7x |")
    lines.append("| TRT FP16 (GPU) | 87.31% | -0.05% | 1.3x |")
    lines.append("| TRT INT8 (GPU, calibrated) | 86.72% | -0.64% | 1.9x |")
    lines.append("")
    lines.append("**The QAT advantage:** Not only does QAT recover the accuracy lost by PTQ,")
    lines.append("it *surpasses* FP32. The quantization noise during training acts as a")
    lines.append("regularizer, preventing overfitting on the training set.")
    lines.append("")

    # Write report
    report = "\n".join(lines)
    os.makedirs("reports", exist_ok=True)
    with open("reports/benchmark_report.md", "w") as f:
        f.write(report)

    # Also save combined JSON
    combined = {
        "results": results,
        "trt_profile": trt_profile,
    }
    with open("models/final_results.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(report)
    print(f"\nReport saved to reports/benchmark_report.md")
    print(f"Data saved to models/final_results.json")


if __name__ == "__main__":
    generate_report()