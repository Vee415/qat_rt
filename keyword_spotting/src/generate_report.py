"""
Auto-generate markdown benchmark report from results.
"""

import json
import os
from datetime import datetime
from collections import OrderedDict


REPORT_TEMPLATE = """# Keyword Spotting — Benchmark Report

Generated: {timestamp}

## System Info
- **Device:** {device}
- **Date:** {date}

## Results

| Variant | Accuracy | Size (MB) | Latency Mean (ms) | Latency P95 (ms) | Latency P99 (ms) | Throughput (fps) |
|---------|----------|-----------|-------------------|------------------|------------------|-------------------|
{results_table}

## Analysis

### Accuracy vs Size
{accuracy_analysis}

### Latency Comparison
{latency_analysis}

### Key Takeaways
{takeaways}

## Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| 40 MFCC coefficients | Standard for speech; covers 0-8kHz perceptual range |
| 1-second fixed window | Speech Commands clips are ~1s; consistent input shape |
| Static PTQ over dynamic | Dynamic dequantizes at runtime — slower on small models |
| HistogramObserver for QAT | More accurate range estimation for non-uniform activations |
| Freeze BN at epoch 3 | Stabilizes quantization parameters after initial adaptation |
| FP16 TRT before INT8 TRT | FP16 needs no calibration — cleaner baseline for speedup attribution |

## Methodology
- All benchmarks use {iterations} iterations after {warmup} warmup iterations
- Same hardware for all variants
- Accuracy measured on test set with speaker-independent split
"""


def generate_report(results_path: str = "models/benchmark_results.json",
                    output_path: str = "reports/benchmark_report.md"):
    """Generate markdown report from benchmark results."""
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}. Run benchmark first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    if not results:
        print("No benchmark results to report.")
        return

    # Build results table
    rows = []
    for r in results:
        rows.append(
            f"| {r['variant']} | {r['accuracy']} | {r['size_mb']} | "
            f"{r['latency_mean_ms']} | {r['latency_p95_ms']} | "
            f"{r['latency_p99_ms']} | {r['throughput_fps']} |"
        )
    results_table = "\n".join(rows)

    # Simple analysis
    baseline = results[0] if results else None
    accuracy_analysis = []
    latency_analysis = []

    if baseline:
        for r in results[1:]:
            acc_diff = float(r["accuracy"]) - float(baseline["accuracy"])
            size_ratio = float(baseline["size_mb"]) / float(r["size_mb"]) if float(r["size_mb"]) > 0 else 0
            accuracy_analysis.append(
                f"- **{r['variant']}**: {acc_diff:+.4f} accuracy vs baseline, "
                f"{size_ratio:.1f}x size reduction"
            )

            lat_ratio = float(baseline["latency_mean_ms"]) / float(r["latency_mean_ms"]) if float(r["latency_mean_ms"]) > 0 else 0
            latency_analysis.append(
                f"- **{r['variant']}**: {lat_ratio:.1f}x faster than baseline "
                f"(p95: {r['latency_p95_ms']}ms vs {baseline['latency_p95_ms']}ms)"
            )

    takeaways = (
        "- PTQ provides model size reduction but may lose accuracy on hard classes\n"
        "- QAT recovers most PTQ accuracy loss by training through quantization noise\n"
        "- TensorRT adds latency improvement through layer fusion and kernel auto-tuning\n"
        "- The contribution of each technique (quantization, QAT, TRT) should be analyzed independently"
    )

    report = REPORT_TEMPLATE.format(
        timestamp=datetime.now().isoformat(),
        device="CUDA" if os.environ.get("CUDA_VISIBLE_DEVICES") else "CPU",
        date=datetime.now().strftime("%Y-%m-%d"),
        results_table=results_table,
        accuracy_analysis="\n".join(accuracy_analysis),
        latency_analysis="\n".join(latency_analysis),
        takeaways=takeaways,
        iterations=1000,
        warmup=50,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()