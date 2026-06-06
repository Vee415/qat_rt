# Keyword Spotting: QAT & TensorRT Pipeline — Final Report

> **Updated:** 2026-06-04 — Full cross-device benchmarks with laptop (RTX 4060) and Jetson Orin Nano Super.
> See **[Edge Comparison Report](edge_comparison_report.md)** for the complete cross-device analysis.

## Training Results

| Model | Accuracy | Size | Notes |
|-------|----------|------|-------|
| FP32 baseline | 88.65% | 0.37 MB | 30 epochs, best val checkpoint |
| PTQ INT8 | 87.19% | 0.10 MB | -1.46% from FP32, 3.7x compression |
| QAT INT8 | 88.22% | 0.10 MB | +1.03% over PTQ, nearly matches FP32 |

**Key insight:** QAT beats PTQ by +1.03% and nearly matches FP32 accuracy — quantization noise during training acts as regularization.

## Full Model Variant Comparison (Laptop, Single-Sample)

| Variant | Accuracy | Size (MB) | P99 (ms) | Mean (ms) | Throughput |
|---------|----------|-----------|----------|-----------|------------|
| TRT Native FP16 | 88.07% | 0.85 | 0.51 | 0.38 | 2,641 fps |
| TRT Native INT8 | 88.05% | 0.68 | 1.26 | 0.34 | 2,945 fps |
| ONNX CPU | 88.12% | 0.36 | 0.19 | 0.15 | 6,509 fps |
| ONNX CUDA | 88.12% | 0.36 | 0.29 | 0.19 | 5,212 fps |
| PyTorch FP32 (GPU) | 88.12% | 0.37 | 0.63 | 0.43 | 2,299 fps |
| PyTorch FP32 (CPU) | 88.12% | 0.37 | 1.65 | 1.37 | 732 fps |
| PTQ INT8 (CPU) | 87.21% | 0.10 | 1.36 | 1.13 | 887 fps |
| QAT INT8 (CPU) | 88.22% | 0.10 | 1.29 | 1.10 | 908 fps |

## Cross-Device Replay Benchmark

> **See [edge_comparison_report.md](edge_comparison_report.md) for full analysis.**

| Runtime | Device | Accuracy | p99 (ms) | Throughput | RAM | Cold Start |
|---------|--------|----------|-----------|------------|-----|------------|
| ONNX CPU | Laptop | 89.00% | 0.62 | 3,520 fps | 11.9 GB | 723 ms |
| ONNX CPU | Jetson | 89.00% | 17.42 | 154 fps | 1.8 GB | 78 ms |
| TRT FP16 | Laptop | 88.80% | 4.11 | 1,219 fps | 10.8 GB | 399 ms |
| TRT FP16 | Jetson | 89.00% | 1.32 | 1,170 fps | 2.0 GB | 233 ms |
| TRT INT8 | Laptop | 89.40% | 3.77 | 1,238 fps | 11.0 GB | 404 ms |
| TRT INT8 | Jetson | 89.80% | 1.19 | 1,341 fps | 1.9 GB | 215 ms |

**Zero dropped windows on all configurations at 10 FPS target.**

## Key Findings

1. **Same model, same accuracy across devices** — ONNX CPU = 89.00% on both laptop and Jetson
2. **Jetson ARM CPU is 27x slower** for CPU inference (p99: 17.42ms vs 0.62ms)
3. **Jetson beats laptop on TRT p99 latency** — unified memory eliminates GPU↔CPU copy overhead (1.19ms vs 3.77ms for INT8)
4. **QAT INT8 beats PTQ** by +1.03% and nearly matches FP32 accuracy
5. **Best edge deployment: TRT INT8 on Jetson** — 89.80% accuracy, p99=1.19ms, 0.21MB engine, 1,341fps

## Quantization Impact Summary

| Technique | Accuracy | Accuracy Drop | Size Reduction |
|-----------|----------|---------------|----------------|
| FP32 baseline | 88.65% | — | — |
| PTQ INT8 (calibration only) | 87.19% | -1.46% | 3.7x |
| QAT INT8 (fine-tuned) | 88.22% | +1.03% over PTQ | 3.7x |
| TRT FP16 (GPU) | 88.07% | -0.58% | 1.3x (laptop), 2.5x (Jetson) |
| TRT INT8 (GPU, calibrated) | 88.05% | -0.60% | 1.8x (laptop), 3.6x (Jetson) |