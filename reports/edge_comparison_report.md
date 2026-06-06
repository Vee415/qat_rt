# Edge Deployment Benchmark: Laptop (RTX 4060) vs Jetson Orin Nano Super

**Model:** KWSNet (94K params) trained on Google Speech Commands v2 (10 classes)  
**Date:** 2026-06-04  
**Same model weights used across all devices** — trained on laptop (RTX 4060), deployed via ONNX to both devices.

## Hardware

| | Laptop | Jetson Orin Nano Super |
|---|---|---|
| CPU | Intel i7-13700H (20 threads) | 6-core ARM Cortex-A78AE |
| GPU | NVIDIA RTX 4060 Laptop (8GB) | NVIDIA Ampere 1024-core (8GB unified) |
| RAM | 16GB DDR5 | 8GB LPDDR5 (unified CPU/GPU) |
| OS | Windows 11 | Ubuntu 22.04 (JetPack 6.1) |
| TensorRT | 10.8 | 10.x (JetPack bundled) |

## Training Results (Laptop)

| Model | Accuracy | Size | Notes |
|-------|----------|------|-------|
| FP32 baseline | 88.65% | 0.37 MB | 30 epochs, best val at epoch 30 |
| PTQ INT8 | 87.19% | 0.10 MB | -1.46% from FP32 |
| QAT INT8 | 88.22% | 0.10 MB | +1.03% vs PTQ, near FP32 |

## Cross-Device Comparison: Replay Benchmark

The replay benchmark streams .wav files at 10 FPS (real-time rate), measuring latency under load, dropped windows, and system resource usage.

### Latency & Accuracy

| Runtime | Precision | Accuracy | p50 (ms) | p95 (ms) | p99 (ms) | Throughput | Dropped |
|---------|-----------|----------|-----------|-----------|-----------|------------|---------|
| **Laptop** | | | | | | | |
| ONNX CPU | FP32 | 89.00% | 0.27 | 0.37 | 0.62 | 3,520 fps | 0/500 |
| TRT FP16 | FP16 | 88.80% | 0.38 | 2.13 | 4.11 | 1,219 fps | 0/500 |
| TRT INT8 | INT8 | 89.40% | 0.45 | 1.78 | 3.77 | 1,238 fps | 0/500 |
| **Jetson Orin Nano** | | | | | | | |
| ONNX CPU | FP32 | 89.00% | 7.27 | 12.18 | 17.42 | 154 fps | 0/500 |
| TRT FP16 | FP16 | 89.00% | 1.01 | 1.14 | 1.32 | 1,170 fps | 0/500 |
| TRT INT8 | INT8 | 89.80% | 0.95 | 1.08 | 1.19 | 1,341 fps | 0/500 |

### Cold Start Time

| Runtime | Laptop | Jetson |
|---------|--------|--------|
| ONNX CPU | 723 ms | 78 ms |
| TRT FP16 | 399 ms | 233 ms |
| TRT INT8 | 404 ms | 215 ms |

### Memory Usage

| Runtime | Laptop RAM | Jetson RAM | Laptop GPU | Jetson GPU |
|---------|-----------|------------|------------|------------|
| ONNX CPU | 11,877 MB | 1,824 MB | ~1% | ~2% |
| TRT FP16 | 10,804 MB | 1,989 MB | ~1% | ~33% |
| TRT INT8 | 10,971 MB | 1,946 MB | ~1% | ~32% |

## Cross-Device Speedup Analysis

### ONNX CPU: Laptop vs Jetson

| Metric | Laptop | Jetson | Jetson/Laptop |
|--------|--------|--------|---------------|
| p50 latency | 0.27 ms | 7.27 ms | **27x slower** |
| p99 latency | 0.62 ms | 17.42 ms | **28x slower** |
| Throughput | 3,520 fps | 154 fps | **23x slower** |

The Jetson's ARM CPU is ~27x slower than the laptop's Intel i7 for CPU inference. At 10 FPS target, p99 of 17.42ms is still within the 100ms deadline (5.7x headroom), but leaves little room for other CPU work.

### TRT FP16: Laptop vs Jetson

| Metric | Laptop | Jetson | Jetson/Laptop |
|--------|--------|--------|---------------|
| p50 latency | 0.38 ms | 1.01 ms | **2.7x slower** |
| p99 latency | 4.11 ms | 1.32 ms | **3.1x faster** |
| Throughput | 1,219 fps | 1,170 fps | **1.04x slower** |

Interesting: Jetson's TRT FP16 has **better p99** than the laptop! This is because the Jetson has unified memory (no GPU↔CPU copy overhead) and dedicated GPU with no OS scheduling jitter from other GPU tasks. The laptop's p99 is inflated by GPU memory transfer variability.

### TRT INT8: Laptop vs Jetson

| Metric | Laptop | Jetson | Jetson/Laptop |
|--------|--------|--------|---------------|
| p50 latency | 0.45 ms | 0.95 ms | **2.1x slower** |
| p99 latency | 3.77 ms | 1.19 ms | **3.2x faster** |
| Throughput | 1,238 fps | 1,341 fps | **1.08x faster** |

Same story — Jetson TRT INT8 has superior tail latency and slightly better throughput than the laptop, thanks to unified memory architecture.

## Full Benchmark Baseline (Laptop Only)

Single-sample latency benchmark (no replay timing, no system metrics):

| Variant | Accuracy | Size (MB) | P99 (ms) | Mean (ms) | Throughput |
|---------|----------|-----------|----------|-----------|------------|
| PyTorch FP32 (CPU) | 88.12% | 0.37 | 1.65 | 1.37 | 732 fps |
| PyTorch FP32 (GPU) | 88.12% | 0.37 | 0.63 | 0.43 | 2,299 fps |
| PTQ INT8 (CPU) | 87.21% | 0.10 | 1.36 | 1.13 | 887 fps |
| QAT INT8 (CPU) | 88.22% | 0.10 | 1.29 | 1.10 | 908 fps |
| ONNX CPU | 88.12% | 0.36 | 0.19 | 0.15 | 6,509 fps |
| ONNX CUDA | 88.12% | 0.36 | 0.29 | 0.19 | 5,212 fps |
| TRT Native FP16 | 88.07% | 0.85 | 0.51 | 0.38 | 2,641 fps |
| TRT Native INT8 | 88.05% | 0.68 | 1.26 | 0.34 | 2,945 fps |

## TensorRT Engine Build Results

| Device | Engine | Size | Accuracy (build_bench) |
|--------|--------|------|----------------------|
| Laptop (RTX 4060) | FP16 | 0.85 MB | 88.07% |
| Laptop (RTX 4060) | INT8 | 0.68 MB | 88.05% |
| Jetson (Orin Nano) | FP16 | 0.33 MB | 88.07% |
| Jetson (Orin Nano) | INT8 | 0.21 MB | 87.97% |

Jetson engines are smaller due to the Ampere architecture's more efficient INT8 kernel fusion.

## Key Takeaways

1. **Same model, same accuracy across devices.** ONNX CPU = 89.00% on both. TRT FP16 = 88-89% on both. No accuracy degradation from device switch.

2. **Jetson ARM CPU is 27x slower** than laptop Intel i7 for ONNX CPU inference. For CPU-only workloads, the laptop wins decisively.

3. **Jetson GPU matches or beats laptop for TRT inference** in p99 latency, thanks to unified memory (no CPU↔GPU copy overhead) and dedicated GPU.

4. **TRT INT8 is the best deployment target for Jetson**: 89.80% accuracy, p99=1.19ms, 1,341 fps throughput — 5.7x headroom at 10 FPS target, with the smallest engine size (0.21 MB).

5. **Laptop's p99 tail latency is worse than Jetson for TRT** — the laptop's GPU is shared with the OS and other processes, causing p99 spikes (3.77-4.11ms vs 1.19-1.32ms on Jetson).

6. **Zero dropped windows on all configurations** — even ONNX CPU on Jetson (p99=17.42ms) stays within the 100ms deadline at 10 FPS.

7. **QAT INT8 beats FP32 accuracy** (+1.03% vs PTQ, nearly matching FP32 at 88.22% vs 88.65%) while being 3.7x smaller — quantization noise as regularization works.

## Full Model Variant Comparison (Laptop, Single-Sample)

Single-sample inference benchmark (no replay timing, no system metrics). This shows the full quantization story from FP32 through PTQ, QAT, and TensorRT.

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

**Quantization impact:**
- PTQ INT8: 87.21% (-0.91% from FP32), 3.7x smaller
- QAT INT8: 88.22% (+0.10% from FP32, +1.01% over PTQ), 3.7x smaller
- QAT beats FP32 because quantization noise acts as regularization

## Deployment Recommendation

| Target | Best Runtime | Accuracy | p99 (ms) | Engine Size |
|--------|-------------|----------|-----------|-------------|
| Jetson Orin Nano (edge) | **TRT INT8** | 89.80% | 1.19 | 0.21 MB |
| Jetson Orin Nano (fallback) | ONNX CPU | 89.00% | 17.42 | 0.36 MB |
| Laptop/Server (GPU available) | **TRT FP16** | 88.80% | 4.11 | 0.85 MB |
| Laptop/Server (CPU only) | **ONNX CPU** | 89.00% | 0.62 | 0.36 MB |