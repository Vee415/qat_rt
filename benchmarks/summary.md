# Edge Inference Benchmark Summary

**Date:** 2026-06-04 00:47
**Platform:** AMD64 (Intel64 Family 6 Model 186 Stepping 2, GenuineIntel)

## Results

| Runtime | Precision | Accuracy | p50 (ms) | p95 (ms) | p99 (ms) | Throughput | Dropped | RAM (MB) | Cold Start (ms) |
|---------|-----------|----------|----------|----------|----------|-----------|---------|---------|-----------------|
| trt_int8 | int8 | 89.40% | 0.45 | 1.78 | 3.77 | 1238 fps | 0/500 (0.0%) | 10971 | 404 |

## Key Findings

- **Best latency (p99):** trt_int8 (int8) at 3.77ms
- **Best accuracy:** trt_int8 (int8) at 89.40%
- **Best throughput:** trt_int8 (int8) at 1238 fps
- **Smallest model:** trt_int8 (int8) at 0.68 MB

## Dropped Window Analysis

All runtimes kept up with the target FPS — zero dropped windows. This is expected for the KWS model (94K params) which is very lightweight.

## Deployment Recommendations

- **GPU deployment (Jetson/RTX):** trt_int8 — p99 latency 3.77ms, accuracy 89.40%
- **Accuracy-critical:** trt_int8 (int8) — best accuracy at 89.40%
- **Memory-constrained:** trt_int8 (int8) — smallest model at 0.68 MB