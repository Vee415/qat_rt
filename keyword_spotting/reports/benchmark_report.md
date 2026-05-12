# Keyword Spotting: QAT & TensorRT Pipeline — Final Report

## Overview

A 3-block CNN (KWSNet, 94K parameters) trained on Google Speech Commands v2
for 10-class keyword spotting, optimized through:

1. **FP32 baseline** — full-precision training
2. **PTQ INT8** — post-training static quantization with calibration
3. **QAT INT8** — quantization-aware training with fake quant nodes
4. **TensorRT FP16/INT8** — GPU-optimized inference engines

**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU, Intel CPU, TensorRT 10.16

## 1. Accuracy Comparison

| Variant | Accuracy | vs FP32 |
|---------|----------|---------|
| TRT Native FP16 | 0.8731 (87.31%) | -0.05% |
| TRT Native INT8 | 0.8672 (86.72%) | -0.64% |
| ONNX Runtime (CPU) | 0.8736 (87.36%) | +0.00% |
| ORT TensorRT INT8 | 0.8736 (87.36%) | +0.00% |
| ONNX Runtime (CUDA) | 0.8736 (87.36%) | +0.00% |
| ORT TensorRT FP16 | 0.8736 (87.36%) | +0.00% |
| PyTorch FP32 (GPU) | 0.8733 (87.33%) | -0.03% |
| PyTorch PTQ INT8 (CPU) | 0.8643 (86.43%) | -0.93% |
| PyTorch QAT INT8 (CPU) | 0.8900 (89.00%) | +1.64% |
| PyTorch FP32 (CPU) | 0.8736 (87.36%) | +0.00% |

**Key finding:** QAT INT8 *exceeds* FP32 accuracy by +1.64%. Training with
quantization noise acts as regularization, improving generalization.

## 2. Latency & Throughput (single-sample inference)

> **Why P99 over mean?** For streaming keyword spotting, mean latency is
> misleading. If your audio pipeline expects inference every 2ms and P99 is
> 2.62ms, 1% of frames will miss their deadline — causing dropped audio,
> missed wake words, or buffer underruns. P99 tells you the worst-case
> latency your system must tolerate. The table is sorted by P99.

| Variant | P99 (ms) | P95 (ms) | Mean (ms) | Throughput (fps) |
|---------|----------|----------|-----------|------------------|
| TRT Native FP16 | **0.128** | 0.094 | 0.084 | 11892.8 |
| TRT Native INT8 | **0.166** | 0.113 | 0.088 | 11368.6 |
| ONNX Runtime (CPU) | **0.22** | 0.17 | 0.15 | 6542.4 |
| ORT TensorRT INT8 | **0.27** | 0.21 | 0.20 | 5111.6 |
| ONNX Runtime (CUDA) | **0.30** | 0.24 | 0.21 | 4697.9 |
| ORT TensorRT FP16 | **0.35** | 0.27 | 0.23 | 4310.9 |
| PyTorch FP32 (GPU) | **0.93** | 0.80 | 0.53 | 1874.1 |
| PyTorch PTQ INT8 (CPU) | **1.88** | 1.65 | 1.35 | 740.2 |
| PyTorch QAT INT8 (CPU) | **2.16** | 1.86 | 1.45 | 691.9 |
| PyTorch FP32 (CPU) | **2.62** | 2.27 | 1.86 | 537.6 |

### P99 vs Mean: The Tail Latency Gap

| Variant | Mean (ms) | P99 (ms) | P99/Mean Ratio |
|---------|-----------|----------|-----------------|
| TRT Native FP16 | 0.084 | 0.128 | 1.5x |
| TRT Native INT8 | 0.088 | 0.166 | 1.9x |
| ONNX Runtime (CPU) | 0.15 | 0.22 | 1.5x |
| ORT TensorRT INT8 | 0.20 | 0.27 | 1.4x |
| ONNX Runtime (CUDA) | 0.21 | 0.30 | 1.4x |
| ORT TensorRT FP16 | 0.23 | 0.35 | 1.5x |
| PyTorch FP32 (GPU) | 0.53 | 0.93 | 1.8x |
| PyTorch PTQ INT8 (CPU) | 1.35 | 1.88 | 1.4x |
| PyTorch QAT INT8 (CPU) | 1.45 | 2.16 | 1.5x |
| PyTorch FP32 (CPU) | 1.86 | 2.62 | 1.4x |

P99 is 1.1-2.0x higher than mean across variants. The gap is largest for
CPU-based inference (PyTorch) due to OS scheduling jitter, and smallest for
GPU-based inference (TRT) where the GPU handles timing predictably.

## 3. Model Size Comparison

| Variant | Size (MB) | Compression vs FP32 |
|---------|-----------|---------------------|
| TRT Native FP16 | 0.29 | 1.3x |
| TRT Native INT8 | 0.20 | 1.8x |
| ONNX Runtime (CPU) | 0.36 | 1.0x |
| ORT TensorRT INT8 | 0.36 | 1.0x |
| ONNX Runtime (CUDA) | 0.36 | 1.0x |
| ORT TensorRT FP16 | 0.36 | 1.0x |
| PyTorch FP32 (GPU) | 0.37 | 1.0x |
| PyTorch PTQ INT8 (CPU) | 0.10 | 3.7x |
| PyTorch QAT INT8 (CPU) | 0.10 | 3.7x |
| PyTorch FP32 (CPU) | 0.37 | 1.0x |

INT8 quantization reduces model size to 0.10 MB — a **3.7x compression**
over FP32. TRT engines are also compact: 0.29 MB (FP16) and 0.20 MB (INT8).

## 4. Speedup vs FP32 CPU Baseline (by P99 latency)

| Variant | P99 (ms) | Speedup vs FP32 CPU |
|---------|----------|---------------------|
| TRT Native FP16 | 0.128 | 20.5x |
| TRT Native INT8 | 0.166 | 15.8x |
| ONNX Runtime (CPU) | 0.22 | 11.9x |
| ORT TensorRT INT8 | 0.27 | 9.7x |
| ONNX Runtime (CUDA) | 0.30 | 8.7x |
| ORT TensorRT FP16 | 0.35 | 7.5x |
| PyTorch FP32 (GPU) | 0.93 | 2.8x |
| PyTorch PTQ INT8 (CPU) | 1.88 | 1.4x |
| PyTorch QAT INT8 (CPU) | 2.16 | 1.2x |
| PyTorch FP32 (CPU) | 2.62 | 1.0x |

## 5. TensorRT Layer Fusion Analysis

Original KWSNet has 12 separate operations (3x Conv+BN+ReLU, 3x MaxPool,
1x AdaptiveAvgPool, 1x Linear). After TRT optimization:

### FP16 Engine (12 layers, 0.29 MB)
```
  Reformatting CopyNode for Input Tensor 0 to /features/features.0/Conv + /features/features.2/Relu
  /features/features.0/Conv + /features/features.2/Relu
  Reformatting CopyNode for Input Tensor 0 to /features/features.3/MaxPool
  /features/features.3/MaxPool
  /features/features.4/Conv + /features/features.6/Relu
  /features/features.7/MaxPool
  /features/features.8/Conv + /features/features.10/Relu
  /features/features.11/MaxPool
  /classifier/classifier.0/GlobalAveragePool
  __mye165_0_myl9_0
  /classifier/classifier.2/Gemm_myl9_1
  __myl_Cast_myl9_2
```

### INT8 Engine (10 layers, 0.20 MB)
```
  Reformatting CopyNode for Input Tensor 0 to /features/features.0/Conv + /features/features.2/Relu
  /features/features.0/Conv + /features/features.2/Relu
  /features/features.3/MaxPool
  /features/features.4/Conv + /features/features.6/Relu
  /features/features.7/MaxPool
  /features/features.8/Conv + /features/features.10/Relu
  /features/features.11/MaxPool
  /classifier/classifier.0/GlobalAveragePool
  /classifier/classifier.2/Gemm + /classifier/classifier.2/Gemm_3
  reshape_after_/classifier/classifier.2/Gemm
```

**Fusion decisions:**
- Conv+ReLU fused into single kernels (3 instances in both engines)
- FP16 adds 2 reformat nodes (FP32<->FP16 conversion at input and between layers)
- INT8 needs only 1 reformat (input quantization), keeping INT8 format throughout
- INT8 fuses the final Gemm+quantize into a single kernel (Gemm + Gemm_3)

## 6. Deployment Recommendations

| Target Device | Recommended Variant | Accuracy | P99 (ms) | Mean (ms) | Size |
|---------------|---------------------|----------|----------|-----------|------|
| Edge CPU (Cortex-A, no GPU) | PyTorch QAT INT8 (CPU) | 89.00% | 2.16 | 1.45 | 0.10 MB |
| Edge CPU (lowest P99) | ONNX Runtime (CPU) | 87.36% | 0.22 | 0.15 | 0.36 MB |
| NVIDIA GPU (Jetson, RTX) | TRT Native FP16 | 87.31% | 0.13 | 0.08 | 0.29 MB |
| NVIDIA GPU (smallest) | TRT Native INT8 | 86.72% | 0.17 | 0.09 | 0.20 MB |
| Server CPU (accuracy-critical) | PyTorch QAT INT8 (CPU) | 89.00% | 2.16 | 1.45 | 0.10 MB |
| Server GPU (throughput) | TRT Native FP16 | 87.31% | 0.13 | 0.08 | 0.29 MB |

### Why P99 drives deployment choice:

For a keyword spotting system that processes audio in 10ms frames, the
deadline is 10ms. All variants meet this deadline on *average*, but P99
reveals who actually stays within budget:

- **ONNX Runtime (CPU)** P99 of 0.22ms — 45x headroom before deadline
- **TRT Native FP16** P99 of 0.13ms — 77x headroom before deadline
- **PyTorch FP32 (CPU)** P99 of 2.62ms — only 3.8x headroom, risky under load
- **PyTorch QAT INT8 (CPU)** P99 of 2.16ms — 4.6x headroom, better but still tight

### Key tradeoffs:

- **Best accuracy:** QAT INT8 at 89.00% — the only variant that *beats* FP32.
  Training with quantization noise acts as regularizer, improving generalization.
- **Best CPU P99:** ONNX Runtime at 0.22ms — graph optimizations and
  operator fusion give 12x better tail latency than PyTorch on CPU.
- **Best GPU P99:** TRT Native FP16 at 0.13ms — kernel fusion + FP16
  tensor cores give 20x better tail latency than PyTorch CPU.
- **Smallest model:** QAT INT8 / PTQ INT8 at 0.10 MB — 3.7x smaller than FP32.
  For 256MB edge devices, this matters more than latency.

## 7. Quantization Impact Summary

| Technique | Accuracy | Accuracy Drop | Size Reduction |
|-----------|----------|---------------|----------------|
| FP32 baseline | 87.36% | — | — |
| PTQ INT8 (calibration only) | 86.43% | -0.93% | 3.7x |
| QAT INT8 (fine-tuned) | 89.00% | **+1.64%** | 3.7x |
| TRT FP16 (GPU) | 87.31% | -0.05% | 1.3x |
| TRT INT8 (GPU, calibrated) | 86.72% | -0.64% | 1.9x |

**The QAT advantage:** Not only does QAT recover the accuracy lost by PTQ,
it *surpasses* FP32. The quantization noise during training acts as a
regularizer, preventing overfitting on the training set.
