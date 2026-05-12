# QAT & TensorRT Keyword Spotting — Session Update

## Project Overview

A keyword spotting pipeline built on Google Speech Commands v2 (10 classes), optimized through quantization (PTQ, QAT) and TensorRT deployment. The goal: learn QAT and TensorRT concepts through building, with Socratic check-ins before each module.

---

## Completed Modules

### M01 — Data Pipeline & MFCC
- Downloaded Google Speech Commands v2 dataset
- Extracted 40-dimensional MFCC features from 1-second audio clips
- Speaker-aware train/val/test split (no speaker overlap between splits)
- Per-sample normalization (zero mean, unit variance)
- Dataset sizes: train 30,769 / val 3,703 / test 4,074

### M02 — CNN Model & FP32 Baseline
- Built KWSNet: 3-block CNN (Conv+BN+ReLU+MaxPool) → AdaptiveAvgPool → Linear(128, 10)
- 94K parameters total
- Trained FP32 baseline to 87.36% test accuracy
- Added `KWSNetQuantizable` variant with QuantStub/DeQuantStub and `fuse_model()` for Conv+BN+ReLU fusion
- Exported FP32 ONNX model (0.36 MB)

### M03 — Post-Training Quantization (PTQ)
- Static INT8 quantization with calibration (100 batches)
- Conv+BN+ReLU fusion is critical: without it, accuracy drops from 87% to 57%
- PTQ INT8 result: 86.43% accuracy (-0.93% from FP32), model size 0.10 MB (3.7x compression)
- Exported FP32 ONNX (quantized PyTorch ONNX export fails — `quantized::batch_norm2d` unsupported)

### M04-M06 — Quantization-Aware Training (QAT)
- `prepare_qat()` inserts FakeQuantize nodes (learnable scale/zero_point)
- Trained 15 epochs with Adam lr=1e-4, froze BatchNorm at epoch 3
- BN freeze prevents oscillation: fake quant noise changes activation distributions, BN keeps chasing them
- STE (Straight-Through Estimator) allows gradients to flow through rounding: forward uses quantized values, backward treats quantization as identity
- QAT INT8 result: 89.00% accuracy (+1.64% above FP32), model size 0.10 MB
- QAT beats FP32 because quantization noise during training acts as regularizer, pushing weights toward flat minima (bucket centers)

### M07 — TensorRT Engine Build
- TensorRT Python API (not ORT) for engine building — ORT's TensorRT EP couldn't load nvinfer.dll
- Built TRT FP16 engine (0.29 MB) and TRT INT8 engine (0.20 MB) with calibration
- TRT requires optimization profile for dynamic batch shapes (our ONNX has `batch_size` as dynamic axis)
- INT8 calibrator pre-loads batches to GPU memory and returns device pointers (not numpy arrays)
- IHostMemory API changed in TRT 10: use `.nbytes` not `.size()` or `len()`

### M08 — TensorRT Profiling & Fusion Analysis
- Used `EngineInspector` API (TRT 10+) to inspect layer fusion decisions
- FP16 engine: 12 layers (3 fused Conv+ReLU, 2 reformat nodes, standalone Gemm)
- INT8 engine: 10 layers (3 fused Conv+ReLU, 1 fused Gemm+quantize, 1 reformat node)
- INT8 has fewer layers because it keeps INT8 format through the middle of the network, eliminating a reformat node
- INT8 also fuses the final Gemm+quantize into one kernel
- Profiling with 100 warmup iterations: FP16 P99 0.128ms, INT8 P99 0.166ms

### M09 — Benchmark & Report
- Benchmarked 10 variants: PyTorch FP32/PTQ/QAT (CPU/CUDA), ONNX Runtime (CPU/CUDA/TRT FP16/TRT INT8), TRT Native FP16/INT8
- P99 (not mean) is the primary deployment metric: it reveals tail latency that causes dropped frames
- ONNX Runtime CPU (0.22ms P99) beats PyTorch CUDA (0.93ms P99) at batch_size=1 because GPU transfer overhead dominates for small models
- TRT Native FP16 achieves best P99 at 0.128ms (20x better than PyTorch CPU)
- Report saved to `reports/benchmark_report.md`, data to `models/final_results.json`

### Key Files

| File | Purpose |
|------|---------|
| `src/model.py` | KWSNet + KWSNetQuantizable model definitions |
| `src/dataset.py` | MFCC extraction, speaker-aware splits, DataLoader |
| `models/kws_fp32.pth` | FP32 baseline checkpoint (87.36%) |
| `models/kws_ptq_int8.pth` | PTQ INT8 checkpoint (86.43%) |
| `models/kws_qat_best.pth` | Best QAT checkpoint (val 90.44%) |
| `models/kws_qat_int8.pth` | QAT INT8 converted checkpoint (89.00%) |
| `models/kws_fp32.onnx` | FP32 ONNX export (0.36 MB) |
| `models/kws_trt_fp16.engine` | TensorRT FP16 engine (0.29 MB) |
| `models/kws_trt_int8.engine` | TensorRT INT8 engine (0.20 MB) |
| `models/benchmark_results.json` | All 10 variant benchmark results |
| `models/trt_profile_report.json` | TRT profiling data |
| `models/final_results.json` | Combined benchmark + profile data |
| `reports/benchmark_report.md` | Final markdown report |
| `quantize_ptq_baseline.py` | PTQ pipeline script |
| `train_qat.py` | QAT training script |
| `benchmark_baseline.py` | Full benchmark script (PyTorch + ONNX) |
| `build_trt_engines.py` | TRT engine build + benchmark script |
| `profile_trt_engines.py` | TRT profiling + fusion analysis script |
| `generate_report.py` | Report generation script |

---

## Socratic Quiz Results

| Module | Score | Notes |
|--------|-------|-------|
| M01 (Data/MFCC) | 2/3 | Understood spatial relationships and per-sample normalization; missed mel scale reasoning |
| M02 (Model) | 1/3 | Knew MaxPool picks strongest feature; couldn't trace spatial dims or predict noise-class impact |
| M03 (PTQ) | 1/3 | Understood fusion; didn't know why PTQ can fail badly or what bad calibration does |
| M04-M06 (QAT) | 2/3 | Knew fake quant adds noise during training; partial on BN freeze; didn't know where scales go after convert |
| M07-M08 (TRT) | 1/3 | Knew format reformatting; couldn't name TRT optimizations or explain INT8 fewer layers |
| M09 (Benchmark) | 1/2 | Correctly identified GPU variants benefit from batching; didn't know why CPU ONNX beats GPU PyTorch |
| Capstone | 0/1 | Skipped Raspberry Pi deployment question |

**Overall assessment:** Can run the pipeline and interpret results. Gaps in quantization mechanics (calibration, scale storage, BN freeze feedback loop), hardware reasoning (when INT8 wins, GPU transfer overhead), and model architecture math (spatial dimension tracing). Deep-dive explanation provided for all gap areas.

---

### M10 — Real-Time Deployment Pipeline

- Built complete real-time keyword spotting demo with 5 new source modules
- `src/audio_capture.py`: sounddevice callback-based mic capture with thread-safe ring buffer (16000 samples = 1 second)
- `src/realtime_mfcc.py`: MFCC extraction matching training pipeline (n_mfcc=40, center=False, per-sample z-score normalization), produces (1, 1, 40, 98)
- `src/inference_engine.py`: ONNX and TensorRT backends with latency tracking (rolling P99 deque of 1000 samples), 10-iteration warmup
- `src/keyword_detector.py`: softmax + threshold (0.7) + 500ms cooldown to prevent repeated triggers
- `realtime_demo.py`: CLI app with --backend/--threshold/--stride/--cooldown args, stride-based inference loop, stats on exit
- Fixed time-frame bug: `input_length=101` changed to `98` in export_onnx.py, benchmark.py, and m08 module
- ONNX inference pipeline verified: MFCC shape (1,1,40,98), logits shape (1,10), random audio correctly rejected below threshold, cooldown blocks rapid re-triggers
- Ring buffer verified: write/read with zero-padding and wraparound works correctly

### Key Files

| File | Purpose |
|------|---------|
| `src/audio_capture.py` | AudioRingBuffer + AudioCapture (sounddevice callback) |
| `src/realtime_mfcc.py` | RealtimeMFCC extraction matching training params |
| `src/inference_engine.py` | ONNXBackend, TRTBackend, InferenceEngine with P99 tracking |
| `src/keyword_detector.py` | softmax, threshold, cooldown, keyword mapping |
| `realtime_demo.py` | CLI entry point integrating all modules |
| `modules/m10_realtime_demo/` | Socratic module (quiz + run wrapper) |

### Bug Fixes

- `src/export_onnx.py`: Fixed `input_length` default from 101 to 98
- `src/benchmark.py`: Fixed 3 hardcoded `40, 101` shapes to `40, 98`
- `modules/m08_tensorrt_infer/run.py`: Fixed `1, 1, 40, 101` shape to `1, 1, 40, 98`