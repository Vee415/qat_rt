# QAT & TensorRT Keyword Spotting

A keyword spotting pipeline built on Google Speech Commands v2 (10 classes), optimized through quantization (PTQ, QAT) and TensorRT deployment.

## Results

### Accuracy

| Variant | Accuracy | vs FP32 |
|---------|----------|---------|
| PyTorch QAT INT8 (CPU) | **89.00%** | **+1.64%** |
| PyTorch FP32 (CPU) | 87.36% | baseline |
| ONNX Runtime (CPU) | 87.36% | +0.00% |
| TRT Native FP16 | 87.31% | -0.05% |
| TRT Native INT8 | 86.72% | -0.64% |
| PyTorch PTQ INT8 (CPU) | 86.43% | -0.93% |

### Latency (P99, single-sample inference)

| Variant | P99 (ms) | Size (MB) |
|---------|----------|-----------|
| TRT Native FP16 | **0.128** | 0.29 |
| TRT Native INT8 | 0.166 | 0.20 |
| ONNX Runtime (CPU) | 0.22 | 0.36 |
| PyTorch FP32 (CPU) | 2.62 | 0.37 |

QAT INT8 achieves the best accuracy (89.00%) while being 3.7x smaller than FP32. TRT FP16 achieves the best latency at 0.128ms P99 — a **20x speedup** over PyTorch CPU.

### Quantization Impact

| Technique | Accuracy | Size Reduction |
|-----------|----------|----------------|
| FP32 baseline | 87.36% | 1.0x |
| PTQ INT8 | 86.43% (-0.93%) | 3.7x |
| QAT INT8 | 89.00% (+1.64%) | 3.7x |

QAT beats FP32 because quantization noise during training acts as regularization.

## Project Structure

```
keyword_spotting/
├── src/
│   ├── dataset.py            # MFCC extraction, speaker-aware splits
│   ├── model.py              # KWSNet + KWSNetQuantizable (94K params)
│   ├── train.py              # FP32 training loop
│   ├── evaluate.py           # Accuracy, per-class, confusion matrix
│   ├── quantize_ptq.py       # Post-training static INT8 quantization
│   ├── quantize_qat.py       # Quantization-aware training pipeline
│   ├── export_onnx.py        # ONNX export utility
│   ├── export_tensorrt.py    # TRT engine build + inference wrapper
│   ├── benchmark.py           # Multi-backend latency/accuracy benchmark
│   ├── generate_report.py    # Markdown report generation
│   ├── audio_capture.py      # Real-time mic capture with ring buffer
│   ├── realtime_mfcc.py      # MFCC extraction matching training pipeline
│   ├── inference_engine.py   # ONNX + TRT inference backends + cold-start timing
│   ├── keyword_detector.py   # Softmax, threshold, cooldown post-processing
│   ├── replay_benchmark.py  # Config-driven replay benchmark runner
│   ├── health_check.py       # Pipeline health check CLI
│   ├── logging_config.py    # Structured logging setup
│   ├── replay/
│   │   └── file_replay_engine.py  # File-based replay at configurable FPS
│   └── metrics/
│       ├── system_metrics.py # RAM/CPU/GPU collection (psutil + jtop)
│       └── metrics_logger.py # BenchmarkResult dataclass + CSV/JSON/MD output
├── configs/
│   ├── kws_onnx.yaml          # ONNX CPU benchmark config
│   ├── kws_tensorrt_fp16.yaml # TRT FP16 benchmark config
│   └── kws_tensorrt_int8.yaml # TRT INT8 benchmark config
├── tests/
│   ├── test_preprocessing.py  # MFCC extraction unit tests
│   └── test_postprocessing.py # Softmax/threshold/cooldown unit tests
├── docs/
│   ├── runbook.md           # Setup, run, debug, known issues
│   ├── failure_cases.md     # Common failures and diagnosis
│   └── jetson_setup.md      # Jetson Orin Nano Super setup guide
├── benchmarks/              # Output directory for benchmark results
├── realtime_demo.py          # Real-time keyword spotting CLI
├── train_baseline.py         # FP32 training script
├── train_qat.py              # QAT 3-stage training script
├── quantize_ptq_baseline.py  # PTQ pipeline script
├── benchmark_baseline.py     # Full benchmark across backends
├── build_trt_engines.py      # TRT engine build + benchmark
├── profile_trt_engines.py    # TRT layer fusion analysis
├── generate_report.py        # Report generation script
├── params.yaml               # Central hyperparameter config
├── requirements.txt
├── Dockerfile                # Multi-platform (x86 + Jetson)
└── reports/
    └── benchmark_report.md   # Full benchmark report
```

## Quick Start

### Setup

```bash
pip install -r requirements.txt
# TensorRT requires separate install: pip install tensorrt pycuda
```

### Real-Time Demo

```bash
# ONNX Runtime (CPU) — works on any machine
python realtime_demo.py --backend onnx --threshold 0.7

# TensorRT (GPU) — requires CUDA + TensorRT
python realtime_demo.py --backend trt --threshold 0.7

# List available microphones
python -c "import sounddevice as sd; print(sd.query_devices())"

# Use a specific mic device
python realtime_demo.py --backend onnx --device 2

# Adjust confidence threshold and inference stride
python realtime_demo.py --backend onnx --threshold 0.5 --stride 100 --cooldown 500
```

Keywords: **yes, no, up, down, left, right, on, off, stop, go**

### Training & Quantization

```bash
# FP32 baseline
python train_baseline.py

# PTQ INT8
python quantize_ptq_baseline.py

# QAT INT8
python train_qat.py

# Export to ONNX
python -c "from src.export_onnx import export_to_onnx; export_to_onnx('models/kws_fp32.pth', 'models/kws_fp32.onnx')"
```

### Benchmarking

```bash
# All backends (PyTorch, ONNX Runtime, TRT)
python benchmark_baseline.py

# TRT engine build + benchmark
python build_trt_engines.py

# TRT profiling + fusion analysis
python profile_trt_engines.py
```

## Architecture

```
Audio (16kHz)
     │
     ▼
MFCC Extraction (40-dim, center=False → 98 frames)
     │
     ▼
KWSNet: 3x Conv+BN+ReLU+MaxPool → AdaptiveAvgPool → Linear(128, 10)
     │
     ▼
Softmax → Keyword Detection (threshold + cooldown)
```

### Replay Benchmark Pipeline

```
.wav files (Speech Commands test set)
     │
     ▼
FileReplayEngine (streams at configurable FPS, tracks dropped windows)
     │
     ▼
MFCC Extraction (RealtimeMFCC, per-sample normalization)
     │
     ▼
Inference Engine (ONNX Runtime or TensorRT, p50/p95/p99 latency)
     │
     ▼
Keyword Detection (softmax + threshold + cooldown)
     │
     ▼
MetricsLogger (CSV + JSON + markdown summary)
     +
SystemMetricsCollector (RAM, CPU, GPU in background thread)
```

**KWSNet**: 94K parameters, 3-block CNN. Input: (1, 1, 40, 98), Output: (1, 10).

Quantizable variant (`KWSNetQuantizable`) adds `QuantStub`/`DeQuantStub` and `fuse_model()` for Conv+BN+ReLU fusion.

## Key Learnings

- **Conv+BN+ReLU fusion is critical** for PTQ — without it, accuracy drops from 87% to 57%
- **QAT beats FP32** (+1.64%) because quantization noise acts as regularization
- **P99 > mean** for real-time systems — mean hides tail latency from OS scheduling and cache misses
- **ONNX Runtime CPU beats PyTorch CUDA** at batch_size=1 because GPU transfer overhead dominates for small models
- **TRT INT8 has fewer layers** than FP16 because it keeps INT8 format through the network, eliminating reformat nodes and fusing Gemm+quantize
- **center=False** in MFCC is essential — it produces 98 time frames matching the ONNX model, not 101

## Hardware

Benchmarked on NVIDIA GeForce RTX 4060 Laptop GPU with TensorRT 10.16.

## Edge Replay Benchmark

Replay-based inference benchmarking for edge deployment. Reads .wav files at real-time rates and measures latency, throughput, accuracy, RAM/CPU/GPU utilization, cold-start time, and dropped windows.

### Quick Start

```bash
# Health check
python -m src.health_check --config configs/kws_onnx.yaml

# Run ONNX CPU benchmark
python src/replay_benchmark.py --config configs/kws_onnx.yaml

# Run TensorRT FP16 benchmark (requires CUDA + TensorRT)
python src/replay_benchmark.py --config configs/kws_tensorrt_fp16.yaml

# Run TensorRT INT8 benchmark (optional)
python src/replay_benchmark.py --config configs/kws_tensorrt_int8.yaml
```

### Benchmark Metrics

| Metric | Why It Matters |
|--------|---------------|
| p50/p95/p99 latency | Real-time deployment discipline — p99 reveals tail latency |
| Throughput (fps) | Whether the system can keep up with input rate |
| Dropped windows | Overload behavior — how many frames missed their deadline |
| RAM usage | Edge-device awareness — Jetson has 8GB unified memory |
| CPU/GPU utilization | Hardware awareness — is the device being used efficiently? |
| Cold-start time | Production startup cost — time from load to first inference |
| Model size | Deployment footprint — matters for memory-constrained devices |
| Accuracy before/after FP16/INT8 | Optimization tradeoff awareness |

### Jetson Orin Nano Super

TRT engines must be rebuilt on the Jetson (architecture-specific). See [docs/jetson_setup.md](docs/jetson_setup.md) for full setup instructions.

```bash
# On Jetson: rebuild engines first
python build_trt_engines.py

# Then run benchmark
python src/replay_benchmark.py --config configs/kws_tensorrt_fp16.yaml
```

### Configuration

Benchmark configs are in `configs/`. Each YAML specifies:
- `runtime`: backend, model path, precision, providers
- `replay`: data dir, target FPS, max files, shuffle, speed
- `benchmark`: warmup, max windows, metrics interval, output dir

See [docs/runbook.md](docs/runbook.md) for full documentation.