# Next Immediate Steps — QAT_RT / Jetson Orin Nano Super

## What We Completed

- Full KWS pipeline: FP32 training → PTQ → QAT → ONNX → TRT FP16/INT8
- Cross-device replay benchmark: Laptop (RTX 4060) vs Jetson Orin Nano Super
- Edge comparison report with deployment recommendations
- All 22 planned items from edge-replay-bench spec, plus 7 additional items

## The Gap

The KWS model (94K params) is too small to stress the Jetson Orin Nano's 40 TOPS GPU. Single-sample latency at batch=1 is dominated by overhead, not compute. The GPU only shines with bigger models, batched inference, or concurrent multi-model workloads.

## Immediate Next Steps

### 1. Batch Size Sweep on Jetson

Show GPU scaling — at batch=1 overhead dominates, at batch=8+ compute dominates.

- Modify `inference_engine.py` and `replay_benchmark.py` to support batched inference
- Test batch sizes: 1, 2, 4, 8, 16, 32
- Run on both devices (Laptop RTX 4060 + Jetson Orin Nano)
- Expect: Jetson throughput scales much better with batch size (unified memory, no PCIe bottleneck)

**Files to modify:**
- `src/inference_engine.py` — add batch dimension support to ONNXBackend and TRTBackend
- `src/replay_benchmark.py` — add `batch_size` config option
- `configs/kws_onnx.yaml`, `configs/kws_tensorrt_fp16.yaml`, `configs/kws_tensorrt_int8.yaml` — add `batch_size` field

**Expected result:** At batch=1, Jetson TRT INT8 is ~3x slower than laptop TRT INT8 in mean latency. At batch=8+, Jetson should match or beat laptop on throughput/watt.

### 2. YOLOv8-nano on Jetson Orin Nano

A real vision model (3.2M params) where GPU actually matters.

- Export YOLOv8-nano to ONNX → TRT FP16/INT8
- Benchmark single-stream and multi-stream inference
- Compare against CPU-only inference on Jetson
- Measure power consumption (MAXN vs 15W mode)

**New files:**
- `src/yolo_benchmark.py` or extend `replay_benchmark.py` for vision models
- `configs/yolo_tensorrt_fp16.yaml`, `configs/yolo_tensorrt_int8.yaml`

**Expected result:** TRT FP16 should be 10-50x faster than CPU for YOLO (vs only 14x for KWS). This is where the Orin Nano's GPU justifies itself.

### 3. Concurrent Multi-Model Pipeline

Run KWS + vision simultaneously on the Jetson GPU — the realistic edge workload.

- KWS on mic input (continuous streaming at 10 FPS)
- YOLOv8-nano on camera input (continuous at 15-30 FPS)
- Both sharing the GPU, measuring interference and throughput drop
- Show that the Orin Nano can handle both with <5% throughput loss per model

**New files:**
- `src/concurrent_pipeline.py` — multi-threaded inference with shared GPU
- `configs/concurrent_kws_yolo.yaml`

**Expected result:** Both models run concurrently on the Orin Nano with acceptable throughput. This is the story for "why you need a GPU on the edge."

### 4. Stress Test — High FPS Replay

Show where CPU falls behind and GPU pulls ahead.

- Run `replay_benchmark.py` with `speed: 2.0` (20 FPS), `speed: 5.0` (50 FPS), `speed: 10.0` (100 FPS)
- At 100 FPS, the CPU (ONNX) will drop frames. TRT will not.
- This demonstrates the "dropped windows" feature of FileReplayEngine

**Config changes:**
- `configs/kws_onnx_stress.yaml` — speed: 5.0
- `configs/kws_tensorrt_int8_stress.yaml` — speed: 10.0

**Expected result:** ONNX CPU starts dropping windows around 50-60 FPS on Jetson. TRT INT8 stays at 0 drops past 100 FPS.

### 5. Power & Thermal Measurement

The Orin Nano's perf/watt story is its strongest edge selling point.

- Run benchmarks in MAXN mode (25W) vs 15W mode
- Use `tegrastats` to capture power draw (VDD_IN) during inference
- Calculate FPS/watt for each runtime/mode combination

**New files:**
- `src/power_benchmark.py` — wraps replay_benchmark with power measurement via tegrastats

**Expected result:** TRT INT8 at 15W should deliver ~80% of MAXN throughput at 60% power — the real edge deployment story.

## Priority Order

1. **Batch size sweep** (1-2 hours) — most impactful for the existing KWS model
2. **Stress test** (30 min) — easy config change, dramatic results
3. **YOLOv8-nano** (4-6 hours) — needs new model, TRT export, but biggest impact
4. **Concurrent pipeline** (3-4 hours) — the real edge story, depends on YOLO being set up
5. **Power measurement** (1-2 hours) — easy to add once benchmarks are running

## Jetson Access

See memory file `jetson-orin-access.md` for SSH, environment, and known issues.