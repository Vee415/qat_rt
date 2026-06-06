# Jetson Orin Nano Super Setup Guide

## Hardware

- **Device:** NVIDIA Jetson Orin Nano Super (8GB)
- **GPU:** Ampere architecture, 1024 CUDA cores, 40 TOPS (INT8)
- **CPU:** 6-core Arm Cortex-A78AE
- **Memory:** 8GB unified (CPU + GPU)
- **Storage:** MicroSD card (recommend 64GB+ UHS-I)

## Software Stack

### JetPack 6.x (Recommended)

JetPack 6.x includes:
- Ubuntu 22.04
- CUDA 12.x
- cuDNN 9.x
- TensorRT 10.x
- PyTorch 2.x (pre-installed)
- Python 3.10

### Installation Steps

```bash
# 1. Flash Jetson with JetPack using SDK Manager
#    https://developer.nvidia.com/sdk-manager
#    Select JetPack 6.x for Orin Nano

# 2. Verify CUDA
nvcc --version
nvidia-smi  # Note: GPU utilization won't show on Jetson

# 3. Verify TensorRT
python -c "import tensorrt; print(tensorrt.__version__)"

# 4. Install jetson-stats (for jtop GPU monitoring)
sudo apt update
sudo apt install python3-pip
pip3 install jetson-stats
sudo systemctl restart jetson_stats.service

# 5. Install project dependencies
cd ~/edge-replay-bench  # or wherever you cloned the project
pip3 install -r requirements.txt
pip3 install pycuda  # for TRT backend

# 6. Verify installation
python3 -m src.health_check --config configs/kws_onnx.yaml
```

### Performance Mode

For consistent benchmark results, lock Jetson to maximum performance:

```bash
# Set MAXN power mode (maximum performance, ~15W)
sudo nvpmodel -m 0

# Maximize clocks
sudo jetson_clocks

# Verify
jtop  # Should show MAXN mode, high clock speeds
```

To return to power-saving mode:
```bash
sudo nvpmodel -m 1  # 15W mode (lower clocks)
```

### TRT Engine Build (MUST be done on Jetson)

```bash
# Build FP16 engine
python3 src/export_tensorrt.py --onnx_path models/kws_fp32.onnx --engine_path models/kws_trt_fp16.engine --precision fp16

# Build INT8 engine (requires calibration data)
python3 build_trt_engines.py

# Verify engines
ls -la models/*.engine
```

**Critical:** TRT engines are GPU-architecture specific. An engine built on your x86 dev machine (RTX 4060) will NOT run on Jetson Orin Nano. You must rebuild engines on the Jetson.

### Running the Benchmark

```bash
# ONNX CPU benchmark (works on any device)
python3 src/replay_benchmark.py --config configs/kws_onnx.yaml

# TensorRT FP16 benchmark (GPU accelerated)
python3 src/replay_benchmark.py --config configs/kws_tensorrt_fp16.yaml

# TensorRT INT8 benchmark (fastest path)
python3 src/replay_benchmark.py --config configs/kws_tensorrt_int8.yaml
```

### Monitoring During Benchmark

```bash
# In a separate terminal, monitor GPU/CPU/RAM
jtop

# Or use tegrastats for raw output
tegrastats --interval 1000
```

### Expected Performance on Jetson Orin Nano Super

Based on the model size (94K params) and Jetson's 40 TOPS capability:

| Runtime | Expected p95 Latency | Expected Throughput |
|---------|----------------------|---------------------|
| ONNX CPU | 1-3ms | 300-1000 fps |
| TRT FP16 | 0.1-0.5ms | 2000-10000 fps |
| TRT INT8 | 0.05-0.3ms | 3000-20000 fps |

The KWS model is very lightweight — expect near-zero dropped windows at 10 FPS (100ms deadline).

### Docker on Jetson

If using Docker on Jetson, use the NVIDIA L4T base image:

```bash
# Build with Jetson base image
docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t/pytorch:l4t-r36.4.0 \
  -t edge-replay-bench-jetson .

# Run with GPU access
docker run --runtime=nvidia --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  edge-replay-bench-jetson \
  --config configs/kws_tensorrt_fp16.yaml
```

### Troubleshooting Jetson-Specific Issues

| Issue | Solution |
|-------|----------|
| `nvidia-smi` shows N/A for GPU | Use `jtop` instead — nvidia-smi doesn't work properly on Jetson |
| TRT engine won't load | Rebuild engine on Jetson — engines are architecture-specific |
| Out of memory | Check `jtop` for memory usage; kill unnecessary processes; reduce `max_files` in config |
| Slow inference | Run `sudo jetson_clocks` to maximize clocks; check power mode with `nvpmodel -q` |
| `pip install tensorrt` fails | TensorRT comes pre-installed with JetPack — don't pip install it separately |
| Docker GPU access fails | Use `--runtime=nvidia` flag, not `--gpus all` (Jetson uses L4T runtime) |