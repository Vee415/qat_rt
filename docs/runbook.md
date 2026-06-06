# Runbook — Edge Replay Benchmark

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# For Jetson Orin Nano Super:
pip install jetson-stats   # for jtop GPU metrics
pip install tensorrt pycuda  # for TRT benchmarks (usually pre-installed on Jetson)

# Download dataset (if not already present)
python -c "from src.dataset import download_speech_commands; download_speech_commands('data')"
```

### 2. Health Check

```bash
# Verify ONNX pipeline works
python -m src.health_check --config configs/kws_onnx.yaml

# Verify TRT pipeline (requires GPU + TensorRT)
python -m src.health_check --config configs/kws_tensorrt_fp16.yaml

# Quick check without config
python -m src.health_check --backend onnx --model models/kws_fp32.onnx
```

### 3. Run Benchmarks

```bash
# ONNX Runtime CPU (works on any machine)
python src/replay_benchmark.py --config configs/kws_onnx.yaml

# TensorRT FP16 (requires CUDA + TensorRT)
python src/replay_benchmark.py --config configs/kws_tensorrt_fp16.yaml

# TensorRT INT8 (optional, fastest path)
python src/replay_benchmark.py --config configs/kws_tensorrt_int8.yaml

# Custom output directory
python src/replay_benchmark.py --config configs/kws_onnx.yaml --output-dir benchmarks/run_2024_01_15
```

### 4. View Results

Results are written to `benchmarks/` (or custom output directory):
- `results.csv` — tabular results for all runtimes
- `results.json` — structured JSON with full details
- `summary.md` — markdown summary with plain-English interpretation

### 5. Run Tests

```bash
# Unit tests (no GPU required)
pytest tests/

# Just preprocessing tests
pytest tests/test_preprocessing.py

# Just postprocessing tests
pytest tests/test_postprocessing.py
```

## On Jetson Orin Nano Super

### First-Time Setup

```bash
# 1. Flash Jetson with JetPack 6.x (includes PyTorch, CUDA, TensorRT)
# 2. Install dependencies
pip install -r requirements.txt
pip install jetson-stats

# 3. Rebuild TRT engines ON THE JETSON (engines built on x86 will NOT work)
#    Engines are architecture-specific
python build_trt_engines.py  # builds kws_trt_fp16.engine and kws_trt_int8.engine

# 4. Run health check
python -m src.health_check --config configs/kws_tensorrt_fp16.yaml

# 5. Run benchmark
python src/replay_benchmark.py --config configs/kws_tensorrt_fp16.yaml
```

### Important Notes

- **TRT engines are not portable**: An engine built on your x86 dev machine will NOT run on Jetson. You must rebuild engines on the Jetson itself.
- **Unified memory**: On Jetson, CPU and GPU share 8GB RAM. The `SystemMetricsCollector` uses `jtop` (jetson-stats) to read unified memory stats.
- **Performance**: Jetson Orin Nano Super has a 40 TOPS GPU. The KWS model (94K params) is very lightweight — expect near-zero dropped windows at 10 FPS.
- **Power modes**: Jetson has multiple power modes (15W, 25W, MAXN). For benchmarking, use MAXN mode for maximum performance:
  ```bash
  sudo nvpmodel -m 0   # MAXN mode (maximum performance)
  sudo jetson_clocks    # Maximize clocks
  ```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: tensorrt` | TensorRT must be installed separately. On Jetson: comes with JetPack. On x86: `pip install tensorrt` |
| `ModuleNotFoundError: pycuda` | Install with `pip install pycuda`. On Jetson: `pip install pycuda` |
| `ModuleNotFoundError: jetson_stats` | Only needed on Jetson: `pip install jetson-stats` then `sudo systemctl restart jetson_stats.service` |
| `nvidia-smi` shows no GPU utilization on Jetson | Expected! Use `jtop` instead — nvidia-smi doesn't report utilization on Jetson |
| TRT engine won't load on different GPU | Rebuild the engine on the target device. Engines are GPU-architecture specific. |
| `FileNotFoundError: No .wav files found` | Download dataset first: `python -c "from src.dataset import download_speech_commands; download_speech_commands('data')"` |
| Accuracy is 0% | Check that `data/speech_commands/` has class subdirectories (yes/, no/, etc.) with .wav files |