# Failure Cases — Edge Replay Benchmark

## Common Failures and Diagnosis

### 1. TRT Engine Won't Load on Different GPU

**Symptom:** `RuntimeError` or segfault when loading a `.engine` file built on a different machine.

**Root cause:** TensorRT engines are serialized for a specific GPU architecture. An engine built on RTX 4060 (Ada architecture) will not run on Jetson Orin Nano (Ampere architecture).

**Fix:** Rebuild the engine on the target device:
```bash
python build_trt_engines.py
```

**Prevention:** Always rebuild engines on the deployment device. Never copy `.engine` files between machines.

---

### 2. nvidia-smi Shows No GPU Utilization on Jetson

**Symptom:** `nvidia-smi` shows `N/A` for GPU utilization or 0% even during active inference.

**Root cause:** Jetson uses unified memory architecture. `nvidia-smi` is designed for discrete GPUs and does not report utilization correctly on Tegra SoCs.

**Fix:** Use `jtop` (from `jetson-stats` package) instead:
```bash
sudo jetson_stats  # or: jtop
```

**In code:** `SystemMetricsCollector` automatically detects Jetson and uses `jtop`/`tegrastats` instead of `nvidia-smi`.

---

### 3. Cold Start Takes >5 Seconds on TRT

**Symptom:** First inference with TensorRT takes several seconds.

**Root cause:** TRT engines require CUDA context initialization and memory allocation on first use. The 10-iteration warmup in `TRTBackend.__init__` covers most of this, but the very first CUDA call can be slow.

**Diagnosis:** Check `cold_start_ms` in the benchmark output. If it's >5000ms, something is wrong.

**Expected values:**
- ONNX CPU: 50-200ms
- TRT FP16 on RTX 4060: 200-500ms
- TRT FP16 on Jetson Orin Nano: 300-800ms
- TRT INT8: similar to FP16

**Fix:** If cold start is unexpectedly high, check that GPU memory isn't occupied by other processes.

---

### 4. Dropped Windows Despite Low Latency

**Symptom:** The benchmark reports dropped windows even though p99 latency is well below the deadline.

**Root cause:** The `FileReplayEngine` measures wall-clock time including audio loading and MFCC preprocessing. If `torchaudio.load()` or MFCC extraction is slow on the first few files (due to lazy initialization), windows may appear dropped.

**Diagnosis:** Check if drops are concentrated at the start. The first 5-10 files may be slower due to:
- `torchaudio` lazy initialization on first `load()`
- GPU context setup on first inference
- Python import overhead

**Fix:** The warmup phase (50 iterations with random data) handles GPU warmup. For audio loading warmup, add a few dummy loads before the benchmark:
```python
# Pre-warm torchaudio
import torchaudio
torchaudio.load("data/speech_commands/yes/0a7c2a8d_nohash_0.wav")
```

---

### 5. Accuracy Drops Significantly from Training Results

**Symptom:** Benchmark accuracy is much lower than the ~87-89% expected from training.

**Possible causes:**

| Cause | Diagnosis | Fix |
|-------|-----------|-----|
| MFCC parameters mismatch | Check `n_mfcc=40`, `center=False` in `realtime_mfcc.py` matches `dataset.py` | Ensure both use `center=False` |
| Wrong normalization | RealtimeMFCC uses per-sample normalization `(x - mean) / (std + 1e-8)` | Must match `dataset.py` |
| Wrong model file | Loading FP32 model for ONNX benchmark but evaluating with different preprocessing | Check config `model_path` matches the intended variant |
| Too few test files | `max_files: 50` gives unstable accuracy estimates | Use `max_files: null` or ≥500 files |

---

### 6. ONNX Runtime Uses Wrong Provider

**Symptom:** ONNX benchmark runs on CPU even when GPU is available.

**Root cause:** Config `providers: ["CPUExecutionProvider"]` explicitly selects CPU.

**Fix:** To use GPU with ONNX Runtime:
```yaml
providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
```
ONNX Runtime will try providers in order and fall back if CUDA is unavailable.

---

### 7. Out of Memory on Jetson

**Symptom:** `RuntimeError: CUDA out of memory` or `MemoryError` on Jetson.

**Root cause:** Jetson Orin Nano has only 8GB unified memory shared between CPU and GPU. If other processes are running, available memory may be insufficient.

**Fix:**
```bash
# Check memory usage
jtop  # or: free -h

# Kill unnecessary processes
sudo systemctl stop gnome-shell  # if running desktop environment

# Reduce max_windows in config to use less memory
max_files: 200
max_windows: 2000
```

---

### 8. Model Shape Mismatch

**Symptom:** `RuntimeError: Expected input shape (1,1,40,98) but got (1,1,40,101)` or similar.

**Root cause:** MFCC extraction produces wrong number of time frames. This happens when `center=True` is used instead of `center=False`.

**Key detail:** With `center=False`, `time_frames = (clip_length - n_fft) // hop_length + 1 = (16000 - 400) // 160 + 1 = 98`. With `center=True`, you get 101 frames, which doesn't match the ONNX model input.

**Fix:** Ensure `realtime_mfcc.py` uses `center=False` in the melkwargs:
```python
melkwargs={"center": False, ...}
```

---

### 9. Permission Denied on Jetson for GPU Stats

**Symptom:** `jtop` fails with permission error when run as non-root user.

**Fix:**
```bash
# Add user to jetson_stats group
sudo usermod -aG jetson_stats $USER

# Restart the service
sudo systemctl restart jetson_stats.service

# Log out and back in for group changes to take effect
```

---

### 10. Benchmark Results Differ Between Runs

**Symptom:** p99 latency varies by 20-50% between benchmark runs.

**Root cause:** OS scheduling jitter, thermal throttling, or background processes.

**Mitigation:**
- Close other applications before benchmarking
- On Jetson: run `sudo jetson_clocks` to lock clocks at maximum
- Run multiple times and report median results
- Use p95 instead of p99 if variance is too high