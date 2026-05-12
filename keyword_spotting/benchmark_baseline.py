"""Standalone benchmark: accuracy, size, and latency across all model variants.

Tests all available backends:
- PyTorch FP32 (CPU and CUDA)
- PyTorch PTQ INT8 (CPU)
- PyTorch QAT INT8 (CPU)
- ONNX Runtime FP32 (CPU, CUDA, TensorRT)
- ONNX Runtime INT8 quantized via ORT (CPU, CUDA, TensorRT)
"""
import sys
sys.path.insert(0, 'src')

import os
import time
import json
from collections import OrderedDict

import numpy as np
import torch
import torch.ao.quantization as quant
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model


def evaluate_model(model, loader, device, desc="Evaluating"):
    """Evaluate model accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total


def benchmark_latency(model, device, input_shape=(1, 1, 40, 98),
                       warmup=50, iterations=500):
    """Benchmark inference latency for a PyTorch model."""
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    return np.array(latencies)


def benchmark_onnx(onnx_path, loader, providers, input_name,
                    warmup=50, iterations=500, batch_size=1):
    """Benchmark ONNX Runtime inference with given providers."""
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Filter to only available providers
    available = ort.get_available_providers()
    active_providers = [p for p in providers if p in available]
    if not active_providers:
        print(f"    No requested providers available (wanted {providers}, have {available})")
        return None

    try:
        sess = ort.InferenceSession(onnx_path, sess_options=session_options,
                                     providers=active_providers)
    except Exception as e:
        print(f"    Failed to create session with {active_providers}: {e}")
        return None

    # Accuracy
    correct, total = 0, 0
    for inputs, labels in loader:
        outputs = sess.run(None, {input_name: inputs.numpy()})[0]
        predicted = outputs.argmax(axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
    accuracy = correct / total

    # Latency
    dummy = np.random.randn(batch_size, 1, 40, 98).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {input_name: dummy})

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    model_size = os.path.getsize(onnx_path) / (1024 * 1024)
    provider_str = "+".join(active_providers)

    return OrderedDict([
        ("variant", f"ONNX {provider_str}"),
        ("accuracy", f"{accuracy:.4f}"),
        ("size_mb", f"{model_size:.2f}"),
        ("latency_mean_ms", f"{latencies.mean():.2f}"),
        ("latency_std_ms", f"{latencies.std():.2f}"),
        ("latency_p95_ms", f"{np.percentile(latencies, 95):.2f}"),
        ("latency_p99_ms", f"{np.percentile(latencies, 99):.2f}"),
        ("throughput_fps", f"{1000/latencies.mean():.1f}"),
    ])


def make_result_row(variant, accuracy, size_bytes, latencies):
    """Create a result row from latency array."""
    lat = np.array(latencies)
    size_mb = size_bytes / (1024 * 1024)
    return OrderedDict([
        ("variant", variant),
        ("accuracy", f"{accuracy:.4f}"),
        ("size_mb", f"{size_mb:.2f}"),
        ("latency_mean_ms", f"{lat.mean():.2f}"),
        ("latency_std_ms", f"{lat.std():.2f}"),
        ("latency_p95_ms", f"{np.percentile(lat, 95):.2f}"),
        ("latency_p99_ms", f"{np.percentile(lat, 99):.2f}"),
        ("throughput_fps", f"{1000/lat.mean():.1f}"),
    ])


def main():
    print("=" * 70)
    print("Full Benchmark: Accuracy, Size, and Latency")
    print("=" * 70)

    import onnxruntime as ort
    print(f"\nONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")

    has_cuda = torch.cuda.is_available()
    device_gpu = torch.device("cuda") if has_cuda else None
    device_cpu = torch.device("cpu")

    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading dataset...")
    loaders = get_dataloaders('data/speech_commands', batch_size=256, num_workers=0)
    results = []

    # ================================================================
    # 1. PyTorch FP32 (CPU)
    # ================================================================
    print("\n--- PyTorch FP32 (CPU) ---")
    fp32_model = get_model(num_classes=10, quantizable=True)
    fp32_model.load_state_dict(
        torch.load("models/kws_fp32.pth", map_location="cpu", weights_only=True)
    )
    fp32_model.eval()

    fp32_cpu_acc = evaluate_model(fp32_model, loaders["test"], device_cpu,
                                   desc="FP32 CPU Acc")
    fp32_cpu_lat = benchmark_latency(fp32_model, device_cpu, iterations=500)
    print(f"  Accuracy: {fp32_cpu_acc:.4f} | Latency: {fp32_cpu_lat.mean():.2f} ms")
    results.append(make_result_row("PyTorch FP32 (CPU)", fp32_cpu_acc,
                                    os.path.getsize("models/kws_fp32.pth"),
                                    fp32_cpu_lat))

    # ================================================================
    # 2. PyTorch FP32 (GPU) — use base KWSNet (no QuantStub) for GPU
    # ================================================================
    if has_cuda:
        print("\n--- PyTorch FP32 (CUDA) ---")
        from model import KWSNet
        fp32_gpu = KWSNet(num_classes=10)
        # Load weights from quantizable model (same architecture minus QuantStub)
        quantizable_state = torch.load("models/kws_fp32.pth", map_location="cuda",
                                        weights_only=True)
        # Remove quant/dequant keys
        clean_state = {k: v for k, v in quantizable_state.items()
                       if not k.startswith(('quant.', 'dequant.'))}
        fp32_gpu.load_state_dict(clean_state, strict=False)
        fp32_gpu.to(device_gpu)
        fp32_gpu.eval()

        fp32_gpu_acc = evaluate_model(fp32_gpu, loaders["test"], device_gpu,
                                       desc="FP32 GPU Acc")
        fp32_gpu_lat = benchmark_latency(fp32_gpu, device_gpu, iterations=500)
        print(f"  Accuracy: {fp32_gpu_acc:.4f} | Latency: {fp32_gpu_lat.mean():.2f} ms")
        results.append(make_result_row("PyTorch FP32 (CUDA)", fp32_gpu_acc,
                                        os.path.getsize("models/kws_fp32.pth"),
                                        fp32_gpu_lat))
        del fp32_gpu
        torch.cuda.empty_cache()

    # ================================================================
    # 3. PyTorch PTQ INT8 (CPU)
    # ================================================================
    print("\n--- PyTorch PTQ INT8 (CPU) ---")
    ptq_model = get_model(num_classes=10, quantizable=True)
    ptq_model.load_state_dict(
        torch.load("models/kws_fp32.pth", map_location="cpu", weights_only=True)
    )
    ptq_model.eval()
    ptq_model.fuse_model()
    ptq_model.qconfig = quant.get_default_qconfig("x86")
    ptq_prepared = quant.prepare(ptq_model)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loaders["train"]):
            if i >= 100:
                break
            ptq_prepared(inputs)
    ptq_int8 = quant.convert(ptq_prepared)

    ptq_acc = evaluate_model(ptq_int8, loaders["test"], device_cpu,
                              desc="PTQ INT8 Acc")
    ptq_lat = benchmark_latency(ptq_int8, device_cpu, iterations=500)
    print(f"  Accuracy: {ptq_acc:.4f} | Latency: {ptq_lat.mean():.2f} ms")
    results.append(make_result_row("PyTorch PTQ INT8 (CPU)", ptq_acc,
                                    os.path.getsize("models/kws_ptq_int8.pth"),
                                    ptq_lat))

    # ================================================================
    # 4. PyTorch QAT INT8 (CPU)
    # ================================================================
    if os.path.exists("models/kws_qat_int8.pth"):
        print("\n--- PyTorch QAT INT8 (CPU) ---")
        # Rebuild QAT INT8 from best checkpoint
        qat_model = get_model(num_classes=10, quantizable=True)
        qat_model.load_state_dict(
            torch.load("models/kws_fp32.pth", map_location="cpu", weights_only=True)
        )
        qat_model.eval()
        qat_model.fuse_model()
        qat_model.qconfig = quant.get_default_qat_qconfig("x86")
        qat_prepared = quant.prepare_qat(qat_model.train())
        qat_prepared.load_state_dict(
            torch.load("models/kws_qat_best.pth", map_location="cpu", weights_only=True)
        )
        qat_prepared.eval()
        qat_int8 = quant.convert(qat_prepared)

        qat_acc = evaluate_model(qat_int8, loaders["test"], device_cpu,
                                   desc="QAT INT8 Acc")
        qat_lat = benchmark_latency(qat_int8, device_cpu, iterations=500)
        print(f"  Accuracy: {qat_acc:.4f} | Latency: {qat_lat.mean():.2f} ms")
        results.append(make_result_row("PyTorch QAT INT8 (CPU)", qat_acc,
                                        os.path.getsize("models/kws_qat_int8.pth"),
                                        qat_lat))

    # ================================================================
    # 5. ONNX Runtime (CPU)
    # ================================================================
    fp32_onnx = "models/kws_fp32.onnx"
    if os.path.exists(fp32_onnx):
        print("\n--- ONNX Runtime FP32 (CPU) ---")
        result = benchmark_onnx(fp32_onnx, loaders["test"],
                                  ["CPUExecutionProvider"], "mfcc_input",
                                  iterations=500)
        if result:
            print(f"  Accuracy: {result['accuracy']} | Latency: {result['latency_mean_ms']} ms")
            results.append(result)

    # ================================================================
    # 6. ONNX Runtime (CUDA)
    # ================================================================
    if has_cuda and os.path.exists(fp32_onnx):
        print("\n--- ONNX Runtime FP32 (CUDA) ---")
        result = benchmark_onnx(fp32_onnx, loaders["test"],
                                  ["CUDAExecutionProvider", "CPUExecutionProvider"],
                                  "mfcc_input", iterations=500)
        if result:
            print(f"  Accuracy: {result['accuracy']} | Latency: {result['latency_mean_ms']} ms")
            results.append(result)

    # ================================================================
    # 7. ONNX Runtime (TensorRT FP16)
    # ================================================================
    if "TensorrtExecutionProvider" in ort.get_available_providers() and os.path.exists(fp32_onnx):
        print("\n--- ONNX Runtime (TensorRT FP16) ---")
        # TRT EP with FP16 enabled
        trt_options = ort.SessionOptions()
        trt_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess = ort.InferenceSession(fp32_onnx, sess_options=trt_options,
                                         providers=[("TensorrtExecutionProvider", {
                                             "trt_fp16_enable": True,
                                             "trt_engine_cache_enable": True,
                                             "trt_engine_cache_path": "models/trt_cache_fp16",
                                         }), "CUDAExecutionProvider", "CPUExecutionProvider"])
            input_name = sess.get_inputs()[0].name

            # Accuracy
            correct, total = 0, 0
            for inputs, labels in loaders["test"]:
                outputs = sess.run(None, {input_name: inputs.numpy()})[0]
                predicted = outputs.argmax(axis=1)
                total += labels.size(0)
                correct += (predicted == labels.numpy()).sum()
            trt_fp16_acc = correct / total

            # Latency
            dummy = np.random.randn(1, 1, 40, 98).astype(np.float32)
            for _ in range(50):
                sess.run(None, {input_name: dummy})
            latencies = []
            for _ in range(500):
                start = time.perf_counter()
                sess.run(None, {input_name: dummy})
                latencies.append((time.perf_counter() - start) * 1000)
            lat = np.array(latencies)
            size_mb = os.path.getsize(fp32_onnx) / (1024 * 1024)
            print(f"  Accuracy: {trt_fp16_acc:.4f} | Latency: {lat.mean():.2f} ms")
            results.append(OrderedDict([
                ("variant", "ORT TensorRT FP16"),
                ("accuracy", f"{trt_fp16_acc:.4f}"),
                ("size_mb", f"{size_mb:.2f}"),
                ("latency_mean_ms", f"{lat.mean():.2f}"),
                ("latency_std_ms", f"{lat.std():.2f}"),
                ("latency_p95_ms", f"{np.percentile(lat, 95):.2f}"),
                ("latency_p99_ms", f"{np.percentile(lat, 99):.2f}"),
                ("throughput_fps", f"{1000/lat.mean():.1f}"),
            ]))
        except Exception as e:
            print(f"  TensorRT FP16 failed: {e}")

        # ================================================================
        # 8. ONNX Runtime (TensorRT INT8)
        # ================================================================
        print("\n--- ONNX Runtime (TensorRT INT8) ---")
        try:
            sess_int8 = ort.InferenceSession(fp32_onnx, sess_options=trt_options,
                                               providers=[("TensorrtExecutionProvider", {
                                                   "trt_int8_enable": True,
                                                   "trt_fp16_enable": False,
                                                   "trt_engine_cache_enable": True,
                                                   "trt_engine_cache_path": "models/trt_cache_int8",
                                               }), "CUDAExecutionProvider", "CPUExecutionProvider"])
            # Warmup + accuracy + latency (same pattern)
            correct, total = 0, 0
            for inputs, labels in loaders["test"]:
                outputs = sess_int8.run(None, {input_name: inputs.numpy()})[0]
                predicted = outputs.argmax(axis=1)
                total += labels.size(0)
                correct += (predicted == labels.numpy()).sum()
            trt_int8_acc = correct / total

            for _ in range(50):
                sess_int8.run(None, {input_name: dummy})
            latencies = []
            for _ in range(500):
                start = time.perf_counter()
                sess_int8.run(None, {input_name: dummy})
                latencies.append((time.perf_counter() - start) * 1000)
            lat = np.array(latencies)
            print(f"  Accuracy: {trt_int8_acc:.4f} | Latency: {lat.mean():.2f} ms")
            results.append(OrderedDict([
                ("variant", "ORT TensorRT INT8"),
                ("accuracy", f"{trt_int8_acc:.4f}"),
                ("size_mb", f"{size_mb:.2f}"),
                ("latency_mean_ms", f"{lat.mean():.2f}"),
                ("latency_std_ms", f"{lat.std():.2f}"),
                ("latency_p95_ms", f"{np.percentile(lat, 95):.2f}"),
                ("latency_p99_ms", f"{np.percentile(lat, 99):.2f}"),
                ("throughput_fps", f"{1000/lat.mean():.1f}"),
            ]))
        except Exception as e:
            print(f"  TensorRT INT8 failed: {e}")

    # ================================================================
    # Print Results Table
    # ================================================================
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}")
    if results:
        headers = list(results[0].keys())
        col_widths = [max(len(h), max(len(r[h]) for r in results)) for h in headers]
        header_line = " | ".join(f"{h:>{w}}" for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        for r in results:
            print(" | ".join(f"{r[h]:>{w}}" for h, w in zip(headers, col_widths)))

    # Save results
    results_path = "models/benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump([dict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()