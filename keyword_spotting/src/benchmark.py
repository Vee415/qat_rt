"""
Latency and accuracy benchmarking across all model variants.

Variants: FP32 PyTorch, PTQ ONNX INT8, QAT ONNX INT8, TRT FP16, TRT INT8
"""

import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from dataset import get_dataloaders
from model import get_model


def benchmark_pytorch(model_path: str, loaders, device: str = "cpu",
                      warmup: int = 50, iterations: int = 1000):
    """Benchmark PyTorch model latency."""
    model = get_model(num_classes=10, quantizable=True)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    # Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loaders["test"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = correct / total

    # Latency
    dummy = torch.randn(1, 1, 40, 98).to(device)
    for _ in range(warmup):
        model(dummy)

    latencies = []
    for _ in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    model_size = os.path.getsize(model_path) / (1024 * 1024)

    return OrderedDict([
        ("variant", "FP32 PyTorch"),
        ("accuracy", f"{accuracy:.4f}"),
        ("size_mb", f"{model_size:.2f}"),
        ("latency_mean_ms", f"{latencies.mean():.2f}"),
        ("latency_std_ms", f"{latencies.std():.2f}"),
        ("latency_p95_ms", f"{np.percentile(latencies, 95):.2f}"),
        ("latency_p99_ms", f"{np.percentile(latencies, 99):.2f}"),
        ("throughput_fps", f"{1000/latencies.mean():.1f}"),
    ])


def benchmark_onnx(onnx_path: str, loaders, warmup: int = 50,
                    iterations: int = 1000):
    """Benchmark ONNX Runtime model latency."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime not installed. Skipping ONNX benchmark.")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Accuracy
    correct = 0
    total = 0
    for inputs, labels in loaders["test"]:
        outputs = session.run(None, {input_name: inputs.numpy()})[0]
        predicted = outputs.argmax(axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
    accuracy = correct / total

    # Latency
    dummy = np.random.randn(1, 1, 40, 98).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    model_size = os.path.getsize(onnx_path) / (1024 * 1024)

    return OrderedDict([
        ("variant", os.path.basename(onnx_path).replace(".onnx", "")),
        ("accuracy", f"{accuracy:.4f}"),
        ("size_mb", f"{model_size:.2f}"),
        ("latency_mean_ms", f"{latencies.mean():.2f}"),
        ("latency_std_ms", f"{latencies.std():.2f}"),
        ("latency_p95_ms", f"{np.percentile(latencies, 95):.2f}"),
        ("latency_p99_ms", f"{np.percentile(latencies, 99):.2f}"),
        ("throughput_fps", f"{1000/latencies.mean():.1f}"),
    ])


def benchmark_trt(engine_path: str, loaders, warmup: int = 50,
                  iterations: int = 1000):
    """Benchmark TensorRT engine latency."""
    from export_tensorrt import TRTInference

    trt_model = TRTInference(engine_path)

    # Accuracy
    correct = 0
    total = 0
    for inputs, labels in loaders["test"]:
        for i in range(inputs.size(0)):
            output = trt_model.infer(inputs[i].numpy())
            pred = output.argmax()
            total += 1
            correct += (pred == labels[i].item())
    accuracy = correct / total

    # Latency
    dummy = np.random.randn(1, 1, 40, 98).astype(np.float32)
    for _ in range(warmup):
        trt_model.infer(dummy)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        trt_model.infer(dummy)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    model_size = os.path.getsize(engine_path) / (1024 * 1024)

    return OrderedDict([
        ("variant", os.path.basename(engine_path).replace(".engine", "")),
        ("accuracy", f"{accuracy:.4f}"),
        ("size_mb", f"{model_size:.2f}"),
        ("latency_mean_ms", f"{latencies.mean():.2f}"),
        ("latency_std_ms", f"{latencies.std():.2f}"),
        ("latency_p95_ms", f"{np.percentile(latencies, 95):.2f}"),
        ("latency_p99_ms", f"{np.percentile(latencies, 99):.2f}"),
        ("throughput_fps", f"{1000/latencies.mean():.1f}"),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/speech_commands")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    loaders = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    # FP32 PyTorch
    fp32_path = os.path.join(args.models_dir, "kws_fp32.pth")
    if os.path.exists(fp32_path):
        print("Benchmarking FP32 PyTorch...")
        results.append(benchmark_pytorch(fp32_path, loaders, device,
                                          iterations=args.iterations))

    # PTQ ONNX INT8
    ptq_onnx = os.path.join(args.models_dir, "kws_ptq_int8.onnx")
    if os.path.exists(ptq_onnx):
        print("Benchmarking PTQ ONNX INT8...")
        results.append(benchmark_onnx(ptq_onnx, loaders,
                                       iterations=args.iterations))

    # QAT ONNX INT8
    qat_onnx = os.path.join(args.models_dir, "kws_qat_int8.onnx")
    if os.path.exists(qat_onnx):
        print("Benchmarking QAT ONNX INT8...")
        results.append(benchmark_onnx(qat_onnx, loaders,
                                       iterations=args.iterations))

    # TRT FP16
    trt_fp16 = os.path.join(args.models_dir, "kws_trt_fp16.engine")
    if os.path.exists(trt_fp16):
        print("Benchmarking TRT FP16...")
        results.append(benchmark_trt(trt_fp16, loaders,
                                      iterations=args.iterations))

    # TRT INT8
    trt_int8 = os.path.join(args.models_dir, "kws_trt_int8.engine")
    if os.path.exists(trt_int8):
        print("Benchmarking TRT INT8...")
        results.append(benchmark_trt(trt_int8, loaders,
                                      iterations=args.iterations))

    # Print results table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    if results:
        headers = list(results[0].keys())
        print(" | ".join(f"{h:>18}" for h in headers))
        print("-" * 80)
        for r in results:
            print(" | ".join(f"{r[h]:>18}" for h in headers))

    # Save results for report generation
    import json
    results_path = os.path.join(args.models_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()