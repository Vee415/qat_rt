"""Build TensorRT engines and run inference benchmarks.

Uses TensorRT Python API directly (not through ORT) to:
1. Build TRT FP16 engine from ONNX model
2. Build TRT INT8 engine from ONNX model (with calibration)
3. Benchmark inference latency and accuracy for all variants
"""
import sys
sys.path.insert(0, 'src')

import os
import time
import json
import numpy as np
from collections import OrderedDict

import torch
import torch.ao.quantization as quant
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model


def evaluate_model(model, loader, device, desc="Evaluating"):
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


def build_trt_engine(onnx_path, engine_path, fp16=True, int8=False,
                     calibration_loader=None, max_batch=256):
    """Build a TensorRT engine from an ONNX model."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"  TRT Parse Error: {parser.get_error(error)}")
            return False

    config = builder.create_builder_config()

    # Add optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    input_name = None
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if inp.is_shape_tensor:
            continue
        input_name = inp.name
        shape = inp.shape
        # shape = [-1, 1, 40, 98] where -1 is dynamic batch
        min_shape = (1, shape[1], shape[2], shape[3])
        opt_shape = (max_batch // 2, shape[1], shape[2], shape[3])
        max_shape = (max_batch, shape[1], shape[2], shape[3])
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # FP16 mode
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled")

    # INT8 mode with calibration
    if int8:
        import pycuda.driver as cuda
        import pycuda.autoinit

        config.set_flag(trt.BuilderFlag.INT8)
        print("  INT8 enabled")

        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, loader, batch_size, cache_file="models/trt_int8.cache"):
                trt.IInt8EntropyCalibrator2.__init__(self)
                self.batch_size = batch_size
                self.cache_file = cache_file
                self.current_idx = 0
                # Pre-load all batches and copy to GPU
                self.device_buffers = []
                for inputs, _ in loader:
                    batch_np = inputs.numpy().astype(np.float32).ravel()
                    dev_mem = cuda.mem_alloc(batch_np.nbytes)
                    cuda.memcpy_htod(dev_mem, batch_np)
                    self.device_buffers.append(dev_mem)

            def get_batch_size(self):
                return self.batch_size

            def get_batch(self, name):
                if self.current_idx < len(self.device_buffers):
                    dev_ptr = int(self.device_buffers[self.current_idx])
                    self.current_idx += 1
                    return [dev_ptr]
                return None

            def read_calibration_cache(self):
                if os.path.isfile(self.cache_file):
                    with open(self.cache_file, 'rb') as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                with open(self.cache_file, 'wb') as f:
                    f.write(cache)

        calibrator = Calibrator(calibration_loader, max_batch)
        config.int8_calibrator = calibrator

    # Set max workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    # Build engine
    print(f"  Building engine (this may take a minute)...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("  ERROR: Failed to build TensorRT engine")
        return False

    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)

    size_mb = engine_bytes.nbytes / (1024 * 1024)
    print(f"  Engine saved to: {engine_path} ({size_mb:.2f} MB)")
    return True


def benchmark_trt_engine(engine_path, loader, warmup=50, iterations=500):
    """Benchmark a TensorRT engine for accuracy and latency."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    logger = trt.Logger(trt.Logger.WARNING)

    # Load engine
    runtime = trt.Runtime(logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("  ERROR: Failed to load TensorRT engine")
        return None

    context = engine.create_execution_context()

    # Get I/O info
    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    # Fixed single-sample shape for latency benchmark
    single_shape = (1, 1, 40, 98)
    output_shape = (1, 10)
    print(f"  Input: {input_name} shape={single_shape}")
    print(f"  Output: {output_name}")

    # Set input shape for dynamic profile
    context.set_input_shape(input_name, single_shape)

    # Allocate single-sample buffers for latency benchmark
    input_size = int(np.prod(single_shape))
    input_host = cuda.pagelocked_empty(input_size, np.float32)
    input_device = cuda.mem_alloc(input_host.nbytes)
    output_size = int(np.prod(output_shape))
    output_host = cuda.pagelocked_empty(output_size, np.float32)
    output_device = cuda.mem_alloc(output_host.nbytes)

    stream = cuda.Stream()

    # Accuracy evaluation — process one sample at a time
    correct, total = 0, 0
    for inputs, labels in tqdm(loader, desc="TRT Accuracy", ncols=100):
        batch_inputs = inputs.numpy()
        for i in range(batch_inputs.shape[0]):
            np.copyto(input_host, batch_inputs[i].ravel())
            cuda.memcpy_htod_async(input_device, input_host, stream)

            context.set_tensor_address(input_name, int(input_device))
            context.set_tensor_address(output_name, int(output_device))
            context.execute_async_v3(stream_handle=stream.handle)

            cuda.memcpy_dtoh_async(output_host, output_device, stream)
            stream.synchronize()

            predicted = output_host.argmax()
            if predicted == labels[i].item():
                correct += 1
            total += 1

    accuracy = correct / total

    # Latency benchmark (single sample)
    np.copyto(input_host, np.random.randn(*single_shape).astype(np.float32).ravel())

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod_async(input_device, input_host, stream)
        context.set_tensor_address(input_name, int(input_device))
        context.set_tensor_address(output_name, int(output_device))
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_host, output_device, stream)
        stream.synchronize()

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        cuda.memcpy_htod_async(input_device, input_host, stream)
        context.set_tensor_address(input_name, int(input_device))
        context.set_tensor_address(output_name, int(output_device))
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_host, output_device, stream)
        stream.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    engine_size = os.path.getsize(engine_path) / (1024 * 1024)

    return OrderedDict([
        ("variant", os.path.basename(engine_path).replace(".engine", "")),
        ("accuracy", f"{accuracy:.4f}"),
        ("size_mb", f"{engine_size:.2f}"),
        ("latency_mean_ms", f"{latencies.mean():.2f}"),
        ("latency_std_ms", f"{latencies.std():.2f}"),
        ("latency_p95_ms", f"{np.percentile(latencies, 95):.2f}"),
        ("latency_p99_ms", f"{np.percentile(latencies, 99):.2f}"),
        ("throughput_fps", f"{1000/latencies.mean():.1f}"),
    ])


def main():
    print("=" * 70)
    print("TensorRT Engine Build + Benchmark")
    print("=" * 70)

    import tensorrt as trt
    print(f"TensorRT version: {trt.__version__}")
    print(f"GPU FP16 support: {trt.Builder(trt.Logger(trt.Logger.WARNING)).platform_has_fast_fp16}")

    # Load data for calibration
    print("\nLoading dataset for calibration...")
    loaders = get_dataloaders('data/speech_commands', batch_size=256, num_workers=0)

    fp32_onnx = "models/kws_fp32.onnx"
    if not os.path.exists(fp32_onnx):
        print("ERROR: models/kws_fp32.onnx not found. Run quantize_ptq_baseline.py first.")
        return

    results = []

    # ================================================================
    # 1. Build TRT FP16 Engine
    # ================================================================
    print("\n--- Building TensorRT FP16 Engine ---")
    fp16_engine_path = "models/kws_trt_fp16.engine"
    if not os.path.exists(fp16_engine_path):
        success = build_trt_engine(fp32_onnx, fp16_engine_path,
                                    fp16=True, int8=False)
        if not success:
            print("FP16 engine build failed. Skipping TRT benchmarks.")
            return
    else:
        print(f"  Engine already exists: {fp16_engine_path}")

    # Benchmark TRT FP16
    print("\n--- Benchmarking TRT FP16 ---")
    try:
        result = benchmark_trt_engine(fp16_engine_path, loaders["test"])
        if result:
            print(f"  Accuracy: {result['accuracy']} | Latency: {result['latency_mean_ms']} ms")
            results.append(result)
    except Exception as e:
        print(f"  TRT FP16 benchmark failed: {e}")

    # ================================================================
    # 2. Build TRT INT8 Engine
    # ================================================================
    print("\n--- Building TensorRT INT8 Engine ---")
    int8_engine_path = "models/kws_trt_int8.engine"
    if not os.path.exists(int8_engine_path):
        success = build_trt_engine(fp32_onnx, int8_engine_path,
                                    fp16=True, int8=True,
                                    calibration_loader=loaders["train"])
        if not success:
            print("INT8 engine build failed.")
        else:
            print(f"  Engine saved to: {int8_engine_path}")
    else:
        print(f"  Engine already exists: {int8_engine_path}")

    # Benchmark TRT INT8
    if os.path.exists(int8_engine_path):
        print("\n--- Benchmarking TRT INT8 ---")
        try:
            result = benchmark_trt_engine(int8_engine_path, loaders["test"])
            if result:
                print(f"  Accuracy: {result['accuracy']} | Latency: {result['latency_mean_ms']} ms")
                results.append(result)
        except Exception as e:
            print(f"  TRT INT8 benchmark failed: {e}")

    # ================================================================
    # Print Results
    # ================================================================
    if results:
        print(f"\n{'='*80}")
        print("TENSORRT BENCHMARK RESULTS")
        print(f"{'='*80}")
        headers = list(results[0].keys())
        col_widths = [max(len(h), max(len(r[h]) for r in results)) for h in headers]
        header_line = " | ".join(f"{h:>{w}}" for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        for r in results:
            print(" | ".join(f"{r[h]:>{w}}" for h, w in zip(headers, col_widths)))

        # Save results
        results_path = "models/trt_benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump([dict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()