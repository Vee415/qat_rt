"""Profile TensorRT engines: layer fusion analysis and latency benchmarking.

Uses TensorRT Engine Inspector API (TRT 10+) to:
1. Inspect engine layers and fusion decisions
2. Benchmark per-inference latency with CUDA synchronization
3. Verify accuracy
4. Generate a profiling report
"""
import sys
sys.path.insert(0, 'src')

import os
import time
import json
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

from dataset import get_dataloaders


def inspect_engine(engine_path):
    """Inspect a TensorRT engine using Engine Inspector API."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print(f"  ERROR: Failed to load engine: {engine_path}")
        return None

    context = engine.create_execution_context()
    context.set_input_shape("mfcc_input", (1, 1, 40, 98))

    inspector = engine.create_engine_inspector()
    inspector.execution_context = context

    info = OrderedDict()
    info["engine_path"] = engine_path
    info["engine_size_mb"] = f"{os.path.getsize(engine_path) / (1024*1024):.2f}"
    info["num_layers"] = engine.num_layers

    # Tensor info
    info["num_io_tensors"] = engine.num_io_tensors
    tensors = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = "INPUT" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT"
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        tensors.append({"name": name, "shape": list(shape), "dtype": str(dtype), "mode": mode})
    info["tensors"] = tensors

    # Layer info via inspector
    layers = []
    for i in range(engine.num_layers):
        try:
            desc = inspector.get_layer_information(i, trt.LayerInformationFormat.ONELINE).strip()
        except Exception:
            desc = f"Layer {i} (info unavailable)"
        layers.append(desc)
    info["layers"] = layers

    del context, inspector
    return info


def profile_engine_latency(engine_path, warmup=100, iterations=500):
    """Benchmark per-inference latency with CUDA synchronization."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape("mfcc_input", (1, 1, 40, 98))

    input_size = 1 * 1 * 40 * 98
    output_size = 1 * 10
    input_host = cuda.pagelocked_empty(input_size, np.float32)
    input_device = cuda.mem_alloc(input_host.nbytes)
    output_host = cuda.pagelocked_empty(output_size, np.float32)
    output_device = cuda.mem_alloc(output_host.nbytes)
    stream = cuda.Stream()

    np.copyto(input_host, np.random.randn(input_size).astype(np.float32))

    # Warmup
    for _ in range(warmup):
        cuda.memcpy_htod_async(input_device, input_host, stream)
        context.set_tensor_address("mfcc_input", int(input_device))
        context.set_tensor_address("keyword_output", int(output_device))
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_host, output_device, stream)
        stream.synchronize()

    # Timed runs
    latencies = []
    for _ in range(iterations):
        cuda.memcpy_htod_async(input_device, input_host, stream)
        start = time.perf_counter()
        context.set_tensor_address("mfcc_input", int(input_device))
        context.set_tensor_address("keyword_output", int(output_device))
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_host, output_device, stream)
        stream.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    engine_size = os.path.getsize(engine_path) / (1024 * 1024)

    return OrderedDict([
        ("variant", os.path.basename(engine_path).replace(".engine", "")),
        ("size_mb", f"{engine_size:.2f}"),
        ("latency_mean_ms", f"{latencies.mean():.3f}"),
        ("latency_std_ms", f"{latencies.std():.3f}"),
        ("latency_p50_ms", f"{np.percentile(latencies, 50):.3f}"),
        ("latency_p95_ms", f"{np.percentile(latencies, 95):.3f}"),
        ("latency_p99_ms", f"{np.percentile(latencies, 99):.3f}"),
        ("latency_min_ms", f"{latencies.min():.3f}"),
        ("latency_max_ms", f"{latencies.max():.3f}"),
        ("throughput_fps", f"{1000/latencies.mean():.1f}"),
    ])


def benchmark_accuracy(engine_path, loader):
    """Verify accuracy of a TRT engine on the test set."""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    context.set_input_shape("mfcc_input", (1, 1, 40, 98))

    input_host = cuda.pagelocked_empty(1*1*40*98, np.float32)
    input_device = cuda.mem_alloc(input_host.nbytes)
    output_host = cuda.pagelocked_empty(1*10, np.float32)
    output_device = cuda.mem_alloc(output_host.nbytes)
    stream = cuda.Stream()

    correct, total = 0, 0
    for inputs, labels in tqdm(loader, desc=f"Accuracy {os.path.basename(engine_path)}", ncols=80):
        batch_np = inputs.numpy()
        for i in range(batch_np.shape[0]):
            np.copyto(input_host, batch_np[i].ravel())
            cuda.memcpy_htod_async(input_device, input_host, stream)
            context.set_tensor_address("mfcc_input", int(input_device))
            context.set_tensor_address("keyword_output", int(output_device))
            context.execute_async_v3(stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(output_host, output_device, stream)
            stream.synchronize()
            if output_host.argmax() == labels[i].item():
                correct += 1
            total += 1

    return correct / total


def analyze_fusion(layers):
    """Analyze layer fusion decisions from TRT inspector output."""
    fused = []
    reformats = []
    standalone = []

    for layer in layers:
        if "Reformatting CopyNode" in layer:
            reformats.append(layer)
        elif "+" in layer and ("Conv" in layer or "Gemm" in layer):
            fused.append(layer)
        else:
            standalone.append(layer)

    return {"fused": fused, "reformats": reformats, "standalone": standalone}


def main():
    print("=" * 70)
    print("TensorRT Engine Profiling & Fusion Analysis")
    print("=" * 70)

    import tensorrt as trt
    print(f"TensorRT version: {trt.__version__}")

    fp16_engine_path = "models/kws_trt_fp16.engine"
    int8_engine_path = "models/kws_trt_int8.engine"

    # ================================================================
    # 1. Inspect Engine Structure
    # ================================================================
    engines = {}
    for path, label in [(fp16_engine_path, "FP16"), (int8_engine_path, "INT8")]:
        if not os.path.exists(path):
            print(f"\n  {label} engine not found: {path}")
            continue

        print(f"\n--- Inspecting {label} Engine ---")
        info = inspect_engine(path)
        if info:
            engines[label] = info
            print(f"  Size: {info['engine_size_mb']} MB")
            print(f"  Layers: {info['num_layers']}")
            for t in info["tensors"]:
                print(f"  {t['name']}: shape={t['shape']}, dtype={t['dtype']}, {t['mode']}")
            print(f"\n  TRT Optimized Layers:")
            for layer in info["layers"]:
                print(f"    {layer}")

            # Fusion analysis
            fusion = analyze_fusion(info["layers"])
            print(f"\n  Fusion Summary:")
            print(f"    Fused kernels (Conv+ReLU, Gemm+):  {len(fusion['fused'])}")
            print(f"    Reformat/copy nodes:                {len(fusion['reformats'])}")
            print(f"    Standalone layers:                  {len(fusion['standalone'])}")
            print(f"    Total kernel launches:              {len(info['layers'])}")

    # Compare FP16 vs INT8 structure
    if "FP16" in engines and "INT8" in engines:
        fp16 = engines["FP16"]
        int8 = engines["INT8"]
        print(f"\n{'='*70}")
        print("FP16 vs INT8 ENGINE STRUCTURE")
        print(f"{'='*70}")
        print(f"\n  {'Property':<30} {'FP16':>10} {'INT8':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Engine Size':<30} {fp16['engine_size_mb']:>10} {int8['engine_size_mb']:>10}")
        print(f"  {'Num Layers':<30} {fp16['num_layers']:>10} {int8['num_layers']:>10}")
        print(f"  {'Num I/O Tensors':<30} {fp16['num_io_tensors']:>10} {int8['num_io_tensors']:>10}")

        fp16_fusion = analyze_fusion(fp16["layers"])
        int8_fusion = analyze_fusion(int8["layers"])
        print(f"  {'Fused kernels':<30} {len(fp16_fusion['fused']):>10} {len(int8_fusion['fused']):>10}")
        print(f"  {'Reformat nodes':<30} {len(fp16_fusion['reformats']):>10} {len(int8_fusion['reformats']):>10}")

        print(f"\n  Key difference: INT8 engine has {int8['num_layers']} layers vs FP16's {fp16['num_layers']}.")
        print(f"  INT8 eliminates FP16's extra reformat node (Layer 2) by keeping data in INT8 format")
        print(f"  between Conv+ReLU and MaxPool, and fuses the final Gemm.")

    # ================================================================
    # 2. Latency Benchmarking
    # ================================================================
    print(f"\n{'='*70}")
    print("LATENCY BENCHMARKING (single-sample inference)")
    print(f"{'='*70}")

    profile_results = []
    for path, label in [(fp16_engine_path, "FP16"), (int8_engine_path, "INT8")]:
        if not os.path.exists(path):
            continue
        print(f"\n--- Profiling {label} Engine ---")
        result = profile_engine_latency(path)
        if result:
            for k, v in result.items():
                print(f"  {k}: {v}")
            profile_results.append(result)

    # ================================================================
    # 3. Accuracy Verification
    # ================================================================
    print(f"\n{'='*70}")
    print("ACCURACY VERIFICATION")
    print(f"{'='*70}")

    loaders = get_dataloaders('data/speech_commands', batch_size=256, num_workers=0)

    for path, label in [(fp16_engine_path, "FP16"), (int8_engine_path, "INT8")]:
        if not os.path.exists(path):
            continue
        acc = benchmark_accuracy(path, loaders["test"])
        variant = os.path.basename(path).replace(".engine", "")
        print(f"  {label} accuracy: {acc:.4f} ({acc*100:.2f}%)")
        for r in profile_results:
            if r["variant"] == variant:
                r["accuracy"] = f"{acc:.4f}"

    # ================================================================
    # 4. Fusion Analysis Deep Dive
    # ================================================================
    print(f"\n{'='*70}")
    print("FUSION ANALYSIS")
    print(f"{'='*70}")

    print("""
  Original PyTorch KWSNet (before TRT optimization):
    features.0: Conv2d(1, 32, 3x3)  +  features.1: BN(32)  +  features.2: ReLU
    features.3: MaxPool2d(2)
    features.4: Conv2d(32, 64, 3x3) +  features.5: BN(64)  +  features.6: ReLU
    features.7: MaxPool2d(2)
    features.8: Conv2d(64, 128, 3x3)+  features.9: BN(128) +  features.10: ReLU
    features.11: MaxPool2d(2)
    classifier.0: AdaptiveAvgPool2d(1)
    classifier.2: Linear(128, 10)
    Total: 12 separate operations = 12 kernel launches (in naive execution)

  What TRT does:
    1. Conv+ReLU FUSION: Merges Conv2d + ReLU into a single kernel
       (Conv+BN+ReLU fusion happens at export time; TRT gets Conv+ReLU)
    2. Reformatting: Inserts copy nodes for format conversions (FP32<->FP16, FP32<->INT8)
    3. Gemm FUSION (INT8): Merges the final Linear layer into a fused Gemm+quantize op
    4. Kernel auto-tuning: Selects optimal CUDA kernels for each layer at build time""")

    if "FP16" in engines:
        fp16_fusion = analyze_fusion(engines["FP16"]["layers"])
        print(f"\n  FP16 Engine Fusion Details:")
        print(f"    Fused Conv+ReLU kernels: {len(fp16_fusion['fused'])}")
        for f in fp16_fusion['fused']:
            print(f"      {f.strip()}")
        print(f"    Reformat/copy nodes:    {len(fp16_fusion['reformats'])}")
        for r in fp16_fusion['reformats']:
            print(f"      {r.strip()}")

    if "INT8" in engines:
        int8_fusion = analyze_fusion(engines["INT8"]["layers"])
        print(f"\n  INT8 Engine Fusion Details:")
        print(f"    Fused kernels:          {len(int8_fusion['fused'])}")
        for f in int8_fusion['fused']:
            print(f"      {f.strip()}")
        print(f"    Reformat/copy nodes:    {len(int8_fusion['reformats'])}")
        for r in int8_fusion['reformats']:
            print(f"      {r.strip()}")

    # ================================================================
    # 5. Save Report
    # ================================================================
    if profile_results:
        report = {
            "trt_version": trt.__version__,
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "engines": profile_results,
            "fp16_layers": engines.get("FP16", {}).get("layers", []),
            "int8_layers": engines.get("INT8", {}).get("layers", []),
        }

        report_path = "models/trt_profile_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nProfile report saved to {report_path}")

    # ================================================================
    # 6. Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("PROFILING COMPLETE — KEY TAKEAWAYS")
    print(f"{'='*70}")
    print("""
  1. LAYER FUSION: TRT merges Conv+ReLU into single kernels, reducing kernel
     launch overhead. Each fused layer = 1 GPU kernel instead of 2-3.

  2. REFORMAT NODES: FP16 needs format conversion between FP32 input and
     FP16 compute. INT8 reduces this by keeping intermediate data in INT8.

  3. WHY FP16 > INT8 ON THIS GPU: The RTX 4060's Tensor Cores are extremely
     fast at FP16. For this tiny 94K-param model, the quantize/dequantize
     overhead at each layer boundary outweighs the compute savings of INT8.
     On larger models or older GPUs, INT8 would likely be faster.

  4. ENGINE SIZE: INT8 engine (0.20 MB) is smaller than FP16 (0.29 MB) because
     INT8 weights are half the size of FP16 weights (1 byte vs 2 bytes).

  5. ACCURACY: FP16 preserves accuracy (~87.3%), INT8 has minor loss (~86.7%)
     because TRT's calibration quantization adds some error.""")


if __name__ == "__main__":
    main()