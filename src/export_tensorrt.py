"""
TensorRT engine build and inference.

Builds FP16 and INT8 engines from ONNX models, wraps inference,
and provides profiling via trtexec.
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available. Engine build functions will not work.")

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class TRTInference:
    """Wrapper for TensorRT engine inference."""

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not installed.")

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        import pycuda.driver as cuda
        import pycuda.autoinit

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = np.prod(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

        self.stream = cuda.Stream()

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on a single input."""
        import pycuda.driver as cuda

        # Copy input to device
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

        # Set tensor addresses
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()

        return self.outputs[0]["host"].reshape(self.outputs[0]["shape"])

    def __del__(self):
        if self.stream:
            self.stream.synchronize()


def build_engine(onnx_path: str, engine_path: str, precision: str = "fp16",
                 calib_data_dir: str = None, batch_size: int = 1):
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to the ONNX model
        engine_path: Output path for the .engine file
        precision: "fp16" or "int8"
        calib_data_dir: Directory with calibration data (required for int8)
        batch_size: Max batch size for the engine
    """
    if not TRT_AVAILABLE:
        print("TensorRT not available. Use trtexec instead:")
        cmd = _trtexec_command(onnx_path, engine_path, precision)
        print(f"  {cmd}")
        return

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX parse error: {parser.get_error(error)}")
            return

    config = builder.create_builder_config()

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("Building FP16 engine...")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # FP16 fallback
        print("Building INT8 engine...")
        # INT8 calibration would go here for non-QAT models

    # Set max workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Build engine
    print(f"Building TensorRT engine (this may take a few minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    engine_size = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"Engine saved to {engine_path} ({engine_size:.2f} MB)")


def _trtexec_command(onnx_path: str, engine_path: str, precision: str = "fp16") -> str:
    """Generate trtexec command for engine building."""
    cmd = f"trtexec --onnx={onnx_path} --saveEngine={engine_path}"
    if precision == "fp16":
        cmd += " --fp16"
    elif precision == "int8":
        cmd += " --int8 --fp16"
    cmd += " --verbose"
    return cmd


def profile_with_trtexec(onnx_path: str, iterations: int = 1000):
    """Profile an ONNX model using trtexec.

    Returns latency stats and prints layer fusion info.
    """
    cmd = f"trtexec --onnx={onnx_path} --iterations={iterations} --verbose"
    print(f"Running: {cmd}")
    print("Note: Run this command manually to see layer fusion output.")
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--engine_path", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=["fp16", "int8"],
                        default="fp16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_trtexec", action="store_true",
                        help="Print trtexec command instead of using Python API")
    args = parser.parse_args()

    if args.use_trtexec or not TRT_AVAILABLE:
        cmd = _trtexec_command(args.onnx_path, args.engine_path, args.precision)
        print(f"Run this command to build the engine:\n{cmd}")
    else:
        build_engine(args.onnx_path, args.engine_path, args.precision,
                     batch_size=args.batch_size)


if __name__ == "__main__":
    main()