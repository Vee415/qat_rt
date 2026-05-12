"""
Inference engine for keyword spotting.

Supports two backends:
  - ONNX Runtime (CPU): Default, works on any machine
  - TensorRT (GPU): Requires CUDA + TensorRT installation

Both backends accept input shape (1, 1, 40, 98) and produce
output shape (1, 10) with raw logits (no softmax applied).
"""

import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def infer(self, mfcc: np.ndarray) -> np.ndarray:
        """Run inference on a single MFCC input.

        Args:
            mfcc: Input array of shape (1, 1, 40, 98), float32

        Returns:
            Output array of shape (1, 10), float32 raw logits
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        pass


class ONNXBackend(InferenceBackend):
    """ONNX Runtime inference backend (CPU by default)."""

    def __init__(self, model_path: str,
                 providers: Optional[list] = None):
        import onnxruntime as ort

        if providers is None:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        print(f"  ONNX model input: {self.input_name} shape={input_shape}")
        print(f"  ONNX model output: {self.output_name}")

        # Warm up
        dummy = np.random.randn(1, 1, 40, 98).astype(np.float32)
        for _ in range(10):
            self.session.run(None, {self.input_name: dummy})

    def infer(self, mfcc: np.ndarray) -> np.ndarray:
        """Run ONNX inference. Returns raw logits (1, 10)."""
        return self.session.run(
            [self.output_name],
            {self.input_name: mfcc.astype(np.float32)}
        )[0]

    def close(self) -> None:
        del self.session


class TRTBackend(InferenceBackend):
    """TensorRT inference backend (requires CUDA + TensorRT)."""

    def __init__(self, engine_path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Set fixed input shape
        self.context.set_input_shape("mfcc_input", (1, 1, 40, 98))

        # Allocate GPU buffers
        input_size = 1 * 1 * 40 * 98
        output_size = 1 * 10
        self.input_host = cuda.pagelocked_empty(input_size, np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_host = cuda.pagelocked_empty(output_size, np.float32)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)
        self.stream = cuda.Stream()

        # Warm up
        np.copyto(self.input_host, np.random.randn(input_size).astype(np.float32))
        for _ in range(10):
            cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
            self.context.set_tensor_address("mfcc_input", int(self.input_device))
            self.context.set_tensor_address("keyword_output", int(self.output_device))
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
            self.stream.synchronize()

    def infer(self, mfcc: np.ndarray) -> np.ndarray:
        """Run TensorRT inference. Returns raw logits (1, 10)."""
        import pycuda.driver as cuda

        np.copyto(self.input_host, mfcc.ravel())
        cuda.memcpy_htod_async(self.input_device, self.input_host, self.stream)
        self.context.set_tensor_address("mfcc_input", int(self.input_device))
        self.context.set_tensor_address("keyword_output", int(self.output_device))
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()

        return self.output_host.reshape(1, 10)

    def close(self) -> None:
        if self.stream:
            self.stream.synchronize()
        del self.context
        del self.engine


class InferenceEngine:
    """High-level inference engine with latency tracking.

    Wraps a backend and measures per-inference latency,
    maintaining a rolling window for P99 calculation.
    """

    def __init__(self, backend: InferenceBackend):
        self.backend = backend
        self._latencies = deque(maxlen=1000)

    def predict(self, mfcc: np.ndarray) -> tuple:
        """Run inference and return (logits, latency_ms)."""
        start = time.perf_counter()
        logits = self.backend.infer(mfcc)
        latency_ms = (time.perf_counter() - start) * 1000
        self._latencies.append(latency_ms)
        return logits, latency_ms

    def get_p99_latency(self) -> float:
        """Get P99 latency in milliseconds over recent inferences."""
        if not self._latencies:
            return 0.0
        return float(np.percentile(list(self._latencies), 99))

    def get_stats(self) -> dict:
        """Get latency statistics."""
        if not self._latencies:
            return {"mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "n": 0}
        lat = np.array(self._latencies)
        return {
            "mean_ms": float(lat.mean()),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "n": len(lat),
        }

    def close(self) -> None:
        self.backend.close()


def create_engine(backend: str, model_path: str) -> InferenceEngine:
    """Factory function to create the appropriate inference engine.

    Args:
        backend: 'onnx' or 'trt'
        model_path: Path to model file (.onnx or .engine)

    Returns:
        InferenceEngine instance
    """
    if backend == 'onnx':
        print(f"Loading ONNX model: {model_path}")
        b = ONNXBackend(model_path)
    elif backend == 'trt':
        print(f"Loading TensorRT engine: {model_path}")
        b = TRTBackend(model_path)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'onnx' or 'trt'.")

    return InferenceEngine(b)