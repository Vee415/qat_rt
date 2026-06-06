# Edge Replay Benchmark - Keyword Spotting
# Multi-platform: works on x86 (dev) and Jetson Orin Nano Super (deployment)
#
# Build for x86:
#   docker build -t edge-replay-bench .
#
# Build for Jetson (use NVIDIA L4T base image):
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t/pytorch:l4t-r36.4.0 -t edge-replay-bench-jetson .
#
# Run ONNX CPU benchmark:
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models edge-replay-bench --config configs/kws_onnx.yaml
#
# Run TRT benchmark (requires GPU):
#   docker run --gpus all --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models edge-replay-bench --config configs/kws_tensorrt_fp16.yaml

ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Note: TensorRT and pycuda must be installed separately on Jetson
# They come pre-installed in the NVIDIA L4T base image
# On x86, install with: pip install tensorrt pycuda

# Copy project
COPY . .

# Default: run ONNX CPU benchmark
ENTRYPOINT ["python", "src/replay_benchmark.py"]
CMD ["--config", "configs/kws_onnx.yaml"]