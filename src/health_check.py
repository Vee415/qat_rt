"""
Health check CLI for edge inference pipeline.

Verifies that the inference engine loads correctly, model shapes match
expectations, MFCC extraction produces valid output, and basic inference
works.

Usage:
    python -m src.health_check --config configs/kws_onnx.yaml
    python -m src.health_check --backend onnx --model models/kws_fp32.onnx
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from inference_engine import create_engine
from keyword_detector import KEYWORDS, KeywordDetector
from realtime_mfcc import RealtimeMFCC, CLIP_LENGTH

logger = logging.getLogger(__name__)


def check_mfcc_pipeline() -> bool:
    """Verify MFCC extraction produces (1, 1, 40, 98) from 1s audio."""
    try:
        extractor = RealtimeMFCC()
        audio = np.random.randn(CLIP_LENGTH).astype(np.float32)
        mfcc = extractor.extract(audio)

        if mfcc.shape != (1, 1, 40, 98):
            logger.error(f"MFCC shape mismatch: expected (1, 1, 40, 98), got {mfcc.shape}")
            return False

        if not np.all(np.isfinite(mfcc)):
            logger.error("MFCC contains NaN or Inf values")
            return False

        logger.info(f"MFCC pipeline: OK (shape={mfcc.shape})")
        return True
    except Exception as e:
        logger.error(f"MFCC pipeline failed: {e}")
        return False


def check_model_loads(backend: str, model_path: str) -> bool:
    """Try to create inference engine, verify it loads without error."""
    try:
        engine = create_engine(backend, model_path)
        logger.info(f"Model loads: OK (cold_start={engine.cold_start_ms:.1f}ms)")
        engine.close()
        return True
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return False


def check_inference_sanity(backend: str, model_path: str) -> bool:
    """Run one inference on random input, verify output shape and values."""
    try:
        engine = create_engine(backend, model_path)
        dummy = np.random.randn(1, 1, 40, 98).astype(np.float32)

        logits, latency_ms = engine.predict(dummy)

        if logits.shape[0] != 1 or logits.shape[1] != 10:
            logger.error(f"Output shape mismatch: expected (1, 10), got {logits.shape}")
            engine.close()
            return False

        if not np.all(np.isfinite(logits)):
            logger.error("Output contains NaN or Inf values")
            engine.close()
            return False

        logger.info(f"Inference sanity: OK (latency={latency_ms:.2f}ms, output_shape={logits.shape})")
        engine.close()
        return True
    except Exception as e:
        logger.error(f"Inference sanity check failed: {e}")
        return False


def check_keyword_detector() -> bool:
    """Verify keyword detection logic works with known input."""
    try:
        detector = KeywordDetector(threshold=0.5, cooldown_ms=0)

        # Strong "yes" signal
        logits = np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        keyword, confidence = detector.detect(logits)

        if keyword != "yes":
            logger.error(f"Keyword detection failed: expected 'yes', got '{keyword}'")
            return False

        if confidence < 0.5:
            logger.error(f"Confidence too low: {confidence}")
            return False

        # Uniform logits should not trigger
        detector2 = KeywordDetector(threshold=0.7)
        logits_uniform = np.zeros((1, 10))
        keyword2, _ = detector2.detect(logits_uniform)

        if keyword2 != "---":
            logger.error(f"Low-confidence detection should be suppressed, got '{keyword2}'")
            return False

        logger.info(f"Keyword detector: OK (keywords={KEYWORDS})")
        return True
    except Exception as e:
        logger.error(f"Keyword detector check failed: {e}")
        return False


def check_system_info() -> bool:
    """Print system info and check for Jetson/GPU."""
    import platform

    logger.info(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    logger.info(f"Python: {platform.python_version()}")

    # Check Jetson
    if os.path.exists("/etc/nv_tegra_release"):
        with open("/etc/nv_tegra_release") as f:
            logger.info(f"Jetson: {f.read().strip()}")
    else:
        logger.info("Not running on Jetson")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA: {torch.version.cuda}")
        else:
            logger.info("No CUDA GPU available")
    except ImportError:
        logger.info("PyTorch not installed")

    # Check TensorRT
    try:
        import tensorrt
        logger.info(f"TensorRT: {tensorrt.__version__}")
    except ImportError:
        logger.info("TensorRT not installed")

    # Check ONNX Runtime
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        logger.info(f"ONNX Runtime: {onnxruntime.__version__} (providers: {providers})")
    except ImportError:
        logger.info("ONNX Runtime not installed")

    return True


def run_health_check(config_path: str = None, backend: str = None,
                     model_path: str = None) -> bool:
    """Run all health checks.

    Returns True if all checks pass, False otherwise.
    """
    checks = []

    # System info (always runs)
    logger.info("=" * 50)
    logger.info("SYSTEM INFO")
    logger.info("=" * 50)
    checks.append(("System Info", check_system_info()))

    # MFCC pipeline (always runs, no model needed)
    logger.info("")
    logger.info("=" * 50)
    logger.info("MFCC PIPELINE")
    logger.info("=" * 50)
    checks.append(("MFCC Pipeline", check_mfcc_pipeline()))

    # Keyword detector (always runs)
    logger.info("")
    logger.info("=" * 50)
    logger.info("KEYWORD DETECTOR")
    logger.info("=" * 50)
    checks.append(("Keyword Detector", check_keyword_detector()))

    # Model checks (requires config or explicit args)
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        backend = config['runtime']['backend']
        model_path = config['runtime']['model_path']

    if backend and model_path:
        logger.info("")
        logger.info("=" * 50)
        logger.info(f"MODEL: {backend} - {model_path}")
        logger.info("=" * 50)
        checks.append(("Model Load", check_model_loads(backend, model_path)))
        checks.append(("Inference Sanity", check_inference_sanity(backend, model_path)))

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("HEALTH CHECK SUMMARY")
    logger.info("=" * 50)
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        logger.info(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\nAll health checks passed!")
    else:
        logger.warning("\nSome health checks failed!")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Edge Replay Benchmark Health Check")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (uses runtime settings)")
    parser.add_argument("--backend", type=str, choices=["onnx", "trt"], default=None,
                        help="Inference backend to check")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    success = run_health_check(
        config_path=args.config,
        backend=args.backend,
        model_path=args.model,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()