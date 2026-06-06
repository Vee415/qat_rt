"""
Config-driven replay benchmark runner for edge inference.

Reads a YAML config, instantiates the replay engine + inference backend +
system metrics collector, runs the benchmark, and writes results to
CSV/JSON/markdown.

Usage:
    python src/replay_benchmark.py --config configs/kws_onnx.yaml
    python src/replay_benchmark.py --config configs/kws_tensorrt_fp16.yaml

On Jetson Orin Nano Super:
    python src/replay_benchmark.py --config configs/kws_tensorrt_fp16.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from inference_engine import create_engine, InferenceEngine
from keyword_detector import KeywordDetector, KEYWORDS
from metrics.metrics_logger import BenchmarkResult, MetricsLogger
from metrics.system_metrics import SystemMetricsCollector, is_jetson
from realtime_mfcc import RealtimeMFCC
from replay.file_replay_engine import FileReplayEngine

logger = logging.getLogger(__name__)


def get_platform_name() -> str:
    """Get a descriptive platform name for benchmark results."""
    if is_jetson():
        try:
            with open("/etc/nv_tegra_release") as f:
                release = f.read().strip()
            return f"Jetson ({release})"
        except Exception:
            return "Jetson Orin Nano Super"
    else:
        import platform
        return f"{platform.machine()} ({platform.processor() or 'x86'})"


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def compute_accuracy(engine: InferenceEngine, mfcc_extractor: RealtimeMFCC,
                     replay: FileReplayEngine) -> float:
    """Run inference on all replay files and compute top-1 accuracy.

    Returns accuracy as a float between 0 and 1.
    """
    correct = 0
    total = 0

    for audio, metadata in replay.replay_no_timing():
        mfcc = mfcc_extractor.extract(audio)
        logits, _ = engine.predict(mfcc)
        predicted = int(np.argmax(logits))
        if predicted == metadata['label']:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def run_benchmark(config: dict) -> BenchmarkResult:
    """Run a single benchmark configuration.

    Args:
        config: Loaded YAML config dict.

    Returns:
        BenchmarkResult with all collected metrics.
    """
    runtime_cfg = config['runtime']
    replay_cfg = config['replay']
    bench_cfg = config['benchmark']
    log_cfg = config.get('logging', {})

    # Setup logging
    setup_logging(
        level=log_cfg.get('level', 'INFO'),
        format_str=log_cfg.get('format'),
    )

    # Start system metrics collection
    metrics_collector = SystemMetricsCollector(
        interval_ms=bench_cfg.get('metrics_interval_ms', 100)
    )
    metrics_collector.start()

    # Create inference engine (measures cold start)
    logger.info(f"Creating {runtime_cfg['backend']} engine: {runtime_cfg['model_path']}")
    engine = create_engine(runtime_cfg['backend'], runtime_cfg['model_path'])

    # Create MFCC extractor
    mfcc_extractor = RealtimeMFCC()

    # Create file replay engine
    replay = FileReplayEngine(
        data_dir=replay_cfg['data_dir'],
        target_fps=replay_cfg.get('target_fps', 10),
        max_files=replay_cfg.get('max_files'),
        shuffle=replay_cfg.get('shuffle', True),
        seed=replay_cfg.get('seed', 42),
        speed=replay_cfg.get('speed', 1.0),
    )

    # Get model size
    model_path = runtime_cfg['model_path']
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0

    # Compute accuracy if requested
    accuracy = 0.0
    if bench_cfg.get('accuracy', True):
        logger.info("Computing accuracy on replay dataset...")
        # Reset replay stats for accuracy pass
        replay.reset_stats()
        accuracy = compute_accuracy(engine, mfcc_extractor, replay)

    # Warmup
    logger.info(f"Warming up ({bench_cfg.get('warmup_iterations', 50)} iterations)...")
    dummy_mfcc = np.random.randn(1, 1, 40, 98).astype(np.float32)
    for _ in range(bench_cfg.get('warmup_iterations', 50)):
        engine.predict(dummy_mfcc)

    # Reset replay engine for benchmark pass
    replay.reset_stats()

    # Run benchmark with real-time replay
    logger.info(f"Running replay benchmark ({len(replay)} files, target_fps={replay_cfg.get('target_fps', 10)})...")

    # Re-create replay engine for the timed pass (fresh stats)
    replay = FileReplayEngine(
        data_dir=replay_cfg['data_dir'],
        target_fps=replay_cfg.get('target_fps', 10),
        max_files=replay_cfg.get('max_files'),
        shuffle=replay_cfg.get('shuffle', True),
        seed=replay_cfg.get('seed', 42),
        speed=replay_cfg.get('speed', 1.0),
    )

    max_windows = bench_cfg.get('max_windows', 5000)
    window_count = 0

    for audio, metadata in replay.replay():
        mfcc = mfcc_extractor.extract(audio)
        logits, latency_ms = engine.predict(mfcc)

        window_count += 1
        if window_count >= max_windows:
            logger.info(f"Reached max_windows limit ({max_windows})")
            break

    # Stop metrics collection
    metrics_collector.stop()
    system_stats = metrics_collector.get_summary()

    # Get latency stats from inference engine
    latency_stats = engine.get_stats()

    # Get replay stats
    replay_stats = replay.get_stats()

    # Build result
    result = BenchmarkResult(
        runtime=runtime_cfg['name'],
        precision=runtime_cfg.get('precision', 'unknown'),
        accuracy=accuracy,
        latency_p50_ms=latency_stats.get('p50_ms', 0),
        latency_p95_ms=latency_stats.get('p95_ms', 0),
        latency_p99_ms=latency_stats.get('p99_ms', 0),
        latency_mean_ms=latency_stats.get('mean_ms', 0),
        throughput_fps=1000.0 / latency_stats.get('mean_ms', 1) if latency_stats.get('mean_ms', 0) > 0 else 0,
        ram_mean_mb=system_stats.get('ram_mean_mb', 0),
        ram_peak_mb=system_stats.get('ram_peak_mb', 0),
        cpu_mean_pct=system_stats.get('cpu_mean_pct', 0),
        cpu_peak_pct=system_stats.get('cpu_peak_pct', 0),
        gpu_mean_pct=system_stats.get('gpu_mean_pct', 0),
        gpu_peak_pct=system_stats.get('gpu_peak_pct', 0),
        gpu_mem_mean_mb=system_stats.get('gpu_mem_mean_mb', 0),
        model_size_mb=model_size_mb,
        cold_start_ms=engine.cold_start_ms or 0.0,
        total_windows=replay_stats['total_windows'],
        dropped_windows=replay_stats['dropped_windows'],
        drop_rate=replay_stats['drop_rate'],
        platform=get_platform_name(),
    )

    # Cleanup
    engine.close()

    return result


def setup_logging(level: str = "INFO", format_str: str = None, log_file: str = None):
    """Configure structured logging for the project."""
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Edge Replay Benchmark: replay-based inference benchmarking"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (e.g., configs/kws_onnx.yaml)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory from config")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override output dir if specified
    if args.output_dir:
        config['benchmark']['output_dir'] = args.output_dir

    # Setup logging early
    log_cfg = config.get('logging', {})
    setup_logging(
        level=log_cfg.get('level', 'INFO'),
        format_str=log_cfg.get('format'),
    )

    logger.info(f"Platform: {get_platform_name()}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Runtime: {config['runtime']['name']} ({config['runtime']['backend']})")

    # Run benchmark
    result = run_benchmark(config)

    # Save results
    output_dir = config['benchmark'].get('output_dir', 'benchmarks')
    metrics_logger = MetricsLogger(output_dir)
    metrics_logger.add_result(result)
    metrics_logger.print_summary_table()
    metrics_logger.to_csv()
    metrics_logger.to_json()
    metrics_logger.generate_summary_md()

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()