"""
Benchmark result data structures and metrics logging.

Provides BenchmarkResult dataclass for structured result storage,
and MetricsLogger for CSV/JSON/markdown output.
"""

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark run result."""

    runtime: str = ""              # e.g., "onnx_cpu", "trt_fp16", "trt_int8"
    precision: str = ""            # e.g., "fp32", "fp16", "int8"
    accuracy: float = 0.0         # top-1 accuracy on test set
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0
    throughput_fps: float = 0.0    # inferences per second
    ram_mean_mb: float = 0.0
    ram_peak_mb: float = 0.0
    cpu_mean_pct: float = 0.0
    cpu_peak_pct: float = 0.0
    gpu_mean_pct: float = 0.0
    gpu_peak_pct: float = 0.0
    gpu_mem_mean_mb: float = 0.0
    model_size_mb: float = 0.0
    cold_start_ms: float = 0.0    # engine load + warmup time
    total_windows: int = 0
    dropped_windows: int = 0
    drop_rate: float = 0.0
    platform: str = ""             # e.g., "Jetson Orin Nano Super" or "x86 RTX 4060"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MetricsLogger:
    """Accumulate benchmark results and write to CSV/JSON/markdown.

    Args:
        output_dir: Directory for output files (default: "benchmarks").
    """

    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)
        logger.info(
            f"[{result.runtime}] p99={result.latency_p99_ms:.2f}ms, "
            f"throughput={result.throughput_fps:.0f}fps, "
            f"accuracy={result.accuracy:.4f}, "
            f"dropped={result.dropped_windows}/{result.total_windows} "
            f"({result.drop_rate:.2%})"
        )

    def to_csv(self, filename: str = "results.csv") -> Path:
        """Write results to CSV file."""
        if not self.results:
            logger.warning("No results to write to CSV")
            return self.output_dir / filename

        filepath = self.output_dir / filename
        fieldnames = list(asdict(self.results[0]).keys())

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))

        logger.info(f"Results written to {filepath}")
        return filepath

    def to_json(self, filename: str = "results.json") -> Path:
        """Write results to JSON file."""
        if not self.results:
            logger.warning("No results to write to JSON")
            return self.output_dir / filename

        filepath = self.output_dir / filename
        data = [asdict(r) for r in self.results]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results written to {filepath}")
        return filepath

    def print_summary_table(self) -> None:
        """Print a formatted summary table to the logger."""
        if not self.results:
            logger.info("No results to summarize")
            return

        # Header
        fmt = "{:<15} {:<8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10} {:>8}"
        header = fmt.format(
            "Runtime", "Prec", "Acc%", "p50(ms)", "p95(ms)", "p99(ms)",
            "Thr(fps)", "Drop%", "RAM(MB)", "Cold(ms)"
        )
        logger.info("\n" + "=" * len(header))
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * len(header))
        logger.info(header)
        logger.info("-" * len(header))

        for r in self.results:
            line = fmt.format(
                r.runtime, r.precision,
                f"{r.accuracy*100:.2f}",
                f"{r.latency_p50_ms:.2f}", f"{r.latency_p95_ms:.2f}",
                f"{r.latency_p99_ms:.2f}", f"{r.throughput_fps:.0f}",
                f"{r.drop_rate:.2%}",
                f"{r.ram_mean_mb:.0f}",
                f"{r.cold_start_ms:.0f}",
            )
            logger.info(line)

        logger.info("=" * len(header))

    def generate_summary_md(self, filename: str = "summary.md") -> Path:
        """Generate a markdown summary with plain-English interpretation."""
        if not self.results:
            logger.warning("No results to summarize")
            return self.output_dir / filename

        filepath = self.output_dir / filename

        lines = [
            "# Edge Inference Benchmark Summary",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Platform:** {self.results[0].platform}",
            "",
            "## Results",
            "",
        ]

        # Results table
        lines.append(
            "| Runtime | Precision | Accuracy | p50 (ms) | p95 (ms) | "
            "p99 (ms) | Throughput | Dropped | RAM (MB) | Cold Start (ms) |"
        )
        lines.append(
            "|---------|-----------|----------|----------|----------|----------"
            "|-----------|---------|---------|-----------------|"
        )

        for r in self.results:
            lines.append(
                f"| {r.runtime} | {r.precision} | "
                f"{r.accuracy*100:.2f}% | "
                f"{r.latency_p50_ms:.2f} | {r.latency_p95_ms:.2f} | "
                f"{r.latency_p99_ms:.2f} | {r.throughput_fps:.0f} fps | "
                f"{r.dropped_windows}/{r.total_windows} ({r.drop_rate:.1%}) | "
                f"{r.ram_mean_mb:.0f} | {r.cold_start_ms:.0f} |"
            )

        lines.append("")

        # Best performers
        best_p99 = min(self.results, key=lambda r: r.latency_p99_ms)
        best_accuracy = max(self.results, key=lambda r: r.accuracy)
        best_throughput = max(self.results, key=lambda r: r.throughput_fps)
        smallest = min(self.results, key=lambda r: r.model_size_mb)

        lines.extend([
            "## Key Findings",
            "",
            f"- **Best latency (p99):** {best_p99.runtime} ({best_p99.precision}) "
            f"at {best_p99.latency_p99_ms:.2f}ms",
            f"- **Best accuracy:** {best_accuracy.runtime} ({best_accuracy.precision}) "
            f"at {best_accuracy.accuracy*100:.2f}%",
            f"- **Best throughput:** {best_throughput.runtime} ({best_throughput.precision}) "
            f"at {best_throughput.throughput_fps:.0f} fps",
            f"- **Smallest model:** {smallest.runtime} ({smallest.precision}) "
            f"at {smallest.model_size_mb:.2f} MB",
        ])

        # Dropped windows analysis
        any_dropped = any(r.dropped_windows > 0 for r in self.results)
        if any_dropped:
            lines.extend([
                "",
                "## Dropped Window Analysis",
                "",
                "Some runtimes could not keep up with the target FPS:",
                "",
            ])
            for r in self.results:
                if r.dropped_windows > 0:
                    lines.append(
                        f"- **{r.runtime} ({r.precision}):** "
                        f"{r.dropped_windows}/{r.total_windows} windows dropped "
                        f"({r.drop_rate:.1%}) — p99 latency {r.latency_p99_ms:.2f}ms "
                        f"exceeds deadline"
                    )
        else:
            lines.extend([
                "",
                "## Dropped Window Analysis",
                "",
                "All runtimes kept up with the target FPS — zero dropped windows. "
                "This is expected for the KWS model (94K params) which is very lightweight.",
            ])

        # Deployment recommendations
        lines.extend([
            "",
            "## Deployment Recommendations",
            "",
        ])

        # Find best for different scenarios
        onnx_results = [r for r in self.results if 'onnx' in r.runtime.lower()]
        trt_results = [r for r in self.results if 'trt' in r.runtime.lower()]

        if onnx_results:
            best_onnx = min(onnx_results, key=lambda r: r.latency_p99_ms)
            lines.append(
                f"- **CPU-only deployment:** {best_onnx.runtime} — "
                f"p99 latency {best_onnx.latency_p99_ms:.2f}ms, "
                f"accuracy {best_onnx.accuracy*100:.2f}%"
            )

        if trt_results:
            best_trt = min(trt_results, key=lambda r: r.latency_p99_ms)
            lines.append(
                f"- **GPU deployment (Jetson/RTX):** {best_trt.runtime} — "
                f"p99 latency {best_trt.latency_p99_ms:.2f}ms, "
                f"accuracy {best_trt.accuracy*100:.2f}%"
            )

        lines.append(
            f"- **Accuracy-critical:** {best_accuracy.runtime} ({best_accuracy.precision}) — "
            f"best accuracy at {best_accuracy.accuracy*100:.2f}%"
        )
        lines.append(
            f"- **Memory-constrained:** {smallest.runtime} ({smallest.precision}) — "
            f"smallest model at {smallest.model_size_mb:.2f} MB"
        )

        content = "\n".join(lines)
        with open(filepath, 'w') as f:
            f.write(content)

        logger.info(f"Summary written to {filepath}")
        return filepath