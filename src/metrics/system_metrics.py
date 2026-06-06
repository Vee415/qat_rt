"""
System metrics collector for edge inference benchmarking.

Collects RAM, CPU, and GPU utilization in a background thread.
Designed for Jetson Orin Nano Super as primary target (jtop/tegrastats),
with x86 fallback (nvidia-smi).

On Jetson:
  - CPU/RAM: psutil (works on Linux)
  - GPU: jtop (jetson-stats package) or tegrastats subprocess
  - Note: nvidia-smi does NOT report GPU utilization on Jetson

On x86 (dev machine with NVIDIA GPU):
  - CPU/RAM: psutil
  - GPU: nvidia-smi subprocess
"""

import logging
import os
import platform
import subprocess
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def is_jetson() -> bool:
    """Detect if running on a Jetson device.

    Checks for /etc/nv_tegra_release which is present on all Jetson modules.
    """
    return os.path.exists("/etc/nv_tegra_release")


class SystemMetricsCollector:
    """Collect system metrics (RAM, CPU, GPU) in a background thread.

    Samples metrics at a configurable interval. Call start() before
    benchmarking and stop() after. Call get_summary() for aggregated stats.

    Args:
        interval_ms: Sampling interval in milliseconds (default 100).
    """

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._samples = []
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._jetson = is_jetson()

    def _collect_sample(self) -> dict:
        """Collect one sample of system metrics."""
        import psutil

        sample = {
            'timestamp': time.perf_counter(),
            'ram_used_mb': psutil.virtual_memory().used / (1024 * 1024),
            'ram_total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'cpu_pct': psutil.cpu_percent(interval=None),
        }

        # GPU metrics: platform-specific
        if self._jetson:
            gpu_stats = self._collect_jetson_gpu()
        else:
            gpu_stats = self._collect_nvidia_smi()

        sample.update(gpu_stats)
        return sample

    def _collect_jetson_gpu(self) -> dict:
        """Collect GPU metrics on Jetson via tegrastats (fast, reliable).

        Uses tegrastats as primary method since jtop's context manager
        is too slow for repeated sampling and can hang on some JetPack versions.
        """
        stats = {
            'gpu_pct': 0.0,
            'gpu_mem_mb': 0.0,
        }

        # Primary: tegrastats subprocess (fast, no context manager overhead)
        try:
            result = subprocess.run(
                ['tegrastats', '--interval', '1'],
                capture_output=True, text=True, timeout=2
            )
            # tegrastats output: "RAM 1234/8192MB (lfb 5432MB) CPU [12%@1234] GPU [34%@123MHz]"
            for line in result.stdout.strip().split('\n'):
                if 'GPU' in line:
                    import re
                    gpu_match = re.search(r'GPU \[(\d+)%', line)
                    if gpu_match:
                        stats['gpu_pct'] = float(gpu_match.group(1))
                    # Also extract RAM used from tegrastats
                    ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
                    if ram_match:
                        stats['gpu_mem_mb'] = float(ram_match.group(1))
                    break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return stats

    def _collect_nvidia_smi(self) -> dict:
        """Collect GPU metrics on x86 via nvidia-smi subprocess."""
        stats = {
            'gpu_pct': 0.0,
            'gpu_mem_mb': 0.0,
        }

        try:
            result = subprocess.run(
                [
                    'nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits',
                ],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    stats['gpu_pct'] = float(parts[0].strip())
                    stats['gpu_mem_mb'] = float(parts[1].strip())
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        return stats

    def start(self) -> None:
        """Start background metrics collection."""
        self._samples = []
        self._running.set()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        logger.info(f"SystemMetricsCollector started (interval={self.interval_ms}ms, "
                     f"jetson={self._jetson})")

    def stop(self) -> None:
        """Stop background metrics collection and join thread."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(f"SystemMetricsCollector stopped ({len(self._samples)} samples collected)")

    def _collect_loop(self) -> None:
        """Background loop that collects samples at the configured interval."""
        while self._running.is_set():
            try:
                sample = self._collect_sample()
                self._samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to collect metrics sample: {e}")

            # Sleep in small increments to allow responsive stop()
            sleep_interval = self.interval_ms / 1000.0
            elapsed = 0
            while elapsed < sleep_interval and self._running.is_set():
                time.sleep(min(0.01, sleep_interval - elapsed))
                elapsed += 0.01

    def get_summary(self) -> dict:
        """Compute summary statistics from collected samples.

        Returns:
            dict with mean and peak values for RAM, CPU, GPU metrics.
        """
        if not self._samples:
            return {
                'ram_mean_mb': 0, 'ram_peak_mb': 0,
                'cpu_mean_pct': 0, 'cpu_peak_pct': 0,
                'gpu_mean_pct': 0, 'gpu_peak_pct': 0,
                'gpu_mem_mean_mb': 0,
                'n_samples': 0,
            }

        ram = np.array([s['ram_used_mb'] for s in self._samples])
        cpu = np.array([s['cpu_pct'] for s in self._samples])
        gpu = np.array([s['gpu_pct'] for s in self._samples])
        gpu_mem = np.array([s['gpu_mem_mb'] for s in self._samples])

        return {
            'ram_mean_mb': float(ram.mean()),
            'ram_peak_mb': float(ram.max()),
            'cpu_mean_pct': float(cpu.mean()),
            'cpu_peak_pct': float(cpu.max()),
            'gpu_mean_pct': float(gpu.mean()),
            'gpu_peak_pct': float(gpu.max()),
            'gpu_mem_mean_mb': float(gpu_mem.mean()),
            'n_samples': len(self._samples),
        }

    def get_current(self) -> dict:
        """Get a single sample of current system metrics (non-threaded)."""
        return self._collect_sample()