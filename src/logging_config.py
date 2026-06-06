"""
Structured logging setup for edge-replay-bench.

Replaces print() calls with Python logging module.
Configure via YAML config or function call.
"""

import logging
from typing import Optional


def setup_logging(level: str = "INFO", format_str: Optional[str] = None,
                  log_file: Optional[str] = None) -> None:
    """Configure structured logging for the project.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        format_str: Optional format string.
        log_file: Optional file path for log output.
    """
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

    # Reduce noise from third-party libraries
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    logging.getLogger("tensorrt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)