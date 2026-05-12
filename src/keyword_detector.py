"""
Keyword detection with post-processing for real-time inference.

Applies softmax to raw logits, thresholds by confidence,
enforces a cooldown period to prevent repeated triggers from
the same utterance, and maps class indices to keyword strings.
"""

import time
import numpy as np
from typing import Optional, Tuple

# Must match dataset.py TARGET_CLASSES exactly
KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

NO_KEYWORD = "---"


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


class KeywordDetector:
    """Post-processor for keyword spotting model output.

    Args:
        threshold: Minimum softmax probability to accept a detection (default 0.7)
        cooldown_ms: Minimum milliseconds between detections (default 500)
        keywords: List of keyword strings indexed by class order
    """

    def __init__(self, threshold: float = 0.7,
                 cooldown_ms: float = 500.0,
                 keywords: list = None):
        self.threshold = threshold
        self.cooldown_ms = cooldown_ms
        self.keywords = keywords or KEYWORDS
        self._last_detection_time = 0.0

    def detect(self, logits: np.ndarray) -> Tuple[Optional[str], float]:
        """Process inference output and return detected keyword.

        Args:
            logits: Raw model output of shape (1, 10) or (10,)

        Returns:
            Tuple of (keyword_string, confidence_float).
            keyword_string is one of KEYWORDS or NO_KEYWORD ("---").
        """
        if logits.ndim == 2:
            logits = logits[0]

        probs = softmax(logits)
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        if confidence < self.threshold:
            return NO_KEYWORD, confidence

        # Cooldown check
        now = time.perf_counter() * 1000  # ms
        elapsed = now - self._last_detection_time
        if elapsed < self.cooldown_ms:
            return NO_KEYWORD, confidence

        self._last_detection_time = now
        keyword = self.keywords[max_idx]
        return keyword, confidence

    def reset(self) -> None:
        """Reset cooldown state."""
        self._last_detection_time = 0.0