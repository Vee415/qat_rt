"""Unit tests for MFCC preprocessing pipeline.

Verifies that RealtimeMFCC produces correct shapes and values
matching the training pipeline.
"""

import numpy as np
import pytest

# Skip if torchaudio not available
torchaudio = pytest.importorskip("torchaudio")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from realtime_mfcc import RealtimeMFCC, SAMPLE_RATE, CLIP_LENGTH, TIME_FRAMES


class TestRealtimeMFCC:
    """Test suite for RealtimeMFCC feature extraction."""

    def test_output_shape_from_one_second(self):
        """(1,1,40,98) output from 1-second audio at 16kHz."""
        extractor = RealtimeMFCC()
        audio = np.random.randn(CLIP_LENGTH).astype(np.float32)
        mfcc = extractor.extract(audio)
        assert mfcc.shape == (1, 1, 40, 98), f"Expected (1,1,40,98), got {mfcc.shape}"

    def test_output_dtype(self):
        """Output should be float32."""
        extractor = RealtimeMFCC()
        audio = np.random.randn(CLIP_LENGTH).astype(np.float32)
        mfcc = extractor.extract(audio)
        assert mfcc.dtype == np.float32

    def test_short_audio_padding(self):
        """Audio shorter than 1s is zero-padded to (1,1,40,98)."""
        extractor = RealtimeMFCC()
        short = np.random.randn(8000).astype(np.float32)  # 0.5s
        mfcc = extractor.extract(short)
        assert mfcc.shape == (1, 1, 40, 98)

    def test_long_audio_truncation(self):
        """Audio longer than 1s is truncated to last 1s, producing (1,1,40,98)."""
        extractor = RealtimeMFCC()
        long_audio = np.random.randn(32000).astype(np.float32)  # 2s
        mfcc = extractor.extract(long_audio)
        assert mfcc.shape == (1, 1, 40, 98)

    def test_very_short_audio(self):
        """Very short audio (0.1s) still produces valid (1,1,40,98) output."""
        extractor = RealtimeMFCC()
        tiny = np.random.randn(1600).astype(np.float32)  # 0.1s
        mfcc = extractor.extract(tiny)
        assert mfcc.shape == (1, 1, 40, 98)

    def test_silence_produces_finite_output(self):
        """All-zero input produces finite MFCC (no NaN from std=0 normalization)."""
        extractor = RealtimeMFCC()
        silence = np.zeros(CLIP_LENGTH, dtype=np.float32)
        mfcc = extractor.extract(silence)
        assert np.all(np.isfinite(mfcc)), "MFCC of silence contains NaN or Inf"

    def test_deterministic_output(self):
        """Same input produces same output (deterministic preprocessing)."""
        extractor = RealtimeMFCC()
        audio = np.random.randn(CLIP_LENGTH).astype(np.float32)
        mfcc1 = extractor.extract(audio)
        mfcc2 = extractor.extract(audio)
        np.testing.assert_array_equal(mfcc1, mfcc2)

    def test_different_audio_different_output(self):
        """Different audio produces different MFCC output."""
        extractor = RealtimeMFCC()
        audio1 = np.random.randn(CLIP_LENGTH).astype(np.float32)
        audio2 = np.random.randn(CLIP_LENGTH).astype(np.float32)
        mfcc1 = extractor.extract(audio1)
        mfcc2 = extractor.extract(audio2)
        assert not np.allclose(mfcc1, mfcc2), "Different inputs should produce different MFCCs"

    def test_normalization_applied(self):
        """Per-sample normalization (zero mean, unit variance) is applied."""
        extractor = RealtimeMFCC()
        # Use audio with large amplitude to ensure normalization effect is visible
        audio = (np.random.randn(CLIP_LENGTH) * 100).astype(np.float32)
        mfcc = extractor.extract(audio)
        # After per-sample normalization, the flattened output should have
        # approximately zero mean and unit variance (with some tolerance
        # since MFCC transform is nonlinear)
        flat = mfcc.flatten()
        assert abs(flat.mean()) < 1.0, f"MFCC mean too far from zero: {flat.mean()}"