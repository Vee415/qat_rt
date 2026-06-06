"""
File-based replay engine for edge inference benchmarking.

Reads .wav files from a directory (e.g., Google Speech Commands test set),
pads/truncates to a fixed clip length, and yields audio windows at a
configurable real-time cadence. Tracks dropped windows where inference
cannot keep up with the target FPS.

Designed for Jetson Orin Nano Super as primary target, also works on x86.
"""

import logging
import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Must match dataset.py and realtime_mfcc.py
SAMPLE_RATE = 16000
CLIP_LENGTH = 16000  # 1 second at 16kHz


class FileReplayEngine:
    """Replay .wav files at real-time rates for benchmarking.

    Scans a data directory for .wav files organized by class subdirectories,
    loads each file, resamples to 16kHz mono if needed, pads/truncates to
    CLIP_LENGTH samples, and yields audio windows at a configurable FPS.

    A window is "dropped" when the consumer (preprocessing + inference)
    takes longer than 1/target_fps seconds to process it, meaning the
    system cannot keep up with the simulated sensor rate.

    Args:
        data_dir: Path to dataset root (e.g., data/speech_commands).
                  Expects class subdirectories containing .wav files.
        target_fps: Target inference windows per second (default 10 = 100ms stride).
        sample_rate: Audio sample rate (default 16000).
        clip_length: Number of audio samples per window (default 16000 = 1 second).
        shuffle: Randomize file order for realistic measurement (default True).
        max_files: Limit total files for quick runs (None = all test files).
        seed: Random seed for reproducible shuffling.
        speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed).
               Use speed > 1.0 for stress testing (more dropped windows expected).
    """

    def __init__(
        self,
        data_dir: str,
        target_fps: float = 10.0,
        sample_rate: int = SAMPLE_RATE,
        clip_length: int = CLIP_LENGTH,
        shuffle: bool = True,
        max_files: Optional[int] = None,
        seed: int = 42,
        speed: float = 1.0,
    ):
        self.data_dir = Path(data_dir)
        self.target_fps = target_fps
        self.sample_rate = sample_rate
        self.clip_length = clip_length
        self.shuffle = shuffle
        self.max_files = max_files
        self.seed = seed
        self.speed = speed

        # Stats
        self._total_windows = 0
        self._dropped_windows = 0
        self._last_yield_time = None

        # Scan files
        self._files = self._scan_files()
        if not self._files:
            raise FileNotFoundError(
                f"No .wav files found in {self.data_dir}/<class>/*.wav"
            )

        logger.info(
            f"FileReplayEngine: {len(self._files)} files from {self.data_dir}, "
            f"target_fps={target_fps}, speed={speed}x"
        )

    def _scan_files(self) -> list:
        """Scan data_dir for .wav files in class subdirectories.

        Returns list of (path, label) tuples where label is the class index.
        Uses the same class order as dataset.py: yes, no, up, down, left,
        right, on, off, stop, go.
        """
        target_classes = [
            "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
        ]
        class_to_idx = {c: i for i, c in enumerate(target_classes)}

        files = []
        for cls in target_classes:
            cls_dir = self.data_dir / cls
            if cls_dir.is_dir():
                for wav_file in sorted(cls_dir.glob("*.wav")):
                    files.append((wav_file, class_to_idx[cls]))

        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(files)

        if self.max_files and len(files) > self.max_files:
            files = files[:self.max_files]

        return files

    def _load_audio(self, filepath: Path) -> np.ndarray:
        """Load a .wav file and return float32 mono audio at target sample rate.

        Uses torchaudio for loading when available; falls back to soundfile
        on platforms where torchaudio's C extension doesn't load (e.g., Jetson).
        """
        # Try torchaudio first (consistent with training pipeline)
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(filepath))

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to numpy float32
            audio = waveform.squeeze(0).numpy().astype(np.float32)
        except (ImportError, OSError):
            # Fallback: soundfile (Jetson without torchaudio)
            import soundfile as sf
            audio, sr = sf.read(str(filepath), dtype='float32')

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample if needed (linear interpolation — dataset is 16kHz already)
            if sr != self.sample_rate:
                import torch
                import torch.nn.functional as F
                t = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
                new_len = int(t.shape[-1] * self.sample_rate / sr)
                t = F.interpolate(t, size=new_len, mode='linear', align_corners=False)
                audio = t.squeeze().numpy().astype(np.float32)

        # Pad or truncate to clip_length
        if len(audio) < self.clip_length:
            audio = np.pad(audio, (0, self.clip_length - len(audio)),
                           mode='constant', constant_values=0)
        elif len(audio) > self.clip_length:
            # Take the center portion (matching dataset.py training behavior)
            start = (len(audio) - self.clip_length) // 2
            audio = audio[start:start + self.clip_length]

        return audio

    def replay(self) -> Generator[tuple, None, None]:
        """Yield (audio, metadata) tuples at real-time cadence.

        Each yield is spaced by 1/(target_fps * speed) seconds. If the
        consumer takes longer than that, the window is marked as "dropped"
        and the next window is yielded immediately (no sleep).

        Yields:
            (audio_array, metadata_dict) where:
                audio_array: float32 array of shape (clip_length,) at sample_rate
                metadata_dict: {'path': str, 'label': int, 'window_idx': int,
                               'dropped': bool, 'deadline_ms': float}
        """
        window_deadline_s = 1.0 / (self.target_fps * self.speed)

        for idx, (filepath, label) in enumerate(self._files):
            window_start = time.perf_counter()

            # Check if previous window was dropped (consumer was too slow)
            dropped = False
            if self._last_yield_time is not None:
                elapsed = window_start - self._last_yield_time
                if elapsed > window_deadline_s:
                    dropped = True
                    self._dropped_windows += 1
                    # No sleep — catch up immediately

            # Load and preprocess audio
            try:
                audio = self._load_audio(filepath)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
                continue

            metadata = {
                'path': str(filepath),
                'label': label,
                'window_idx': idx,
                'dropped': dropped,
                'deadline_ms': window_deadline_s * 1000,
            }

            self._total_windows += 1
            yield audio, metadata

            # Sleep to maintain target FPS (only if not behind)
            elapsed = time.perf_counter() - window_start
            remaining = window_deadline_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

            self._last_yield_time = time.perf_counter()

    def replay_no_timing(self) -> Generator[tuple, None, None]:
        """Yield (audio, metadata) without real-time pacing.

        Used for accuracy benchmarks where timing doesn't matter —
        just process all files as fast as possible.
        """
        for idx, (filepath, label) in enumerate(self._files):
            try:
                audio = self._load_audio(filepath)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
                continue

            metadata = {
                'path': str(filepath),
                'label': label,
                'window_idx': idx,
                'dropped': False,
                'deadline_ms': 0,
            }

            self._total_windows += 1
            yield audio, metadata

    def get_stats(self) -> dict:
        """Return replay statistics.

        Returns:
            dict with total_windows, dropped_windows, drop_rate.
        """
        drop_rate = (
            self._dropped_windows / self._total_windows
            if self._total_windows > 0 else 0.0
        )
        return {
            'total_windows': self._total_windows,
            'dropped_windows': self._dropped_windows,
            'drop_rate': drop_rate,
        }

    def reset_stats(self) -> None:
        """Reset replay statistics for a fresh run."""
        self._total_windows = 0
        self._dropped_windows = 0
        self._last_yield_time = None

    def __len__(self) -> int:
        """Number of files available for replay."""
        return len(self._files)