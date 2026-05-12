"""
Audio capture module for real-time keyword spotting.

Uses sounddevice InputStream with a ring buffer to store the
last 1 second of 16kHz audio. A stride-based read interface
allows the inference loop to pull audio snapshots at a
configurable interval (default 100ms).
"""

import numpy as np
import sounddevice as sd
import threading
from typing import Optional

SAMPLE_RATE = 16000
BUFFER_SECONDS = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SECONDS)  # 16000 samples


class AudioRingBuffer:
    """Thread-safe circular buffer for audio samples.

    Written by the sounddevice callback thread, read by the
    inference thread. Stores the last `size` samples.
    """

    def __init__(self, size: int = BUFFER_SIZE):
        self._buffer = np.zeros(size, dtype=np.float32)
        self._write_pos = 0
        self._samples_written = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """Write audio data into the ring buffer (called from audio thread)."""
        with self._lock:
            n = len(data)
            if n >= self._buffer.size:
                self._buffer[:] = data[-self._buffer.size:]
                self._write_pos = 0
            else:
                end_pos = (self._write_pos + n) % self._buffer.size
                if end_pos > self._write_pos:
                    self._buffer[self._write_pos:end_pos] = data
                else:
                    first_chunk = self._buffer.size - self._write_pos
                    self._buffer[self._write_pos:] = data[:first_chunk]
                    self._buffer[:end_pos] = data[first_chunk:]
                self._write_pos = end_pos
            self._samples_written += n

    def read_last(self, n: int) -> np.ndarray:
        """Read the last n samples from the buffer (called from inference thread).

        Returns a contiguous copy of the most recent n samples.
        If fewer than n samples have been written, pads with zeros at the start.
        """
        with self._lock:
            available = min(n, self._samples_written, self._buffer.size)
            if available == 0:
                return np.zeros(n, dtype=np.float32)

            result = np.zeros(n, dtype=np.float32)

            if available >= self._buffer.size:
                # Buffer is full: oldest data starts at _write_pos, wraps around
                result[n - self._buffer.size:] = np.roll(self._buffer, -self._write_pos)
            else:
                # Read `available` samples ending at _write_pos
                start = (self._write_pos - available) % self._buffer.size
                if start + available <= self._buffer.size:
                    result[n - available:n] = self._buffer[start:start + available]
                else:
                    first_chunk = self._buffer.size - start
                    result[n - available:n - available + first_chunk] = self._buffer[start:]
                    result[n - available + first_chunk:n] = self._buffer[:available - first_chunk]
            return result


class AudioCapture:
    """Microphone audio capture using sounddevice InputStream.

    Args:
        sample_rate: Audio sample rate (default 16000)
        buffer_seconds: How many seconds of audio to buffer (default 1.0)
        device: sounddevice device index (default = system default)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 buffer_seconds: float = BUFFER_SECONDS,
                 device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_seconds)
        self._ring = AudioRingBuffer(self.buffer_size)
        self._stream = None
        self._device = device

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status: sd.CallbackFlags):
        """sounddevice callback: called from audio thread."""
        audio = indata[:, 0].astype(np.float32)
        self._ring.write(audio)

    def start(self) -> None:
        """Start microphone capture."""
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=int(self.sample_rate * 0.01),  # 10ms blocks
            device=self._device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop microphone capture."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_audio(self, n_samples: Optional[int] = None) -> np.ndarray:
        """Get the last n_samples of audio from the ring buffer.

        Args:
            n_samples: Number of samples to retrieve. Default = buffer_size (1 second).

        Returns:
            numpy array of shape (n_samples,) with float32 audio at sample_rate.
        """
        if n_samples is None:
            n_samples = self.buffer_size
        return self._ring.read_last(n_samples)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()