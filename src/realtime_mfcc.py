"""
Real-time MFCC feature extraction for keyword spotting.

Uses torchaudio MFCC transform when available, falls back to
a pure PyTorch implementation on platforms where torchaudio's
C extension doesn't load (e.g., Jetson with NVIDIA PyTorch wheels).
"""

import numpy as np
import torch

# Must match dataset.py and params.yaml exactly
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64
CLIP_LENGTH = 16000  # 1 second at 16kHz

# With center=False: time_frames = (clip_length - n_fft) // hop_length + 1 = 98
TIME_FRAMES = (CLIP_LENGTH - N_FFT) // HOP_LENGTH + 1  # 98

# Try to import torchaudio; fall back to pure PyTorch if unavailable
_TORCHAUDIO_AVAILABLE = False
try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    pass


class _MelFilterBank(torch.nn.Module):
    """Pure PyTorch mel filterbank for Jetson fallback."""

    def __init__(self, sample_rate, n_fft, n_mels, f_min=0.0, f_max=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2.0

        # Create mel filterbank
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        bin_points = ((n_fft + 1) * hz_points / sample_rate).long()

        filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            f_left = bin_points[i]
            f_center = bin_points[i + 1]
            f_right = bin_points[i + 2]
            for j in range(f_left, f_right):
                if j < filterbank.shape[1]:
                    if j < f_center:
                        filterbank[i, j] = (j - f_left) / max(f_center - f_left, 1)
                    else:
                        filterbank[i, j] = (f_right - j) / max(f_right - f_center, 1)

        self.register_buffer('filterbank', filterbank)

    @staticmethod
    def _hz_to_mel(hz):
        return 2595.0 * torch.log10(torch.tensor(hz) / 700.0 + 1.0)

    @staticmethod
    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def forward(self, spectrogram):
        return torch.matmul(self.filterbank, spectrogram)


class _PurePytorchMFCC(torch.nn.Module):
    """Pure PyTorch MFCC extraction (no torchaudio dependency).

    Produces the same (n_mfcc, time_frames) output as torchaudio.transforms.MFCC
    with center=False, matching the training pipeline.
    """

    def __init__(self, sample_rate, n_mfcc, n_fft, hop_length, n_mels, center=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.mel_fb = _MelFilterBank(sample_rate, n_fft, n_mels)
        # DCT matrix for MFCC
        self.register_buffer('dct_matrix', self._create_dct_matrix(n_mfcc, n_mels))

    @staticmethod
    def _create_dct_matrix(n_mfcc, n_mels):
        n = torch.arange(n_mels).float()
        k = torch.arange(n_mfcc).float().unsqueeze(1)
        dct = torch.cos(torch.pi * k * (2 * n + 1) / (2 * n_mels))
        dct[0] *= 1.0 / torch.sqrt(torch.tensor(n_mels, dtype=torch.float32))
        dct[1:] *= torch.sqrt(torch.tensor(2.0 / n_mels, dtype=torch.float32))
        return dct

    def forward(self, waveform):
        # Compute STFT
        if self.center:
            waveform = torch.nn.functional.pad(
                waveform, (self.n_fft // 2, self.n_fft // 2))
        spec = torch.stft(
            waveform, self.n_fft, self.hop_length,
            window=torch.hann_window(self.n_fft).to(waveform.device),
            center=False, return_complex=True
        )
        power_spec = spec.abs().pow(2.0)

        # Apply mel filterbank
        mel_spec = self.mel_fb(power_spec)

        # Log mel spectrogram
        log_mel = torch.log(mel_spec + 1e-8)

        # Apply DCT to get MFCC
        mfcc = torch.matmul(self.dct_matrix, log_mel)

        return mfcc


class RealtimeMFCC:
    """Extract MFCC features from raw audio, matching training pipeline.

    Produces (1, 1, 40, 98) tensors suitable for direct inference.
    Applies the same per-sample normalization as dataset.py:
        mfcc = (mfcc - mean) / (std + 1e-8)

    Uses torchaudio when available, falls back to pure PyTorch on Jetson.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC,
                 device: str = 'cpu'):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.device = torch.device(device)

        if _TORCHAUDIO_AVAILABLE:
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": N_FFT,
                    "hop_length": HOP_LENGTH,
                    "n_mels": N_MELS,
                    "center": False,
                },
            ).to(self.device)
        else:
            self.mfcc_transform = _PurePytorchMFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                center=False,
            ).to(self.device)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from raw audio samples.

        Args:
            audio: float32 array of shape (n_samples,).
                   Should be 16000 samples (1 second at 16kHz).
                   If shorter, zero-padded; if longer, truncated to last CLIP_LENGTH.

        Returns:
            float32 array of shape (1, 1, 40, 98), per-sample normalized.
        """
        # Pad or truncate to exactly CLIP_LENGTH samples
        if len(audio) < CLIP_LENGTH:
            audio = np.pad(audio, (0, CLIP_LENGTH - len(audio)),
                           mode='constant', constant_values=0)
        elif len(audio) > CLIP_LENGTH:
            audio = audio[-CLIP_LENGTH:]

        # Convert to torch tensor: (1, n_samples) for torchaudio
        waveform = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        # Extract MFCC: output shape (1, n_mfcc, time_frames) = (1, 40, 98)
        with torch.no_grad():
            mfcc = self.mfcc_transform(waveform)

        # Per-sample normalization (same as dataset.py)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        # Reshape to (1, 1, 40, 98) for model input: (batch, channel, freq, time)
        mfcc = mfcc.unsqueeze(0)

        return mfcc.cpu().numpy()