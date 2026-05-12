"""
Real-time MFCC feature extraction for keyword spotting.

Uses the same torchaudio MFCC transform and per-sample normalization
as the training pipeline in dataset.py to ensure feature consistency.
"""

import numpy as np
import torch
import torchaudio

# Must match dataset.py and params.yaml exactly
SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64
CLIP_LENGTH = 16000  # 1 second at 16kHz

# With center=False: time_frames = (clip_length - n_fft) // hop_length + 1 = 98
TIME_FRAMES = (CLIP_LENGTH - N_FFT) // HOP_LENGTH + 1  # 98


class RealtimeMFCC:
    """Extract MFCC features from raw audio, matching training pipeline.

    Produces (1, 1, 40, 98) tensors suitable for direct inference.
    Applies the same per-sample normalization as dataset.py:
        mfcc = (mfcc - mean) / (std + 1e-8)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 n_mfcc: int = N_MFCC,
                 device: str = 'cpu'):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.device = torch.device(device)

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