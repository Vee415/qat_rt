"""
Audio dataset pipeline for Google Speech Commands v2.

Handles download, MFCC feature extraction, speaker-aware splits,
and DataLoader creation.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# Try torchaudio first; fall back to soundfile on platforms where
# torchaudio's C extension doesn't load (e.g., Jetson with NVIDIA wheels).
_TORCHAUDIO_AVAILABLE = False
try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    pass

if not _TORCHAUDIO_AVAILABLE:
    import soundfile as sf


# 10-class subset: yes/no/up/down/left/right/on/off/stop/go
TARGET_CLASSES = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]

SAMPLE_RATE = 16000
N_MFCC = 40
CLIP_LENGTH = 16000  # 1 second at 16kHz


class SpeechCommandsDataset(Dataset):
    """Google Speech Commands v2 with MFCC feature extraction."""

    def __init__(self, root: str, subset: str = "train", n_mfcc: int = N_MFCC,
                 clip_length: int = CLIP_LENGTH, augment: bool = False):
        self.root = Path(root)
        self.n_mfcc = n_mfcc
        self.clip_length = clip_length
        self.augment = augment and (subset == "train")
        self.classes = TARGET_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if _TORCHAUDIO_AVAILABLE:
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=SAMPLE_RATE,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": 400,
                    "hop_length": 160,
                    "n_mels": 64,
                    "center": False,
                },
            )
        else:
            from src.realtime_mfcc import _PurePytorchMFCC
            self.mfcc_transform = _PurePytorchMFCC(
                sample_rate=SAMPLE_RATE,
                n_mfcc=n_mfcc,
                n_fft=400,
                hop_length=160,
                n_mels=64,
                center=False,
            )

        self.file_list = []
        self.labels = []
        self._load_file_list(subset)

    def _load_file_list(self, subset: str):
        """Load file list for train/val/test split (speaker-aware)."""
        # Speech Commands v2 provides pre-defined train/val/test splits
        # train: training_list.txt, val: validation_list.txt, test: testing_list.txt
        # Speech Commands v2 has validation_list.txt and testing_list.txt
        # Training set = all files NOT in val or test
        val_file = self.root / "validation_list.txt"
        test_file = self.root / "testing_list.txt"

        val_set = set()
        test_set = set()

        if val_file.exists():
            with open(val_file) as f:
                val_set = {line.strip() for line in f if line.strip()
                           and line.strip().split("/")[0] in self.class_to_idx}

        if test_file.exists():
            with open(test_file) as f:
                test_set = {line.strip() for line in f if line.strip()
                            and line.strip().split("/")[0] in self.class_to_idx}

        # Build full file list per class
        all_files = {}
        for cls in self.classes:
            cls_dir = self.root / cls
            if cls_dir.exists():
                for wav_file in sorted(cls_dir.glob("*.wav")):
                    rel_path = f"{cls}/{wav_file.name}"
                    all_files[rel_path] = wav_file

        if subset == "val":
            file_set = val_set
        elif subset == "test":
            file_set = test_set
        else:
            file_set = set(all_files.keys()) - val_set - test_set

        for rel_path in sorted(file_set):
            if rel_path in all_files:
                self.file_list.append(all_files[rel_path])
                cls = rel_path.split("/")[0]
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.file_list)

    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad short clips or crop long clips to clip_length samples."""
        if waveform.shape[1] < self.clip_length:
            padding = self.clip_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > self.clip_length:
            start = torch.randint(0, waveform.shape[1] - self.clip_length, (1,)).item()
            waveform = waveform[:, start:start + self.clip_length]
        return waveform

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply training augmentations: time shift and noise injection."""
        if not self.augment:
            return waveform

        # Time shift: shift by up to 1600 samples (100ms)
        shift = torch.randint(-1600, 1600, (1,)).item()
        waveform = torch.roll(waveform, shifts=shift, dims=1)

        # Noise injection: add Gaussian noise at SNR ~20dB
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(waveform) * 0.01
            waveform = waveform + noise

        return waveform

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        label = self.labels[idx]

        # Load audio — torchaudio if available, soundfile fallback for Jetson
        if _TORCHAUDIO_AVAILABLE:
            waveform, sr = torchaudio.load(str(filepath))
        else:
            audio, sr = sf.read(str(filepath), dtype='float32')
            waveform = torch.from_numpy(audio)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # (1, n_samples)
            elif waveform.ndim == 2:
                waveform = waveform.T  # soundfile returns (n_samples, channels)

        # Resample if needed
        if sr != SAMPLE_RATE:
            if _TORCHAUDIO_AVAILABLE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
            else:
                # Simple linear interpolation resampling for Jetson fallback
                # Speech Commands dataset is already 16kHz, so this rarely triggers
                import torch.nn.functional as F
                waveform = waveform.unsqueeze(0)  # (1, 1, N) for interpolate
                new_len = int(waveform.shape[-1] * SAMPLE_RATE / sr)
                waveform = F.interpolate(waveform, size=new_len, mode='linear',
                                         align_corners=False)
                waveform = waveform.squeeze(0)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or crop to fixed length
        waveform = self._pad_or_crop(waveform)

        # Apply augmentations
        waveform = self._augment(waveform)

        # Extract MFCC features
        mfcc = self.mfcc_transform(waveform)  # (1, n_mfcc, time_frames)

        # Normalize per-sample
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        return mfcc, label


def download_speech_commands(root: str) -> str:
    """Download and extract Google Speech Commands v2."""
    root = Path(root)
    data_dir = root / "speech_commands"
    # Check for actual content, not just directory existence
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Dataset already exists at {data_dir}")
        return str(data_dir)

    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    archive_path = root / "speech_commands_v0.02.tar.gz"

    print(f"Downloading Speech Commands v2 from {url}...")
    urllib.request.urlretrieve(url, str(archive_path))

    print("Extracting...")
    data_dir.mkdir(parents=True, exist_ok=True)
    import tarfile
    with tarfile.open(str(archive_path), "r:gz") as tar:
        tar.extractall(str(data_dir))

    archive_path.unlink()
    print(f"Dataset ready at {data_dir}")
    return str(data_dir)


def get_dataloaders(data_dir: str, batch_size: int = 64,
                    num_workers: int = 4) -> dict:
    """Create train/val/test DataLoaders."""
    train_ds = SpeechCommandsDataset(data_dir, subset="train", augment=True)
    val_ds = SpeechCommandsDataset(data_dir, subset="val", augment=False)
    test_ds = SpeechCommandsDataset(data_dir, subset="test", augment=False)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True),
    }

    print(f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return loaders