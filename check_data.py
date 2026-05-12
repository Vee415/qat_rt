"""Quick dataset verification - just check split sizes, no MFCC."""
import os
import sys
sys.path.insert(0, 'src')

from dataset import TARGET_CLASSES
from pathlib import Path

root = Path('data/speech_commands')

val_file = root / "validation_list.txt"
test_file = root / "testing_list.txt"

class_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}

val_set = set()
test_set = set()
all_files = {}

with open(val_file) as f:
    val_set = {line.strip() for line in f if line.strip() and line.strip().split("/")[0] in class_to_idx}

with open(test_file) as f:
    test_set = {line.strip() for line in f if line.strip() and line.strip().split("/")[0] in class_to_idx}

for cls in TARGET_CLASSES:
    cls_dir = root / cls
    if cls_dir.exists():
        for wav_file in sorted(cls_dir.glob("*.wav")):
            rel_path = f"{cls}/{wav_file.name}"
            all_files[rel_path] = wav_file

train_set = set(all_files.keys()) - val_set - test_set

print(f"Total files: {len(all_files)}")
print(f"Train: {len(train_set)}")
print(f"Val: {len(val_set)}")
print(f"Test: {len(test_set)}")

# Quick single-file MFCC test
import torch
import torchaudio

waveform, sr = torchaudio.load(str(list(all_files.values())[0]))
print(f"\nSample audio: shape={waveform.shape}, sr={sr}")
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000, n_mfcc=40,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64, "center": False}
)
mfcc = mfcc_transform(waveform)
print(f"MFCC output: shape={mfcc.shape}")