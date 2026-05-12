"""
CNN model for keyword spotting.

3-block CNN architecture designed for MFCC input (1 x 40 x T).
Simple enough to clearly demonstrate QAT effects.
"""

import torch
import torch.nn as nn


class KWSNet(nn.Module):
    """Simple CNN for keyword spotting on MFCC features.

    Architecture: 3 conv blocks (conv + bn + relu + pool) + classifier.
    Input: (batch, 1, n_mfcc, time_frames)
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes: int = 10, n_mfcc: int = 40):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class KWSNetQuantizable(KWSNet):
    """Same architecture with QuantStub/DeQuantStub and fusion support."""

    def __init__(self, num_classes: int = 10, n_mfcc: int = 40):
        super().__init__(num_classes, n_mfcc)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv+BN+ReLU layers for quantization."""
        torch.ao.quantization.fuse_modules(self, [
            ['features.0', 'features.1', 'features.2'],  # Conv1+BN1+ReLU1
            ['features.4', 'features.5', 'features.6'],  # Conv2+BN2+ReLU2
            ['features.8', 'features.9', 'features.10'],  # Conv3+BN3+ReLU3
        ], inplace=True)


def get_model(num_classes: int = 10, quantizable: bool = False) -> nn.Module:
    """Create a fresh model instance."""
    if quantizable:
        return KWSNetQuantizable(num_classes=num_classes)
    return KWSNet(num_classes=num_classes)