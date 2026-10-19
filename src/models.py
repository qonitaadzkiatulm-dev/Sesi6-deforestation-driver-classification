"""CNN kecil untuk klasifikasi driver dari tumpukan citra multi-pita (radar + optik)."""
import torch
import torch.nn as nn


def _blok(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """Seluruh pita ditumpuk menjadi satu citra multi-kanal (pendekatan sederhana seperti di materi).
    Model belajar sendiri kapan memakai radar atau optik, tanpa aturan eksplisit."""

    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.fitur = nn.Sequential(
            _blok(in_ch, 16), _blok(16, 16), nn.MaxPool2d(2),
            _blok(16, 32), _blok(32, 32), nn.MaxPool2d(2),
            _blok(32, 64),
        )
        self.kepala = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.kepala(self.fitur(x))
