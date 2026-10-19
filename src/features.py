"""Fitur buatan tangan untuk baseline machine learning konvensional (Random Forest).

Sesuai saran di seminar: selalu sediakan benchmark metode konvensional sebelum memakai deep learning.
Fitur memakai bentuk alert (ukuran, kelonjongan, jumlah komponen) dan kontras spektral di dalam vs di luar alert.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from config import MASK_CH

NAMA_FITUR = (
    ["luas", "extent_bbox", "elongasi", "rasio_keliling_luas", "n_komponen", "frak_komponen_terbesar"]
    + [f"{p}_{b}" for b in ["VV", "VH", "B2", "B3", "B4", "B8"] for p in ("dalam", "luar", "selisih")]
    + ["ndvi_dalam", "vh_minus_vv_dalam", "std_B8_dalam"]
)


def _fitur_satu(patch: np.ndarray) -> np.ndarray:
    mask = patch[MASK_CH] > 0.5
    n = mask.sum()
    ys, xs = np.nonzero(mask)

    luas = n / mask.size
    bbox = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) if n else 1
    extent = n / bbox
    if n > 2:
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([ys, xs]))))
        elong = float(np.sqrt((ev[1] + 1e-6) / (ev[0] + 1e-6)))
    else:
        elong = 1.0
    keliling = (mask & ~ndi.binary_erosion(mask)).sum()
    lab, ncomp = ndi.label(mask)
    terbesar = np.bincount(lab.ravel())[1:].max() / n if ncomp else 0.0

    spektral = []
    for b in range(6):
        dalam = patch[b][mask].mean() if n else 0.0
        luar = patch[b][~mask].mean() if (~mask).any() else 0.0
        spektral += [dalam, luar, dalam - luar]
    b4, b8 = patch[4][mask].mean(), patch[5][mask].mean()
    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    vh_vv = patch[1][mask].mean() - patch[0][mask].mean()
    std_b8 = patch[5][mask].std()
    return np.array([luas, extent, elong, keliling / max(n, 1), ncomp, terbesar, *spektral, ndvi, vh_vv, std_b8], np.float32)


def ekstrak_fitur(X: np.ndarray) -> np.ndarray:
    return np.stack([_fitur_satu(p) for p in X])
