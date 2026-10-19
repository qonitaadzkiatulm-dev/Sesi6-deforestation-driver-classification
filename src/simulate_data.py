"""Simulator patch citra satelit untuk klasifikasi driver deforestasi.

PENTING: ini DATA SINTETIS. Bentuk spasial dan tanda spektral dirancang meniru deskripsi di materi seminar
(jalan = garis tipis, logging selektif = celah kecil di sepanjang jalur, pertanian skala besar = petak
lurus besar, tambang = area sangat terang + kolam, kebakaran = bekas bakar gelap, banjir = pita di sepanjang
sungai dengan backscatter radar rendah). Hasil model pada data ini TIDAK mewakili akurasi pada citra Sentinel nyata.

Jalankan:  python src/simulate_data.py --n-per-class 300 --out data/patches.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from config import BANDS, CLASSES, MASK_CH, N_BANDS, PATCH, WILAYAH

# ---------------------------------------------------------------- tanda spektral rata-rata
# Urutan: VV(dB), VH(dB), B2, B3, B4, B8 (reflektansi 0-1)
HUTAN = np.array([-7.0, -13.0, 0.03, 0.05, 0.03, 0.32])
SIGNATURE = {
    "pertanian_skala_kecil": np.array([-10.0, -16.5, 0.08, 0.11, 0.13, 0.24]),
    "pertanian_skala_besar": np.array([-10.5, -17.0, 0.07, 0.10, 0.12, 0.30]),
    "jalan": np.array([-9.5, -16.0, 0.10, 0.13, 0.16, 0.24]),
    "logging_selektif": np.array([-8.5, -14.5, 0.05, 0.07, 0.07, 0.27]),
    "tambang": np.array([-12.0, -19.0, 0.14, 0.17, 0.19, 0.21]),
    "kebakaran": np.array([-8.5, -14.5, 0.03, 0.04, 0.05, 0.14]),
    "banjir": np.array([-15.0, -21.0, 0.05, 0.06, 0.04, 0.09]),
}
# Pasangan kelas yang sengaja dibuat mirip (sumber kebingungan, seperti dibahas di materi)
CONFUSER = {
    "kebakaran": "banjir",
    "banjir": "kebakaran",
    "pertanian_skala_kecil": "pertanian_skala_besar",
    "pertanian_skala_besar": "pertanian_skala_kecil",
    "jalan": "logging_selektif",
    "logging_selektif": "jalan",
    "tambang": "pertanian_skala_besar",
}
# Peluang wilayah per kelas (kasar, mengikuti pola yang diceritakan di materi)
WILAYAH_BOBOT = {
    "pertanian_skala_kecil": [0.15, 0.65, 0.20],
    "pertanian_skala_besar": [0.65, 0.05, 0.30],
    "jalan": [0.35, 0.30, 0.35],
    "logging_selektif": [0.15, 0.45, 0.40],
    "tambang": [0.25, 0.25, 0.50],
    "kebakaran": [0.70, 0.15, 0.15],
    "banjir": [0.40, 0.30, 0.30],
}


# ---------------------------------------------------------------- utilitas geometri
def _smooth_noise(rng, size, sigma):
    n = ndi.gaussian_filter(rng.normal(size=(size, size)), sigma, mode="reflect")
    return (n - n.mean()) / (n.std() + 1e-9)


def _grid(size):
    yy, xx = np.mgrid[0:size, 0:size]
    return yy.astype(float), xx.astype(float)


def _dist_polyline(points, size):
    yy, xx = _grid(size)
    d = np.full((size, size), np.inf)
    for (y0, x0), (y1, x1) in zip(points[:-1], points[1:]):
        vy, vx = y1 - y0, x1 - x0
        t = np.clip(((yy - y0) * vy + (xx - x0) * vx) / (vy * vy + vx * vx + 1e-9), 0, 1)
        d = np.minimum(d, np.hypot(yy - (y0 + t * vy), xx - (x0 + t * vx)))
    return d


def _random_polyline(rng, size, n_mid=1, jitter=5.0):
    if rng.random() < 0.5:
        p0, p1 = (0.0, rng.uniform(4, size - 4)), (size - 1.0, rng.uniform(4, size - 4))
    else:
        p0, p1 = (rng.uniform(4, size - 4), 0.0), (rng.uniform(4, size - 4), size - 1.0)
    pts = [p0]
    for i in range(1, n_mid + 1):
        f = i / (n_mid + 1)
        pts.append(
            (p0[0] + (p1[0] - p0[0]) * f + rng.normal(0, jitter), p0[1] + (p1[1] - p0[1]) * f + rng.normal(0, jitter))
        )
    pts.append(p1)
    return pts


def _blob(rng, size, center, radius, irregular=0.3):
    yy, xx = _grid(size)
    noise = _smooth_noise(rng, size, 2.0)
    return np.hypot(yy - center[0], xx - center[1]) < radius * (1 + irregular * noise)


def _rand_center(rng, size, margin=8):
    return rng.uniform(margin, size - margin, 2)


# ---------------------------------------------------------------- pembuat mask per kelas
def _mask_for(cls: str, rng, size):
    """Mengembalikan (mask_alert, mask_air). mask_air hanya dipakai tambang (kolam)."""
    water = np.zeros((size, size), bool)
    if cls == "pertanian_skala_kecil":
        anchor = _rand_center(rng, size)
        m = np.zeros((size, size), bool)
        for _ in range(rng.integers(1, 4)):
            c = anchor + rng.normal(0, 4, 2)
            m |= _blob(rng, size, c, rng.uniform(2.5, 4.5), 0.25)
    elif cls == "pertanian_skala_besar":
        h, w = rng.integers(10, 19, 2)
        y0, x0 = rng.integers(2, size - h - 1), rng.integers(2, size - w - 1)
        m = np.zeros((size, size), bool)
        m[y0 : y0 + h, x0 : x0 + w] = True
        m &= _smooth_noise(rng, size, 1.2) > -2.2  # tepi hampir lurus
    elif cls == "jalan":
        d = _dist_polyline(_random_polyline(rng, size, 1, 4.0), size)
        m = d <= rng.uniform(0.9, 1.6)
    elif cls == "logging_selektif":
        trail = _dist_polyline(_random_polyline(rng, size, 2, 5.0), size) <= 0.8
        trail &= _smooth_noise(rng, size, 1.5) > -0.3  # jalur putus-putus
        m = trail.copy()
        ys, xs = np.nonzero(trail)
        for _ in range(rng.integers(5, 10)):
            if len(ys) == 0:
                break
            k = rng.integers(len(ys))
            m |= _blob(rng, size, (ys[k] + rng.normal(0, 3), xs[k] + rng.normal(0, 3)), rng.uniform(1.3, 2.2), 0.2)
    elif cls == "tambang":
        c = _rand_center(rng, size, 9)
        m = _blob(rng, size, c, rng.uniform(5, 8), 0.4)
        for _ in range(rng.integers(0, 3)):
            m |= _blob(rng, size, c + rng.normal(0, 7, 2), rng.uniform(2, 3.5), 0.3)
        water = _blob(rng, size, c + rng.normal(0, 2, 2), rng.uniform(1.5, 3.0), 0.3) & m
    elif cls == "kebakaran":
        c = _rand_center(rng, size, 9)
        yy, xx = _grid(size)
        field = _smooth_noise(rng, size, 3.0)
        m = (field > rng.uniform(0.0, 0.7)) & (np.hypot(yy - c[0], xx - c[1]) < rng.uniform(9, 14))
        if m.sum() < 40:
            m = _blob(rng, size, c, 7, 0.6)
    elif cls == "banjir":
        d = _dist_polyline(_random_polyline(rng, size, 2, 6.0), size)
        m = d < rng.uniform(3.0, 5.0) * (1 + 0.3 * _smooth_noise(rng, size, 2.0))
    else:
        raise ValueError(cls)
    if m.sum() < 6:  # jaga-jaga: pastikan alert tidak kosong
        m = _blob(rng, size, _rand_center(rng, size), 3.0, 0.2)
    return m, water


def _korupsi_mask(m, rng):
    """Alert nyata tidak sempurna: tepi meleset, sebagian piksel terlewat, ada piksel positif palsu."""
    r = rng.random()
    if r < 0.4:
        m = ndi.binary_dilation(m)
    elif r < 0.6:
        e = ndi.binary_erosion(m)
        m = e if e.sum() >= 6 else m
    keep = rng.random(m.shape) > rng.uniform(0.0, 0.25)
    m2 = m & keep
    m = m2 if m2.sum() >= 6 else m
    for _ in range(rng.integers(0, 7)):
        m[rng.integers(0, m.shape[0]), rng.integers(0, m.shape[1])] = True
    return m


def render_patch(cls: str, rng, size: int = PATCH):
    """Membuat satu patch (7, size, size): 6 pita + mask alert. Kondisi bebas awan."""
    # Sebagian kecil kejadian punya bentuk "tidak khas" (mirip kelas lain) -> tidak bisa ditebak dari bentuk saja
    shape_cls = cls
    if cls in CONFUSER and rng.random() < 0.2:
        shape_cls = CONFUSER[cls]
    mask, water = _mask_for(shape_cls, rng, size)
    mask = _korupsi_mask(mask, rng)
    water &= mask

    sig = SIGNATURE[cls].copy()
    conf = CONFUSER.get(cls)
    if conf is not None:  # campur sebagian ke kelas "mirip" -> tidak trivial
        a = rng.uniform(0.0, 0.6)
        sig = (1 - a) * sig + a * SIGNATURE[conf]
    gain = rng.normal(1.0, 0.10, N_BANDS)  # variasi antar-adegan (musim, sudut, dll.)
    sig_s = sig.copy()
    sig_s[2:] *= gain[2:]
    sig_s[:2] += (gain[:2] - 1.0) * 3.0

    alpha = ndi.gaussian_filter(mask.astype(float), 0.7)
    tex = _smooth_noise(rng, size, 1.5)
    out = np.empty((N_BANDS + 1, size, size), np.float32)
    for b in range(N_BANDS):
        base = HUTAN[b] * (1 + 0.06 * tex) if b >= 2 else HUTAN[b] + 0.8 * tex
        val = base * (1 - alpha) + sig_s[b] * alpha
        if cls == "tambang" and water.any():  # kolam: gelap di optik & radar
            w = ndi.gaussian_filter(water.astype(float), 0.6)
            pond = np.array([-20.0, -26.0, 0.05, 0.06, 0.04, 0.03])[b]
            val = val * (1 - w) + pond * w
        if b < 2:  # speckle radar (derau dalam dB)
            val = val + rng.normal(0, 1.3, (size, size))
        else:
            val = np.clip(val + rng.normal(0, 0.012, (size, size)), 0.0, 1.0)
        out[b] = val
    out[MASK_CH] = mask.astype(np.float32)
    return out


def apply_clouds(X: np.ndarray, fractions, seed: int = 0) -> np.ndarray:
    """Menutup sebagian piksel Sentinel-2 dengan awan. Radar (S1) dan mask alert tidak terpengaruh."""
    rng = np.random.default_rng(seed)
    fractions = np.broadcast_to(np.asarray(fractions, float), (len(X),))
    Y = X.copy()
    cloud_ref = np.array([0.55, 0.55, 0.52, 0.45], np.float32).reshape(4, 1, 1)
    for i, f in enumerate(fractions):
        if f <= 0:
            continue
        field = _smooth_noise(rng, X.shape[-1], 3.0)
        thr = np.quantile(field, 1 - f)
        a = np.clip((field - thr) * 3 + 0.5, 0, 1).astype(np.float32)
        Y[i, 2:6] = Y[i, 2:6] * (1 - a) + (cloud_ref + rng.normal(0, 0.02, (4, 1, 1))) * a
    return Y


def generate_dataset(n_per_class: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y, meta = [], [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(render_patch(cls, rng))
            y.append(ci)
            wil = rng.choice(WILAYAH, p=WILAYAH_BOBOT[cls])
            if cls == "kebakaran" and wil != "Kalimantan":
                bulan = int(rng.choice([7, 8, 9, 10], p=[0.15, 0.3, 0.35, 0.2]))  # musim kemarau
            else:
                bulan = int(rng.integers(1, 13))
            meta.append((wil, bulan))
    idx = rng.permutation(len(y))
    X = np.stack(X)[idx]
    y = np.array(y)[idx]
    meta = pd.DataFrame(meta, columns=["wilayah", "bulan"]).iloc[idx].reset_index(drop=True)
    return X, y, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "patches.npz"))
    args = ap.parse_args()
    X, y, meta = generate_dataset(args.n_per_class, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, wilayah=meta["wilayah"].to_numpy(), bulan=meta["bulan"].to_numpy())
    print(f"Tersimpan {out}  X={X.shape}  kelas={np.bincount(y).tolist()}  pita={BANDS}+mask")
