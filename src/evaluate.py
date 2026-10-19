"""Visualisasi hasil. Mengikuti prinsip visualisasi: tanpa chartjunk, skala jujur, palet aman buta warna."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import CLASSES, MASK_CH

WARNA = {"RF": "#9AA0A6", "CNN S1+S2": "#0072B2", "CNN S1": "#E69F00", "CNN S2": "#009E73"}


def _bersih(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_contoh(X, y, path: Path):
    """Satu contoh per kelas: komposit optik (B4,B3,B2), radar VH, dan mask alert."""
    fig, axes = plt.subplots(3, len(CLASSES), figsize=(2.0 * len(CLASSES), 6.2))
    for ci, nama in enumerate(CLASSES):
        p = X[np.where(y == ci)[0][0]]
        rgb = np.clip(np.stack([p[4], p[3], p[2]], -1) * 5.0, 0, 1)
        axes[0, ci].imshow(rgb)
        axes[1, ci].imshow(p[1], cmap="gray", vmin=-26, vmax=-8)
        axes[2, ci].imshow(p[MASK_CH], cmap="gray_r")
        axes[0, ci].set_title(nama.replace("_", "\n"), fontsize=9)
        for r in range(3):
            axes[r, ci].axis("off")
    for r, t in enumerate(["Sentinel-2 (RGB)", "Sentinel-1 (VH)", "Mask alert"]):
        axes[r, 0].text(-0.08, 0.5, t, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("Contoh patch sintetis per kelas driver", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, judul, path: Path):
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), [c.replace("_", "\n") for c in CLASSES], fontsize=7)
    ax.set_yticks(range(len(CLASSES)), [c.replace("_", " ") for c in CLASSES], fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "#1F2933")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(judul, fontsize=11)
    fig.colorbar(im, fraction=0.046, label="Proporsi per kelas")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ablasi_awan(hasil: dict, tingkat_awan, path: Path):
    """Bar chart berkelompok: akurasi per model pada tiap tingkat tutupan awan (sumbu mulai dari nol)."""
    modelnames = list(hasil)
    lebar = 0.8 / len(modelnames)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for k, m in enumerate(modelnames):
        vals = [hasil[m][f"{c:g}"]["akurasi"] for c in tingkat_awan]
        xs = np.arange(len(tingkat_awan)) + k * lebar
        ax.bar(xs, vals, lebar, label=m, color=WARNA.get(m, "#444"))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(tingkat_awan)) + lebar * (len(modelnames) - 1) / 2,
                  [f"{int(c * 100)}%" for c in tingkat_awan])
    ax.set_xlabel("Tutupan awan pada citra optik (Sentinel-2)")
    ax.set_ylabel("Akurasi (data uji)")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncol=len(modelnames), loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
    ax.set_title("Ketahanan terhadap awan: radar vs optik vs gabungan", loc="left", fontsize=11)
    _bersih(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coverage_akurasi(tau, cakupan, akurasi, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(cakupan, akurasi, color="#0072B2", lw=2.5, marker="o", ms=4)
    for t, c, a in zip(tau, cakupan, akurasi):
        if t in (0.5, 0.7, 0.9):
            ax.annotate(f"τ={t}", (c, a), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.set_xlabel("Cakupan (proporsi alert yang diberi label)")
    ax.set_ylabel("Akurasi pada alert berlabel")
    ax.set_title("Ambang keyakinan: akurasi naik, cakupan turun", loc="left", fontsize=11)
    _bersih(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ringkasan_wilayah(df, path: Path):
    """Bar bertumpuk: komposisi driver hasil prediksi per wilayah (analog ringkasan pola antar-wilayah)."""
    tab = df.groupby(["wilayah", "driver_prediksi"]).size().unstack(fill_value=0)
    tab = tab.div(tab.sum(1), axis=0)
    palet = ["#E69F00", "#D55E00", "#0072B2", "#009E73", "#CC79A7", "#7A3E3E", "#56B4E9", "#9AA0A6"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    kiri = np.zeros(len(tab))
    for k, kolom in enumerate(tab.columns):
        ax.barh(tab.index, tab[kolom], left=kiri, label=kolom.replace("_", " "), color=palet[k % len(palet)])
        kiri += tab[kolom].to_numpy()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proporsi alert")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Komposisi driver hasil prediksi per wilayah (data sintetis)", loc="left", fontsize=11)
    _bersih(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
