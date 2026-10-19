"""Simulasi alur bulanan: klasifikasikan alert baru -> driver + keyakinan -> label "tidak_pasti" bila keyakinan rendah.

Jalankan:  python src/predict.py --awan 0.5 --tau 0.6
Membutuhkan models/cnn_s1s2.pt (hasil src/train.py) dan data/patches_test.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate as ev
from config import CLASSES, N_CLASSES
from models import SmallCNN
from simulate_data import apply_clouds
from train import normalisasi, prediksi_proba

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--awan", type=float, default=0.5, help="fraksi tutupan awan pada citra optik (0-1)")
    ap.add_argument("--tau", type=float, default=0.6, help="ambang keyakinan; di bawahnya diberi label tidak_pasti")
    ap.add_argument("--model", default=str(ROOT / "models" / "cnn_s1s2.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "patches_test.npz"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    model = SmallCNN(len(ck["kanal"]), N_CLASSES)
    model.load_state_dict(ck["state_dict"])

    d = np.load(args.data, allow_pickle=True)
    X, y = d["X"], d["y"]
    Xc = normalisasi(apply_clouds(X, args.awan, seed=123), ck["mu"], ck["sd"])
    proba = prediksi_proba(model, Xc, ck["kanal"])
    conf, pred = proba.max(1), proba.argmax(1)

    df = pd.DataFrame({
        "wilayah": d["wilayah"], "bulan": d["bulan"],
        "driver_sebenarnya": [CLASSES[i] for i in y],
        "driver_prediksi": [CLASSES[i] if c >= args.tau else "tidak_pasti" for i, c in zip(pred, conf)],
        "keyakinan": conf.round(3),
    })
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "prediksi_alert.csv", index=False)
    ev.plot_ringkasan_wilayah(df, out / "ringkasan_driver_per_wilayah.png")

    berlabel = df["driver_prediksi"] != "tidak_pasti"
    akurasi = (df.loc[berlabel, "driver_prediksi"] == df.loc[berlabel, "driver_sebenarnya"]).mean()
    print(f"Alert: {len(df)} | awan={args.awan:.0%} | tau={args.tau}")
    print(f"Cakupan berlabel: {berlabel.mean():.1%} | akurasi pada yang berlabel: {akurasi:.1%}")
    print(f"Tersimpan: {out / 'prediksi_alert.csv'}")
    print("\nCatatan latensi: pada sistem nyata, klasifikasi driver diperbarui bulanan sehingga tidak cocok "
          "untuk intervensi lapangan yang membutuhkan respons dalam hitungan hari.")


if __name__ == "__main__":
    main()
