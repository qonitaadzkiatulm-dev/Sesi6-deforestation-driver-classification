"""Pelatihan & eksperimen: benchmark RF vs CNN, ablasi sensor di bawah awan, dan ambang keyakinan.

Jalankan:  python src/train.py                (default: ukuran sedang, beberapa menit di CPU)
           python src/train.py --quick        (uji cepat)
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import evaluate as ev
from config import CHANNEL_SETS, CLASSES, MASK_CH, N_BANDS, N_CLASSES
from features import ekstrak_fitur
from models import SmallCNN
from simulate_data import apply_clouds, generate_dataset

ROOT = Path(__file__).resolve().parents[1]
TINGKAT_AWAN = [0.0, 0.5, 0.9]


# ---------------------------------------------------------------- data
def tambah_awan_acak(X, seed):
    """Gabungkan versi bersih + versi berawan (fraksi acak) agar model terbiasa dengan awan."""
    rng = np.random.default_rng(seed)
    return np.concatenate([X, apply_clouds(X, rng.uniform(0.2, 0.95, len(X)), seed + 1)])


def hitung_normalisasi(X):
    mu = X[:, :N_BANDS].mean((0, 2, 3))
    sd = X[:, :N_BANDS].std((0, 2, 3)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def normalisasi(X, mu, sd):
    Z = X.copy()
    Z[:, :N_BANDS] = (Z[:, :N_BANDS] - mu[None, :, None, None]) / sd[None, :, None, None]
    return Z


# ---------------------------------------------------------------- CNN
def augment(xb):
    k = int(np.random.randint(4))
    xb = torch.rot90(xb, k, (2, 3))
    return torch.flip(xb, (3,)) if np.random.rand() < 0.5 else xb


@torch.no_grad()
def prediksi_proba(model, X, kanal, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i : i + bs][:, kanal])
        out.append(torch.softmax(model(xb), 1).numpy())
    return np.concatenate(out)


def latih_cnn(Xtr, ytr, Xva, yva, kanal, epochs, seed, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallCNN(len(kanal), N_CLASSES)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * int(np.ceil(len(Xtr) / 64)))
    xt, yt = torch.from_numpy(Xtr[:, kanal]), torch.from_numpy(ytr).long()
    terbaik, state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):
            idx = perm[i : i + 64]
            loss = nn.functional.cross_entropy(model(augment(xt[idx])), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc = accuracy_score(yva, prediksi_proba(model, Xva, kanal).argmax(1))
        if acc > terbaik:
            terbaik, state = acc, copy.deepcopy(model.state_dict())
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"    epoch {ep + 1:>2}/{epochs}  akurasi validasi={acc:.3f} (terbaik {terbaik:.3f})")
    model.load_state_dict(state)
    return model, terbaik


def metrik(y, pred):
    return {"akurasi": round(float(accuracy_score(y, pred)), 4), "f1_makro": round(float(f1_score(y, pred, average="macro")), 4)}


# ---------------------------------------------------------------- utama
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=450, help="patch per kelas untuk latih+validasi")
    ap.add_argument("--n-test", type=int, default=150, help="patch per kelas untuk uji")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.epochs = 60, 30, 4

    torch.set_num_threads(max(1, torch.get_num_threads()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.1f}s] {s}", flush=True)  # noqa: E731

    log("Membangkitkan data sintetis ...")
    X, y, meta = generate_dataset(args.n_train, args.seed)
    Xte, yte, meta_te = generate_dataset(args.n_test, args.seed + 1000)
    np.savez_compressed(ROOT / "data" / "patches_test.npz", X=Xte, y=yte, wilayah=meta_te["wilayah"].to_numpy(),
                        bulan=meta_te["bulan"].to_numpy())
    ev.plot_contoh(X, y, out / "contoh_patch.png")

    # pisah latih / validasi (acak, 85/15)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    n_va = int(0.15 * len(y))
    va, tr = idx[:n_va], idx[n_va:]
    Xtr, ytr = tambah_awan_acak(X[tr], args.seed + 10), np.tile(y[tr], 2)
    Xva, yva = tambah_awan_acak(X[va], args.seed + 20), np.tile(y[va], 2)
    mu, sd = hitung_normalisasi(Xtr)
    Xtr_n, Xva_n = normalisasi(Xtr, mu, sd), normalisasi(Xva, mu, sd)
    Xte_awan = {c: apply_clouds(Xte, c, args.seed + 30) for c in TINGKAT_AWAN}
    Xte_awan_n = {c: normalisasi(v, mu, sd) for c, v in Xte_awan.items()}
    log(f"Latih={len(Xtr)} validasi={len(Xva)} uji={len(Xte)} (x {len(TINGKAT_AWAN)} tingkat awan)")

    hasil: dict[str, dict] = {}

    # ---- 1) Baseline konvensional: Random Forest + fitur buatan tangan
    log("Baseline Random Forest ...")
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1)
    rf.fit(ekstrak_fitur(Xtr), ytr)
    hasil["RF"] = {}
    for c in TINGKAT_AWAN:
        p = rf.predict(ekstrak_fitur(Xte_awan[c]))
        hasil["RF"][f"{c:g}"] = metrik(yte, p)
        if c == 0.0:
            cm_rf = confusion_matrix(yte, p, labels=range(N_CLASSES))
    log(f"  RF akurasi (awan 0/50/90%): {[hasil['RF'][f'{c:g}']['akurasi'] for c in TINGKAT_AWAN]}")

    # ---- 2) CNN dengan beberapa kombinasi sensor (ablasi)
    modelcnn = {}
    for nama, kanal in CHANNEL_SETS.items():
        log(f"Melatih CNN [{nama}] ({len(kanal)} kanal) ...")
        m, acc_va = latih_cnn(Xtr_n, ytr, Xva_n, yva, kanal, args.epochs, args.seed, log)
        modelcnn[nama] = (m, kanal)
        key = f"CNN {nama}"
        hasil[key] = {}
        for c in TINGKAT_AWAN:
            pr = prediksi_proba(m, Xte_awan_n[c], kanal)
            hasil[key][f"{c:g}"] = metrik(yte, pr.argmax(1))
        log(f"  {key} akurasi (awan 0/50/90%): {[hasil[key][f'{c:g}']['akurasi'] for c in TINGKAT_AWAN]}")

    # ---- 3) Laporan rinci model utama (CNN S1+S2, tanpa awan)
    m_utama, k_utama = modelcnn["S1+S2"]
    pr0 = prediksi_proba(m_utama, Xte_awan_n[0.0], k_utama)
    cm_cnn = confusion_matrix(yte, pr0.argmax(1), labels=range(N_CLASSES))
    laporan = classification_report(yte, pr0.argmax(1), target_names=CLASSES, digits=3, zero_division=0)
    (out / "laporan_klasifikasi_cnn.txt").write_text(laporan)
    ev.plot_confusion(cm_cnn, "CNN S1+S2 (tanpa awan)", out / "confusion_cnn.png")
    ev.plot_confusion(cm_rf, "Random Forest (tanpa awan)", out / "confusion_rf.png")
    ev.plot_ablasi_awan(hasil, TINGKAT_AWAN, out / "ablasi_awan.png")

    # ---- 4) Ambang keyakinan -> label "tidak pasti" (pada awan 50%)
    pr50 = prediksi_proba(m_utama, Xte_awan_n[0.5], k_utama)
    conf, pred50 = pr50.max(1), pr50.argmax(1)
    taus = [round(t, 2) for t in np.arange(0.3, 0.96, 0.05)]
    cakupan, akur = [], []
    for t in taus:
        ok = conf >= t
        cakupan.append(float(ok.mean()))
        akur.append(float((pred50[ok] == yte[ok]).mean()) if ok.any() else float("nan"))
    ev.plot_coverage_akurasi(taus, cakupan, akur, out / "coverage_akurasi.png")
    hasil["ambang_keyakinan_awan50"] = {
        f"{t}": {"cakupan": round(c, 4), "akurasi_berlabel": None if np.isnan(a) else round(a, 4)}
        for t, c, a in zip(taus, cakupan, akur)
    }

    # ---- 5) simpan model utama + normalisasi
    torch.save({"state_dict": m_utama.state_dict(), "kanal": k_utama, "mu": mu, "sd": sd}, ROOT / "models" / "cnn_s1s2.pt")
    hasil["_info"] = {
        "seed": args.seed, "n_train_per_kelas": args.n_train, "n_test_per_kelas": args.n_test, "epochs": args.epochs,
        "waktu_detik": round(time.time() - t0, 1),
        "catatan": "Data sintetis; hasil tidak mewakili akurasi pada citra Sentinel nyata.",
    }
    (out / "metrics.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False))
    log("Selesai. Hasil di " + str(out))


if __name__ == "__main__":
    main()
