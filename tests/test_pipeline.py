import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import CLASSES, MASK_CH, N_BANDS, PATCH  # noqa: E402
from features import NAMA_FITUR, ekstrak_fitur  # noqa: E402
from simulate_data import apply_clouds, generate_dataset  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return generate_dataset(12, seed=3)


def test_bentuk_dan_label(data):
    X, y, meta = data
    assert X.shape == (12 * len(CLASSES), N_BANDS + 1, PATCH, PATCH)
    assert set(y) == set(range(len(CLASSES)))
    assert len(meta) == len(y)
    assert np.isfinite(X).all()


def test_deterministik():
    a, _, _ = generate_dataset(3, seed=11)
    b, _, _ = generate_dataset(3, seed=11)
    assert np.array_equal(a, b)


def test_mask_alert_tidak_kosong_dan_biner(data):
    X, _, _ = data
    m = X[:, MASK_CH]
    assert set(np.unique(m)) <= {0.0, 1.0}
    assert (m.reshape(len(m), -1).sum(1) >= 6).all()


def test_awan_hanya_mengubah_optik(data):
    X, _, _ = data
    Xc = apply_clouds(X, 0.8, seed=1)
    assert np.array_equal(Xc[:, :2], X[:, :2])          # radar tidak terpengaruh awan
    assert np.array_equal(Xc[:, MASK_CH], X[:, MASK_CH])  # mask alert tetap
    assert np.abs(Xc[:, 2:6] - X[:, 2:6]).mean() > 0.05


def test_tanpa_awan_tidak_mengubah_data(data):
    X, _, _ = data
    assert np.array_equal(apply_clouds(X, 0.0, seed=1), X)


def test_fitur_bentuk(data):
    X, _, _ = data
    F = ekstrak_fitur(X)
    assert F.shape == (len(X), len(NAMA_FITUR))
    assert np.isfinite(F).all()


def test_jalan_lebih_lonjong_daripada_pertanian_besar(data):
    X, y, _ = data
    F = ekstrak_fitur(X)
    i = NAMA_FITUR.index("elongasi")
    assert F[y == CLASSES.index("jalan"), i].mean() > F[y == CLASSES.index("pertanian_skala_besar"), i].mean()
