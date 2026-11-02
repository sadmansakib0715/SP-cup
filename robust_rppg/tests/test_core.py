import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robust_rppg.spectral import hr_spectrum, interp_peak, band_mask, detrend_sparse
from robust_rppg.methods import METHODS
from robust_rppg.evaluate import metrics, loa_repeated, hr_from_ppg
from robust_rppg.analysis import HRTracker
from robust_rppg.extract import resample_uniform


def _rgb(hr, fs=30, sec=20, seed=0):
    t = np.arange(int(sec * fs)) / fs
    p = np.sin(2 * np.pi * hr / 60 * t)
    rng = np.random.default_rng(seed)
    sig = np.array([0.27, 0.80, 0.54])
    return t, p, np.array([150, 110, 90.]) * (1 - 0.004 * np.outer(p, sig)) + 0.05 * rng.standard_normal((len(t), 3))


def test_offgrid_peak():
    for hr in [47.3, 73.4, 121.9]:
        t, p, _ = _rgb(hr, sec=10)
        f, P = hr_spectrum(p, 30)
        i = np.flatnonzero(band_mask(f, 40, 180))[np.argmax(P[band_mask(f, 40, 180)])]
        assert abs(interp_peak(f, P, i) - hr) < 0.15


def test_methods_polarity_and_hr():
    t, p, rgb = _rgb(83.0)
    for name, m in METHODS.items():
        b = m(rgb, 30)
        assert np.corrcoef(b, p)[0, 1] > 0.8, name           # +ve = blood volume
        f, P = hr_spectrum(b, 30); mk = band_mask(f, 40, 180)
        assert abs(f[mk][np.argmax(P[mk])] - 83.0) < 0.5, name


def test_low_fps():
    t, p, rgb = _rgb(66.0, fs=10, sec=30)
    b = METHODS["POS"](rgb, 10)
    f, P = hr_spectrum(b, 10); mk = band_mask(f, 40, 180)
    assert abs(f[mk][np.argmax(P[mk])] - 66.0) < 0.5


def test_detrend_sparse_matches_dense():
    x = np.cumsum(np.random.default_rng(1).standard_normal(400))
    n = len(x); D = np.diff(np.eye(n), 2, axis=0)
    dense = x - np.linalg.solve(np.eye(n) + 1e4 * D.T @ D, x)
    assert np.allclose(detrend_sparse(x, 100), dense, atol=1e-6)


def test_viterbi_rejects_outlier():
    grid = np.arange(40, 180.5, 0.5)
    specs = np.array([np.exp(-0.5 * ((grid - 75) / 1.5) ** 2) for _ in range(12)])
    specs[6] = np.exp(-0.5 * ((grid - 150) / 1.5) ** 2) * 1.2 + 0.4 * specs[5]   # harmonic jump
    path, post = HRTracker(grid, 1.0, 3.0).viterbi(specs)
    assert abs(grid[path[6]] - 75) < 1.0


def test_loa_and_metrics():
    rng = np.random.default_rng(0)
    ref = rng.uniform(60, 100, 200); pred = ref + rng.normal(1, 2, 200)
    m = metrics(pred, ref, np.repeat(np.arange(20), 10), n_boot=200)
    assert abs(m["bias"] - 1) < 0.5 and m["LoA_high"] - m["LoA_low"] < 4 * 1.96 * 2
    assert m["MAE_CI95"][0] < m["MAE"] < m["MAE_CI95"][1]


def test_resample_gaps():
    t = np.r_[np.arange(0, 5, 0.1), np.arange(6.5, 10, 0.1)]
    x = np.c_[t, t, t]
    tg, y, v = resample_uniform(t, x, 10)
    assert not v[(tg > 5.2) & (tg < 6.3)].any() and v[tg < 4.8].all()


def test_ppg_reference():
    t, p, _ = _rgb(91.2, fs=60, sec=20)
    assert abs(hr_from_ppg(p, 60) - 91.2) < 0.2
    assert abs(hr_from_ppg(p, 60, method="peaks") - 91.2) < 1.0


def test_presets():
    from robust_rppg.analysis import Config
    lean = Config.preset("lean")
    assert lean.rois == ("face_all",) and lean.methods == ("POS",) and lean.use_tracker
    assert Config.preset("full", win_sec=8).win_sec == 8
