"""Evaluation with the rPPG-Toolbox's statistical gaps fixed.

 * reference HR from contact PPG via robust peak detection (IBI) *and* a
   0.1-bpm interpolated spectrum - not snapped to the FFT grid the prediction
   uses; sensor-reported HR can be supplied instead when the dataset has it
 * overlapping windows, partial trailing windows dropped (toolbox scores them)
 * Bland-Altman limits of agreement for repeated measures per subject
   (Bland & Altman 2007, variance components) - no sqrt(2) inflation, no
   in-place jitter of the data
 * subject-level bootstrap 95% CIs (windows of one subject are not independent)
 * stratified tables (ITA category, lux, fps, mask, ...)
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from .spectral import bandpass, hr_spectrum, interp_peak, band_mask


def hr_from_ppg(ppg, fs, lo=40.0, hi=180.0, method="spectral"):
    x = bandpass(np.asarray(ppg, float), fs, lo / 60 * 0.9, hi / 60 * 1.1)
    if method == "peaks":
        pk, _ = signal.find_peaks(x, distance=max(1, int(fs * 60 / hi * 0.8)),
                                  prominence=0.3 * np.std(x))
        if len(pk) < 3:
            return np.nan
        ibi = np.diff(pk) / fs
        ibi = ibi[np.abs(ibi - np.median(ibi)) < 0.3 * np.median(ibi)]   # reject missed/extra beats
        return float(60.0 / np.mean(ibi)) if len(ibi) else np.nan
    f, p = hr_spectrum(x, fs)
    b = band_mask(f, lo, hi)
    i = np.flatnonzero(b)[np.argmax(p[b])]
    return interp_peak(f, p, i)


def windowed_reference(t, ppg, centers, win_sec, fs=None, method="spectral"):
    t = np.asarray(t, float); ppg = np.asarray(ppg, float)
    fs = fs or 1.0 / np.median(np.diff(t))
    tg = np.arange(t[0], t[-1], 1 / fs)
    x = np.interp(tg, t, ppg)
    out = []
    for c in centers:
        m = (tg >= c - win_sec / 2) & (tg < c + win_sec / 2)
        out.append(hr_from_ppg(x[m], fs, method=method) if m.sum() >= 0.9 * win_sec * fs else np.nan)
    return np.asarray(out)


def loa_repeated(diff, subj):
    """Bland-Altman 2007 LoA when several pairs come from each subject and the
    true value varies within subject. Returns bias, sd_total, (lo, hi)."""
    diff = np.asarray(diff, float); subj = np.asarray(subj)
    bias = diff.mean()
    ids = np.unique(subj)
    if len(ids) < 2 or len(ids) == len(diff):
        sd = diff.std(ddof=1)
        return bias, sd, (bias - 1.96 * sd, bias + 1.96 * sd)
    n_i = np.array([np.sum(subj == s) for s in ids])
    m_i = np.array([diff[subj == s].mean() for s in ids])
    n, k = len(diff), len(ids)
    ssw = sum(((diff[subj == s] - m) ** 2).sum() for s, m in zip(ids, m_i))
    msw = ssw / (n - k)
    msb = (n_i * (m_i - bias) ** 2).sum() / (k - 1)
    d = (n ** 2 - (n_i ** 2).sum()) / ((k - 1) * n)
    s2b = max((msb - msw) / d, 0.0)
    sd = np.sqrt(s2b + msw)
    return bias, sd, (bias - 1.96 * sd, bias + 1.96 * sd)


def metrics(pred, ref, subj=None, n_boot=1000, seed=0):
    pred = np.asarray(pred, float); ref = np.asarray(ref, float)
    ok = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[ok], ref[ok]
    subj = np.arange(len(pred)) if subj is None else np.asarray(subj)[ok]
    e = pred - ref
    out = dict(n=int(len(e)), n_subjects=int(len(np.unique(subj))),
               MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
               MAPE=float(np.mean(np.abs(e) / ref) * 100),
               r=float(np.corrcoef(pred, ref)[0, 1]) if len(e) > 2 and np.std(ref) > 0 and np.std(pred) > 0 else np.nan,
               within5=float(np.mean(np.abs(e) <= 5) * 100))
    bias, sd, (lo, hi) = loa_repeated(e, subj)
    out.update(bias=float(bias), LoA_low=float(lo), LoA_high=float(hi))
    if n_boot and len(np.unique(subj)) > 2:
        rng = np.random.default_rng(seed)
        ids = np.unique(subj)
        idx = {s: np.flatnonzero(subj == s) for s in ids}
        maes, rmses = [], []
        for _ in range(n_boot):
            pick = np.concatenate([idx[s] for s in rng.choice(ids, len(ids))])
            maes.append(np.mean(np.abs(e[pick]))); rmses.append(np.sqrt(np.mean(e[pick] ** 2)))
        out["MAE_CI95"] = [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))]
        out["RMSE_CI95"] = [float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))]
    return out


def stratified(rows: list[dict], key: str, pred="pred", ref="ref", subj="subject"):
    groups = {}
    for r in rows:
        groups.setdefault(r.get(key, "unknown"), []).append(r)
    return {g: metrics([r[pred] for r in v], [r[ref] for r in v], [r[subj] for r in v], n_boot=500)
            for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def bland_altman_plot(pred, ref, subj=None, path="bland_altman.png", title=""):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pred = np.asarray(pred, float); ref = np.asarray(ref, float)
    ok = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[ok], ref[ok]
    subj = np.arange(len(pred)) if subj is None else np.asarray(subj)[ok]
    d = pred - ref; a = (pred + ref) / 2
    bias, sd, (lo, hi) = loa_repeated(d, subj)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(a, d, s=10, alpha=0.6)
    for y, ls, lab in [(bias, "-", f"bias {bias:.2f}"), (lo, "--", f"-1.96SD {lo:.2f}"), (hi, "--", f"+1.96SD {hi:.2f}")]:
        ax.axhline(y, color="k", ls=ls, lw=1, label=lab)
    ax.set_xlabel("Mean of rPPG and reference HR [bpm]")
    ax.set_ylabel("rPPG HR - reference HR [bpm]")
    ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
    return path
