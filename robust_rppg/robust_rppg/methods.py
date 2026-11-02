"""Training-free rPPG projections, vectorised, operating on uniformly sampled
mean-RGB traces of shape (T, 3). Implemented from the original papers:
  POS   - Wang, den Brinker, Stuijk, de Haan, IEEE TBME 2017
  CHROM - de Haan & Jeanne, IEEE TBME 2013
  OMIT  - Casado & Bordallo Lopez, 2023 (QR-based orthogonal projection)
Sign convention: every method returns +ve = blood-volume increase (= reflected
intensity decrease), matching contact-PPG pleth waveforms, so outputs can be fused.
All use short (1.6 s) overlap-added windows so they are local in time and can
run on a full recording or a live ring buffer identically.
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .spectral import bandpass

EPS = 1e-9


def _windows(rgb: np.ndarray, l: int):
    return sliding_window_view(rgb, (l, 3))[:, 0]  # (K, l, 3)


def _ola(h: np.ndarray, total: int, l: int) -> np.ndarray:
    out = np.zeros(total)
    k = h.shape[0]
    for j in range(l):
        out[j:j + k] += h[:, j]
    return out


def pos(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    t = rgb.shape[0]
    l = max(int(np.ceil(win_sec * fs)), 4)
    if t < l + 1:
        return np.zeros(t)
    w = _windows(rgb, l)
    cn = w / (w.mean(axis=1, keepdims=True) + EPS)
    p = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    s = cn @ p.T                                   # (K, l, 2)
    alpha = s[..., 0].std(axis=1) / (s[..., 1].std(axis=1) + EPS)
    h = s[..., 0] + alpha[:, None] * s[..., 1]
    h -= h.mean(axis=1, keepdims=True)
    # POS output follows skin intensity; flip so +ve = blood volume (contact-PPG convention)
    return -_ola(h, t, l)


def _moving_norm(rgb: np.ndarray, l: int) -> np.ndarray:
    k = np.ones(l)
    s = np.stack([np.convolve(rgb[:, c], k, mode="same") for c in range(3)], 1)
    cnt = np.convolve(np.ones(len(rgb)), k, mode="same")
    mu = s / (cnt[:, None] + EPS)
    return rgb / (mu + EPS) - 1.0


def chrom(rgb: np.ndarray, fs: float, win_sec: float = 1.6, lo=0.6, hi=3.5) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    t = rgb.shape[0]
    l = max(int(np.ceil(win_sec * fs)), 4)
    l += l % 2
    if t < 2 * l:
        return np.zeros(t)
    cn = _moving_norm(rgb, l)
    x = 3 * cn[:, 0] - 2 * cn[:, 1]
    y = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    xf, yf = bandpass(x, fs, lo, hi), bandpass(y, fs, lo, hi)
    hop = l // 2
    han = np.hanning(l)
    out = np.zeros(t)
    for s0 in range(0, t - l + 1, hop):
        xs, ys = xf[s0:s0 + l], yf[s0:s0 + l]
        a = xs.std() / (ys.std() + EPS)
        seg = xs - a * ys
        out[s0:s0 + l] += han * (seg - seg.mean())
    return out


def omit(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    t = rgb.shape[0]
    l = max(int(np.ceil(win_sec * fs)), 4)
    if t < l + 1:
        return np.zeros(t)
    w = _windows(rgb, l)
    cn = w / (w.mean(axis=1, keepdims=True) + EPS)       # (K, l, 3)
    m = np.transpose(cn, (0, 2, 1))                      # (K, 3, l)
    q, _ = np.linalg.qr(m)                               # (K, 3, 3)
    s = q[:, :, 0]                                       # dominant (intensity) direction
    proj = np.eye(3)[None] - s[:, :, None] * s[:, None, :]
    y = proj @ m
    h = -y[:, 1, :]                                      # green residual; blood absorbs green
    h -= h.mean(axis=1, keepdims=True)
    return _ola(h, t, l)


METHODS = {"POS": pos, "CHROM": chrom, "OMIT": omit}
