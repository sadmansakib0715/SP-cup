"""Spectral utilities: O(N) detrending, zero-phase filtering, high-resolution
HR spectra with sub-bin peak interpolation, and signal-quality indices.

Fixes relative to rPPG-Toolbox/evaluation/post_process.py:
  * detrend is solved with a sparse banded system (O(N)) instead of a dense
    N x N matrix inverse (O(N^3) time, O(N^2) memory).
  * spectra are zero-padded to a fixed fine grid (default 0.1 bpm) and the peak
    is refined by parabolic interpolation on the log spectrum, so the reported
    HR is not snapped to an FFT bin (3.5 bpm wide for a 10 s window in the toolbox).
  * 4th-order SOS Butterworth (numerically stable) instead of 1st-order (b, a).
"""
from __future__ import annotations
import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

EPS = 1e-12


def detrend_sparse(x: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """Smoothness-priors detrending (Tarvainen 2002), sparse O(N) solve."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 3:
        return x - x.mean()
    d = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    a = sparse.eye(n) + (lam ** 2) * (d.T @ d)
    trend = spsolve(a.tocsc(), x)
    return x - trend


def bandpass(x: np.ndarray, fs: float, lo_hz: float, hi_hz: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    hi_hz = min(hi_hz, 0.45 * fs)  # stay below Nyquist for low-fps (C5) streams
    sos = signal.butter(order, [lo_hz, hi_hz], btype="bandpass", fs=fs, output="sos")
    padlen = min(3 * (2 * len(sos) + 1) * 3, len(x) - 1)
    if padlen < 1:
        return x - x.mean()
    return signal.sosfiltfilt(sos, x, padlen=padlen)


def _nfft_for(n: int, fs: float, df_bpm: float) -> int:
    need = int(np.ceil(fs * 60.0 / df_bpm))
    return int(2 ** np.ceil(np.log2(max(n, need))))


def hr_spectrum(x: np.ndarray, fs: float, df_bpm: float = 0.1, fmax_bpm: float = 420.0):
    """Hann-tapered, zero-padded power spectrum on a fine bpm grid.
    Returns (freq_bpm, power) for 0..min(fmax_bpm, Nyquist)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    w = signal.windows.hann(len(x), sym=False)
    nfft = _nfft_for(len(x), fs, df_bpm)
    p = np.abs(np.fft.rfft(x * w, n=nfft)) ** 2
    f = np.fft.rfftfreq(nfft, d=1.0 / fs) * 60.0
    keep = f <= min(fmax_bpm, fs * 30.0)
    return f[keep], p[keep]


def interp_peak(freq: np.ndarray, power: np.ndarray, idx: int) -> float:
    """Parabolic interpolation of a spectral peak on log power."""
    if idx <= 0 or idx >= len(power) - 1:
        return float(freq[idx])
    a, b, c = np.log(power[idx - 1:idx + 2] + EPS)
    den = a - 2 * b + c
    delta = 0.0 if abs(den) < 1e-12 else 0.5 * (a - c) / den
    delta = float(np.clip(delta, -0.5, 0.5))
    return float(freq[idx] + delta * (freq[1] - freq[0]))


def band_mask(freq, lo_bpm, hi_bpm):
    return (freq >= lo_bpm) & (freq <= hi_bpm)


def harmonic_sum(freq: np.ndarray, power: np.ndarray, weight2: float = 0.5) -> np.ndarray:
    """Sub-harmonic summation: S(f) + w*S(2f). Favors the true fundamental when
    motion or a strong dicrotic notch inflates the 2nd harmonic."""
    p2 = np.interp(2 * freq, freq, power, right=0.0)
    return power + weight2 * p2


def spectral_quality(freq, power, peak_bpm, lo_bpm=40.0, hi_bpm=180.0, half_bw=6.0) -> float:
    """Fraction of in-band (+2nd harmonic) power that sits at the peak and its
    harmonic. 1.0 = pure sinusoid, ~0 = white noise. Robust SQI in [0, 1]."""
    band = band_mask(freq, lo_bpm, 2.2 * hi_bpm)
    tot = power[band].sum() + EPS
    near = (np.abs(freq - peak_bpm) <= half_bw) | (np.abs(freq - 2 * peak_bpm) <= half_bw)
    return float(power[band & near].sum() / tot)


def snr_db(freq, power, ref_bpm, lo_bpm=36.0, hi_bpm=198.0, half_bw=6.0) -> float:
    """de Haan-style SNR around a reference HR (used for evaluation only)."""
    band = band_mask(freq, lo_bpm, hi_bpm)
    sig = (np.abs(freq - ref_bpm) <= half_bw) | (np.abs(freq - 2 * ref_bpm) <= half_bw)
    s = power[band & sig].sum()
    n = power[band & ~sig].sum()
    return float(10 * np.log10((s + EPS) / (n + EPS)))
