"""Candidate generation (ROI x method), SQI-weighted spectral fusion,
Viterbi / forward HR tracking and BVP reconstruction.

Why this lowers MAE versus a single POS-on-face-box + argmax(FFT):
  1. many partially independent candidates (6 ROIs + whole-skin, 3 projections)
     -> a motion/specular/occlusion burst rarely corrupts all of them at once;
  2. each candidate is weighted by its own spectral concentration (SQI) and by
     agreement with the consensus, so bad ROIs/methods are suppressed per window;
  3. HR is decoded as the most likely *trajectory* through the fused
     spectrogram (HMM with a physiological rate-of-change prior) instead of an
     independent argmax per window -> removes isolated harmonic/noise jumps,
     which dominate MAE/RMSE in practice;
  4. the final value is refined off-grid (parabolic interpolation on a
     0.1 bpm zero-padded spectrum) rather than snapped to an FFT bin.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .methods import METHODS
from .spectral import bandpass, hr_spectrum, interp_peak, harmonic_sum, spectral_quality, band_mask
from .extract import FaceTrace, resample_uniform, occluded_rois, ita_category, ROI_NAMES


@dataclass
class Config:
    lo_bpm: float = 40.0          # elderly bradycardia (C6) - keep low edge at 40
    hi_bpm: float = 180.0
    win_sec: float = 10.0
    step_sec: float = 1.0
    grid_bpm: float = 0.5         # HMM state resolution (final value is refined off-grid)
    methods: tuple = ("POS", "CHROM", "OMIT")
    rois: tuple = tuple(ROI_NAMES)
    sqi_power: float = 2.0
    min_valid: float = 0.6        # min fraction of valid samples in a window
    occlusion_frac: float = 0.35
    max_rate_bpm_s: float = 3.0   # HR prior: typical |dHR/dt|; exercise can exceed -> raise
    emission_gamma: float = 1.0
    fs: float | None = None       # processing rate; None = native median frame rate
    use_tracker: bool = True      # False = independent argmax per window (ablation)
    fuse: bool = True             # False = single best-a-priori candidate (ablation)
    debug: bool = False
    hierarchical: bool = True     # False = flat SQI fusion of all candidates (ablation)
    color_prior: bool = True      # pulse colour-signature (G/R) prior on the fused spectrum


    @classmethod
    def preset(cls, name: str, **kw):
        """'full' : 7 ROIs x 3 projections, hierarchical coherence fusion + colour prior + HMM.
                   Best worst-case behaviour and overall MAE on the synthetic suite.
        'lean' : skin-masked whole-face POS + HMM tracking. Ties 'full' on recoverable
                   conditions, ~20x cheaper analysis -> Raspberry Pi / live default.
        Choose between them on YOUR validation split (see README results)."""
        base = {"full": {}, "lean": dict(fuse=False, color_prior=False, methods=("POS",), rois=("face_all",))}[name]
        return cls(**{**base, **kw})


@dataclass
class FaceResult:
    tid: int
    t_hr: np.ndarray
    hr: np.ndarray
    conf: np.ndarray
    hr_video: float
    hr_video_spectral: float
    conf_video: float
    t_bvp: np.ndarray
    bvp: np.ndarray
    fs: float
    ita: float | None
    ita_category: str
    rois_used: list
    occluded: list
    weights: dict = field(default_factory=dict)
    diag: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    fused_spec: np.ndarray | None = None
    grid: np.ndarray | None = None


def _norm(x):
    s = np.std(x)
    return x / s if s > 0 else x


class HRTracker:
    """HMM over an HR grid. offline: Viterbi + forward-backward;
    online: causal forward filtering (one call per new fused spectrum)."""

    def __init__(self, grid: np.ndarray, step_sec: float, max_rate: float, gamma: float = 1.0):
        self.grid = grid
        d = grid[:, None] - grid[None, :]
        sig = max_rate * step_sec
        tr = np.exp(-0.5 * (d / sig) ** 2) + 1e-4 * np.exp(-np.abs(d) / 40.0)  # rare big jumps allowed
        self.T = tr / tr.sum(0, keepdims=True)        # T[to, from]
        self.logT = np.log(self.T)
        self.gamma = gamma
        self.belief = None

    def _emis(self, spec):
        p = np.maximum(spec, 0) + 1e-12
        p = p / p.sum()
        return (p ** self.gamma) / (p ** self.gamma).sum()

    def viterbi(self, specs: np.ndarray):
        k, s = specs.shape
        le = np.log(np.stack([self._emis(x) for x in specs]))
        dp = le[0] - np.log(s)
        bp = np.zeros((k, s), int)
        for i in range(1, k):
            cand = dp[None, :] + self.logT          # [to, from]
            bp[i] = np.argmax(cand, 1)
            dp = cand[np.arange(s), bp[i]] + le[i]
        path = np.zeros(k, int)
        path[-1] = int(np.argmax(dp))
        for i in range(k - 1, 0, -1):
            path[i - 1] = bp[i, path[i]]
        # forward-backward posteriors -> confidence
        e = np.stack([self._emis(x) for x in specs])
        a = np.zeros((k, s)); b = np.ones((k, s))
        a[0] = e[0] / e[0].sum()
        for i in range(1, k):
            a[i] = e[i] * (self.T @ a[i - 1]); a[i] /= a[i].sum()
        for i in range(k - 2, -1, -1):
            b[i] = self.T.T @ (e[i + 1] * b[i + 1]); b[i] /= b[i].sum()
        post = a * b; post /= post.sum(1, keepdims=True)
        return path, post

    def step(self, spec: np.ndarray):
        e = self._emis(spec)
        self.belief = e if self.belief is None else e * (self.T @ self.belief)
        self.belief /= self.belief.sum()
        i = int(np.argmax(self.belief))
        return i, self.belief


def color_prior(rgb_seg, fs, grid, floor=0.15):
    """Haemoglobin absorbs green far more than red, so at the true pulse rate the
    normalised G amplitude exceeds R (G/R ~1.5-3 for all skin tones), while
    illumination/specular/intensity artefacts are near-achromatic (G/R ~1).
    Returns a soft per-frequency prior in [floor, 1]."""
    n = rgb_seg / (rgb_seg.mean(0) + 1e-9) - 1
    n = np.stack([bandpass(n[:, c], fs, grid[0] / 60 * 0.9, min(grid[-1] / 60 * 1.1, 0.45 * fs)) for c in range(3)], 1)
    w = np.hanning(len(n)); tt = np.arange(len(n)) / fs
    E = np.exp(-2j * np.pi * np.outer(tt, grid / 60.0))          # (T, F)
    A = np.abs((n * w[:, None]).T @ E)                           # (3, F)
    gr = A[1] / (A[0] + 1e-12)
    return np.clip((gr - 1.0) / 0.6, floor, 1.0)


def _posterior_conf(post_row, grid, center, half=3.0):
    return float(post_row[np.abs(grid - center) <= half].sum())


def candidate_signals(trace: FaceTrace, cfg: Config, extra: dict | None = None):
    t, rgb, frac, npx = trace.arrays()
    if len(t) < 8:
        return None
    dt = np.median(np.diff(t))
    fs = cfg.fs or float(np.round(1.0 / dt, 3))
    occl = occluded_rois(frac, cfg.occlusion_frac)
    cands = {}
    tg = valid_all = None
    ref_rgb = None
    for roi in cfg.rois:
        if roi in occl:
            continue
        tg, x, valid = resample_uniform(t, rgb[roi], fs)
        if valid.mean() < 0.5:
            continue
        if roi == "face_all":
            ref_rgb = x
        for m in cfg.methods:
            bvp = METHODS[m](x, fs)
            if not np.any(bvp):
                continue
            bvp = bandpass(bvp, fs, cfg.lo_bpm / 60 * 0.9, min(cfg.hi_bpm / 60 * 1.6, 0.45 * fs))
            cands[(roi, m)] = (_norm(bvp), valid)
    for name, (te, xe) in (extra or {}).items():
        if tg is None:
            break
        xe = np.interp(tg, np.asarray(te, float), np.asarray(xe, float))
        xe = bandpass(xe, fs, cfg.lo_bpm / 60 * 0.9, min(cfg.hi_bpm / 60 * 1.6, 0.45 * fs))
        cands[(name, "ext")] = (_norm(xe), np.ones(len(tg), bool))
    if not cfg.fuse and cands:
        pref = [k for k in [("face_all", cfg.methods[0])] if k in cands] or list(cands)[:1]
        cands = {pref[0]: cands[pref[0]]}
    return tg, fs, cands, sorted(occl), ref_rgb


def analyze(trace: FaceTrace, cfg: Config = Config(), extra: dict | None = None) -> FaceResult | None:
    """extra: optional {name: (t_sec, bvp)} candidates (e.g. a neural model's output)
    that enter the same SQI-weighted fusion."""
    cs = candidate_signals(trace, cfg, extra)
    if cs is None or not cs[2]:
        return None
    tg, fs, cands, occl, ref_rgb = cs
    n = len(tg)
    wl = int(min(round(cfg.win_sec * fs), n))
    hop = max(int(round(cfg.step_sec * fs)), 1)
    starts = list(range(0, max(n - wl, 0) + 1, hop))
    grid = np.arange(cfg.lo_bpm, cfg.hi_bpm + 1e-9, cfg.grid_bpm)
    keys = list(cands)
    fused_specs, fine_specs, wsum = [], [], {k: 0.0 for k in keys}
    priors = []
    bvp = np.zeros(n); bw = np.zeros(n); diag = []
    han = np.hanning(wl) + 1e-3
    for s0 in starts:
        sl = slice(s0, s0 + wl)
        specs, peaks, ws, segs = [], [], [], []
        for k in keys:
            seg, valid = cands[k]
            v = valid[sl].mean()
            if v < cfg.min_valid:
                specs.append(None); peaks.append(np.nan); ws.append(0.0); segs.append(None); continue
            f, p = hr_spectrum(seg[sl], fs)
            band = band_mask(f, cfg.lo_bpm, cfg.hi_bpm)
            hs = harmonic_sum(f, p)
            i = np.flatnonzero(band)[np.argmax(hs[band])]
            pk = interp_peak(f, p, i)
            q = spectral_quality(f, p, pk, cfg.lo_bpm, cfg.hi_bpm)
            pb = np.interp(grid, f, hs)
            pb = pb / (pb.sum() + 1e-12)
            specs.append((f, p, pb)); peaks.append(pk); ws.append((q ** cfg.sqi_power) * v); segs.append(seg[sl])
        ws = np.asarray(ws); peaks = np.asarray(peaks)
        if ws.sum() <= 0:
            fused_specs.append(np.ones_like(grid) / len(grid)); fine_specs.append(None); priors.append(None); continue
        # Hierarchical fusion. Candidates of one projection share failure modes
        # (e.g. CHROM's fixed skin-tone model on dark skin), so they must not
        # vote independently. Level 1: fuse ROIs within each method. Level 2:
        # weight each method by the SPATIAL COHERENCE of its ROI peaks - a real
        # pulse has the same rate in every skin region, artefacts scatter.
        good = ws > 0
        meths = sorted({k[1] for k in keys}) if cfg.hierarchical else ["_all"]
        mspec, mw = {}, {}
        for m in meths:
            idx = [i for i, k in enumerate(keys) if (k[1] == m or m == "_all") and good[i]]
            if not idx:
                continue
            wm = ws[idx]
            fm = sum(ws[i] * specs[i][2] for i in idx) / wm.sum()
            pm = grid[np.argmax(fm)]
            near = np.exp(-0.5 * ((peaks[idx] - pm) / 4.0) ** 2)
            coh = float((wm * near).sum() / wm.sum()) if (len(idx) > 1 and cfg.hierarchical) else 1.0
            mspec[m], mw[m] = fm, (coh ** 2) * float(wm.mean())
        if not mw or sum(mw.values()) <= 0:
            fused_specs.append(np.ones_like(grid) / len(grid)); fine_specs.append(None); priors.append(None); continue
        tot = sum(mw.values())
        fused = sum(mw[m] * mspec[m] for m in mw) / tot
        prior = color_prior(ref_rgb[sl], fs, grid) if (cfg.color_prior and ref_rgb is not None) else None
        priors.append(prior)
        if prior is not None:
            fused = fused * prior
        cons = grid[np.argmax(fused)]
        agree = np.where(good, np.exp(-0.5 * ((peaks - cons) / 6.0) ** 2), 0)
        ws2 = np.array([ws[i] * (mw.get(k[1], 0.0) if cfg.hierarchical else 1.0) * (0.25 + 0.75 * agree[i])
                        for i, k in enumerate(keys)])
        if ws2.sum() <= 0:
            ws2 = ws.copy()
        ws2 /= ws2.sum()
        ff = specs[int(np.argmax(ws2))][0]
        fine = sum(w * np.interp(ff, s[0], s[1] / (s[1].sum() + 1e-12)) for w, s in zip(ws2, specs) if w > 0)
        if prior is not None:
            fine = fine * np.interp(ff, grid, prior, left=prior[0], right=prior[-1])
        fused_specs.append(fused); fine_specs.append((ff, fine))
        if cfg.debug:
            diag.append([(k, float(pk), float(w0), float(w1)) for k, pk, w0, w1 in zip(keys, peaks, ws, ws2)])
        for k, w in zip(keys, ws2):
            wsum[k] += w
        # BVP: weighted, sign-aligned sum, Hann overlap-add
        ref = segs[int(np.argmax(ws2))]
        acc = np.zeros(wl)
        for w, sg in zip(ws2, segs):
            if w > 0 and sg is not None:
                acc += w * np.sign(np.dot(sg, ref) or 1.0) * sg
        bvp[sl] += han * acc; bw[sl] += han
    fused_specs = np.asarray(fused_specs)
    trk = HRTracker(grid, cfg.step_sec, cfg.max_rate_bpm_s, cfg.emission_gamma)
    path, post = trk.viterbi(fused_specs)
    if not cfg.use_tracker:
        path = np.argmax(fused_specs, 1)
    hr = np.empty(len(path)); conf = np.empty(len(path))
    peaked = np.empty(len(path)); colr = np.empty(len(path)); post_c = np.empty(len(path))
    for i, si in enumerate(path):
        c = grid[si]
        if fine_specs[i] is not None:
            ff, fine = fine_specs[i]
            m = np.abs(ff - c) <= 4.0
            j = np.flatnonzero(m)[np.argmax(fine[m])]
            c = interp_peak(ff, fine, j)
        hr[i] = c; conf[i] = _posterior_conf(post[i], grid, c)
        fsp = fused_specs[i] / (fused_specs[i].sum() + 1e-12)
        peaked[i] = float(fsp[np.abs(grid - c) <= 3.0].sum())
        pr = priors[i] if i < len(priors) else None
        colr[i] = float(np.interp(c, grid, pr)) if pr is not None else 1.0
        post_c[i] = conf[i]
        # reported confidence = geometric mean of trajectory consistency (HMM posterior)
        # and spectral evidence (fused-spectrum mass within +/-3 bpm of the estimate).
        conf[i] = float(np.sqrt(conf[i] * peaked[i]))
    t_hr = tg[np.asarray(starts)] + (wl / fs) / 2
    # video-level: whole-record fused spectrum, constrained near the track median
    med = float(np.median(hr))
    tot = None; wtot = 0
    for k in keys:
        seg, valid = cands[k]
        if valid.mean() < cfg.min_valid:
            continue
        f, p = hr_spectrum(seg, fs)
        p = p / (p[band_mask(f, cfg.lo_bpm, cfg.hi_bpm)].sum() + 1e-12)
        w = wsum[k]
        tot = w * p if tot is None else tot + w * p; wtot += w
    hv_spec, cv = med, float(np.mean(conf))
    if tot is not None and wtot > 0:
        m = np.abs(f - med) <= 8.0
        j = np.flatnonzero(m)[np.argmax(tot[m])]
        hv_spec = interp_peak(f, tot, j)
    # Video-level HR = confidence-weighted mean of the tracked series (= average HR,
    # what an oximeter-averaged reference reports). The whole-record spectral peak
    # (hv_spec, the toolbox's definition) is biased when HR drifts within the video.
    hv = float(np.average(hr, weights=conf + 1e-3)) if len(hr) >= 3 else hv_spec
    bvp = np.where(bw > 0, bvp / np.maximum(bw, 1e-9), 0.0)
    total = sum(wsum.values()) or 1.0
    weights = {f"{r}|{m}": round(v / total, 4) for (r, m), v in sorted(wsum.items(), key=lambda kv: -kv[1])}
    ita = trace.ita()
    return FaceResult(trace.tid, t_hr, hr, conf, float(hv), float(hv_spec), cv, tg, _norm(bvp), fs, ita, ita_category(ita),
                      sorted({r for r, _ in keys}), occl, weights, diag,
                      dict(peaked=peaked, color=colr, posterior=post_c), fused_specs, grid)
