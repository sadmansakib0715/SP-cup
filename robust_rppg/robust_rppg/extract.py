"""Per-frame ROI signal extraction with adaptive skin masking, occlusion
(mask) detection, ITA skin-tone estimation and timestamp-aware resampling.

Fixes relative to rPPG-Toolbox:
  * only skin pixels enter the average (toolbox: whole 1.5x box incl. hair and
    background, resized to 72x72) -> large SNR gain on dark skin & low light
  * per-ROI skin fraction is tracked, so masked / occluded regions (C7) drop
    out automatically instead of polluting the average
  * real frame timestamps are kept and traces are resampled onto a uniform
    grid (toolbox: assumes perfectly uniform frames; PURE timestamps ignored),
    which matters for <=10 fps or variable-rate live streams (C5).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2
from scipy.interpolate import interp1d
from .face import roi_polygons, ROI_DEFS, LOWER_FACE

ROI_NAMES = list(ROI_DEFS) + ["face_all"]


@dataclass
class SkinModel:
    """Adaptive chroma model, anchored on the forehead (least often occluded)."""
    cr: float | None = None
    cb: float | None = None
    scr: float = 6.0
    scb: float = 6.0
    alpha: float = 0.05

    def update(self, ycc_forehead: np.ndarray):
        if len(ycc_forehead) < 20:
            return
        cr, cb = np.median(ycc_forehead[:, 1]), np.median(ycc_forehead[:, 2])
        scr = 1.4826 * np.median(np.abs(ycc_forehead[:, 1] - cr))
        scb = 1.4826 * np.median(np.abs(ycc_forehead[:, 2] - cb))
        if self.cr is None:
            self.cr, self.cb, self.scr, self.scb = cr, cb, max(scr, 3), max(scb, 3)
        else:
            a = self.alpha
            self.cr += a * (cr - self.cr); self.cb += a * (cb - self.cb)
            self.scr += a * (max(scr, 3) - self.scr); self.scb += a * (max(scb, 3) - self.scb)

    def mask(self, ycc: np.ndarray, k: float = 3.0) -> np.ndarray:
        y = ycc[:, 0]
        ok = (y < 245) & (y > max(4.0, 0.35 * np.median(y) if len(y) else 4.0))
        if self.cr is None:
            return ok
        d = ((ycc[:, 1] - self.cr) / self.scr) ** 2 + ((ycc[:, 2] - self.cb) / self.scb) ** 2
        return ok & (d < k * k)


@dataclass
class FaceTrace:
    tid: int
    t: list = field(default_factory=list)
    rgb: dict = field(default_factory=lambda: {r: [] for r in ROI_NAMES})
    frac: dict = field(default_factory=lambda: {r: [] for r in ROI_NAMES})
    npx: dict = field(default_factory=lambda: {r: [] for r in ROI_NAMES})
    lab_samples: list = field(default_factory=list)
    skin: SkinModel = field(default_factory=SkinModel)

    def arrays(self):
        t = np.asarray(self.t, float)
        rgb = {r: np.asarray(v, float) for r, v in self.rgb.items()}
        frac = {r: np.asarray(v, float) for r, v in self.frac.items()}
        npx = {r: np.asarray(v, float) for r, v in self.npx.items()}
        return t, rgb, frac, npx

    def ita(self) -> float | None:
        """Individual Typology Angle (deg) = atan((L*-50)/b*). Median over the
        first frames' skin pixels. NB: depends on illumination/white balance."""
        if not self.lab_samples:
            return None
        lab = np.concatenate(self.lab_samples, 0)
        L, b = np.median(lab[:, 0]), np.median(lab[:, 2])
        return float(np.degrees(np.arctan2(L - 50.0, b)))


def ita_category(ita: float | None) -> str:
    if ita is None:
        return "unknown"
    for th, name in [(55, "very_light"), (41, "light"), (28, "intermediate"), (10, "tan"), (-30, "brown")]:
        if ita > th:
            return name
    return "dark"


def _poly_mask(poly, x0, y0, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(poly - [x0, y0]).astype(np.int32), 1)
    return m.astype(bool)


def extract_frame(frame_rgb: np.ndarray, anchors: np.ndarray, trace: FaceTrace, t: float,
                  ita_frames: int = 60):
    h, w = frame_rgb.shape[:2]
    polys = roi_polygons(anchors)
    allp = np.concatenate(list(polys.values()))
    x0, y0 = np.floor(allp.min(0)).astype(int)
    x1, y1 = np.ceil(allp.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        _append_missing(trace, t); return
    crop = frame_rgb[y0:y1, x0:x1]
    ycc = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb).reshape(-1, 3).astype(np.float32)
    flat = crop.reshape(-1, 3).astype(np.float64)
    masks = {n: _poly_mask(p, x0, y0, crop.shape[:2]).reshape(-1) for n, p in polys.items()}
    fh = masks["forehead_l"] | masks["forehead_r"]
    trace.skin.update(ycc[fh])
    skin = trace.skin.mask(ycc)
    union = np.zeros_like(skin)
    trace.t.append(t)
    for n, m in masks.items():
        area = m.sum()
        sel = m & skin
        k = sel.sum()
        frac = k / area if area else 0.0
        if k >= 8:
            trace.rgb[n].append(flat[sel].mean(0)); union |= sel
        else:
            trace.rgb[n].append([np.nan] * 3)
        trace.frac[n].append(frac); trace.npx[n].append(k)
    k = union.sum()
    trace.rgb["face_all"].append(flat[union].mean(0) if k >= 8 else [np.nan] * 3)
    trace.frac["face_all"].append(k / max(sum(m.sum() for m in masks.values()), 1))
    trace.npx["face_all"].append(k)
    if len(trace.lab_samples) < ita_frames:
        sel = (fh | masks["cheek_l"] | masks["cheek_r"]) & skin
        if sel.sum() > 20:
            px = (flat[sel][::4] / 255.0).astype(np.float32).reshape(-1, 1, 3)
            trace.lab_samples.append(cv2.cvtColor(px, cv2.COLOR_RGB2Lab).reshape(-1, 3))


def _append_missing(trace: FaceTrace, t: float):
    trace.t.append(t)
    for n in ROI_NAMES:
        trace.rgb[n].append([np.nan] * 3); trace.frac[n].append(0.0); trace.npx[n].append(0)


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float, max_gap: float = 0.5):
    """Resample (possibly irregular, gappy) samples onto a uniform grid.
    Gaps <= max_gap s are linearly bridged; longer gaps are marked invalid."""
    t = np.asarray(t, float); x = np.asarray(x, float)
    tg = np.arange(t[0], t[-1] + 1e-9, 1.0 / fs)
    x2 = x.reshape(len(t), -1)
    good = np.all(np.isfinite(x2), axis=1)
    out = np.full((len(tg), x2.shape[1]), np.nan)
    valid = np.zeros(len(tg), bool)
    if good.sum() >= 2:
        f = interp1d(t[good], x2[good], axis=0, kind="linear", bounds_error=False)
        out = f(tg)
        tgood = t[good]
        idx = np.searchsorted(tgood, tg).clip(1, len(tgood) - 1)
        gap = tgood[idx] - tgood[idx - 1]
        valid = np.isfinite(out).all(1) & (gap <= max_gap + 1e-9)
        # fill invalid with nearest valid to keep filters stable (weights drop to 0 later)
        if valid.any():
            vi = np.flatnonzero(valid)
            near = vi[np.clip(np.searchsorted(vi, np.arange(len(tg))), 0, len(vi) - 1)]
            out[~valid] = out[near[~valid]]
    return tg, out.reshape((len(tg),) + x.shape[1:]), valid


def occluded_rois(frac: dict, thresh: float = 0.35) -> set:
    """ROIs whose median skin fraction is low (mask, beard, hand, hair)."""
    return {n for n, f in frac.items() if n != "face_all" and len(f) and np.nanmedian(f) < thresh}


__all__ = ["FaceTrace", "extract_frame", "resample_uniform", "ita_category", "ROI_NAMES",
           "occluded_rois", "LOWER_FACE"]
