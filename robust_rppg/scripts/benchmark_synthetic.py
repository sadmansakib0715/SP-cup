"""Controlled benchmark: rPPG-Toolbox's own unsupervised path (first-frame Haar
box x1.5 -> 72x72 -> toolbox POS_WANG -> toolbox calculate_metric_per_video)
versus this system and its ablations, on synthetic videos stressing SP Cup
constraints. Truth = exact simulated HR (not a PPG-derived label).

usage: python scripts/benchmark_synthetic.py --toolbox ../rPPG-Toolbox --out results
"""
import argparse, json, os, sys, time
from dataclasses import replace, asdict
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robust_rppg.simulate import SimCfg, generate
from robust_rppg.pipeline import RPPGPipeline
from robust_rppg.analysis import analyze, Config

CONDS = {
    "clean":             dict(),
    "motion_talking":    dict(motion=10, rotation=5, shading=0.15, bob_px=2.5),
    "low_light":         dict(lux_scale=0.12, noise=1.0, exposure_steps=0.04),
    "dark_skin":         dict(skin_dark=1.0),
    "mask":              dict(mask=True, motion=4),
    "lowres_jpeg_10fps": dict(size=(320, 240), fps=10, jpeg_q=25, motion=4),
    "multi_face":        dict(n_faces=2, size=(640, 360), motion=4),
    "hard_combo":        dict(lux_scale=0.25, skin_dark=0.7, motion=8, rotation=4, shading=0.12, bob_px=2.0,
                              jpeg_q=40, size=(320, 240), fps=15, exposure_steps=0.03),
}
HR_RANGES = [(72, 88), (52, 60), (96, 118)]
WIN = 10.0


def toolbox_baseline(frames, fs, tb):
    sys.path.insert(0, tb)
    if not hasattr(np, "mat"):          # toolbox uses np.mat, removed in NumPy 2.0
        np.mat = np.asmatrix
    from unsupervised_methods.methods.POS_WANG import POS_WANG
    from evaluation.post_process import calculate_metric_per_video
    f0 = frames[0][1]
    det = cv2.CascadeClassifier(os.path.join(tb, "dataset/haarcascade_frontalface_default.xml"))
    z = det.detectMultiScale(f0.astype(np.uint8))
    if len(z) < 1:
        box = [0, 0, f0.shape[0], f0.shape[1]]          # toolbox fallback (H/W swapped, as in BaseLoader)
    else:
        box = list(z[np.argmax(z[:, 2])])
    c = 1.5                                              # LARGE_BOX_COEF
    box[0] = max(0, box[0] - (c - 1) / 2 * box[2]); box[1] = max(0, box[1] - (c - 1) / 2 * box[3])
    box[2] = c * box[2]; box[3] = c * box[3]
    box = np.asarray(box, int)
    crops = np.stack([cv2.resize(fr[max(box[1], 0):min(box[1] + box[3], fr.shape[0]),
                                     max(box[0], 0):min(box[0] + box[2], fr.shape[1])], (72, 72),
                                 interpolation=cv2.INTER_AREA).astype(np.float32) for _, fr in frames])
    bvp = POS_WANG(crops, fs)
    hr_video = calculate_metric_per_video(bvp, bvp, fs=fs, diff_flag=False)[1]
    wl = int(WIN * fs); centers, hrs = [], []
    for s in range(0, len(bvp) - wl + 1, wl):                 # toolbox: non-overlapping windows
        hrs.append(calculate_metric_per_video(bvp[s:s + wl], bvp[s:s + wl], fs=fs, diff_flag=False)[1])
        centers.append((s + wl / 2) / fs)
    cx = box[0] + box[2] / 2
    return dict(hr_video=float(hr_video), t=centers, hr=hrs, box_cx=float(cx), face_found=bool(len(z)))


def true_window(gt_t, gt_hr, centers, taper="uniform"):
    """Window HR truth. 'uniform' = plain mean (matches a rectangular periodogram,
    the toolbox's estimator); 'hann' = Hann-weighted mean (matches this system's
    tapered spectrum). Both are reported for every method."""
    out = []
    for c in centers:
        m = (gt_t >= c - WIN / 2) & (gt_t < c + WIN / 2)
        w = np.hanning(m.sum()) + 1e-6 if taper == "hann" else np.ones(m.sum())
        out.append(float(np.average(gt_hr[m], weights=w)))
    return out


def render_one(cond, seed, tb, out, seconds):
    """Stage 1 (slow): render video, run tracker + ROI extraction, run toolbox baseline; pickle."""
    import pickle
    raw = os.path.join(out, "raw", f"{cond}_s{seed}.pkl")
    if os.path.exists(raw):
        return raw
    hr0, hr1 = HR_RANGES[seed % len(HR_RANGES)]
    sc = replace(SimCfg(seconds=seconds, seed=seed, hr0=hr0, hr1=hr1), **CONDS[cond])
    t0 = time.time()
    frames = list(generate(sc)); gt = generate.last_gt
    pl = RPPGPipeline(); t1 = time.time()
    for t, fr in frames:
        pl.push(t, fr)
    ext_fps = len(frames) / (time.time() - t1)
    W = sc.size[0]
    face_x = [W * (k + 0.5) / sc.n_faces for k in range(sc.n_faces)]
    tmap = {}
    for tr in pl.tracker.tracks:
        bx = tr.box[0] + tr.box[2] / 2
        tmap[tr.tid] = int(np.argmin([abs(bx - fx) for fx in face_x]))
    b = toolbox_baseline(frames, sc.fps, tb)
    b["face_index"] = int(np.argmin([abs(b["box_cx"] - fx) for fx in face_x]))
    traces = {k: v for k, v in pl.traces.items() if len(v.t) >= 0.5 * len(frames)}
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    with open(raw, "wb") as f:
        pickle.dump(dict(cond=cond, seed=seed, cfg=asdict(sc), gt=gt, traces=traces, track_face=tmap,
                         toolbox=b, extract_fps=ext_fps), f)
    print(cond, seed, "rendered in %.0fs" % (time.time() - t0), flush=True)
    return raw


VARIANTS = {
    "toolbox": None,
    "single_roi_no_tracker": dict(fuse=False, use_tracker=False, color_prior=False),
    "flat_fusion_no_tracker": dict(use_tracker=False, color_prior=False, hierarchical=False),
    "fusion_no_tracker": dict(use_tracker=False),
    "full": dict(),
}


def analyze_one(raw, out):
    """Stage 2 (fast): all analysis variants from the saved traces -> json."""
    import pickle
    d = pickle.load(open(raw, "rb"))
    gt = d["gt"]
    rec = dict(cond=d["cond"], seed=d["seed"], extract_fps=d["extract_fps"], variants={})
    b = d["toolbox"]; k = b["face_index"]
    rec["variants"]["toolbox"] = [dict(face=k, hr=b["hr"], true_win=true_window(gt["t"], gt["hr"][k], b["t"]),
                                       true_win_hann=true_window(gt["t"], gt["hr"][k], b["t"], "hann"),
                                       hr_video=b["hr_video"], true_video=float(np.mean(gt["hr"][k])))]
    for name, kw in VARIANTS.items():
        if kw is None:
            continue
        rows = []
        for tid, tr in d["traces"].items():
            k = d["track_face"].get(tid, 0)
            ta = time.time(); r = analyze(tr, Config(**kw)); dt = time.time() - ta
            if r is None:
                continue
            rows.append(dict(face=k, hr=r.hr.tolist(), conf=r.conf.tolist(), true_win=true_window(gt["t"], gt["hr"][k], r.t_hr),
                             true_win_hann=true_window(gt["t"], gt["hr"][k], r.t_hr, "hann"), hr_video_spectral=r.hr_video_spectral,
                             hr_video=r.hr_video, true_video=float(np.mean(gt["hr"][k])), analyze_s=dt,
                             ita=r.ita, ita_cat=r.ita_category, occluded=r.occluded))
        rec["variants"][name] = rows
    with open(os.path.join(out, f"{d['cond']}_s{d['seed']}.json"), "w") as f:
        json.dump(rec, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolbox", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--conds", nargs="*", default=list(CONDS))
    ap.add_argument("--reanalyze", action="store_true", help="only stage 2 on existing raw/*.pkl")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.reanalyze:
        import glob
        for raw in sorted(glob.glob(os.path.join(a.out, "raw", "*.pkl"))):
            analyze_one(raw, a.out)
        sys.exit(0)
    ap2 = a
    for seed in range(a.seeds):
        for c in a.conds:
            raw = render_one(c, seed, os.path.abspath(a.toolbox), a.out, a.seconds)
            analyze_one(raw, a.out)
