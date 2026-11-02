"""Evaluate on UBFC-rPPG / PURE / generic folder with corrected metrics.
python scripts/eval_dataset.py --dataset ubfc-rppg --root /data/UBFC/DATASET_2 --out eval_ubfc
Reports, per reference type (PPG-spectral, PPG-peaks, sensor HR):
  window-level (10 s, 1 s step) and video-level MAE/RMSE/MAPE/r/within-5,
  repeated-measures Bland-Altman, subject-bootstrap CIs, ITA-stratified tables.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robust_rppg.pipeline import RPPGPipeline
from robust_rppg.analysis import Config
from robust_rppg.datasets import READERS
from robust_rppg.evaluate import windowed_reference, metrics, stratified, bland_altman_plot, hr_from_ppg
from robust_rppg.export import to_mat

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(READERS)); ap.add_argument("--root", required=True)
ap.add_argument("--out", default="eval_out"); ap.add_argument("--detector", default="auto"); ap.add_argument("--model")
ap.add_argument("--preset", default="full", choices=["full", "lean"])
ap.add_argument("--win", type=float, default=10.0); ap.add_argument("--subject-from", default="name",
                help="'name' or 'prefix2' (PURE: '01-02' -> subject '01')")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rows_w, rows_v = [], []
for name, frames, ref in READERS[a.dataset](a.root):
    pl = RPPGPipeline(a.detector, a.model, Config.preset(a.preset, win_sec=a.win))
    res = pl.run(frames())
    if not res:
        print(name, "no face track"); continue
    r = res[RPPGPipeline.target(res, pl.traces)]
    to_mat(r, os.path.join(a.out, f"{name}.mat"))
    subj = name[:2] if a.subject_from == "prefix2" else name
    fs_ppg = 1 / np.median(np.diff(ref["t_ppg"]))
    refs = {"ppg_spec": windowed_reference(ref["t_ppg"], ref["ppg"], r.t_hr, a.win, fs_ppg, "spectral"),
            "ppg_peaks": windowed_reference(ref["t_ppg"], ref["ppg"], r.t_hr, a.win, fs_ppg, "peaks")}
    if "hr_sensor" in ref:
        hs = np.asarray(ref["hr_sensor"], float); th = np.asarray(ref["t_hr"], float)
        refs["sensor"] = np.array([np.nanmedian(hs[(th >= c - a.win / 2) & (th < c + a.win / 2)]) for c in r.t_hr])
    for i, c in enumerate(r.t_hr):
        rows_w.append(dict(video=name, subject=subj, t=float(c), pred=float(r.hr[i]), conf=float(r.conf[i]),
                           ita_cat=r.ita_category, **{f"ref_{k}": float(v[i]) for k, v in refs.items()}))
    vref = {"ppg_spec": hr_from_ppg(np.interp(np.arange(ref["t_ppg"][0], ref["t_ppg"][-1], 1 / fs_ppg), ref["t_ppg"], ref["ppg"]), fs_ppg)}
    if "hr_sensor" in ref:
        vref["sensor"] = float(np.nanmedian(ref["hr_sensor"]))
    rows_v.append(dict(video=name, subject=subj, pred=r.hr_video, pred_spectral=r.hr_video_spectral, ita=r.ita, ita_cat=r.ita_category,
                       **{f"ref_{k}": v for k, v in vref.items()}))
    print(f"{name}: pred {r.hr_video:.1f} | " + " ".join(f"{k} {v:.1f}" for k, v in vref.items()), flush=True)

report = {}
for k in sorted({c[4:] for r in rows_w for c in r if c.startswith("ref_")}):
    rr = [r for r in rows_w if np.isfinite(r.get(f"ref_{k}", np.nan))]
    report[f"window_{k}"] = metrics([r["pred"] for r in rr], [r[f"ref_{k}"] for r in rr], [r["subject"] for r in rr])
    report[f"window_{k}_by_ITA"] = stratified([dict(r, ref=r[f"ref_{k}"]) for r in rr], "ita_cat")
    bland_altman_plot([r["pred"] for r in rr], [r[f"ref_{k}"] for r in rr], [r["subject"] for r in rr],
                      os.path.join(a.out, f"BA_window_{k}.png"), f"{a.dataset} 10s windows vs {k}")
for k in sorted({c[4:] for r in rows_v for c in r if c.startswith("ref_")}):
    rr = [r for r in rows_v if np.isfinite(r.get(f"ref_{k}", np.nan))]
    report[f"video_{k}"] = metrics([r["pred"] for r in rr], [r[f"ref_{k}"] for r in rr], [r["subject"] for r in rr])
json.dump(dict(report=report, windows=rows_w, videos=rows_v), open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
for k, v in report.items():
    if "by_ITA" not in k:
        print(f"{k:22s} MAE {v['MAE']:.2f}  RMSE {v['RMSE']:.2f}  r {v['r']:.3f}  bias {v['bias']:.2f}  LoA [{v['LoA_low']:.2f}, {v['LoA_high']:.2f}]  n={v['n']}")
