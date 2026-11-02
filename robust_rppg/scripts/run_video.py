"""Offline: one video -> per-face HR series, video HR, rPPG waveform, ITA.
python scripts/run_video.py video.mp4 --out out_dir [--detector yunet --model face_detection_yunet_2023mar.onnx]
"""
import argparse, os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robust_rppg.pipeline import RPPGPipeline, read_video
from robust_rppg.analysis import Config
from robust_rppg.export import to_mat, to_json

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--out", default="out")
ap.add_argument("--detector", default="auto"); ap.add_argument("--model", default=None)
ap.add_argument("--preset", default="full", choices=["full", "lean"])
ap.add_argument("--max-side", type=int, default=640)
ap.add_argument("--win", type=float, default=10.0); ap.add_argument("--max-rate", type=float, default=3.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
pl = RPPGPipeline(a.detector, a.model, Config.preset(a.preset, win_sec=a.win, max_rate_bpm_s=a.max_rate))
t0 = time.time(); res = pl.run(read_video(a.video, a.max_side))
stem = os.path.splitext(os.path.basename(a.video))[0]
tgt = RPPGPipeline.target(res, pl.traces)
for tid, r in res.items():
    to_mat(r, os.path.join(a.out, f"{stem}_face{tid}.mat")); to_json(r, os.path.join(a.out, f"{stem}_face{tid}.json"))
    print(f"face {tid}{' (target)' if tid == tgt else ''}: HR {r.hr_video:.1f} bpm, conf {r.conf_video:.2f}, "
          f"spectral-HR {r.hr_video_spectral:.1f}, ITA {r.ita if r.ita is None else round(r.ita, 1)} ({r.ita_category}), occluded {r.occluded}")
print(f"{pl.timing['frames']} frames in {time.time() - t0:.1f}s")
