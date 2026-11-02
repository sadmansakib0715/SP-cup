"""Dataset readers returning (frames iterator, reference dict) per video.
Reference keeps BOTH the raw PPG (with its own timestamps) and any HR the
sensor itself reported, so evaluation does not depend on one HR-from-PPG
recipe (the toolbox's main labelling weakness)."""
from __future__ import annotations
import glob, json, os
import numpy as np
import cv2
from .pipeline import read_video


def ubfc_rppg(root):
    """UBFC-rPPG DATASET_2: <root>/subjectN/vid.avi + ground_truth.txt
    (line1 PPG, line2 sensor HR, line3 time in s)."""
    for d in sorted(glob.glob(os.path.join(root, "subject*")), key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or 0)):
        gt = np.loadtxt(os.path.join(d, "ground_truth.txt"))
        ppg, hr, t = gt[0], gt[1], gt[2]
        yield os.path.basename(d), (lambda d=d: read_video(os.path.join(d, "vid.avi"))), \
            dict(t_ppg=t, ppg=ppg, t_hr=t, hr_sensor=hr)


def pure(root):
    """PURE: <root>/XX-YY/XX-YY.json + XX-YY/XX-YY/*.png. Uses the real image and
    pulse timestamps (ns) - the toolbox linearly stretches the PPG instead."""
    for d in sorted(glob.glob(os.path.join(root, "*-*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        js = json.load(open(os.path.join(d, name + ".json")))
        pk = js["/FullPackage"]
        tp = np.array([p["Timestamp"] for p in pk], float) * 1e-9
        ppg = np.array([p["Value"]["waveform"] for p in pk], float)
        hr = np.array([p["Value"].get("pulseRate", np.nan) for p in pk], float)
        ti = np.array([p["Timestamp"] for p in js["/Image"]], float) * 1e-9
        t0 = ti[0]
        pngs = sorted(glob.glob(os.path.join(d, name, "*.png")))

        def frames(pngs=pngs, ti=ti, t0=t0):
            for p, t in zip(pngs, ti):
                yield t - t0, cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        yield name, frames, dict(t_ppg=tp - t0, ppg=ppg, t_hr=tp - t0, hr_sensor=hr)


def generic(root):
    """<root>/<id>.(mp4|avi|mkv) + <id>.csv with columns time_s,ppg[,hr]."""
    for v in sorted(glob.glob(os.path.join(root, "*"))):
        stem, ext = os.path.splitext(v)
        if ext.lower() not in (".mp4", ".avi", ".mkv", ".mov") or not os.path.exists(stem + ".csv"):
            continue
        a = np.genfromtxt(stem + ".csv", delimiter=",", names=True)
        ref = dict(t_ppg=a["time_s"], ppg=a["ppg"])
        if "hr" in a.dtype.names:
            ref.update(t_hr=a["time_s"], hr_sensor=a["hr"])
        yield os.path.basename(stem), (lambda v=v: read_video(v)), ref


READERS = {"ubfc-rppg": ubfc_rppg, "pure": pure, "generic": generic}
