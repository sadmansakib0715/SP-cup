"""Result export (.mat / .csv / .json). Variable and file names are
configurable because the SP Cup 2027 template fixes them - set NAMES to the
exact names in the official document before submitting."""
from __future__ import annotations
import json
import numpy as np
from scipy.io import savemat

NAMES = dict(hr_series="HR", t_series="t", hr_video="HR_video", hr_video_spectral="HR_video_spectral", conf="confidence",
             bvp="rPPG", t_bvp="t_rPPG", fs="fs", ita="ITA", ita_cat="ITA_category")


def to_mat(res, path, names=NAMES, extra=None):
    d = {names["hr_series"]: np.asarray(res.hr, float), names["t_series"]: np.asarray(res.t_hr, float),
         names["hr_video"]: float(res.hr_video), names["hr_video_spectral"]: float(res.hr_video_spectral), names["conf"]: np.asarray(res.conf, float),
         names["bvp"]: np.asarray(res.bvp, float), names["t_bvp"]: np.asarray(res.t_bvp, float),
         names["fs"]: float(res.fs), names["ita"]: np.nan if res.ita is None else float(res.ita),
         names["ita_cat"]: res.ita_category}
    d.update(extra or {})
    savemat(path, d, do_compression=True)
    return path


def to_json(res, path):
    d = dict(tid=res.tid, hr_video=res.hr_video, hr_video_spectral=res.hr_video_spectral, conf_video=res.conf_video, t=res.t_hr.tolist(),
             hr=res.hr.tolist(), conf=res.conf.tolist(), ita=res.ita, ita_category=res.ita_category,
             rois_used=res.rois_used, occluded=res.occluded, top_weights=dict(list(res.weights.items())[:10]))
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
    return path
