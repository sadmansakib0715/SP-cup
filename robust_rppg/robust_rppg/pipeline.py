"""End-to-end offline pipeline: frames (+timestamps) -> per-face results."""
from __future__ import annotations
import time
from typing import Iterable
import numpy as np
import cv2
from .face import FaceTracker, make_detector
from .extract import FaceTrace, extract_frame
from .analysis import analyze, Config, FaceResult


def read_video(path: str, max_side: int | None = None):
    """Yields (t_sec, frame_rgb). Uses container timestamps when available
    (variable-frame-rate safe); falls back to index / nominal fps."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last_t = 0, -1.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not np.isfinite(t) or t <= last_t:
            t = i / fps
        last_t = t
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        if max_side and max(fr.shape[:2]) > max_side:
            s = max_side / max(fr.shape[:2])
            fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        yield t, fr
        i += 1
    cap.release()


class RPPGPipeline:
    def __init__(self, detector="auto", model_path=None, cfg: Config | None = None,
                 min_track_frames: int = 30):
        det = make_detector(detector, model_path) if isinstance(detector, str) else detector
        self.tracker = FaceTracker(det)
        self.cfg = cfg or Config()
        self.traces: dict[int, FaceTrace] = {}
        self.min_track = min_track_frames
        self.timing = {"frames": 0, "extract_s": 0.0, "analyze_s": 0.0}

    def push(self, t: float, frame_rgb: np.ndarray):
        t0 = time.perf_counter()
        for tr in self.tracker.update(frame_rgb, t):
            ft = self.traces.setdefault(tr.tid, FaceTrace(tr.tid))
            extract_frame(frame_rgb, tr.smoothed, ft, t)
        self.timing["frames"] += 1
        self.timing["extract_s"] += time.perf_counter() - t0

    def run(self, frames: Iterable) -> dict[int, FaceResult]:
        for t, fr in frames:
            self.push(t, fr)
        return self.finish()

    def finish(self) -> dict[int, FaceResult]:
        t0 = time.perf_counter()
        out = {}
        for tid, tr in self.traces.items():
            if len(tr.t) >= self.min_track:
                r = analyze(tr, self.cfg)
                if r is not None:
                    out[tid] = r
        self.timing["analyze_s"] += time.perf_counter() - t0
        return out

    @staticmethod
    def target(results: dict[int, FaceResult], traces: dict[int, FaceTrace] | None = None, policy="longest"):
        """Pick the measured subject when several faces are present (C3)."""
        if not results:
            return None
        if policy == "longest" and traces:
            return max(results, key=lambda k: len(traces[k].t))
        return max(results, key=lambda k: results[k].conf_video)
