"""Real-time multi-face HR from a webcam / RTSP / file stream.

Each face keeps a ring buffer of ROI traces (win_sec + margin). Once per
step_sec the last window is analysed (same candidates + SQI fusion as offline)
and the fused spectrum is fed to a *causal* HMM forward filter, so the live HR
uses only past samples (no whole-video normalisation as in the toolbox).
Designed to run on a Raspberry Pi 4/5 at 320x240 (see README for timing).
"""
from __future__ import annotations
import time
import numpy as np
import cv2
from .face import FaceTracker, make_detector
from .extract import FaceTrace, extract_frame
from .analysis import analyze, Config, HRTracker
from .spectral import interp_peak


class LiveHR:
    def __init__(self, detector="auto", model_path=None, cfg: Config | None = None, margin_sec=2.0, preset="lean"):
        self.cfg = cfg or Config.preset(preset, win_sec=8.0)
        det = make_detector(detector, model_path) if isinstance(detector, str) else detector
        self.tracker = FaceTracker(det)
        self.traces: dict[int, FaceTrace] = {}
        self.hmm: dict[int, HRTracker] = {}
        self.last_eval: dict[int, float] = {}
        self.state: dict[int, dict] = {}
        self.keep = self.cfg.win_sec + margin_sec
        self.lat = []

    def _trim(self, tr: FaceTrace, now):
        k = int(np.searchsorted(np.asarray(tr.t), now - self.keep))
        if k > 0:
            tr.t = tr.t[k:]
            for d in (tr.rgb, tr.frac, tr.npx):
                for n in d:
                    d[n] = d[n][k:]

    def push(self, t: float, frame_rgb: np.ndarray) -> dict[int, dict]:
        t0 = time.perf_counter()
        active = self.tracker.update(frame_rgb, t)
        for tr in active:
            ft = self.traces.setdefault(tr.tid, FaceTrace(tr.tid))
            extract_frame(frame_rgb, tr.smoothed, ft, t)
            self._trim(ft, t)
            span = ft.t[-1] - ft.t[0]
            if span >= self.cfg.win_sec * 0.95 and t - self.last_eval.get(tr.tid, -1e9) >= self.cfg.step_sec:
                self.last_eval[tr.tid] = t
                one = Config(**{**self.cfg.__dict__, "win_sec": span + 1.0})   # one window = whole buffer
                r = analyze(ft, one)
                if r is not None and r.fused_spec is not None:
                    hmm = self.hmm.setdefault(tr.tid, HRTracker(r.grid, self.cfg.step_sec, self.cfg.max_rate_bpm_s))
                    i, bel = hmm.step(r.fused_spec[-1])
                    c = r.grid[i]
                    conf = float(bel[np.abs(r.grid - c) <= 3].sum())
                    if abs(r.hr[-1] - c) <= 4:          # sub-grid refinement from the window estimate
                        c = r.hr[-1]
                    self.state[tr.tid] = dict(hr=float(c), conf=conf, box=tr.box.copy(), t=t,
                                              ita_category=r.ita_category, occluded=r.occluded)
        self.lat.append(time.perf_counter() - t0)
        live_ids = {tr.tid for tr in active}
        return {k: v for k, v in self.state.items() if k in live_ids}


def run_camera(src=0, detector="auto", model_path=None, width=320, height=240, show=True, preset="lean"):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    live = LiveHR(detector, model_path, preset=preset)
    t_start = time.time()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = time.time() - t_start
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        st = live.push(t, rgb)
        if show:
            for tid, s in st.items():
                x, y, w, h = s["box"].astype(int)
                col = (0, 200, 0) if s["conf"] > 0.5 else (0, 160, 255)
                cv2.rectangle(fr, (x, y), (x + w, y + h), col, 2)
                cv2.putText(fr, f"#{tid} {s['hr']:.0f} bpm ({s['conf']:.2f})", (x, max(15, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.putText(fr, f"{1000 * np.mean(live.lat[-30:]):.1f} ms/frame", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("robust_rppg live", fr)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release(); cv2.destroyAllWindows()
