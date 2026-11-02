"""Face detection, multi-face tracking (SP Cup C3) and landmark-anchored ROIs.

Detector back-ends (best first):
  * MediaPipe FaceLandmarker (.task model, 478 landmarks) - recommended on PC/RPi
  * OpenCV YuNet (.onnx, box + 5 landmarks)               - fast, robust at 320x240
  * Haar cascade (ships with OpenCV)                      - fallback / zero-download
All are reduced to the same 4 anchor points (eye L, eye R, nose, mouth centre),
from which ROIs are placed in a face-normalised frame (origin = mid-eyes,
unit = inter-ocular distance). This makes ROIs rotation/scale invariant and lets
every back-end share one ROI definition.

Fixes relative to rPPG-Toolbox BaseLoader.crop_face_resize:
  * detection/tracking every frame + One-Euro smoothing (toolbox default: ONE
    box from frame 0 for the whole video; with dynamic detection the box jumps
    every N frames, injecting a periodic artefact at fs/N Hz, i.e. 60 bpm for
    N=30 at 30 fps - right inside the HR band)
  * all faces are tracked with persistent IDs instead of 'largest face only'
  * no silent fallback to a full-frame crop with swapped H/W when no face found.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import cv2

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class Detection:
    box: np.ndarray                 # x, y, w, h (float)
    anchors: np.ndarray             # (4, 2): eyeL, eyeR, nose, mouth (image coords)
    score: float = 1.0
    landmarks: np.ndarray | None = None


def _anchors_from_box(box) -> np.ndarray:
    x, y, w, h = box
    return np.array([[x + 0.31 * w, y + 0.40 * h], [x + 0.69 * w, y + 0.40 * h],
                     [x + 0.50 * w, y + 0.60 * h], [x + 0.50 * w, y + 0.79 * h]], float)


class HaarDetector:
    """Haar cascade on a downscaled grey image (<= det_width px wide). Tries the
    plain image first and a CLAHE-equalised one only if nothing is found, which
    rescues dark frames (C1) without hurting well-lit ones."""

    def __init__(self, min_rel_size: float = 0.08, det_width: int = 320):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.det = cv2.CascadeClassifier(path)           # created once (toolbox: per call)
        self.min_rel = min_rel_size
        self.det_w = det_width
        self.clahe = cv2.createCLAHE(2.0, (8, 8))

    def _run(self, gray, ms):
        f = self.det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(ms, ms))
        return f if len(f) else []

    def __call__(self, frame_rgb, search: tuple | None = None):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        ox = oy = 0
        if search is not None:
            x0, y0, x1, y1 = search
            gray = gray[y0:y1, x0:x1]
            ox, oy = x0, y0
        sc = min(1.0, self.det_w / max(gray.shape[1], 1))
        g = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else gray
        ms = max(int(self.min_rel * min(frame_rgb.shape[:2]) * sc), 20)
        faces = self._run(g, ms)
        if not len(faces):
            faces = self._run(self.clahe.apply(g), ms)
        out = []
        for (x, y, w, h) in faces:
            b = np.array([x / sc + ox, y / sc + oy, w / sc, h / sc], float)
            out.append(Detection(b, _anchors_from_box(b)))
        return out


class YuNetDetector:
    def __init__(self, onnx_path: str, score: float = 0.6):
        self.det = cv2.FaceDetectorYN.create(onnx_path, "", (320, 320), score, 0.3, 50)

    def __call__(self, frame_rgb, search=None):
        h, w = frame_rgb.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        out = []
        for f in (faces if faces is not None else []):
            b = f[:4].astype(float)
            lm = f[4:14].reshape(5, 2)            # reye, leye, nose, rmouth, lmouth (subject)
            eye_a, eye_b = lm[0], lm[1]
            el, er = (eye_a, eye_b) if eye_a[0] < eye_b[0] else (eye_b, eye_a)
            anchors = np.array([el, er, lm[2], (lm[3] + lm[4]) / 2], float)
            out.append(Detection(b, anchors, float(f[14])))
        return out


class MediaPipeDetector:
    """MediaPipe Tasks FaceLandmarker. Download face_landmarker.task from the
    MediaPipe model page and pass its path."""
    L_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
    R_EYE = [362, 263, 387, 386, 385, 373, 374, 380]

    def __init__(self, task_path: str, max_faces: int = 4):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=task_path),
                                            num_faces=max_faces, running_mode=vision.RunningMode.IMAGE)
        self.mp = mp
        self.lm = vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, frame_rgb, search=None):
        h, w = frame_rgb.shape[:2]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        res = self.lm.detect(img)
        out = []
        for face in res.face_landmarks:
            p = np.array([[q.x * w, q.y * h] for q in face])
            x0, y0 = p.min(0); x1, y1 = p.max(0)
            a, b = p[self.L_EYE].mean(0), p[self.R_EYE].mean(0)
            el, er = (a, b) if a[0] < b[0] else (b, a)
            anchors = np.array([el, er, p[1], (p[13] + p[14]) / 2], float)
            out.append(Detection(np.array([x0, y0, x1 - x0, y1 - y0]), anchors, 1.0, p))
        return out


def make_detector(kind: str = "auto", model_path: str | None = None):
    if kind in ("mediapipe", "auto") and model_path and model_path.endswith(".task"):
        return MediaPipeDetector(model_path)
    if kind in ("yunet", "auto") and model_path and model_path.endswith(".onnx"):
        return YuNetDetector(model_path)
    return HaarDetector()


class OneEuro:
    """One-Euro filter (Casiez 2012): heavy smoothing when still, low lag when moving."""

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = None
        self.t = None

    @staticmethod
    def _a(cut, dt):
        r = 2 * math.pi * cut * dt
        return r / (r + 1)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx = self.dx + self._a(self.dc, dt) * (dx - self.dx)
        cut = self.mc + self.beta * np.abs(self.dx)
        a = self._a(cut, dt)
        self.x = self.x + a * (x - self.x)
        self.t = t
        return self.x


@dataclass
class Track:
    tid: int
    det: Detection
    filt: OneEuro = field(default_factory=lambda: OneEuro(1.0, 0.02))
    misses: int = 0
    hits: int = 1
    smoothed: np.ndarray | None = None   # smoothed anchors (4,2)

    @property
    def box(self):
        return self.det.box


def _iou(a, b):
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


class FaceTracker:
    """Multi-face tracker: Hungarian matching on (1 - IoU) + centre distance,
    One-Euro smoothed anchors, search-window re-detection for speed."""

    def __init__(self, detector, max_misses: int = 15, full_scan_every: int = 15):
        self.det = detector
        self.tracks: list[Track] = []
        self.next_id = 0
        self.max_misses = max_misses
        self.full_every = full_scan_every
        self.k = 0

    def _detect(self, frame):
        h, w = frame.shape[:2]
        if self.tracks and self.k % self.full_every and isinstance(self.det, HaarDetector):
            dets = []
            for tr in self.tracks:                         # local search = fast + stable
                x, y, bw, bh = tr.box
                s = (int(max(0, x - 0.6 * bw)), int(max(0, y - 0.6 * bh)),
                     int(min(w, x + 1.6 * bw)), int(min(h, y + 1.6 * bh)))
                dets += self.det(frame, s)
            return dets
        return self.det(frame)

    def update(self, frame, t: float) -> list[Track]:
        dets = self._detect(frame)
        self.k += 1
        n, m = len(self.tracks), len(dets)
        matched_t, matched_d = set(), set()
        if n and m:
            cost = np.zeros((n, m))
            for i, tr in enumerate(self.tracks):
                c0 = tr.box[:2] + tr.box[2:] / 2
                for j, d in enumerate(dets):
                    c1 = d.box[:2] + d.box[2:] / 2
                    dist = np.linalg.norm(c0 - c1) / (tr.box[2] + 1e-9)
                    cost[i, j] = (1 - _iou(tr.box, d.box)) + 0.5 * dist
            if linear_sum_assignment is not None:
                rows, cols = linear_sum_assignment(cost)
            else:
                rows, cols = np.unravel_index(np.argsort(cost, axis=None), cost.shape)
            for i, j in zip(rows, cols):
                if i in matched_t or j in matched_d or cost[i, j] > 1.3:
                    continue
                tr = self.tracks[i]
                tr.det, tr.misses, tr.hits = dets[j], 0, tr.hits + 1
                matched_t.add(i); matched_d.add(j)
        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.misses += 1                                # keep last box (coasting)
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks.append(Track(self.next_id, d)); self.next_id += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses]
        for tr in self.tracks:
            if tr.misses == 0:
                tr.smoothed = tr.filt(tr.det.anchors, t).reshape(4, 2)
        return [tr for tr in self.tracks if tr.misses == 0 and tr.smoothed is not None]


# --- ROI geometry ---------------------------------------------------------
# Face frame: origin = mid-eyes, +x toward subject's left (image right), +y down,
# unit = inter-ocular distance (IOD). Rectangles: (x0, y0, x1, y1).
ROI_DEFS = {
    "forehead_l": (-0.85, -1.05, 0.00, -0.42),
    "forehead_r": (0.00, -1.05, 0.85, -0.42),
    "cheek_l":    (-1.05, 0.35, -0.38, 0.95),
    "cheek_r":    (0.38, 0.35, 1.05, 0.95),
    "nose":       (-0.18, 0.10, 0.18, 0.55),
    "chin":       (-0.45, 1.50, 0.45, 1.95),
}
LOWER_FACE = {"cheek_l", "cheek_r", "nose", "chin"}   # occluded by masks (C7)


def face_frame(anchors: np.ndarray):
    el, er, nose, mouth = anchors
    origin = (el + er) / 2
    ex = er - el
    iod = np.linalg.norm(ex) + 1e-9
    ux = ex / iod
    uy = np.array([-ux[1], ux[0]])
    if np.dot(mouth - origin, uy) < 0:   # make +y point toward the mouth
        uy = -uy
    return origin, ux * iod, uy * iod


def roi_polygons(anchors: np.ndarray) -> dict[str, np.ndarray]:
    o, ax, ay = face_frame(anchors)
    polys = {}
    for name, (x0, y0, x1, y1) in ROI_DEFS.items():
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        polys[name] = o + pts[:, :1] * ax + pts[:, 1:] * ay
    return polys
