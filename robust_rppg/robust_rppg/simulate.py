"""Physics-inspired synthetic rPPG video generator for regression testing.

Not a substitute for real data - used to stress each SP Cup constraint in a
controlled way with exactly known ground truth: low light + 8-bit quantisation
(C1), darker skin with weaker pulsatility (C2), multiple faces (C3), low
resolution (C4), JPEG-compression & low fps (C5), masks (C7), plus motion and
illumination drift. Face texture: scikit-image 'astronaut' (public domain).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

PULSE_SIG = np.array([0.27, 0.80, 0.54])   # relative pulsatile strength R,G,B (blood absorption)


@dataclass
class SimCfg:
    seconds: float = 30.0
    fps: float = 30.0
    size: tuple = (480, 360)       # W, H of output video
    hr0: float = 72.0
    hr1: float = 88.0              # linear HR ramp hr0 -> hr1, plus HRV
    amp: float = 0.003             # pulsatile modulation depth at G (real faces: ~0.1-0.5 %)
    shading: float = 0.06          # motion-coupled directional shading across the face
    bob_px: float = 0.0            # in-band periodic motion (talking/nodding) amplitude
    bob_hz: float = 1.55           # ... at 93 bpm, inside the HR band
    exposure_steps: float = 0.0    # auto-exposure/gain jumps (fraction) every few seconds
    lux_scale: float = 1.0         # <1 darkens the scene (low light) before quantisation
    noise: float = 1.5             # sensor read noise (DN, before quantisation)
    skin_dark: float = 0.0         # 0 = original, 1 = very dark (Fitzpatrick V-VI like)
    motion: float = 0.0            # px amplitude of head translation
    rotation: float = 0.0          # deg amplitude of roll
    illum_drift: float = 0.05      # slow global illumination change (fraction)
    jpeg_q: int | None = None      # JPEG quality (emulates low-bitrate compression)
    mask: bool = False             # surgical mask over lower face
    n_faces: int = 1
    seed: int = 0


def _face_texture():
    from skimage.data import astronaut
    im = astronaut()
    face = im[40:200, 150:300].copy()            # 160x150 crop around the face
    ycc = cv2.cvtColor(face, cv2.COLOR_RGB2YCrCb)
    skin = ((ycc[..., 1] > 135) & (ycc[..., 1] < 180) & (ycc[..., 2] > 85) & (ycc[..., 2] < 135)).astype(np.float32)
    skin = cv2.GaussianBlur(skin, (7, 7), 2)
    return face.astype(np.float32), skin


def ppg_wave(t, hr_bpm, rng):
    ph = 2 * np.pi * np.cumsum(hr_bpm / 60.0) * (t[1] - t[0])
    w = np.sin(ph) + 0.35 * np.sin(2 * ph - 0.9) + 0.12 * np.sin(3 * ph - 1.6)  # systolic + dicrotic
    return (w - w.mean()) / w.std()


def generate(cfg: SimCfg):
    """Yields (t, frame_rgb_uint8); returns ground truth via .gt attribute of the generator wrapper."""
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.seconds * cfg.fps)
    t = np.arange(n) / cfg.fps
    face, skin = _face_texture()
    if cfg.skin_dark > 0:
        lab = cv2.cvtColor(face / 255.0, cv2.COLOR_RGB2Lab)
        lab[..., 0] = lab[..., 0] * (1 - 0.6 * cfg.skin_dark * skin)
        lab[..., 2] = lab[..., 2] + 10 * cfg.skin_dark * skin
        face = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1) * 255
    W, H = cfg.size
    faces = []
    for k in range(cfg.n_faces):
        hr0 = cfg.hr0 + 17 * k
        hr1 = cfg.hr1 - 9 * k
        hrv = 1.5 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6)) + np.cumsum(rng.standard_normal(n)) * 0.02
        hr = np.linspace(hr0, hr1, n) + hrv
        faces.append(dict(hr=hr, ppg=ppg_wave(t, hr, rng), phase=rng.uniform(0, 6)))
    amp_map = cv2.GaussianBlur(rng.uniform(0.5, 1.5, skin.shape).astype(np.float32), (31, 31), 10) * skin
    scale = min(H / 360.0, W / (220.0 * cfg.n_faces)) * 1.25
    fh, fw = int(face.shape[0] * scale), int(face.shape[1] * scale)
    face_s = cv2.resize(face, (fw, fh)); amp_s = cv2.resize(amp_map, (fw, fh)); skin_s = cv2.resize(skin, (fw, fh))
    yy, xx = np.mgrid[0:fh, 0:fw].astype(np.float32)
    xn, yn = (xx / fw - 0.5) * 2, (yy / fh - 0.5) * 2
    gain_t = np.ones(n)
    if cfg.exposure_steps > 0:
        g = 1.0
        for i0 in range(n):
            if rng.random() < 1.0 / (3.0 * cfg.fps):
                g = 1 + rng.uniform(-cfg.exposure_steps, cfg.exposure_steps)
            gain_t[i0] = g
    bg = np.full((H, W, 3), (70, 80, 95), np.float32)
    bg += cv2.GaussianBlur(rng.uniform(-25, 25, (H, W, 1)).astype(np.float32), (0, 0), 25)[..., None]
    for i in range(n):
        fr = bg.copy()
        for k, fd in enumerate(faces):
            mod = 1.0 - cfg.amp * (1 - 0.55 * cfg.skin_dark) * fd["ppg"][i] * amp_s[..., None] * (PULSE_SIG / PULSE_SIG[1])
            yaw = np.sin(2 * np.pi * 0.23 * t[i] + fd["phase"])
            shade = 1 + cfg.shading * (yaw * xn + 0.5 * yaw ** 2 * xn ** 2 - 0.3 * np.sin(2 * np.pi * 0.17 * t[i]) * yn)
            img = face_s * mod * shade[..., None]
            cx = W * (k + 0.5) / cfg.n_faces + cfg.motion * (np.sin(2 * np.pi * 0.23 * t[i] + fd["phase"]) + 0.3 * np.sin(2 * np.pi * 0.9 * t[i]))
            cy = H * 0.5 + 0.6 * cfg.motion * np.sin(2 * np.pi * 0.17 * t[i] + 1 + fd["phase"]) \
                + cfg.bob_px * np.sin(2 * np.pi * cfg.bob_hz * t[i])
            ang = cfg.rotation * np.sin(2 * np.pi * 0.13 * t[i] + fd["phase"])
            M = cv2.getRotationMatrix2D((fw / 2, fh / 2), ang, 1.0)
            M[:, 2] += [cx - fw / 2, cy - fh / 2]
            warped = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            alpha = cv2.warpAffine(np.ones((fh, fw), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
            if cfg.mask:
                mk = np.zeros((fh, fw), np.float32)
                pts = np.array([[0.12 * fw, 0.62 * fh], [0.88 * fw, 0.62 * fh], [0.80 * fw, 1.0 * fh], [0.2 * fw, 1.0 * fh]], np.int32)
                cv2.fillConvexPoly(mk, pts, 1.0)
                mkw = cv2.warpAffine(mk, M, (W, H))[..., None]
                warped = warped * (1 - mkw) + mkw * np.array([150, 200, 225], np.float32) * alpha[..., None]
            fr = fr * (1 - alpha[..., None]) + warped
        illum = 1 + cfg.illum_drift * np.sin(2 * np.pi * 0.05 * t[i]) + 0.01 * np.sin(2 * np.pi * 0.31 * t[i])
        fr = fr * illum * cfg.lux_scale * gain_t[i]
        fr = fr + rng.normal(0, cfg.noise, fr.shape) + rng.normal(0, 1, fr.shape) * np.sqrt(np.maximum(fr, 0) * 0.05)
        fr = np.clip(np.round(fr), 0, 255).astype(np.uint8)
        if cfg.jpeg_q:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_q])
            fr = cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)
        yield t[i], fr
    generate.last_gt = dict(t=t, hr=[fd["hr"] for fd in faces], ppg=[fd["ppg"] for fd in faces])


def gt_for(cfg: SimCfg):
    """Re-create the ground truth without rendering frames."""
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.seconds * cfg.fps); t = np.arange(n) / cfg.fps
    _ = rng  # same draw order as generate() for HR/HRV
    hrs, ppgs = [], []
    for k in range(cfg.n_faces):
        hrv = 1.5 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6)) + np.cumsum(rng.standard_normal(n)) * 0.02
        hr = np.linspace(cfg.hr0 + 17 * k, cfg.hr1 - 9 * k, n) + hrv
        hrs.append(hr); ppgs.append(ppg_wave(t, hr, rng)); rng.uniform(0, 6)
    return dict(t=t, hr=hrs, ppg=ppgs)
