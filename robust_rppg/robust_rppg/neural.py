"""Optional: plug a deep rPPG model into the fusion as one more candidate.

The fusion is model-agnostic: any callable that maps aligned face crops
(T, H, W, 3, uint8) at fs -> BVP (T,) can be added. Its window-wise SQI then
decides how much it counts, so a network that collapses under low light or
masks is down-weighted automatically instead of dominating.

`aligned_crops` warps each frame with the tracked similarity transform
(eyes fixed) - a stabilised input, unlike the toolbox's fixed/jumping box.

`ToolboxModel` wraps rPPG-Toolbox checkpoints (e.g. PURE_PhysNet_DiffNormalized,
UBFC-rPPG_FactorizePhys_FSAM_Res). Requires torch; NOT exercised in this
package's tests (no torch in the build sandbox) - verify on your machine and
mind the toolbox's RAIL license before shipping its weights in a submission.
"""
from __future__ import annotations
import numpy as np
import cv2
from .face import face_frame


def aligned_crop(frame_rgb, anchors, size=72, scale=3.5):
    """scale=3.5 IOD ~ the toolbox's 1.5x Haar box, i.e. the crop the pretrained models saw."""
    o, ax, ay = face_frame(anchors)
    iod = np.linalg.norm(ax)
    centre = o + 0.55 * ay                      # between eyes and mouth
    half = scale * iod / 2
    ux, uy = ax / iod, ay / np.linalg.norm(ay)
    src = np.float32([centre - half * ux - half * uy, centre + half * ux - half * uy, centre - half * ux + half * uy])
    dst = np.float32([[0, 0], [size, 0], [0, size]])
    return cv2.warpAffine(frame_rgb, cv2.getAffineTransform(src, dst), (size, size), flags=cv2.INTER_AREA)


def diff_normalize_causal(x: np.ndarray) -> np.ndarray:
    """DiffNormalized frames using only the chunk itself (toolbox: whole-video std)."""
    x = x.astype(np.float32)
    d = (x[1:] - x[:-1]) / (x[1:] + x[:-1] + 1e-7)
    d = d / (d.std() + 1e-7)
    return np.concatenate([d, np.zeros_like(d[:1])], 0)


class ToolboxModel:
    """Input/output conventions taken from the toolbox's own inference configs:
      PhysNet       : 128-frame chunks, DiffNormalized input, DiffNormalized output -> cumsum
      FactorizePhys : 160-frame chunks, Raw 0-255 input + repeated last frame (the model
                      differences internally), Standardized pulse output
    """
    SPECS = {"PhysNet": dict(chunk=128, inp="diff", out="diff"),
             "FactorizePhys": dict(chunk=160, inp="raw+1", out="pulse")}

    def __init__(self, toolbox_path: str, arch: str, weights: str, size=72, device="cpu"):
        import sys, torch
        sys.path.insert(0, toolbox_path)
        if arch not in self.SPECS:
            raise ValueError(f"supported: {list(self.SPECS)}")
        self.torch, self.dev, self.size, self.arch = torch, device, size, arch
        self.chunk = self.SPECS[arch]["chunk"]
        if arch == "PhysNet":
            from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX as Net
            self.net = Net(frames=self.chunk)
        else:
            from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys as Net
            md = dict(FRAME_NUM=self.chunk, MD_TYPE="NMF", MD_FSAM=False, MD_TRANSFORM="T_KAB", MD_S=1, MD_R=1,
                      MD_STEPS=4, MD_INFERENCE=True, MD_RESIDUAL=True)
            self.net = Net(frames=self.chunk, md_config=md, in_channels=3, dropout=0.1, device=torch.device(device))
        sd = torch.load(weights, map_location=device)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        if missing:
            print(f"[ToolboxModel] warning: {len(missing)} missing keys, e.g. {missing[:3]}")
        self.net.eval().to(device)

    def __call__(self, crops: np.ndarray, fs: float) -> np.ndarray:
        """crops: (T, size, size, 3) uint8 RGB from aligned_crop. Returns BVP (T,), +ve = blood volume."""
        torch, spec = self.torch, self.SPECS[self.arch]
        T = len(crops); out = np.zeros(T); wsum = np.zeros(T)
        hop = self.chunk // 2; han = np.hanning(self.chunk) + 1e-3
        for s in range(0, max(T - self.chunk, 0) + 1, hop):
            seg = crops[s:s + self.chunk].astype(np.float32)
            if len(seg) < self.chunk:
                break
            if spec["inp"] == "diff":
                x = diff_normalize_causal(seg)
            else:
                x = np.concatenate([seg, seg[-1:]], 0)
            xt = torch.from_numpy(np.ascontiguousarray(np.transpose(x, (3, 0, 1, 2))[None])).float().to(self.dev)
            with torch.no_grad():
                y = self.net(xt)
            y = (y[0] if isinstance(y, (tuple, list)) else y).reshape(-1).cpu().numpy()[:self.chunk]
            if spec["out"] == "diff":
                y = np.cumsum(y)
            y = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
            y = y / (y.std() + 1e-9)
            out[s:s + self.chunk] += han * y; wsum[s:s + self.chunk] += han
        return np.where(wsum > 0, out / np.maximum(wsum, 1e-9), 0)


def collect_crops(frames, tracker, size=72):
    """Run a FaceTracker over frames and return {track_id: (t, crops)} for ToolboxModel."""
    out = {}
    for t, fr in frames:
        for tr in tracker.update(fr, t):
            ts, cs = out.setdefault(tr.tid, ([], []))
            ts.append(t); cs.append(aligned_crop(fr, tr.smoothed, size))
    return {k: (np.asarray(ts), np.stack(cs)) for k, (ts, cs) in out.items()}
