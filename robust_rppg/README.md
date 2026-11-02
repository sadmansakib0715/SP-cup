# robust_rppg — multi-ROI, quality-weighted, tracked rPPG heart rate

A training-free, CPU-only rPPG heart-rate system built from a full audit of
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) (see `AUDIT.md`), designed around the
IEEE SP Cup 2027 (FaceVitals) constraints. Independent implementation of published algorithms
(POS, CHROM, OMIT, One-Euro, HMM decoding) — no toolbox code inside (see the license note in AUDIT.md §4).

```
frames ─► multi-face tracker ─► 6 landmark ROIs + whole-skin ─► adaptive skin mask ─► mean RGB / ROI
   (timestamps)  (IDs, One-Euro)      (forehead L/R, cheeks,        (forehead-anchored     (+ skin fraction,
                                       nose, chin)                    YCrCb model)          ITA, occlusion)
        ─► timestamp resampling ─► POS · CHROM · OMIT per ROI  =  up to 21 candidates
        ─► per 10 s window: 0.1-bpm spectrum, harmonic sum, SQI
        ─► fusion: ROIs within each method, methods weighted by spatial coherence, × pulse colour prior (G/R)
        ─► HMM over HR (Viterbi offline / forward filter live) ─► off-grid refinement ─► HR(t), confidence(t),
           video HR, fused rPPG waveform, ITA category, occluded ROIs
```

## What each component fixes in the toolbox path

| Error source in the toolbox path | What this system does |
|---|---|
| One face box from frame 0, or a box that jumps every N frames (60-bpm artefact at N=30) | Per-frame tracking with One-Euro smoothing; ROIs anchored to eyes/nose/mouth in a face-normalised frame (rotation/scale invariant) |
| Average of whole 1.5× box incl. hair/background | Skin pixels only, adaptive chroma model anchored on the forehead |
| Single signal → one corrupted region ruins the estimate | `full` preset: 7 regions × 3 projections, fused hierarchically (methods weighted by cross-region coherence of their peaks) with a haemoglobin colour-signature prior; `lean` preset: one skin-masked whole-face POS signal |
| No notion of reliability | Per-window confidence (HMM posterior × spectral peakedness), AUROC 0.92 for flagging >5 bpm errors on the synthetic suite |
| Video HR = whole-recording FFT peak (biased when HR drifts) | Confidence-weighted mean of the tracked series; spectral value kept as `hr_video_spectral` |
| Independent argmax per window → harmonic/noise jumps | Most-likely HR trajectory (HMM, physiological rate prior); posterior gives a confidence per estimate |
| HR snapped to 3.5-bpm FFT bins (10 s) | 0.1-bpm zero-padded spectrum + parabolic peak interpolation |
| Mask/occlusion pollutes the average | ROIs with low skin fraction are dropped automatically (C7) |
| Assumes uniform frame timing | Uses real timestamps, bridges short gaps, marks long gaps invalid (C5, live streams) |

## SP Cup 2027 constraint coverage

| Constraint | Handling | Status |
|---|---|---|
| C1 low light ≤50 lux | CLAHE fallback detection; skin-only averaging; confidence flags windows below the detection limit | synthetic: 0.95 bpm (toolbox 18.2, detection failure) |
| C2 Fitzpatrick I–VI, ITA strata (synthetic dark skin: 0.71 bpm, toolbox 9.06) | ITA per face (`rgb2lab`-equivalent, sRGB→Lab); stratified metrics in `evaluate.stratified` | ITA estimate is illumination/white-balance dependent — calibrate or report as such |
| C3 multi-face | Hungarian-matched tracks with persistent IDs; all faces measured; `target()` selects subject | synthetic: 0.60 bpm, both faces |
| C4 ≤640×480 / 320×240 | Detection on ≤320-px grey image; ROIs scale with inter-ocular distance | synthetic-tested at 320×240 |
| C5 ≤10 fps, ≤500 kbps | Timestamp resampling, filters capped below Nyquist; offline mode identical code path | synthetic (10 fps, JPEG q25): 1.48 full / 1.05 single ROI (toolbox 2.58) |
| C6 ages 60–70+ | HR band starts at 40 bpm; rate prior configurable | no age-specific data tested |
| C7 masks ≥40 % occlusion | Skin-fraction occlusion detection removes lower-face ROIs; forehead ROIs carry the estimate | synthetic: 0.48 bpm (Haar still found masked faces here; real masks may need YuNet/MediaPipe) |

## Results

### Synthetic stress benchmark (exact simulated truth, 30 s videos, 2 HR ranges per condition)
![benchmark](results/benchmark.png)

Window-level 10 s HR error, 8 conditions × 2 HR ranges (72→88 and 52→60 bpm), 16 videos of 30 s,
1,280 windows per method. "single ROI" = this system's preprocessing with one skin-masked whole-face POS
signal; "lean" = single ROI + HMM tracking; "full" = all candidates + hierarchical fusion + HMM.

| Condition | toolbox | single ROI, no tracking | flat fusion | hierarchical fusion, no tracking | **full** |
|---|---|---|---|---|---|
| clean | 0.33 / 0.43 | 0.58 / 0.67 | 0.60 / 0.69 | 0.60 / 0.69 | 0.60 / 0.69 |
| dark skin | 9.06 / 21.42 | 0.67 / 0.82 | 2.47 / 6.52 | 0.71 / 0.86 | 0.71 / 0.86 |
| low light (≈12 % exposure) | 18.15 / 23.72 | 0.80 / 0.92 | 0.89 / 1.05 | 1.02 / 1.62 | 0.95 / 1.30 |
| 320×240, JPEG q25, 10 fps | 2.58 / 3.56 | 1.05 / 1.31 | 2.49 / 5.80 | 2.36 / 4.13 | 1.48 / 2.12 |
| mask | 0.61 / 0.74 | 0.47 / 0.56 | 0.48 / 0.58 | 0.48 / 0.58 | 0.48 / 0.58 |
| motion + talking at 93 bpm | 0.34 / 0.43 | 0.49 / 0.59 | 1.50 / 3.57 | 4.69 / 15.03 | 0.70 / 0.99 |
| two faces | 0.30 / 0.34 | 0.62 / 0.70 | 0.60 / 0.68 | 0.60 / 0.68 | 0.60 / 0.68 |
| hard combo (dark + dim + motion + JPEG + 320×240 + 15 fps) | 21.64 / 26.27 | 45.85 / 57.94 | 22.76 / 26.65 | 20.36 / 23.12 | 20.77 / 23.74 |
| **all** | **6.63 / 14.68** | **5.69 / 19.33** | **3.60 / 9.44** | **3.49 / 9.32** | **2.99 / 7.98** |
| windows > 5 bpm | 22.9 % | 10.3 % | 11.4 % | 12.2 % | 10.8 % |

MAE / RMSE in bpm; truth = uniform window mean. `results/summary_hann.md` repeats everything against a
Hann-weighted truth (matches this system's tapered spectrum; all of our clean-condition errors drop to
~0.15 bpm, the toolbox's to 0.11 — the ~0.5 bpm "floor" above is a window-definition effect from the
simulated 0.1 Hz HR variability, not an accuracy difference).

`lean` (single ROI + tracking, run separately on the same traces): all 3.31 / 9.55; excluding hard combo
**0.66 / 0.82, 0 % > 5 bpm**; hard combo 24.5. `full` excluding hard combo: 0.77 / 1.10, 0.9 % > 5 bpm.

**What the benchmark actually shows**

1. **The big gain over the toolbox comes from preprocessing**, not fusion: per-frame tracking, landmark
   ROIs, skin-only averaging, CLAHE-assisted detection, and a 0.1-bpm spectrum. The toolbox's low-light
   collapse is a detection failure (Haar finds no face in the dark first frame and falls back to the
   whole image); its dark-skin failure happens at the low HR range.
2. **Naive fusion is harmful.** Flat SQI-weighted fusion of all 21 candidates was worse than one good
   candidate: CHROM/OMIT fail together on dark skin (noisy dim blue channel) and forehead candidates
   lock onto in-band head motion *with high SQI*. Correlated candidates must not vote independently.
3. **Hierarchical fusion + colour prior fixes that**, and full fusion has the best overall and worst-case
   numbers — but on *recoverable* conditions it does not beat the single whole-skin POS candidate.
   Both are shipped as presets (`full`, `lean`); pick on your real validation split.
4. **The HMM tracker matters for in-band motion**: without it, fusion jumps to the 93-bpm talking
   rhythm when true HR is 52–60 (4.69 → 0.70 bpm with tracking).
5. **Hard combo is beyond the detection limit**: the oracle correlation between the face's green signal
   and the true PPG is 0.17–0.27 (0.81–0.84 when clean). Pulse amplitude there is ~0.04 DN, below one
   8-bit step, and JPEG removes the dithering noise. No method recovers it; the right behaviour is to
   say so — see the confidence table below.

**Confidence / abstention** (`full`, all 16 videos pooled; confidence = √(HMM posterior × spectral
peakedness), chosen among 6 candidate scores on this same data):

| threshold | coverage | MAE kept | RMSE kept | > 5 bpm kept | hard-combo windows kept |
|---|---|---|---|---|---|
| 0.0 | 100 % | 2.99 | 7.98 | 10.8 % | 100 % |
| 0.4 | 89.9 % | 1.90 | 6.06 | 4.7 % | 40.5 % |
| 0.5 | 78.6 % | 1.35 | 4.36 | 3.0 % | 23.8 % |
| 0.6 | 58.2 % | 0.88 | 2.85 | 0.9 % | 4.8 % |

AUROC of confidence for flagging |error| > 5 bpm: 0.919. Report MAE *with* coverage if you abstain —
judges will (rightly) distrust a low MAE that silently drops hard windows.

**Cost** (one shared CPU core, Python): extraction 24–41 fps at 320–480 px (13 fps for two faces at
640×360); analysis per second of video 1.0 ms (`lean`) vs 19.6 ms (`full`); live `lean` 21 ms/frame.

**Read this honestly.** These are synthetic videos (static face texture with physically motivated pulse
colour modulation, motion-coupled shading, in-band "talking" motion, auto-exposure jumps, 8-bit
quantisation, JPEG). They test the *engineering* (tracking, masking, fusion, decoding) with exact truth,
not real-world accuracy. Real numbers must come from `scripts/eval_dataset.py` on UBFC-rPPG / PURE /
your own IRB-cleared data. The "toolbox" column runs the toolbox's own `POS_WANG` and
`calculate_metric_per_video` (first-frame Haar box ×1.5 → 72×72, its default unsupervised path).
Window MAE has a ~0.5 bpm floor because the truth is the window-mean of a ramping, HRV-modulated HR.

### Real data
Not run here (datasets not reachable from the build sandbox). Run:
```bash
python scripts/eval_dataset.py --dataset ubfc-rppg --root /data/UBFC-rPPG/DATASET_2 --out eval_ubfc
python scripts/eval_dataset.py --dataset pure --root /data/PURE --subject-from prefix2 --out eval_pure
```
Each run reports window (10 s, 1 s step) and video level metrics against **three references** —
PPG spectral, PPG peak/IBI, and the oximeter's own HR — with repeated-measures Bland–Altman,
subject-bootstrap CIs, ITA-stratified tables, BA plots and one `.mat` per video.

## Usage
```bash
pip install -r requirements.txt
python -m pytest -q tests                                     # 9 unit tests
python scripts/run_video.py clip.mp4 --out out/ [--preset lean]   # offline, all faces -> .mat/.json
python scripts/run_live.py --src 0 [--preset full]              # webcam, 320x240, HR + confidence overlay (lean default)
python scripts/run_live.py --src 0 --detector yunet --model face_detection_yunet_2023mar.onnx
python scripts/benchmark_synthetic.py --toolbox ../rPPG-Toolbox --out results     # render + analyse (~20 min, 1 core)
python scripts/benchmark_synthetic.py --toolbox ../rPPG-Toolbox --out results --reanalyze   # re-analyse saved traces
python scripts/summarize.py results [true_win|true_win_hann]
```
Detectors: Haar (built in, zero download) → YuNet ONNX (OpenCV Zoo; better at low res/angles) →
MediaPipe FaceLandmarker `.task` (best landmarks). Use YuNet or MediaPipe for real data; Haar is the
floor, and it is the weakest link under masks, profile views and very low light.

Python API:
```python
from robust_rppg.pipeline import RPPGPipeline, read_video
from robust_rppg.analysis import Config
pl = RPPGPipeline("yunet", "face_detection_yunet_2023mar.onnx", Config.preset("full", win_sec=10))
res = pl.run(read_video("clip.mp4"))            # {track_id: FaceResult}
r = res[RPPGPipeline.target(res, pl.traces)]
r.hr_video, r.t_hr, r.hr, r.conf, r.bvp, r.ita_category, r.occluded, r.weights
```

### Key parameters (`analysis.Config`)
`win_sec` (10 s; longer = steadier, slower to follow change), `max_rate_bpm_s` (HR prior, 3 bpm/s;
raise to ~6 for exercise recovery), `lo_bpm/hi_bpm` (40–180), `sqi_power`, `occlusion_frac` (0.35),
`methods`, `rois`. Ablation switches: `use_tracker`, `fuse`, `hierarchical`, `color_prior`.
Do not combine `color_prior=True` with a single candidate: on the near-noise hard-combo videos that
pushed `lean` from 24.5 to 78 bpm (the prior amplifies noise when there is no fused consensus).

### Adding a deep model
`analyze(trace, cfg, extra={"physnet": (t, bvp)})` adds any external BVP as one more fused candidate;
its weight is decided per window by its SQI. `robust_rppg/neural.py` provides stabilised face crops
(`aligned_crop`), causal Diff-normalisation and a wrapper for toolbox checkpoints (`ToolboxModel`,
requires torch; untested here; mind the RAIL license if you ship toolbox weights).

## What to do next for maximum accuracy (in priority order)
1. **Swap Haar for YuNet/MediaPipe** and run both presets on UBFC/PURE/your data; inspect `r.weights`,
   `r.occluded`, `r.evidence` on failures. Detection is the single biggest failure point in the dark.
2. **Choose preset and tune on a validation split only** (`win_sec`, `max_rate_bpm_s`, `sqi_power`,
   confidence threshold); report test once, with coverage if you abstain.
3. **Train one compact network** (e.g. EfficientPhys/TS-CAN-size for the RPi criterion) with the fixed
   temporal Pearson loss, HR-range (speed) augmentation, low-light/JPEG/10-fps augmentation, and add it as
   an `extra` candidate — fusion keeps it from hurting where it fails.
4. Report per-ITA-stratum, per-lux, per-mask results with BA plots (judges reward honest subgroup analysis).
5. Rounds 2–3: the fused rPPG waveform and per-ROI RGB traces (`trace.arrays()`) are the inputs for
   ratio-of-ratios SpO2 calibration and PTT/morphology BP features.

## Layout
```
robust_rppg/  spectral.py  methods.py  face.py  extract.py  analysis.py  pipeline.py  live.py
              evaluate.py  datasets.py  export.py  neural.py  simulate.py
scripts/      run_video.py  run_live.py  eval_dataset.py  benchmark_synthetic.py  summarize.py
tests/        test_core.py
toolbox_patches/  0001-*.patch (git apply), verify_quantization.py, verify_negpearson.py
AUDIT.md      full toolbox audit
```
`.mat` variable names live in `export.NAMES` — set them to the exact names in the official SP Cup template.
