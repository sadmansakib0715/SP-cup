# rPPG-Toolbox audit (ubicomplab/rPPG-Toolbox @ b7500b8, 2025-09-15)

Scope: full code read of the data, method, training and evaluation paths; git history (321 commits, all
21 remote branches); the NeurIPS 2023 D&B paper (arXiv 2210.00716). Every quantitative claim below was
reproduced in a sandbox with the toolbox's own functions (scripts in `toolbox_patches/verify_*.py`).
Fixes for the confirmed bugs: `toolbox_patches/0001-*.patch` (`git apply` onto the repo).

## 1. Evaluation — the numbers are softer than they look

| # | Finding | Evidence | Impact |
|---|---|---|---|
| E1 | **Ground-truth HR is not independent of the prediction pipeline.** `calculate_metric_per_video` derives the "label" HR from the contact PPG with the *same* detrend → filter → FFT as the prediction. Sensor-reported HR (UBFC `ground_truth.txt` line 2, PURE `pulseRate`) is never read. | Hidden `wip/label/*_Comparison.csv` (sensor HR vs PPG-FFT vs PPG-peak per video): **UBFC: mean \|FFT−sensor\| 7.94 bpm (median 3.33, 15/42 videos > 5 bpm, max 73.4); PURE: 3.40 bpm (5/59 > 5 bpm, max 44.6)**. Peak detection was "Preferred" for 35/42 UBFC videos. | Label noise of several bpm on a minority of videos; MAE rankings below ~1 bpm are not meaningful. |
| E2 | **Dead code points at the corrected labels.** `metrics.read_label/read_hr_label` load `label/{dataset}_Comparison.csv` — that path no longer exists (moved to `wip/label`) and nothing calls them. | grep | The authors' own manual correction is unused. |
| E3 | **FFT-bin quantisation, shared by prediction and label.** `nfft = next_pow2(len)` → bin = 3.52 bpm (10 s window), 1.76 (30 s), 0.88 (60 s). Both HRs snap to the same grid. | True constant error 1.50 bpm on 10 s windows → toolbox reports **0.00 on 52 % of windows**, ~3.5 on the rest (MAE 1.67); the label itself is off from the true HR by up to 1.89 bpm. After patch: reported 1.50, label error 0.00. | Sub-1 bpm differences (e.g. 0.98 vs 1.29 bpm on UBFC) are within one bin at video level. |
| E4 | **Paper ≠ code for post-processing.** Paper §3.3: 2nd-order Butterworth 0.75–2.5 Hz. Code: 1st-order 0.6–3.3 Hz (a comment admits the paper used different values). | `post_process.py` | Published tables are not reproduced with defaults. |
| E5 | **Default evaluation = one HR per whole video** (`USE_SMALLER_WINDOW: False`) → UBFC has 42 data points; windows are non-overlapping; a trailing partial window of ≥9 frames is still scored. | configs, `metrics.py` | High-variance MAE; no time-resolved accuracy. |
| E6 | **Bland–Altman**: `averaged=True` (always used) multiplies SD by √2 → LoA ~41 % too wide; `scatter_plot` overwrites the data with random jitter *in place*, then `difference_plot` uses the jittered data; windows from one subject treated as independent. | `BlandAltmanPy.py` | Misleading agreement plots — matters for SP Cup (BA is a required metric). |
| E7 | **RMSE standard error** = `sqrt(std(se)/sqrt(n))` (dimensionally wrong). | `metrics.py` | — |
| E8 | **MACC** uses `np.roll` (circular wrap) over all N lags with `abs()` → O(N²) and an inverted waveform scores 1.0. | `post_process._compute_macc` | — |
| E9 | **Dense detrend**: `np.linalg.inv` of an N×N matrix. 0.94 s at N=1800; cubic growth (5-min video ≈ minutes + GBs). | timing | Long/clinical recordings impractical. |
| E10 | **Peak HR** uses `find_peaks` with no distance/prominence. | | Over-counts on noisy signals. |
| E11 | **Video-level HR = peak of the whole-recording spectrum.** When HR drifts within a video this peak is not the mean HR. | Simulated 72→88 bpm ramp at 10 fps: spectral peak off by ~2.9 bpm while the windowed series averaged within 0.05 bpm (found while building; this system's first version had the same flaw). | Video-level MAE depends on how much HR changes during the recording, not only on signal quality. |

## 2. Training

| # | Finding | Evidence |
|---|---|---|
| T1 | **`loss/NegPearsonLoss.py` correlates across the batch, not over time** (`CosineSimilarity(dim=0)` on (B,T)). Used by **FactorizePhys and iBVPNet** trainers. Value depends on batch composition; for B=1 it is the constant 1.0 → **zero gradient**. `drop_last` is not set, so the last mini-batch can be B=1. (PhysNet/PhysFormer/PhysMamba import the correct per-sample version from `PhysNetNegPearsonLoss.py`.) | numpy port: B=1 → 1.000 for any input. |
| T2 | Output/label normalisation uses one mean/std over the whole batch tensor. | trainers |
| T3 | `DiffNormalized`/`Standardized` divide by the std of the **entire video** (computed before chunking) — non-causal, impossible in live mode, leaks future statistics. | `BaseLoader.diff_normalize_*` |
| T4 | Best epoch chosen by validation **loss**, not validation HR MAE. | trainers |
| T5 | Inert config keys: `MODEL.RESUME`, `TRAIN.OPTIMIZER.{BETAS,EPS,MOMENTUM}`; `--model_file_name` argparse `type=float`. | grep |
| T6 | No PPG↔video lag compensation (finger-PPG vs face pulse transit ≈ 0.1–0.3 s) although Pearson-type losses are lag sensitive. An early `dataset/synchronize/align.py` existed and was deleted (commit b82cd33). | history |
| T7 | Paper: 1cycle, max LR 0.009, 30 epochs. Many configs: 10 epochs / constant LR. | configs |

## 3. Pre-processing

| # | Finding |
|---|---|
| P1 | Default `DO_DYNAMIC_DETECTION: False`: **one Haar box from frame 0 for the whole video**. With dynamic detection the box *jumps* every N frames → periodic artefact at fs/N (= 1 Hz = **60 bpm** for N=30 @30 fps), inside the HR band; no box smoothing. |
| P2 | Signal = mean of the whole 1.5× box (hair, background, eyes, mouth) resized to 72×72. No skin mask, no landmarks/ROIs → hurts dark skin, low light, masks, beards. |
| P3 | Only the largest face is kept (no multi-face, SP Cup C3). |
| P4 | No-face fallback box `[0, 0, frame.shape[0], frame.shape[1]]` swaps H and W, silently. Haar XML re-loaded from disk for every detection. |
| P5 | PURE frame/PPG timestamps ignored (PPG linearly stretched to the frame count); UBFC sensor HR ignored. |
| P6 | Chunking drops the tail of each video; test chunks are stitched without overlap (independent per-chunk normalisation → steps at boundaries). |
| P7 | MMPD skin-tone/lighting/motion metadata is parsed into filenames but **metrics are never stratified** — the hook for SP Cup C1/C2 reporting already exists. MMPD covers Fitzpatrick III–VI only (no I–II). |
| P8 | **NumPy ≥ 2 breaks** POS, ICA, pseudo-label generation (`np.mat` removed); requirements pin numpy 1.22 / Python 3.8. |
| P9 | Multiprocessing manager busy-waits and mutates `p_list` while iterating it. |

## 4. Hidden / non-obvious contents

* `wip/label/` — manually compared sensor/FFT/peak HR for UBFC, PURE, COHFACE (COHFACE file has no numeric rows) — see E1.
* Git history: deleted model implementations **rPPGNet, STVEN, 3D-CAN, hybrid-CAN** (commit 970fe8c), early `dataset/synchronize/{align,green_channel}.py` (b82cd33), RetinaFace backend (#384), old TensorBoard runs.
* Unmerged branches: `face_align` (BP4D face-alignment work, still contains "TODO: GIRISH REMOVE THIS"), `tsm_wtsm_compare` (weighted-TSM TS-CAN variant), `girish_tmp` (green-channel PPG viz script), `MMPD` (orphan history), `cpu_test`, `bs_dataset_gen`, `output_vizualization`.
* `final_model_release/` (~311 MB) includes motion-augmented **MA-UBFC_{deepphys,efficientphys,physnet,tscan}.pth** without dedicated configs; `model_outputs/PURE_PURE_UBFC_deepphys_outputs.pickle` sample outputs.
* Two configs have a trailing space in the filename (`PURE_PURE_MMPD_PHYSFORMER_BASIC .yaml`, `PURE_PURE_UBFC-PHYS_PHYSFORMER_BASIC .yaml`).
* Internal cluster paths in configs. No credentials/secrets found in history.
* **License: Responsible AI Source Code License v1.1.** §3.2(a)(ii) forbids using the software or derivatives to "detect or infer … skin color … health and medical conditions" (under the *Surveillance* heading); §3.3 requires passing the restrictions on to all downstream users. SP Cup 2027 requires ITA skin-tone stratification and vital-sign inference, and submitted work may be made available to MathWorks commercially. → keep the competition system an independent implementation (this repo re-implements published algorithms from the papers) and use the toolbox only as an external benchmark; confirm with your supervisor/organizer. Not legal advice.

## 4b. Empirical check of the toolbox's own unsupervised path (synthetic, exact truth)

Running the toolbox's `POS_WANG` + `calculate_metric_per_video` with its default preprocessing on the
16-video synthetic suite (see README): window MAE 6.63 bpm overall, 22.9 % of windows > 5 bpm. In low light the
Haar detector found no face in the dark first frame → whole-frame fallback → 18.2 bpm. Same videos,
same truth: this system 2.99 bpm overall, 0.77 bpm excluding the one physically unrecoverable condition.

## 5. Recommended upgrades to the toolbox (beyond the patch)

1. Labels: use sensor HR or robust IBI-based HR from contact PPG; report against both; ship the `wip/label` corrections.
2. Evaluation: fine-grid/interpolated HR, overlapping windows, repeated-measures BA, subject-bootstrap CIs, stratified tables (skin tone, lux, motion, fps, bitrate).
3. Pre-processing: landmark tracking + temporal smoothing, skin-masked multi-ROI traces, timestamp-aware resampling, causal per-window normalisation.
4. Training: temporal Pearson + frequency-domain loss with lag tolerance; HR-range ("speed") augmentation (Vance et al. 2023 report cross-dataset MAE dropping from >13 to <3 bpm); compression/low-light/low-fps augmentation; model selection by validation HR MAE.
5. Inference: SQI per window, ensembling of model + training-free candidates, HR tracking across windows, abstention when confidence is low.
6. Scope: SpO2/BP/HRV are listed as limitations in the paper — no support exists.
