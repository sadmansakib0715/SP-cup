Window-level (10 s) MAE / RMSE [bpm], truth = Hann-weighted window mean

| condition | toolbox | single_roi_no_tracker | flat_fusion_no_tracker | fusion_no_tracker | full |
|---|---|---|---|---|---|
| clean | 0.11 / 0.12 | 0.14 / 0.18 | 0.14 / 0.19 | 0.14 / 0.18 | 0.14 / 0.18 |
| dark_skin | 8.86 / 21.43 | 0.49 / 0.62 | 2.20 / 6.42 | 0.46 / 0.59 | 0.46 / 0.59 |
| hard_combo | 21.75 / 26.28 | 45.81 / 57.95 | 22.86 / 26.74 | 20.42 / 23.16 | 20.78 / 23.76 |
| low_light | 18.10 / 23.80 | 0.37 / 0.48 | 0.48 / 0.62 | 0.72 / 1.37 | 0.65 / 1.02 |
| lowres_jpeg_10fps | 2.41 / 3.55 | 1.05 / 1.25 | 2.47 / 5.62 | 2.42 / 4.05 | 1.52 / 2.08 |
| mask | 0.40 / 0.60 | 0.19 / 0.24 | 0.21 / 0.25 | 0.21 / 0.24 | 0.21 / 0.24 |
| motion_talking | 0.28 / 0.37 | 0.24 / 0.30 | 1.25 / 3.46 | 4.43 / 15.17 | 0.41 / 0.88 |
| multi_face | 0.08 / 0.11 | 0.18 / 0.23 | 0.18 / 0.22 | 0.18 / 0.23 | 0.18 / 0.23 |
| **all** | **6.50 / 14.70** | **5.41 / 19.32** | **3.33 / 9.43** | **3.24 / 9.34** | **2.72 / 7.97** |

Share of windows with |error| > 5 bpm [%]

| toolbox | single_roi_no_tracker | flat_fusion_no_tracker | fusion_no_tracker | full |
|---|---|---|---|---|
| 22.9 | 10.3 | 11.4 | 12.2 | 10.6 |

Video-level MAE [bpm]

| condition | toolbox | single_roi_no_tracker | flat_fusion_no_tracker | fusion_no_tracker | full |
|---|---|---|---|---|---|
| clean | 0.06 | 0.04 | 0.03 | 0.02 | 0.02 |
| dark_skin | 0.02 | 0.13 | 0.26 | 0.09 | 0.09 |
| hard_combo | 14.92 | 18.15 | 16.17 | 18.84 | 20.26 |
| low_light | 1.43 | 0.07 | 0.09 | 0.19 | 0.20 |
| lowres_jpeg_10fps | 1.44 | 0.21 | 0.75 | 0.52 | 0.74 |
| mask | 0.12 | 0.04 | 0.05 | 0.00 | 0.00 |
| motion_talking | 0.20 | 0.06 | 0.49 | 0.15 | 0.20 |
| multi_face | 0.34 | 0.03 | 0.05 | 0.05 | 0.05 |

Abstention: windows of 'full' kept above a confidence threshold (all conditions pooled)

| threshold | coverage | MAE kept | RMSE kept | >5 bpm kept | hard_combo windows kept |
|---|---|---|---|---|---|
| 0.0 | 100.0% | 2.72 | 7.97 | 10.6% | 100.0% |
| 0.3 | 93.7% | 2.10 | 7.02 | 6.5% | 59.5% |
| 0.4 | 89.9% | 1.61 | 6.04 | 4.7% | 40.5% |
| 0.5 | 78.6% | 1.03 | 4.36 | 3.0% | 23.8% |
| 0.6 | 58.2% | 0.49 | 2.82 | 0.9% | 4.8% |

AUROC of confidence for detecting |error| > 5 bpm: 0.917
