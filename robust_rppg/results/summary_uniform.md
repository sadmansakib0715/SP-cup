Window-level (10 s) MAE / RMSE [bpm], truth = uniform window mean

| condition | toolbox | single_roi_no_tracker | flat_fusion_no_tracker | fusion_no_tracker | full |
|---|---|---|---|---|---|
| clean | 0.33 / 0.43 | 0.58 / 0.67 | 0.60 / 0.69 | 0.60 / 0.69 | 0.60 / 0.69 |
| dark_skin | 9.06 / 21.42 | 0.67 / 0.82 | 2.47 / 6.52 | 0.71 / 0.86 | 0.71 / 0.86 |
| hard_combo | 21.64 / 26.27 | 45.85 / 57.94 | 22.76 / 26.65 | 20.36 / 23.12 | 20.77 / 23.74 |
| low_light | 18.15 / 23.72 | 0.80 / 0.92 | 0.89 / 1.05 | 1.02 / 1.62 | 0.95 / 1.30 |
| lowres_jpeg_10fps | 2.58 / 3.56 | 1.05 / 1.31 | 2.49 / 5.80 | 2.36 / 4.13 | 1.48 / 2.12 |
| mask | 0.61 / 0.74 | 0.47 / 0.56 | 0.48 / 0.58 | 0.48 / 0.58 | 0.48 / 0.58 |
| motion_talking | 0.34 / 0.43 | 0.49 / 0.59 | 1.50 / 3.57 | 4.69 / 15.03 | 0.70 / 0.99 |
| multi_face | 0.30 / 0.34 | 0.62 / 0.70 | 0.60 / 0.68 | 0.60 / 0.68 | 0.60 / 0.68 |
| **all** | **6.63 / 14.68** | **5.69 / 19.33** | **3.60 / 9.44** | **3.49 / 9.32** | **2.99 / 7.98** |

Share of windows with |error| > 5 bpm [%]

| toolbox | single_roi_no_tracker | flat_fusion_no_tracker | fusion_no_tracker | full |
|---|---|---|---|---|
| 22.9 | 10.3 | 11.4 | 12.2 | 10.8 |

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
| 0.0 | 100.0% | 2.99 | 7.98 | 10.8% | 100.0% |
| 0.3 | 93.7% | 2.37 | 7.03 | 6.8% | 59.5% |
| 0.4 | 89.9% | 1.90 | 6.06 | 4.7% | 40.5% |
| 0.5 | 78.6% | 1.35 | 4.36 | 3.0% | 23.8% |
| 0.6 | 58.2% | 0.88 | 2.85 | 0.9% | 4.8% |

AUROC of confidence for detecting |error| > 5 bpm: 0.919
