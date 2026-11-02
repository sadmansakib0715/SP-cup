# Patch for ubicomplab/rPPG-Toolbox (verified against upstream main @ b7500b8)

    cd rPPG-Toolbox
    git apply /path/to/0001-Audit-fixes-*.patch      # (or: git am --keep-cr ... ; several files use CRLF)

Fixes: fine-grid interpolated FFT HR (no shared-bin snapping), sparse O(N) detrend, temporal Neg-Pearson
(FactorizePhys/iBVPNet trainers), Bland-Altman sqrt(2) inflation + in-place jitter, RMSE SE, MACC (linear
lags, signed), peak-detection spacing, partial tail windows, no-face fallback (H/W swap), Haar cached,
NumPy >= 2 (np.mat), --model_file_name type, drop_last, two config filenames with trailing spaces.

NOTE: after this patch, toolbox MAE numbers change (usually upward at window level, because errors are no
longer hidden by bin snapping). Do not compare patched numbers with the paper's tables.

verify_quantization.py  - shows bin snapping with the unpatched toolbox (python verify_quantization.py /path/to/rPPG-Toolbox)
verify_negpearson.py    - numpy port of the dim=0 Neg-Pearson behaviour
