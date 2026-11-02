"""Aggregate benchmark JSONs -> markdown tables."""
import json, glob, os, sys
import numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else "results"
names = ["toolbox", "single_roi_no_tracker", "flat_fusion_no_tracker", "fusion_no_tracker", "full"]
key = sys.argv[2] if len(sys.argv) > 2 else "true_win"
rows = {}
for p in sorted(glob.glob(os.path.join(out, "*.json"))):
    r = json.load(open(p)); d = rows.setdefault(r["cond"], {n: {"win": [], "vid": []} for n in names})
    for n in names:
        for f in r["variants"].get(n, []):
            d[n]["win"] += list(np.abs(np.array(f["hr"]) - np.array(f[key])))
            d[n]["vid"].append(abs(f["hr_video"] - f["true_video"]))
print(f"Window-level (10 s) MAE / RMSE [bpm], truth = {'uniform window mean' if key == 'true_win' else 'Hann-weighted window mean'}\n")
print("| condition | " + " | ".join(names) + " |"); print("|" + "---|" * (len(names) + 1))
allw = {n: [] for n in names}
for c, d in rows.items():
    cells = []
    for n in names:
        w = np.array(d[n]["win"]); allw[n] += list(w)
        cells.append(f"{w.mean():.2f} / {np.sqrt((w ** 2).mean()):.2f}" if len(w) else "n/a")
    print(f"| {c} | " + " | ".join(cells) + " |")
print("| **all** | " + " | ".join(f"**{np.mean(allw[n]):.2f} / {np.sqrt(np.mean(np.square(allw[n]))):.2f}**" for n in names) + " |")
print("\nShare of windows with |error| > 5 bpm [%]\n")
print("| " + " | ".join(names) + " |"); print("|" + "---|" * len(names))
print("| " + " | ".join(f"{100 * np.mean(np.array(allw[n]) > 5):.1f}" for n in names) + " |")
print("\nVideo-level MAE [bpm]\n")
print("| condition | " + " | ".join(names) + " |"); print("|" + "---|" * (len(names) + 1))
for c, d in rows.items():
    print(f"| {c} | " + " | ".join(f"{np.mean(d[n]['vid']):.2f}" if d[n]['vid'] else "n/a" for n in names) + " |")

print("\nAbstention: windows of 'full' kept above a confidence threshold (all conditions pooled)\n")
print("| threshold | coverage | MAE kept | RMSE kept | >5 bpm kept | hard_combo windows kept |"); print("|---|---|---|---|---|---|")
C, E, K = [], [], []
for p in sorted(glob.glob(os.path.join(out, "*.json"))):
    r = json.load(open(p))
    for f in r["variants"]["full"]:
        C += f["conf"]; E += list(np.abs(np.array(f["hr"]) - np.array(f[key]))); K += [r["cond"]] * len(f["conf"])
C, E, K = np.array(C), np.array(E), np.array(K)
for th in [0.0, 0.3, 0.4, 0.5, 0.6]:
    m = C >= th
    print(f"| {th:.1f} | {100 * m.mean():.1f}% | {E[m].mean():.2f} | {np.sqrt((E[m] ** 2).mean()):.2f} | {100 * np.mean(E[m] > 5):.1f}% | {100 * m[K == 'hard_combo'].mean():.1f}% |")
pos, neg = C[E > 5], C[E <= 5]
print(f"\nAUROC of confidence for detecting |error| > 5 bpm: {np.mean(pos[:, None] < neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]):.3f}")
