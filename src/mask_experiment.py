"""Masking experiment: hide detected hand frames, fill them with each method, compare to the truth.

methods: A hold (keep last detected shape), B linear interpolation, C mean shape (sample-average hand shape, last orientation kept)
error: mean 3D distance [cm] over the 21 hand landmarks (MediaPipe world coordinates, hand-centered)
usage: python src/mask_experiment.py
"""
import numpy as np, json, csv
from handfill import ShapePrior
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
LENS = [9, 24, 45]  # frames: 0.3 s / 0.8 s / 1.5 s
METHODS = ["A:hold", "B:linear", "C:mean-shape", "D0:canon-slerp", "D:prior-k8", "D16:prior-k16"]


def assign(pose, hl):
    T = len(pose)
    asg = np.full((T, 2), -1)
    for t in range(T):
        cand = []
        for k, iw in enumerate((16, 15)):  # [R, L]
            wp = pose[t, iw, :2]
            for h in range(2):
                w0 = hl[t, h, 0, :2]
                if np.isnan(w0).any():
                    continue
                dist = np.linalg.norm(w0 - wp)
                if dist < 0.12:
                    cand.append((dist, k, h))
        cand.sort()
        uk, uh = set(), set()
        for dist, k, h in cand:
            if k in uk or h in uh:
                continue
            asg[t, k] = h; uk.add(k); uh.add(h)
    return asg


def rot_align(src, dst):
    """Kabsch rotation that maps src(21x3) to dst(21x3)."""
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


rng = np.random.default_rng(0)
rows = []
STEMS = ("sample1", "sample2", "sample3", "sample4")
DATA = {}
for stem in STEMS:
    body = np.load(f"{OUT}/body_{stem}.npz", allow_pickle=True)
    hands = np.load(f"{OUT}/hands_{stem}.npz", allow_pickle=True)
    DATA[stem] = (body, hands, assign(body["pose"], hands["landmarks"]))
PRIORS = {}
for stem in STEMS:  # leave-one-sample-out library (both hands, left-hand shapes are kept as is)
    lib = []
    for o in STEMS:
        if o == stem:
            continue
        _, hh, aa = DATA[o]
        for t in range(len(aa)):
            for k in range(2):
                if aa[t, k] >= 0:
                    lib.append(hh["world"][t, aa[t, k]])
    PRIORS[stem] = ShapePrior(np.array(lib)[::2], k=8)
    PRIORS[stem + "16"] = ShapePrior(np.array(lib)[::2], k=16)
for stem in STEMS:
    body, hands, asg = DATA[stem]
    hw = hands["world"]
    for k, side in enumerate(("R", "L")):
        T = len(asg)
        valid = asg[:, k] >= 0
        HW = np.full((T, 21, 3), np.nan)
        for t in range(T):
            if valid[t]:
                HW[t] = hw[t, asg[t, k]]
        if valid.sum() < 30:
            continue
        mean_shape = np.nanmean(HW, axis=0)
        for L in LENS:
            errs = {m: [] for m in METHODS}
            for s in range(1, T - L - 1, 3):
                if not valid[s - 1:s + L + 1].all():
                    continue
                truth = HW[s:s + L]
                a, b = HW[s - 1], HW[s + L]
                predA = np.repeat(a[None], L, 0)
                w = (np.arange(1, L + 1) / (L + 1))[:, None, None]
                predB = (1 - w) * a[None] + w * b[None]
                # C: mean shape rotated to the orientation of the last visible hand
                R = rot_align(mean_shape - mean_shape.mean(0), a - a.mean(0))
                predC = np.repeat(((mean_shape - mean_shape.mean(0)) @ R.T + a.mean(0))[None], L, 0)
                predD0 = PRIORS[stem].fill(a, b, L, project=False)
                predD = PRIORS[stem].fill(a, b, L)
                predD16 = PRIORS[stem + "16"].fill(a, b, L)
                for m, pr in zip(METHODS, (predA, predB, predC, predD0, predD, predD16)):
                    errs[m].append(np.linalg.norm(pr - truth, axis=-1).mean() * 100)
            n = len(errs[METHODS[0]])
            if n == 0:
                continue
            for m in METHODS:
                rows.append(dict(sample=stem, side=side, mask_frames=L, method=m, n_windows=n, err_cm=float(np.mean(errs[m]))))

with open(f"{OUT}/mask_experiment.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)

# summary: weighted by windows, over all samples/sides
summ = {}
for L in LENS:
    for m in METHODS:
        rr = [r for r in rows if r["mask_frames"] == L and r["method"] == m]
        n = sum(r["n_windows"] for r in rr)
        summ[(L, m)] = (sum(r["err_cm"] * r["n_windows"] for r in rr) / max(n, 1), n)
lines = ["| mask length | " + " | ".join(METHODS) + " | windows |", "|---|" + "---|" * (len(METHODS) + 1)]
for L in LENS:
    lines.append(f"| {L} frames ({L/29.97:.1f} s) | " + " | ".join("%.2f cm" % summ[(L, m)][0] for m in METHODS) + f" | {summ[(L, METHODS[0])][1]} |")
open(f"{OUT}/mask_experiment.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.arange(len(LENS)); wd = 0.14
for i, m in enumerate(METHODS):
    ax.bar(x + (i - 2.5) * wd, [summ[(L, m)][0] for L in LENS], wd, label=m)
ax.set_xticks(x); ax.set_xticklabels([f"{L} fr" for L in LENS])
ax.set_ylabel("landmark error [cm]"); ax.legend(); ax.set_title("Hand gap filling: masking experiment")
fig.tight_layout(); fig.savefig(f"{OUT}/mask_experiment.png", dpi=150)
