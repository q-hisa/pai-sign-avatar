"""MediaPipe npz (pose / hands / face blendshapes) -> VRM-coordinate motion JSON.

VRM coordinates: +y up, character faces +z, character LEFT = +x.
MediaPipe world: x = image right, y = down, z = toward the camera is negative.
Person faces the camera, so: x_v = x_mp, y_v = -y_mp, z_v = -z_mp.
usage: python export_motion.py <outputs_dir> <stem> <out.json>
"""
import sys, json
import numpy as np
from scipy.ndimage import gaussian_filter1d

outdir, stem, outpath = sys.argv[1:4]
body = np.load(f"{outdir}/body_{stem}.npz", allow_pickle=True)
hands = np.load(f"{outdir}/hands_{stem}.npz", allow_pickle=True)
pose, pw, blend = body["pose"], body["pose_world"], body["blend"]
names = [str(n) for n in body["blend_names"]]
fps = float(body["fps"])
T = len(pose)
hl, hw = hands["landmarks"], hands["world"]
MV = np.diag([1.0, -1.0, -1.0])
MAXGAP = 24

# --- assign detected hands to arms (0 = person's right, 1 = person's left) by wrist proximity
asg = np.full((T, 2), -1)
for t in range(T):
    cand = []
    for k, iw in enumerate((16, 15)):
        for h in range(2):
            w0 = hl[t, h, 0, :2]
            if np.isnan(w0).any():
                continue
            d = np.linalg.norm(w0 - pose[t, iw, :2])
            if d < 0.12:
                cand.append((d, k, h))
    cand.sort()
    uk, uh = set(), set()
    for d, k, h in cand:
        if k in uk or h in uh:
            continue
        asg[t, k] = h; uk.add(k); uh.add(h)

HW = np.full((2, T, 21, 3), np.nan)
for k in range(2):
    for t in range(T):
        if asg[t, k] >= 0:
            HW[k, t] = hw[t, asg[t, k]] @ MV.T
valid = ~np.isnan(HW[:, :, 0, 0])
# linear interpolation of short gaps (method B)
for k in range(2):
    idx = np.where(valid[k])[0]
    for t in range(idx[0], idx[-1] + 1) if len(idx) > 1 else []:
        if valid[k, t]:
            continue
        a = idx[idx < t].max(); b = idx[idx > t].min()
        if b - a - 1 <= MAXGAP:
            w = (t - a) / (b - a)
            HW[k, t] = (1 - w) * HW[k, a] + w * HW[k, b]
usable = ~np.isnan(HW[:, :, 0, 0])
print("hand usable R %.0f%% L %.0f%%" % (100 * usable[0].mean(), 100 * usable[1].mean()))

# smooth where usable (fill holes by hold for filtering only)
def smooth_nan(a, sigma):
    out = a.copy()
    ok = ~np.isnan(a[..., 0, 0]) if a.ndim == 4 else None
    return out

for k in range(2):
    ok = usable[k]
    if ok.sum() > 3:
        idx = np.where(ok)[0]
        seg = HW[k].copy()
        # hold-fill for filtering, then restore NaN
        for t in range(T):
            if not ok[t]:
                j = idx[np.abs(idx - t).argmin()]
                seg[t] = HW[k, j]
        seg = gaussian_filter1d(seg, 1.2, axis=0, mode="nearest")
        HW[k][ok] = seg[ok]

P = pw @ MV.T
P = gaussian_filter1d(P, 1.2, axis=0, mode="nearest")
B = gaussian_filter1d(blend, 0.8, axis=0, mode="nearest")

def bs(n):
    return B[:, names.index(n)]

motion = dict(
    fps=fps, T=T,
    pose=np.round(P, 4).tolist(),
    handR=[None if not usable[0, t] else np.round(HW[0, t], 4).tolist() for t in range(T)],
    handL=[None if not usable[1, t] else np.round(HW[1, t], 4).tolist() for t in range(T)],
    blend={n: np.round(B[:, i], 3).tolist() for i, n in enumerate(names) if n != "_neutral"},
)
json.dump(motion, open(outpath, "w"))
print("wrote", outpath, T, "frames")
