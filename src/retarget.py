"""MediaPipe (pose+hands) -> MuJoCo avatar retargeting and side-by-side rendering.

usage: python src/retarget.py sample4
"""
import sys, os
import numpy as np
import mujoco
import cv2
import mediapy as media
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d

stem = sys.argv[1] if len(sys.argv) > 1 else "sample4"
OUT = "outputs"
body = np.load(f"{OUT}/body_{stem}.npz", allow_pickle=True)
hands = np.load(f"{OUT}/hands_{stem}.npz", allow_pickle=True)
pose, pw, head_blend = body["pose"], body["pose_world"], body["blend"]
fps = float(body["fps"])
T = len(pose)
hl, hw = hands["landmarks"], hands["world"]
print("frames", T, "fps", fps, flush=True)

BN = {str(n): i for i, n in enumerate(body["blend_names"])}
M = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)  # mediapipe -> avatar
mp_map = lambda v: np.asarray(v) @ M.T

AVATAR = os.environ.get("AVATAR", "models/avatar/avatar.xml")
SUF = os.environ.get("SUF", "")
CUTE = "cute" in AVATAR
FACE_DIR = os.environ.get("FACE_FRAMES", "")
FLIP_V = -1 if os.environ.get("FACE_FLIP", "0") == "1" else 1
m = mujoco.MjModel.from_xml_path(AVATAR)
if CUTE:  # glove-white hands, outfit-colored forearm base
    for g in range(m.ngeom):
        bn = m.body(m.geom_bodyid[g]).name
        if bn.startswith(("R_", "L_")):
            m.geom_matid[g] = -1
            m.geom_rgba[g] = [0.98, 0.72, 0.78, 1] if bn.endswith("forearm") else [1.0, 0.94, 0.9, 1]
d = mujoco.MjData(m)
J = lambda n: m.joint(n)
B = lambda n: m.body(n).id
qadr = lambda n: m.joint(n).qposadr[0]
HAS_FACE_JOINTS = any(m.joint(i).name == "brow_l" for i in range(m.njnt))
FACE_TID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "face")
mujoco.mj_forward(m, d)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def frame(w, mf, ff, lf):
    dd = unit(mf - w)
    a = lf - ff
    a = unit(a - dd * (a @ dd))
    return dd, a, np.cross(dd, a)


class Side:
    def __init__(s, side):
        s.side = side
        s.P = "R_rh_" if side == "R" else "L_lh_"
        s.p = "r_" if side == "R" else "l_"
        s.arm = [s.p + n for n in ("sh_flex", "sh_abd", "sh_rot", "elbow_flex", "forearm_roll")]
        s.wr = [s.P + "WRJ2", s.P + "WRJ1"]
        s.vars = s.arm + s.wr
        s.sh = B(s.p + "shoulder")
        s.el = B(s.p + "elbow")
        s.wrist = B(s.P + "wrist")
        s.mfk, s.ffk, s.lfk = (B(s.P + x + "knuckle") for x in ("mf", "ff", "lf"))
        s.va = [qadr(n) for n in s.vars]
        s.lo = np.array([m.jnt_range[J(n).id][0] for n in s.vars])
        s.hi = np.array([m.jnt_range[J(n).id][1] for n in s.vars])
        mujoco.mj_forward(m, d)
        s.S = d.xpos[s.sh].copy()
        s.Lu = np.linalg.norm(d.xpos[s.el] - s.S)
        s.Lf = np.linalg.norm(d.xpos[s.wrist] - d.xpos[s.el])
        s.hl = np.linalg.norm(d.xpos[s.mfk] - d.xpos[s.wrist])
        # pose indices (person's right = avatar R)
        s.i_sh, s.i_el, s.i_wr = (12, 14, 16) if side == "R" else (11, 13, 15)
        s.q = np.zeros(len(s.vars))


sides = [Side("R"), Side("L")]
for s in sides:
    print(s.side, "Lu %.3f Lf %.3f hand %.3f" % (s.Lu, s.Lf, s.hl), flush=True)

# ---- hand assignment per arm
def assign_hands():
    asg = np.full((T, 2), -1)  # for [R, L]
    for t in range(T):
        cand = []
        for k, s in enumerate(sides):
            wp = pose[t, s.i_wr, :2]
            for h in range(2):
                w0 = hl[t, h, 0, :2]
                if np.isnan(w0).any():
                    continue
                dist = np.linalg.norm(w0 - wp)
                if dist < 0.12:
                    cand.append((dist, k, h))
        cand.sort()
        usedk, usedh = set(), set()
        for dist, k, h in cand:
            if k in usedk or h in usedh:
                continue
            asg[t, k] = h
            usedk.add(k); usedh.add(h)
    return asg


asg = assign_hands()
print("hand detected R %.0f%% L %.0f%%" % (100 * (asg[:, 0] >= 0).mean(), 100 * (asg[:, 1] >= 0).mean()), flush=True)

# ---- gap filling: linear interpolation of missing hand landmarks
MAXGAP = 24  # ~0.8 s; longer gaps keep last shape, the pose drives the arm
HW = np.full((2, T, 21, 3), np.nan)
for k in range(2):
    for t in range(T):
        if asg[t, k] >= 0:
            HW[k, t] = hw[t, asg[t, k]]
valid = ~np.isnan(HW[:, :, 0, 0])
filled = np.zeros_like(valid)
for k in range(2):
    idx = np.where(valid[k])[0]
    if len(idx) < 2:
        continue
    for t in range(idx[0], idx[-1] + 1):
        if valid[k, t]:
            continue
        a_ = idx[idx < t].max(); b_ = idx[idx > t].min()
        if b_ - a_ - 1 <= MAXGAP:
            w_ = (t - a_) / (b_ - a_)
            HW[k, t] = (1 - w_) * HW[k, a_] + w_ * HW[k, b_]
            filled[k, t] = True
print("filled frames R %d L %d" % (filled[0].sum(), filled[1].sum()), flush=True)
usable = valid | filled

# ---- finger calibration (abduction sign) at rest
def finger_sign(s, f):
    j = s.P + f.upper() + "J4"
    mid = B(s.P + f + "middle")
    mujoco.mj_forward(m, d)
    base = d.xpos[mid].copy()
    dd, a, n = frame(d.xpos[s.wrist], d.xpos[s.mfk], d.xpos[s.ffk], d.xpos[s.lfk])
    d.qpos[qadr(j)] = 0.3
    mujoco.mj_forward(m, d)
    sg = np.sign((d.xpos[mid] - base) @ a)
    d.qpos[qadr(j)] = 0
    mujoco.mj_forward(m, d)
    return sg if sg != 0 else 1.0


for s in sides:
    s.sg = {f: finger_sign(s, f) for f in ("ff", "mf", "rf")}
    print(s.side, "abd sign", s.sg, flush=True)

FING = {"ff": (5, 6, 7, 8), "mf": (9, 10, 11, 12), "rf": (13, 14, 15, 16), "lf": (17, 18, 19, 20)}


def ang(u, v):
    return float(np.arccos(np.clip(unit(u) @ unit(v), -1, 1)))


def set_fingers(s, hw_t, q):
    p = mp_map(hw_t)  # 21x3 in avatar axes
    dd, a, n = frame(p[0], p[9], p[5], p[17])
    for f, (i0, i1, i2, i3) in FING.items():
        b1, b2, b3 = p[i1] - p[i0], p[i2] - p[i1], p[i3] - p[i2]
        j3 = ang(b1, dd) if f != "lf" else ang(b1, dd)
        j2 = ang(b2, b1)
        j1 = ang(b3, b2)
        j3 = np.clip(j3, 0, 1.57)
        # curl the proximal joint from the angle of the first bone
        P = s.P + f.upper()
        q[qadr(P + "J3")] = j3
        q[qadr(P + "J2")] = np.clip(j2 + 0.0, 0, 1.57)
        q[qadr(P + "J1")] = np.clip(j1 + 0.0, 0, 1.57)
        if f != "lf":
            ab = np.arctan2(b1 @ a, b1 @ dd)
            q[qadr(P + "J4")] = np.clip(s.sg[f] * ab, -0.34, 0.34)
        else:
            q[qadr(P + "J5")] = np.clip(0.4 * j3, 0, 0.78)
            q[qadr(P + "J4")] = 0
    # thumb: IK on positions in hand frame
    sc = s.hl / (np.linalg.norm(p[9] - p[0]) + 1e-9)
    tn = [B(s.P + "thmiddle"), B(s.P + "thdistal")]
    tj = [s.P + "THJ5", s.P + "THJ4", s.P + "THJ3", s.P + "THJ2"]
    ta = [qadr(x) for x in tj]
    lo = np.array([m.jnt_range[J(x).id][0] for x in tj]); hi = np.array([m.jnt_range[J(x).id][1] for x in tj])
    c = [(p[i] - p[0]) @ np.stack([dd, a, n]).T * sc for i in (2, 3)]
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    dA, aA, nA = frame(d.xpos[s.wrist], d.xpos[s.mfk], d.xpos[s.ffk], d.xpos[s.lfk])
    W = d.xpos[s.wrist].copy()
    tg = [W + ci[0] * dA + ci[1] * aA + ci[2] * nA for ci in c]

    def res(x):
        d.qpos[ta] = x
        mujoco.mj_forward(m, d)
        return np.concatenate([d.xpos[tn[0]] - tg[0], d.xpos[tn[1]] - tg[1], 0.01 * x])

    x0 = q[ta]
    sol = least_squares(res, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=25)
    q[ta] = sol.x
    q[qadr(s.P + "THJ1")] = np.clip(ang(p[4] - p[3], p[3] - p[2]), -0.2, 1.5)


# ---- main per-frame solve
Q = np.zeros((T, m.nq))
Q0 = d.qpos.copy()
q = Q0.copy()
err = []
for t in range(T):
    for k, s in enumerate(sides):
        # arm targets
        pe, ps, pwr = pw[t, s.i_el], pw[t, s.i_sh], pw[t, s.i_wr]
        du = unit(mp_map(pe - ps)); df = unit(mp_map(pwr - pe))
        Et = s.S + du * s.Lu
        Wt = Et + df * s.Lf
        h = 1 if usable[k, t] else -1
        if h >= 0:
            p = mp_map(HW[k, t])
            hd = frame(p[0], p[9], p[5], p[17])
        else:
            hd = None

        def res(x):
            d.qpos[:] = q
            d.qpos[s.va] = x
            mujoco.mj_forward(m, d)
            r = [(d.xpos[s.el] - Et) * 3, (d.xpos[s.wrist] - Wt) * 3, 0.03 * x]
            if hd is not None:
                dA, aA, nA = frame(d.xpos[s.wrist], d.xpos[s.mfk], d.xpos[s.ffk], d.xpos[s.lfk])
                r += [(dA - hd[0]) * 0.6, (aA - hd[1]) * 0.6]
            else:
                r += [0.2 * x[5:], 0.05 * x[4:5]]
            return np.concatenate([np.ravel(v) for v in r])

        sol = least_squares(res, np.clip(s.q, s.lo + 1e-6, s.hi - 1e-6), bounds=(s.lo, s.hi), max_nfev=30)
        s.q = sol.x
        q[s.va] = sol.x
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        err.append(np.linalg.norm(d.xpos[s.wrist] - Wt))
        if h >= 0:
            set_fingers(s, HW[k, t], q)
        # else: hold previous finger values
    # head
    ear = mp_map(pw[t, 8] - pw[t, 7])
    mid = mp_map(pw[t, 0] - 0.5 * (pw[t, 7] + pw[t, 8]))
    q[qadr("neck_yaw")] = np.arctan2(ear[1], ear[0])
    q[qadr("neck_roll")] = -np.arctan2(ear[2], ear[0])
    q[qadr("neck_pitch")] = np.arctan2(mid[2], -mid[1] if False else mid[1])
    if HAS_FACE_JOINTS:
        # face (blendshapes -> avatar face joints). "Left" = the subject's left = avatar -x side
        bs = lambda n: float(head_blend[t, BN[n]])
        for sd, S_ in (("l", "Left"), ("r", "Right")):
            up = max(bs("browOuterUp" + S_), 0.6 * bs("browInnerUp")) - bs("browDown" + S_)
            q[qadr("brow_" + sd)] = np.clip(0.04 * up, -0.02, 0.03)
            q[qadr("lid_" + sd)] = m.jnt_range[J("lid_" + sd).id][0] * (1 - np.clip(bs("eyeBlink" + S_) * 1.2, 0, 1))
            out = bs("eyeLookOut" + S_) - bs("eyeLookIn" + S_)
            yaw = 0.6 * out if sd == "l" else -0.6 * out
            q[qadr("eye_%s_yaw" % sd)] = yaw
            q[qadr("eye_%s_pitch" % sd)] = 0.5 * (bs("eyeLookUp" + S_) - bs("eyeLookDown" + S_))
            sm = bs("mouthSmile" + S_) - bs("mouthFrown" + S_)
            q[qadr("corner_%s_z" % sd)] = np.clip(0.03 * sm, -0.02, 0.02)
            q[qadr("corner_%s_x" % sd)] = np.clip(0.02 * (bs("mouthStretch" + S_) + 0.5 * bs("mouthSmile" + S_)) - 0.02 * bs("mouthPucker") - 0.01 * bs("mouthFunnel"), -0.015, 0.015)
        q[qadr("jaw")] = -0.04 * np.clip(bs("jawOpen") * 1.2, 0, 1)
    Q[t] = q
    if t % 20 == 0:
        print("t", t, "wrist err %.3f" % np.mean(err[-2:]), flush=True)

# head pitch: remove median offset
pa = qadr("neck_pitch")
Q[:, pa] -= np.median(Q[:, pa])
for n in ("neck_yaw", "neck_roll"):
    Q[:, qadr(n)] -= np.median(Q[:, qadr(n)])
for n in ("neck_yaw", "neck_pitch", "neck_roll"):
    lo, hi = m.jnt_range[J(n).id]
    Q[:, qadr(n)] = np.clip(Q[:, qadr(n)], lo, hi)
Qs = gaussian_filter1d(Q, sigma=1.5, axis=0, mode="nearest")
print("mean wrist err %.3f m" % np.mean(err), flush=True)
np.savez(f"{OUT}/avatar_q_{stem}{SUF}.npz", q=Qs, fps=fps)

# ---- render
m.vis.global_.offwidth = 1280
m.vis.global_.offheight = 720
ren = mujoco.Renderer(m, 540, 540)
ren2 = mujoco.Renderer(m, 200, 200)
cam2 = mujoco.MjvCamera()
cam2.lookat[:] = [0, 0, 1.52 if CUTE else 1.55]; cam2.distance = 0.6; cam2.azimuth = 270; cam2.elevation = 0
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 1.3]; cam.distance = 1.7; cam.azimuth = 270; cam.elevation = -5
left_path = f"{OUT}/hands_{stem}.mp4"
cap = cv2.VideoCapture(left_path)
frames = []
for t in range(T):
    d.qpos[:] = Qs[t]
    mujoco.mj_forward(m, d)
    if FACE_DIR and FACE_TID >= 0:
        fi = cv2.cvtColor(cv2.imread(f"{FACE_DIR}/f_{t:04d}.jpg"), cv2.COLOR_BGR2RGB)
        fi = cv2.resize(fi, (int(m.tex_width[FACE_TID]), int(m.tex_height[FACE_TID])))
        fi = np.ascontiguousarray(fi[::FLIP_V])
        adr = int(m.tex_adr[FACE_TID]); n = fi.size
        m.tex_data[adr:adr + n] = fi.reshape(-1)
        for r_ in (ren, ren2):
            r_._gl_context.make_current()
            mujoco.mjr_uploadTexture(m, r_._mjr_context, FACE_TID)
    ren.update_scene(d, camera=cam)
    av = ren.render()
    ren2.update_scene(d, camera=cam2)
    av = av.copy()
    av[0:200, 340:540] = ren2.render()
    ok, lf = cap.read()
    if ok:
        lf = cv2.cvtColor(lf, cv2.COLOR_BGR2RGB)
        w = int(lf.shape[1] * 540 / lf.shape[0])
        lf = cv2.resize(lf, (w, 540))
    else:
        lf = np.zeros((540, 720, 3), np.uint8)
    frames.append(np.concatenate([lf, av], axis=1))
media.write_video(f"{OUT}/avatar_{stem}{SUF}.mp4", frames, fps=fps)
print("wrote", f"{OUT}/avatar_{stem}{SUF}.mp4", flush=True)
