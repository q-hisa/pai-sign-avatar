"""手書きのデモ動作を作る（撮影した動画を使わない。誰かの手話の写し取りではない）。

関節角を、時刻ごとのキーフレームで直接指定して作る。出力:
  demo/avatar_q_<id>.npz      関節角（models/avatar/avatar.xml の関節順。キー q, fps）
  demo/motion_<id>.json       顔の動作（表情係数）。体の動きの形式に合わせたもの
これは「手話」ではなく、ROS 2の4ノード・描画・録画の動作確認用のデモ（手を振る、1・2・3の指の本数を示す）。

使い方: python tools_make_demo_motion.py
"""
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "demo"
FPS = 30.0
BLEND_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight"]


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, t0, t1):
    """t0〜t1 で 0→1 に滑らかに変化する。"""
    return smooth((t - t0) / (t1 - t0))


def pulse(t, c, w):
    """時刻 c を中心とする幅 w の山（0→1→0）。"""
    return smooth(1 - np.abs(t - c) / w)


def steps(t, times, values):
    """times の区間ごとに values を取り、境界は0.25秒で滑らかにつなぐ。"""
    out = np.full_like(t, values[0], dtype=float)
    for tm, v0, v1 in zip(times, values[:-1], values[1:]):
        out = out + (v1 - v0) * ramp(t, tm - 0.125, tm + 0.125)
    return out


def wave(t):
    """手を振る（4秒）: 右腕を横に上げ、肘を曲げて手首とひねりで左右に振り、下ろす。"""
    up = ramp(t, 0.0, 0.8) * (1 - ramp(t, 3.2, 4.0))
    sw = np.sin(2 * np.pi * 1.6 * t) * up
    j = {"r_sh_abd": 0.75 * up, "r_sh_flex": 0.0 * up, "r_elbow_flex": 1.5 * up, "r_sh_rot": 1.6 * up,
         "r_forearm_roll": 0.45 * sw, "R_rh_WRJ1": 0.25 * sw, "l_sh_abd": 0.08 + 0 * t,
         "neck_yaw": 0.12 * up, "neck_pitch": -0.05 * up}
    blend = {"mouthSmileLeft": 0.5 * up, "mouthSmileRight": 0.5 * up, "browInnerUp": 0.2 * up,
             "eyeBlinkLeft": 0.9 * pulse(t, 1.3, 0.1) + 0.9 * pulse(t, 2.9, 0.1),
             "eyeBlinkRight": 0.9 * pulse(t, 1.3, 0.1) + 0.9 * pulse(t, 2.9, 0.1)}
    return 4.0, j, blend


def count(t):
    """指の本数で 1・2・3 を示す（5.2秒）: 胸の前に右手を上げ、人差し指→+中指→+薬指と立てる。"""
    up = ramp(t, 0.0, 0.8) * (1 - ramp(t, 4.4, 5.2))
    n = steps(t, [2.0, 3.2], [1, 2, 3])  # 立てる指の数（1〜3）
    j = {"r_sh_flex": 0.9 * up, "r_sh_abd": 0.3 * up, "r_elbow_flex": 1.7 * up, "r_sh_rot": -0.3 * up,
         "r_forearm_roll": -1.4 * up, "l_sh_abd": 0.08 + 0 * t}
    for name, fid in (("MF", 2), ("RF", 3)):  # 中指・薬指: 立てる指の数が足りない間は曲げる
        curl = np.clip(fid - n, 0, 1) * up + (1 - up) * 1.0
        j[f"R_rh_{name}J3"], j[f"R_rh_{name}J2"], j[f"R_rh_{name}J1"] = 1.4 * curl, 1.5 * curl, 1.0 * curl
    one = 1 - up  # 人差し指は、腕を下ろしている間は曲げる
    j["R_rh_FFJ3"], j["R_rh_FFJ2"], j["R_rh_FFJ1"] = 1.4 * one, 1.5 * one, 1.0 * one
    for name in ("LF",):  # 小指は常に曲げる
        j[f"R_rh_{name}J3"], j[f"R_rh_{name}J2"], j[f"R_rh_{name}J1"] = 1.4 + 0 * t, 1.5 + 0 * t, 1.0 + 0 * t
    j["R_rh_THJ4"], j["R_rh_THJ5"], j["R_rh_THJ2"] = 1.0 + 0 * t, 0.5 + 0 * t, -0.3 + 0 * t  # 親指は手のひら側へ
    nod = 0.12 * (pulse(t, 1.1, 0.15) + pulse(t, 2.3, 0.15) + pulse(t, 3.5, 0.15))
    j["neck_pitch"] = nod
    blink = 0.9 * pulse(t, 1.8, 0.1) + 0.9 * pulse(t, 4.0, 0.1)
    blend = {"mouthSmileLeft": 0.35 * up, "mouthSmileRight": 0.35 * up, "browInnerUp": 0.15 * up,
             "eyeBlinkLeft": blink, "eyeBlinkRight": blink, "jawOpen": 0.12 * (nod / 0.12)}
    return 5.2, j, blend


DEMOS = {"demo_wave": wave, "demo_count": count}


def build(fn, model):
    dur, joints, blend = fn(np.arange(0, FPS * 6) / FPS)  # 長さだけ先に取り出す
    t = np.arange(int(round(dur * FPS)) + 1) / FPS
    dur, joints, blend = fn(t)
    q = np.tile(model.qpos0, (len(t), 1))
    for name, vals in joints.items():
        adr = model.jnt_qposadr[model.joint(name).id]
        lo, hi = model.jnt_range[model.joint(name).id]
        q[:, adr] = np.clip(vals, lo, hi)
    n = len(t)
    b = {k: [0.0] * n for k in BLEND_NAMES}
    for k, v in blend.items():
        b[k] = [float(x) for x in np.clip(v, 0, 1)]
    motion = dict(fps=FPS, T=n, pose=[[] for _ in range(n)], handR=[None] * n, handL=[None] * n, blend=b)
    return q, motion


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "avatar" / "avatar.xml"))
    OUT.mkdir(exist_ok=True)
    for name, fn in DEMOS.items():
        q, motion = build(fn, model)
        np.savez(OUT / f"avatar_q_{name}.npz", q=q, fps=FPS)
        (OUT / f"motion_{name}.json").write_text(json.dumps(motion))
        print(name, q.shape)


if __name__ == "__main__":
    main()
