"""手話動画から上半身の姿勢と顔の表情を抽出する（アバター駆動用）。

出力（outputs/、Git管理外）:
  body_<name>.npz : pose[T,33,4] pose_world[T,33,3] blend[T,52] head[T,4,4] fps  欠損はNaN
  body_<name>.mp4 : 棒人形のプレビュー（顔・背景は描かない。眉・口の係数だけバーで表示）
  body_<name>.json: 検出率の統計
使い方: python src/extract_pose_face.py data/samples/sample1.mp4 [...]
"""
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import mediapy
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

BASE = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {
    "pose": ("models/pose_landmarker_full.task", BASE + "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    "face": ("models/face_landmarker.task", BASE + "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"),
}
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
BLEND_NAMES = None


def fetch(key: str) -> str:
    path, url = MODELS[key]
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        print("downloading", url)
        urllib.request.urlretrieve(url, p)
    return str(p)


def analyze(path: Path, out: Path) -> dict:
    global BLEND_NAMES
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=fetch("pose")),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    face_opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=fetch("face")),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    P, PW, BL, HD, frames = [], [], [], [], []
    with vision.PoseLandmarker.create_from_options(pose_opts) as pl, \
            vision.FaceLandmarker.create_from_options(face_opts) as fl:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts = int(i * 1000 / fps)
            pr, fr = pl.detect_for_video(img, ts), fl.detect_for_video(img, ts)
            p = np.full((33, 4), np.nan); pw = np.full((33, 3), np.nan)
            b = np.full(52, np.nan); h = np.full((4, 4), np.nan)
            if pr.pose_landmarks:
                p = np.array([[q.x, q.y, q.z, q.visibility] for q in pr.pose_landmarks[0]])
                pw = np.array([[q.x, q.y, q.z] for q in pr.pose_world_landmarks[0]])
            if fr.face_blendshapes:
                if BLEND_NAMES is None:
                    BLEND_NAMES = [c.category_name for c in fr.face_blendshapes[0]]
                b = np.array([c.score for c in fr.face_blendshapes[0]])
                h = np.array(fr.facial_transformation_matrixes[0])
            P.append(p); PW.append(pw); BL.append(b); HD.append(h)
            # プレビュー: 棒人形のみ（顔や背景は描かない）
            cv = np.zeros((540, 960, 3), np.uint8)
            if not np.isnan(p[0, 0]):
                pts = (p[:, :2] * [960, 540]).astype(int)
                for a_, b_ in BONES:
                    cv2.line(cv, tuple(pts[a_]), tuple(pts[b_]), (200, 200, 200), 4)
                cv2.circle(cv, tuple(pts[0]), 38, (200, 200, 200), 3)
                for k in (15, 16):
                    cv2.circle(cv, tuple(pts[k]), 9, (80, 200, 255), -1)
            if not np.isnan(b[0]) and BLEND_NAMES:
                for row, nm in enumerate(["browInnerUp", "browDownLeft", "jawOpen", "mouthPucker", "cheekPuff"]):
                    v = float(b[BLEND_NAMES.index(nm)])
                    cv2.rectangle(cv, (20, 20 + row * 22), (20 + int(160 * v), 34 + row * 22), (255, 160, 60), -1)
                    cv2.putText(cv, nm, (190, 33 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            frames.append(cv)
            i += 1
    cap.release()
    P, PW, BL, HD = map(np.array, (P, PW, BL, HD))
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"body_{path.stem}.npz", pose=P, pose_world=PW, blend=BL, head=HD, fps=fps,
                        blend_names=np.array(BLEND_NAMES or []))
    mediapy.write_video(out / f"body_{path.stem}.mp4", frames, fps=fps)
    pose_ok, face_ok = ~np.isnan(P[:, 0, 0]), ~np.isnan(BL[:, 0])
    vis = P[:, [11, 12, 13, 14, 15, 16], 3]
    stats = {
        "file": path.name, "frames": int(len(P)),
        "pose_rate": round(float(pose_ok.mean()), 3), "face_rate": round(float(face_ok.mean()), 3),
        "wrist_visibility_mean": round(float(np.nanmean(vis[:, 4:])), 3),
        "blend_peaks": {n: round(float(np.nanmax(BL[:, k])), 2) for k, n in enumerate(BLEND_NAMES or [])
                        if n in ("browInnerUp", "browDownLeft", "browDownRight", "jawOpen", "mouthPucker", "cheekPuff",
                                 "mouthSmileLeft", "eyeBlinkLeft")},
    }
    (out / f"body_{path.stem}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(analyze(Path(arg), Path("outputs")), ensure_ascii=False))
