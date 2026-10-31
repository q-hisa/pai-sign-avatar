"""手話動画から両手のランドマークをMediaPipeで抽出する。

出力（すべて outputs/ 以下、Git管理外）:
  hands_<name>.npz   : landmarks[T,2,21,3](画像座標,正規化) world[T,2,21,3](m) score[T,2] 欠損はNaN
  hands_<name>.mp4   : 骨格のみを黒背景に描いた動画（顔・背景は含めない）
  hands_<name>.json  : 検出率などの統計
使い方: python src/extract_hands.py data/samples/sample1.mp4 [...]
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

MODEL = Path("models/hand_landmarker.task")
URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/latest/hand_landmarker.task")
FINGERS = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
EDGES = [(a, b) for f in FINGERS for a, b in zip(f[:-1], f[1:])] + [(5, 9), (9, 13), (13, 17)]
COLORS = [(80, 200, 255), (255, 160, 60)]  # 手のスロット別の色(BGR→RGB変換は描画側で)


def ensure_model() -> None:
    if not MODEL.exists():
        MODEL.parent.mkdir(exist_ok=True)
        print("downloading", URL)
        urllib.request.urlretrieve(URL, MODEL)


def analyze(path: Path, out: Path) -> dict:
    ensure_model()
    opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    W, H = int(cap.get(3)), int(cap.get(4))
    lms, wl, sc, frames = [], [], [], []
    with vision.HandLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(img, int(i * 1000 / fps))
            a = np.full((2, 21, 3), np.nan); w = np.full((2, 21, 3), np.nan); s = np.full(2, np.nan)
            used = set()
            for hl, hw, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
                slot = 0 if hd[0].category_name == "Left" else 1
                if slot in used:
                    slot = 1 - slot
                if slot in used:
                    continue
                used.add(slot)
                a[slot] = [[p.x, p.y, p.z] for p in hl]
                w[slot] = [[p.x, p.y, p.z] for p in hw]
                s[slot] = hd[0].score
            lms.append(a); wl.append(w); sc.append(s)
            # 骨格のみの描画（背景は黒）
            canvas = np.zeros((540, 960, 3), np.uint8)
            for slot in used:
                pts = (a[slot, :, :2] * [960, 540]).astype(int)
                col = COLORS[slot][::-1]
                for p, q in EDGES:
                    cv2.line(canvas, tuple(pts[p]), tuple(pts[q]), col, 3)
                for p in pts:
                    cv2.circle(canvas, tuple(p), 4, (255, 255, 255), -1)
            frames.append(canvas)
            i += 1
    cap.release()
    lms, wl, sc = np.array(lms), np.array(wl), np.array(sc)
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"hands_{path.stem}.npz", landmarks=lms, world=wl, score=sc, fps=fps)
    mediapy.write_video(out / f"hands_{path.stem}.mp4", frames, fps=fps)
    has = ~np.isnan(lms[:, :, 0, 0])
    any_h, both = has.any(1), has.all(1)
    # 最長の検出欠損(フレーム)
    gap = best = 0
    for v in any_h:
        gap = 0 if v else gap + 1
        best = max(best, gap)
    stats = {
        "file": path.name, "size": [W, H], "fps": round(fps, 2), "frames": int(len(lms)),
        "any_hand_rate": round(float(any_h.mean()), 3), "two_hands_rate": round(float(both.mean()), 3),
        "slot0_rate": round(float(has[:, 0].mean()), 3), "slot1_rate": round(float(has[:, 1].mean()), 3),
        "mean_score": round(float(np.nanmean(sc)), 3) if not np.isnan(sc).all() else None,
        "longest_gap_frames": int(best),
    }
    (out / f"hands_{path.stem}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    return stats


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(analyze(Path(arg), Path("outputs")), ensure_ascii=False))
