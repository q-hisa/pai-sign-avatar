"""動画 → アバター動画を1コマンドで作る（Docker内で実行）。

使い方:
  docker compose run --rm dev python video2avatar.py data/samples/sample4.mp4
  docker compose run --rm dev python video2avatar.py data/samples/*.mp4 --force

段階（各段階は出力が新しければスキップ。--force で全部やり直し）:
  1 extract   MediaPipeで手・姿勢・表情を抽出          -> outputs/hands_<名前>.npz, body_<名前>.npz
  2 retarget  腕IK・指・首へ変換（欠損は線形補間）      -> outputs/avatar_q_<名前>.npz
  3 body      MuJoCoでアンドロイド体を描画（頭は空）     -> android_v2/outputs/robot_<名前>/
  4 motion    顔の動作データ（表情・姿勢・手）          -> android_v2/outputs/motion_<名前>.json
  5 head      VRMの顔・髪を描画（Chromium）              -> android_v2/outputs/head_<名前>/
  6 compose   体と頭を合成してmp4に                      -> outputs/video2avatar/<名前>.mp4
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
V2 = ROOT / "android_v2"
OUT = ROOT / "outputs"


def run(cmd, env=None):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, env=env)


def fresh(target: Path, *sources: Path) -> bool:
    """target があり、すべての入力より新しければ True。"""
    if not target.exists():
        return False
    t = target.stat().st_mtime
    return all(s.exists() and s.stat().st_mtime <= t for s in sources)


def report(stem: str):
    h = np.load(OUT / f"hands_{stem}.npz", allow_pickle=True)["landmarks"]
    got = ~np.isnan(h[:, :, 0, 0])  # [T,2]
    rate = got.mean(axis=0)
    print(f"  手の検出率: スロット1 {rate[0]*100:.0f}% / スロット2 {rate[1]*100:.0f}%（フレーム {len(h)}）")
    if rate.max() < 0.5 or rate.min() < 0.4:
        print("  ⚠ 検出率が低い手があります。手元を大きく・無地の背景で撮り直すと改善します。")


def process(video: Path, a):
    stem = video.stem
    t0 = time.time()
    print(f"\n=== {video.name} ({stem}) ===", flush=True)
    hands, body = OUT / f"hands_{stem}.npz", OUT / f"body_{stem}.npz"
    q = OUT / f"avatar_q_{stem}.npz"
    robot = V2 / "outputs" / f"robot_{stem}"
    motion = V2 / "outputs" / f"motion_{stem}.json"
    head = V2 / "outputs" / f"head_{stem}"
    final = OUT / "video2avatar" / f"{stem}.mp4"
    final.parent.mkdir(parents=True, exist_ok=True)
    (V2 / "outputs").mkdir(parents=True, exist_ok=True)

    print("[1/6] extract")
    if a.force or not fresh(hands, video) or not fresh(body, video):
        run([sys.executable, "src/extract_hands.py", video])
        run([sys.executable, "src/extract_pose_face.py", video])
    else:
        print("  skip（抽出済み）")
    report(stem)

    print("[2/6] retarget")
    if a.force or not fresh(q, hands, body):
        run([sys.executable, "src/retarget.py", stem])
    else:
        print("  skip")

    print("[3/6] body")
    if a.force or not fresh(robot / "headpose.json", q):
        run([sys.executable, V2 / "pipeline/render_android.py", stem,
             "--base-model", ROOT / "models/avatar/avatar.xml", "--motion", q,
             "--out", robot, "--distance", a.distance])
    else:
        print("  skip")

    print("[4/6] motion")
    if a.force or not fresh(motion, hands, body):
        run([sys.executable, "src/export_motion.py", OUT, stem, motion])
    else:
        print("  skip")

    print("[5/6] head")
    if a.force or not fresh(head / "h_0000.png", motion, robot / "headpose.json"):
        run([sys.executable, V2 / "pipeline/render_overlay.py", a.vrm, motion,
             robot / "headpose.json", head, a.scale, a.haircut])
    else:
        print("  skip")

    print("[6/6] compose")
    run([sys.executable, V2 / "pipeline/compose_preview.py", robot, head, final])
    print(f"完了: {final}  ({time.time()-t0:.0f}秒)", flush=True)
    return final


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--vrm", type=Path, default=V2 / "data/model.vrm", help="顔・髪のVRM")
    p.add_argument("--scale", type=float, default=1.4, help="顔の拡大率")
    p.add_argument("--haircut", type=float, default=-0.025, help="髪の切り口（小さいほど長い）")
    p.add_argument("--distance", type=float, default=1.75, help="カメラ距離")
    p.add_argument("--force", action="store_true", help="全段階をやり直す")
    a = p.parse_args()
    if not (V2 / "pipeline/node_modules").exists():
        run(["npm", "--prefix", V2 / "pipeline", "ci"])
    if not a.vrm.exists():
        sys.exit(f"VRMが見つかりません: {a.vrm}")
    done, failed = [], []
    for v in a.videos:
        try:
            done.append(process(v, a))
        except Exception as e:  # 1本の失敗で全体を止めない
            print(f"失敗: {v.name}: {e}", flush=True)
            failed.append(v.name)
    print(f"\n成功 {len(done)} / 失敗 {len(failed)}", ("失敗: " + ", ".join(failed)) if failed else "")
    sys.exit(1 if failed else 0)
