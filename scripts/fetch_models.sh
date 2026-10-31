#!/usr/bin/env bash
# 必要なモデルを取得する（リポジトリには含まれない）。Docker内で実行する: make models
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] MuJoCo Menagerie の shadow_hand（Apache 2.0）"
if [ ! -d models/mujoco_menagerie/shadow_hand ]; then
  mkdir -p models
  git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git models/mujoco_menagerie
  git -C models/mujoco_menagerie sparse-checkout set shadow_hand
fi
mkdir -p android_v2/mujoco_menagerie
rm -rf android_v2/mujoco_menagerie/shadow_hand
cp -R models/mujoco_menagerie/shadow_hand android_v2/mujoco_menagerie/shadow_hand

echo "[2/4] MediaPipe のモデル（手・姿勢・顔）"
BASE=https://storage.googleapis.com/mediapipe-models
for item in hand_landmarker/hand_landmarker pose_landmarker/pose_landmarker_full face_landmarker/face_landmarker; do
  f="models/$(basename "$item").task"
  [ -f "$f" ] || curl -fsSL -o "$f" "$BASE/$item/float16/latest/$(basename "$item").task"
done

echo "[3/4] 顔・髪のVRM（pixiv VRM1_Constraint_Twist_Sample、(c) 2022 pixiv Inc.）"
mkdir -p android_v2/data
if [ ! -f android_v2/data/model.vrm ]; then
  curl -fsSL -o android_v2/data/model.vrm \
    https://github.com/pixiv/three-vrm/raw/dev/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm
fi

echo "[4/4] 顔の描画用 npm パッケージ（three.js、three-vrm）"
(cd android_v2/pipeline && npm ci)
echo "完了"
