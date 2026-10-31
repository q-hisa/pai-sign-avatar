# android_v2

アンドロイド型の体（MuJoCo）と、顔・髪の合成（Three.js + three-vrm）です。

- `model/avatar_android.xml`: 体のモデル。関節名と構造は、元の簡素版（`models/avatar/avatar.xml`）と同じです。外装は滑らかなOBJメッシュ（`model/meshes/`）で、メッシュの磨き込みにChatGPTを使いました（`model/tools_refine_android.py`）。
- `pipeline/render_android.py`: 関節角から体を描画し、頭より手前の領域のマスクと頭の姿勢を出力します。
- `pipeline/render_overlay.py`、`pipeline/solver.js`、`pipeline/index.html`: VRMの顔・髪を、頭の姿勢に重ねて描画します（headless Chromium）。表情は表情係数で動かします。
- `pipeline/compose_preview.py`: 体と顔を合成して動画にします。
- `data/`: 顔のVRM（`make models` で取得。リポジトリには含まれません）。

顔のモデル: VRM1_Constraint_Twist_Sample、(c) 2022 pixiv Inc.、https://vrm.dev/licenses/1.0/
