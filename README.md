# 手話アバター：動画から、ロボットが日本語手話を実演する

Physical AI応用1講座の最終課題の実装です。手話の動画から、手・姿勢・表情を点として取り出し、MuJoCo上の上半身ロボット（アバター）の関節の動きに変えて再現します。再生・描画・録画はROS 2の4つのノードに分けています。

- 体: 白いアンドロイド型の上半身ロボット（MuJoCo、Shadow Hand ×2）
- 顔・髪: VRMモデルを、ロボットの頭の位置に重ねて合成（表情は表情係数で動かす）
- 動きの抽出: MediaPipe（手・姿勢・顔の表情係数）
- 工夫した点: 手が検出できない区間を補完し、手法ごとの誤差を比較（後述）

> **手話の正確さについて**: 抽出した動きをそのまま写し取る仕組みです。再現された手話が正しいかどうかは、元の動画の手話と抽出の精度で決まります。手話話者による確認は済んでいません。通訳の代わりにはなりません。

## 全体の流れ

```
手話の動画 → MediaPipe（手・姿勢・表情） → 欠損の補完 → 関節角への変換
          → ROS 2: planner → motion_player → avatar_sim → recorder
          → MuJoCoでロボットを描画 → VRMの顔を合成 → mp4
```

## すぐ試す（撮影不要のデモ）

話者に由来しない、手書きの動き（手を振る／指で1・2・3を示す）が `demo/` に入っています。手話そのものではなく、ROS 2の4ノードと描画・録画の動作確認用です。

必要なもの: Docker（Apple Siliconでも動作を確認）。GPUは不要です。

```bash
make setup     # イメージのビルド、モデルの取得、ROS 2パッケージのビルド（初回のみ。時間がかかります）
make demo      # デモを再生・録画 -> outputs/ros_runs/demo/final.mp4
```

別の文を指定する場合:

```bash
make ros-demo T="手を振る" RUN=my_run      # 出力は outputs/ros_runs/my_run/
```

## 自分の手話動画で試す

1. 動画を `data/` に置く（`data/` はGit管理外です）。
2. アバター動画を作る。

   ```bash
   make video V="data/my_sign.mp4"          # 出力: outputs/video2avatar/my_sign.mp4
   ```

3. ROS 2で再生する場合は、`ros2_ws/src/sign_avatar/config/sentences.json` に文を登録します。

   ```json
   {"id": "my_sign", "text": "登録する日本語の文",
    "q": "outputs/avatar_q_my_sign.npz", "motion": "android_v2/outputs/motion_my_sign.json"}
   ```

   ```bash
   make ros-demo T="登録する日本語の文" RUN=my_run
   ```

動画の条件: 上半身と両手が正面から映っていること。明るさが十分で、手が小さすぎないこと。手が重なる、速く動く、暗い、といった場合は、手が検出できない区間が増えます。1本あたりの処理時間は、4〜9秒の動画で約1〜1.5分でした（Apple SiliconのDocker上）。

動画の手話を再現した映像は、元の話者の動きと表情を反映します。他人の動画を使う場合は、その人と権利者の許可を得てください。

## ROS 2の構成

| ノード | 役割 |
|---|---|
| `sign_planner` | 入力の日本語文を、登録文に対応づける（既定は文字列の類似度。`use_llm:=true` でLLMも使える） |
| `sign_motion_player` | 登録文の関節角を `/joint_states` に流す。文の間は滑らかにつなぐ |
| `avatar_sim` | MuJoCoで体を描画し、画像・マスク・頭の姿勢を流す |
| `sign_recorder` | 画像などを保存し、顔を合成して `final.mp4` にする |

| トピック | 型 | 内容 |
|---|---|---|
| `/sign/command` | String（JSON） | 再生する登録文 |
| `/sign/frame` | String（JSON） | フレームごとの文と番号（録画側が同期に使う） |
| `/sign/status` | String（JSON） | 再生の状態 |
| `/joint_states` | JointState | 関節名つきの関節角 |
| `/avatar/image/compressed`, `/avatar/mask/compressed` | CompressedImage | 体の映像、頭より手前の領域 |
| `/avatar/head`, `/avatar/camera` | String（JSON） | 頭の位置・回転、カメラ（`/avatar/camera` は遅れて起動したノードにも届く設定） |
| `/avatar/ack` | String | 描画完了の返事 |

- **パラメータ**: 各ノードが宣言しています（例: `lockstep`、`transition_sec`、`hold_sec`、`distance`、`use_llm`、`run_name`）。launchから渡します。
- **同期（lockstep）**: CPU描画が遅いので、playerは1フレーム送るごとに、simの `/avatar/ack` を待ってから次を送ります。録画に抜けが出ません。
- **使っていない通信**: サービスとアクションは使っていません。ackはサービスに、文の再生はアクション（進捗のフィードバックつき）にするのが、本来の使い分けです。改善案として残しています。

## 工夫した点：検出できない手の補完

MediaPipeは、両手の重なりや手の小ささで、手を検出できない区間があります。手話では表現の欠落になるため、補完手法を実装し、検出できた区間を人為的に隠して誤差を比べました（`src/mask_experiment.py`、`src/handfill.py`）。実験の再実行には、手話の動画から抽出した結果が必要です（動画は含めていません）。

| 手法 | 内容 |
|---|---|
| A. 保持 | 直前の手形を保持 |
| B. 線形補間 | 前後の検出から線形補間 |
| C. 平均手形 | 平均の手形に、直前の向きを当てる |
| D0. 正準座標＋球面補間 | 手形と向きを分け、向きを球面補間 |
| D. 手形の事前分布 | D0に、学習した手形の部分空間への射影を加える |

手のランドマークの誤差（cm。小さいほど良い。4つの動画の平均）:

| 隠す長さ | A:保持 | B:線形 | C:平均手形 | D0 | D（k=8） |
|---|---|---|---|---|---|
| 0.3秒 | 1.34 | **0.94** | 2.01 | 0.97 | 1.28 |
| 0.8秒 | 1.97 | **1.38** | 2.21 | 1.43 | 1.66 |
| 1.5秒 | 2.47 | **1.88** | 2.38 | 1.94 | 2.13 |

- 前後が見えている欠損では、線形補間（B）が最良で、保持（A）より誤差が約3割小さくなりました。
- 事前分布（D）は、前後の手形が分かっているこの条件では逆効果でした。
- この実験は、前後が検出できている区間だけを使っています。前後も見えない長い欠損は評価できていません。

## AIの活用

- **MediaPipe**: 手・姿勢・顔の表情係数の推論（学習はしていません）
- **LLM（任意）**: 日本語の入力文の対応づけ（`use_llm:=true`、`ANTHROPIC_API_KEY` が必要。動作の確認は十分にはしていません）
- **開発の支援**: 体の外装メッシュの磨き込みにChatGPT、実装・検証にClaude Codeを使いました

## 既知の限界

- 顔はMuJoCoの描画ではなく、VRMを別に描画してロボットの頭に重ねた合成です（録画のあとに recorder が重ねます）。
- 腕を下ろす場面で、指先が画面の下で切れることがあります。
- 手の重なりでの検出欠損、非手指表現（表情・口形）の取りこぼしがあります。
- 手話の意味の正確さの確認は、手話話者による確認が必要です。

## データの扱い

- 手話の元動画、そこから作った動きのデータ、映像は、このリポジトリに含めていません（顔が映る、または元の話者の動きを写し取ったものであるため）。`.gitignore` で動画ファイルも除外しています。
- `demo/` の動きは、関節角を直接書いたもので、誰かの手話の写し取りではありません。
- 氏名・所属などの個人情報は含めていません。

## ディレクトリ構成

```
video2avatar.py            動画からアバター動画までを1コマンドで実行
src/                       抽出（MediaPipe）、補完、リターゲティング、マスク実験
android_v2/                アンドロイド型の体（MuJoCoモデル、メッシュ）と、顔の合成（Three.js）
ros2_ws/src/sign_avatar/   ROS 2パッケージ（4ノード、launch、登録文の設定）
demo/                      撮影不要の手書きデモの動き
tools_make_demo_motion.py  手書きデモの動きを作る
scripts/fetch_models.sh    モデルの取得
```

## ライセンスとクレジット

`LICENSE` を参照してください。第三者の部品は、それぞれのライセンスに従います。

- MuJoCo、MuJoCo Menagerie（Shadow Hand）: Apache 2.0（Google DeepMind）
- MediaPipe: Apache 2.0（Google）
- 顔・髪のモデル: VRM1_Constraint_Twist_Sample（(c) 2022 pixiv Inc.、https://vrm.dev/licenses/1.0/）。`make models` で取得します
- three.js、@pixiv/three-vrm: MIT
- ROS 2 Humble: Apache 2.0
