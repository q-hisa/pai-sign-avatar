"""ノード間で共通の設定とユーティリティ。"""
import json
import os
from pathlib import Path

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# トピック名
T_TEXT_IN = '/sign/text_in'        # std_msgs/String   : 入力の日本語文（実行中にも受け付ける）
T_COMMAND = '/sign/command'        # std_msgs/String   : JSON {"ids":[...], "texts":[...]}  再生する登録文
T_STATUS = '/sign/status'          # std_msgs/String   : JSON {"state":"playing|done", ...}
T_FRAME = '/sign/frame'            # std_msgs/String   : JSON フレームごとの情報（録画側が同期に使う）
T_JOINTS = '/joint_states'         # sensor_msgs/JointState : アバターの関節角（名前つき）
T_ACK = '/avatar/ack'              # std_msgs/String   : 描画完了の返事（フレーム番号）
T_IMAGE = '/avatar/image/compressed'   # sensor_msgs/CompressedImage : 体の映像（JPEG）
T_MASK = '/avatar/mask/compressed'     # sensor_msgs/CompressedImage : 頭より手前の領域（PNG）
T_HEAD = '/avatar/head'            # std_msgs/String   : JSON 頭の位置・回転（three.js座標）
T_CAMERA = '/avatar/camera'        # std_msgs/String   : JSON カメラ（起動時に1回、遅れて来たノードにも届く）

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
RELIABLE = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)


def repo_root() -> Path:
    """リポジトリのルート（Docker内は /workspace）。環境変数 SIGN_REPO で上書きできる。"""
    return Path(os.environ.get('SIGN_REPO', '/workspace'))


def load_library(path=None):
    """登録文の一覧（config/sentences.json）を読む。"""
    if path:
        p = Path(path)
    else:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory('sign_avatar')) / 'config' / 'sentences.json'
    items = json.loads(p.read_text(encoding='utf-8'))
    return {it['id']: it for it in items}
