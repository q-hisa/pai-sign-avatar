"""sign_motion_player: 登録文の関節データ(avatar_q_*.npz)を /joint_states に流す。

- /sign/command を受けると、文を順に再生する（文と文の間は滑らかにつなぐ）
- lockstep:=true（既定）: avatar_sim の描画完了(/avatar/ack)を待って次のフレームを送る。録画で1フレームも落とさない
- 出力: /joint_states（関節名つき）、/sign/frame（フレームの情報）、/sign/status（playing / done）
"""
import json
import time

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .common import (RELIABLE, T_ACK, T_COMMAND, T_FRAME, T_JOINTS, T_STATUS,
                     load_library, repo_root)


class Player(Node):
    def __init__(self):
        super().__init__('sign_motion_player')
        root = repo_root()
        self.declare_parameter('library', '')
        self.declare_parameter('avatar_model', str(root / 'android_v2/model/avatar_android.xml'))
        self.declare_parameter('base_model', str(root / 'models/avatar/avatar.xml'))
        self.declare_parameter('lockstep', True)
        self.declare_parameter('transition_sec', 0.6)
        self.declare_parameter('hold_sec', 0.5)
        self.declare_parameter('wait_for_recorder', True)
        self.root = root
        self.library = load_library(self.get_parameter('library').value or None)

        # 動作データの並び(元モデルのqpos) → 関節名。アンドロイド体に存在する関節だけを送る
        base = mujoco.MjModel.from_xml_path(self.get_parameter('base_model').value)
        body = mujoco.MjModel.from_xml_path(self.get_parameter('avatar_model').value)
        self.names, self.adr = [], []
        for j in range(body.njnt):
            name = body.joint(j).name
            bj = base.joint(name)
            if base.jnt_type[bj.id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise RuntimeError(f'ヒンジ以外の関節は未対応: {name}')
            self.names.append(name)
            self.adr.append(int(base.jnt_qposadr[bj.id]))
        self.get_logger().info(f'関節 {len(self.names)} 個')

        self.pub_js = self.create_publisher(JointState, T_JOINTS, RELIABLE)
        self.pub_frame = self.create_publisher(String, T_FRAME, RELIABLE)
        self.pub_status = self.create_publisher(String, T_STATUS, RELIABLE)
        self.create_subscription(String, T_COMMAND, self.on_command, RELIABLE)
        self.create_subscription(String, T_ACK, self.on_ack, RELIABLE)
        self.frames = []        # [(q(関節順), info)]
        self.pos = 0
        self.fps = 30.0
        self.waiting = False
        self.acked = -1
        self.playing = False
        self.next_time = 0.0
        self.pending = None
        self.create_timer(0.005, self.tick)

    # ---- 動きの組み立て ----
    def load_q(self, sid):
        d = np.load(self.root / self.library[sid]['q'], allow_pickle=False)
        return d['q'][:, self.adr].astype(float), float(d['fps'])

    def build(self, ids):
        tsec = float(self.get_parameter('transition_sec').value)
        hold = float(self.get_parameter('hold_sec').value)
        seqs = []
        for sid in ids:
            q, fps = self.load_q(sid)
            seqs.append((sid, q))
            self.fps = fps
        n_tr, n_hold = max(1, round(tsec * self.fps)), round(hold * self.fps)
        out = []

        def add(q, kind, m):
            out.append((q, {'kind': kind, 'm': m}))

        for _ in range(n_hold):
            add(seqs[0][1][0], 'hold', [seqs[0][0], 0])
        for k, (sid, q) in enumerate(seqs):
            for i in range(len(q)):
                add(q[i], 'motion', [sid, i])
            if k + 1 < len(seqs):
                nsid, nq = seqs[k + 1]
                for s in range(1, n_tr + 1):
                    w = s / (n_tr + 1)
                    w = w * w * (3 - 2 * w)  # smoothstep
                    src = [sid, len(q) - 1] if s <= (n_tr + 1) // 2 else [nsid, 0]
                    add((1 - w) * q[-1] + w * nq[0], 'transition', src)
        for _ in range(n_hold):
            add(seqs[-1][1][-1], 'hold', [seqs[-1][0], len(seqs[-1][1]) - 1])
        return out

    # ---- コールバック ----
    def on_command(self, msg):
        if self.playing:
            self.get_logger().warn('再生中のため、新しい依頼は無視します')
            return
        cmd = json.loads(msg.data)
        ids = [i for i in cmd['ids'] if i in self.library]
        if not ids:
            return
        self.frames = self.build(ids)
        self.pos, self.acked, self.waiting = 0, -1, False
        self.pending = {'ids': ids, 'texts': cmd.get('texts', [])}
        self.playing = True
        self.started = False
        self.get_logger().info(f'再生を準備: {ids}（{len(self.frames)}フレーム、{self.fps:.2f}fps）')

    def on_ack(self, msg):
        self.acked = int(msg.data)

    def ready(self):
        """描画側（と録画側）がつながっているか。"""
        ok = self.pub_js.get_subscription_count() >= 1 and self.count_publishers(T_ACK) >= 1
        if self.get_parameter('wait_for_recorder').value:
            ok = ok and self.pub_frame.get_subscription_count() >= 1
        return ok

    def status(self, **kw):
        self.pub_status.publish(String(data=json.dumps(kw, ensure_ascii=False)))

    def send(self, i):
        q, info = self.frames[i]
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = str(i)
        js.name = self.names
        js.position = [float(v) for v in q]
        info = dict(info, seq=i)
        self.pub_frame.publish(String(data=json.dumps(info)))
        self.pub_js.publish(js)

    def tick(self):
        if not self.playing:
            return
        if not self.started:
            if not self.ready():
                return
            self.started = True
            self.status(state='playing', fps=self.fps, frames=len(self.frames), **self.pending)
            self.next_time = time.monotonic()
        lock = self.get_parameter('lockstep').value
        if self.waiting:
            if self.acked == self.pos - 1:
                self.waiting = False
            elif time.monotonic() - self.sent_at > 30.0:
                self.get_logger().error('描画の返事が30秒来ません。中止します')
                self.playing = False
                return
            else:
                return
        if self.pos >= len(self.frames):
            self.playing = False
            self.status(state='done', frames=len(self.frames), fps=self.fps, **self.pending)
            self.get_logger().info('再生が完了しました')
            return
        if not lock and time.monotonic() < self.next_time:
            return
        self.send(self.pos)
        self.pos += 1
        self.sent_at = time.monotonic()
        self.next_time += 1.0 / self.fps
        self.waiting = lock


def main():
    rclpy.init()
    node = Player()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
