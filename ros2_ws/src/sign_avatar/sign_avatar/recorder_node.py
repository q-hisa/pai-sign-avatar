"""recorder: 体の映像などを保存し、再生が終わったら顔を合成して mp4 にする。

保存先: outputs/ros_runs/<run_name>/（robot/ 体の画像・マスク・headpose.json、timeline.json、motion.json、final.mp4）
- 受け取ったフレームを欠けなく保存（再生側と同期）
- 完了後、再生した文の顔の動作データ(motion_*.json)をフレーム順につなぎ、VRMの顔・髪を重ねる
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .common import (LATCHED, RELIABLE, T_CAMERA, T_FRAME, T_HEAD, T_IMAGE, T_MASK,
                     T_STATUS, load_library, repo_root)


class Recorder(Node):
    def __init__(self):
        super().__init__('sign_recorder')
        root = repo_root()
        self.root = root
        self.declare_parameter('library', '')
        self.declare_parameter('out_root', str(root / 'outputs/ros_runs'))
        self.declare_parameter('run_name', time.strftime('run_%Y%m%d_%H%M%S'))
        self.declare_parameter('postprocess', True)
        self.declare_parameter('vrm', str(root / 'android_v2/data/model.vrm'))
        self.declare_parameter('scale', 1.4)
        self.declare_parameter('haircut', -0.025)
        self.library = load_library(self.get_parameter('library').value or None)
        name = self.get_parameter('run_name').value or time.strftime('run_%Y%m%d_%H%M%S')
        self.run = Path(self.get_parameter('out_root').value) / name
        (self.run / 'robot').mkdir(parents=True, exist_ok=True)
        self.cam = None
        self.status = None
        self.info, self.heads = {}, {}
        self.n_img, self.n_mask = set(), set()
        self.finished = False
        self.create_subscription(CompressedImage, T_IMAGE, self.on_image, RELIABLE)
        self.create_subscription(CompressedImage, T_MASK, self.on_mask, RELIABLE)
        self.create_subscription(String, T_HEAD, self.on_head, RELIABLE)
        self.create_subscription(String, T_FRAME, self.on_frame, RELIABLE)
        self.create_subscription(String, T_CAMERA, self.on_camera, LATCHED)
        self.create_subscription(String, T_STATUS, self.on_status, RELIABLE)
        self.create_timer(0.2, self.check)
        self.get_logger().info(f'録画先: {self.run}')

    def on_camera(self, msg):
        self.cam = json.loads(msg.data)

    def on_frame(self, msg):
        d = json.loads(msg.data)
        self.info[d['seq']] = d

    def on_head(self, msg):
        d = json.loads(msg.data)
        self.heads[d['seq']] = d

    def on_image(self, msg):
        i = int(msg.header.frame_id)
        (self.run / 'robot' / f'r_{i:04d}.jpg').write_bytes(bytes(msg.data))
        self.n_img.add(i)

    def on_mask(self, msg):
        i = int(msg.header.frame_id)
        (self.run / 'robot' / f'o_{i:04d}.png').write_bytes(bytes(msg.data))
        self.n_mask.add(i)

    def on_status(self, msg):
        s = json.loads(msg.data)
        if s['state'] == 'playing':
            self.get_logger().info(f'録画開始: {s["ids"]}（{s["frames"]}フレーム）')
        self.status = s

    def check(self):
        s = self.status
        if self.finished or not s or s['state'] != 'done':
            return
        n = s['frames']
        want = set(range(n))
        if not (want <= self.n_img and want <= self.n_mask and want <= set(self.heads) and want <= set(self.info)):
            self._t = getattr(self, '_t', time.monotonic())
            if time.monotonic() - self._t > 15:
                self.get_logger().error('フレームが揃いません。中止します')
                self.finished = True
                self.fail = True
            return
        self.finished = True
        self.finalize(s)

    def finalize(self, s):
        n, fps = s['frames'], s['fps']
        (self.run / 'robot' / 'headpose.json').write_text(json.dumps(
            dict(fps=fps, T=n, cam=self.cam,
                 head=[dict(p=self.heads[i]['p'], R=self.heads[i]['R']) for i in range(n)])))
        timeline = [self.info[i] for i in range(n)]
        (self.run / 'timeline.json').write_text(json.dumps(
            dict(ids=s['ids'], texts=s.get('texts', []), fps=fps, frames=timeline), ensure_ascii=False))
        self.get_logger().info(f'体の映像 {n}フレームを保存しました')
        if not self.get_parameter('postprocess').value:
            return
        # 顔の動作データを、再生したフレームの順につなぐ
        cache = {sid: json.loads((self.root / self.library[sid]['motion']).read_text()) for sid in set(s['ids'])}
        pose, hr, hl = [], [], []
        blend = {k: [] for k in next(iter(cache.values()))['blend']}
        for f in timeline:
            sid, i = f['m']
            M = cache[sid]
            pose.append(M['pose'][i]); hr.append(M['handR'][i]); hl.append(M['handL'][i])
            for k in blend:
                blend[k].append(M['blend'][k][i])
        (self.run / 'motion.json').write_text(json.dumps(dict(fps=fps, T=n, pose=pose, handR=hr, handL=hl, blend=blend)))
        pipe = self.root / 'android_v2/pipeline'
        head = self.run / 'head'
        py = sys.executable
        self.get_logger().info('顔・髪を描画しています（Chromium）…')
        subprocess.run([py, str(pipe / 'render_overlay.py'), self.get_parameter('vrm').value,
                        str(self.run / 'motion.json'), str(self.run / 'robot' / 'headpose.json'), str(head),
                        str(self.get_parameter('scale').value), str(self.get_parameter('haircut').value)], check=True)
        subprocess.run([py, str(pipe / 'compose_preview.py'), str(self.run / 'robot'), str(head),
                        str(self.run / 'final.mp4')], check=True)
        self.get_logger().info(f'完成: {self.run / "final.mp4"}')


def main():
    rclpy.init()
    node = Recorder()
    code = 0
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        code = 1 if getattr(node, 'fail', False) else 0
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)
