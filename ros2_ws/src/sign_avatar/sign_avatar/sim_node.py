"""avatar_sim: /joint_states を受けて MuJoCo のアンドロイド体を動かし、映像を配信する。

入力 : /joint_states（関節名つき）
出力 : /avatar/image/compressed（JPEG）、/avatar/mask/compressed（PNG: 頭より手前の領域）、
       /avatar/head（頭の位置・回転）、/avatar/camera（起動後1回）、/avatar/ack（描画完了）
頭（顔・髪）は描かない。顔の合成は recorder が録画の後で行う。
"""
import json

import cv2
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import String

from .common import (LATCHED, RELIABLE, T_ACK, T_CAMERA, T_HEAD, T_IMAGE, T_JOINTS,
                     T_MASK, repo_root)

A = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)  # MuJoCo → three.js


class AvatarSim(Node):
    def __init__(self):
        super().__init__('avatar_sim')
        self.declare_parameter('avatar_model', str(repo_root() / 'android_v2/model/avatar_android.xml'))
        self.declare_parameter('size', 720)
        self.declare_parameter('distance', 1.75)
        self.declare_parameter('jpeg_quality', 95)
        size = int(self.get_parameter('size').value)
        self.size = size
        self.m = mujoco.MjModel.from_xml_path(self.get_parameter('avatar_model').value)
        self.m.vis.global_.offwidth = size
        self.m.vis.global_.offheight = size
        self.d = mujoco.MjData(self.m)
        self.adr = {self.m.joint(j).name: int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt)}
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0, 0, 1.25]
        self.camera.distance = float(self.get_parameter('distance').value)
        self.camera.azimuth, self.camera.elevation = 270, -3
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.hid = self.m.body('head').id
        self.collar_ids = [self.m.geom(n).id for n in ('collar', 'collar_ring')]
        self.ren = mujoco.Renderer(self.m, size, size)
        self.cam_sent = False
        self.pub_img = self.create_publisher(CompressedImage, T_IMAGE, RELIABLE)
        self.pub_mask = self.create_publisher(CompressedImage, T_MASK, RELIABLE)
        self.pub_head = self.create_publisher(String, T_HEAD, RELIABLE)
        self.pub_cam = self.create_publisher(String, T_CAMERA, LATCHED)
        self.pub_ack = self.create_publisher(String, T_ACK, RELIABLE)
        self.create_subscription(JointState, T_JOINTS, self.on_joints, RELIABLE)
        self.get_logger().info(f'アンドロイド体を読み込み: 関節 {self.m.njnt} 個、{size}x{size}')

    def render(self):
        ren = self.ren
        ren.update_scene(self.d, camera=self.camera, scene_option=self.option)
        rgb = ren.render()
        ren.enable_depth_rendering()
        depth = ren.render()
        ren.disable_depth_rendering()
        ren.enable_segmentation_rendering()
        seg = ren.render()
        ren.disable_segmentation_rendering()
        collar = np.isin(seg[..., 0], self.collar_ids) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        gl = mujoco.mjv_averageCamera(ren.scene.camera[0], ren.scene.camera[1])
        pos, fwd, up = np.array(gl.pos), np.array(gl.forward), np.array(gl.up)
        hp, hm = self.d.xpos[self.hid].copy(), self.d.xmat[self.hid].reshape(3, 3).copy()
        center = hp + hm @ np.array([0, 0, .16])
        zc = float((center - pos) @ fwd)
        mask = ((depth < zc - .13) | collar).astype(np.uint8) * 255
        return rgb, mask, pos, fwd, up, hp, hm

    def on_joints(self, msg):
        self.d.qpos[:] = self.m.qpos0
        for name, v in zip(msg.name, msg.position):
            a = self.adr.get(name)
            if a is not None:
                self.d.qpos[a] = v
        mujoco.mj_forward(self.m, self.d)
        rgb, mask, pos, fwd, up, hp, hm = self.render()
        seq = msg.header.frame_id
        if not self.cam_sent:
            cam = dict(pos=(A @ pos).tolist(), forward=(A @ fwd).tolist(), up=(A @ up).tolist(),
                       fovy=float(self.m.vis.global_.fovy), size=self.size)
            self.pub_cam.publish(String(data=json.dumps(cam)))
            self.cam_sent = True
        q = int(self.get_parameter('jpeg_quality').value)
        ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        ok2, png = cv2.imencode('.png', mask)
        for pub, fmt, buf in ((self.pub_img, 'jpeg', jpg), (self.pub_mask, 'png', png)):
            im = CompressedImage()
            im.header = msg.header
            im.format = fmt
            im.data = buf.tobytes()
            pub.publish(im)
        head = dict(seq=int(seq), p=(A @ hp).tolist(), R=(A @ hm @ A.T).reshape(-1).tolist())
        self.pub_head.publish(String(data=json.dumps(head)))
        self.pub_ack.publish(String(data=seq))   # 画像・頭の後に返事を出す


def main():
    rclpy.init()
    node = AvatarSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
