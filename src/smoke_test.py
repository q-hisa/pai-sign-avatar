"""環境のスモークテスト。
1) MuJoCoを画面なしで描画  2) MediaPipeを読込  3) ffmpegでmp4を書き出し
成功すると outputs/smoke_test.mp4 ができる。"""
import os
from pathlib import Path

import mediapy
import mujoco
import numpy as np

# 仮の「3関節の指」。本番ではMenagerieのハンドモデルに置き換える。
XML = """
<mujoco>
  <compiler angle="radian"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="1 1 .1" rgba=".8 .8 .8 1"/>
    <body pos="0 0 .2">
      <geom type="box" size=".05 .05 .02" rgba=".3 .3 .8 1"/>
      <body pos="0 0 .02">
        <joint name="j1" type="hinge" axis="0 1 0" range="-1.6 0" damping=".002" armature=".0002"/>
        <geom type="capsule" fromto="0 0 0 0 0 .08" size=".012" rgba=".9 .6 .5 1"/>
        <body pos="0 0 .08">
          <joint name="j2" type="hinge" axis="0 1 0" range="-1.6 0" damping=".002" armature=".0002"/>
          <geom type="capsule" fromto="0 0 0 0 0 .06" size=".011" rgba=".9 .6 .5 1"/>
          <body pos="0 0 .06">
            <joint name="j3" type="hinge" axis="0 1 0" range="-1.6 0" damping=".002" armature=".0002"/>
            <geom type="capsule" fromto="0 0 0 0 0 .05" size=".01" rgba=".9 .6 .5 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="0.5" kv="0.02"/>
    <position name="a2" joint="j2" kp="0.5" kv="0.02"/>
    <position name="a3" joint="j3" kp="0.5" kv="0.02"/>
  </actuator>
</mujoco>
"""

def main() -> int:
    ok = True
    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    print("mujoco", mujoco.__version__)

    # 1) MediaPipe
    try:
        import mediapipe as mp
        print("[OK] mediapipe", mp.__version__)
    except Exception as e:  # noqa: BLE001
        print("[NG] mediapipe:", e); ok = False

    # 2) 物理シミュレーション（描画なし）
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for _ in range(100):
        data.ctrl[:] = -1.0
        mujoco.mj_step(model, data)
    print("[OK] physics: joint angles =", np.round(data.qpos, 2))

    # 3) 画面なし描画 + mp4書き出し
    try:
        frames = []
        with mujoco.Renderer(model, 480, 640) as renderer:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0, 0, 0.3]; cam.distance = 0.8; cam.azimuth = 150; cam.elevation = -15
            mujoco.mj_resetData(model, data)
            for step in range(240):
                t = step / 240
                data.ctrl[:] = -1.5 * (0.5 - 0.5 * np.cos(2 * np.pi * t))  # 指を曲げ伸ばし
                for _ in range(10):
                    mujoco.mj_step(model, data)
                if step % 4 == 0:
                    renderer.update_scene(data, cam)
                    frames.append(renderer.render())
        out = Path("outputs"); out.mkdir(exist_ok=True)
        mediapy.write_video(out / "smoke_test.mp4", frames, fps=30)
        print(f"[OK] render+video: {len(frames)}フレーム -> outputs/smoke_test.mp4")
    except Exception as e:  # noqa: BLE001
        print("[NG] render/video:", e); ok = False
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
