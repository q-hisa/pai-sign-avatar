"""Render the android and head-occlusion masks from named joints in a base MJCF."""
import argparse
import json
import os
from pathlib import Path
import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def render(args):
    old = mujoco.MjModel.from_xml_path(str(args.base_model))
    m = mujoco.MjModel.from_xml_path(str(args.avatar))
    data = np.load(args.motion, allow_pickle=False)
    Q, fps = data['q'], float(data['fps'])
    if Q.ndim != 2 or Q.shape[1] != old.nq:
        raise ValueError(f'Motion has {Q.shape}; base model requires {old.nq} qpos values.')
    if not np.isfinite(Q).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Motion must contain finite joint values and a positive FPS.')
    pairs = []
    for j in range(m.njnt):
        name = m.joint(j).name
        try:
            oj = old.joint(name).id
        except KeyError as exc:
            raise ValueError(f'Missing joint {name} in the source model.') from exc
        if old.jnt_type[oj] != m.jnt_type[j]:
            raise ValueError(f'Joint type changed: {name}')
        width = 7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 4 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
        pairs.append((int(old.jnt_qposadr[oj]), int(m.jnt_qposadr[j]), width))
    print('mapped joints', len(pairs), 'of', m.njnt, flush=True)
    m.vis.global_.offwidth = args.size
    m.vis.global_.offheight = args.size
    d = mujoco.MjData(m)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 1.25]
    camera.distance, camera.azimuth, camera.elevation = args.distance, 270, -3
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    hid = m.body('head').id
    collar_ids = [m.geom(n).id for n in ('collar', 'collar_ring')]
    A = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    args.out.mkdir(parents=True, exist_ok=True)
    frames = len(Q) if args.limit is None else min(len(Q), args.limit)
    heads, camjson = [], None
    with mujoco.Renderer(m, args.size, args.size) as ren:
        for i in range(frames):
            d.qpos[:] = m.qpos0
            for a, b, width in pairs:
                d.qpos[b:b+width] = Q[i, a:a+width]
            mujoco.mj_forward(m, d)
            ren.update_scene(d, camera=camera, scene_option=option)
            rgb = ren.render()
            ren.enable_depth_rendering()
            depth = ren.render()
            ren.disable_depth_rendering()
            ren.enable_segmentation_rendering()
            seg = ren.render()
            ren.disable_segmentation_rendering()
            collar = np.isin(seg[..., 0], collar_ids) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            # MuJoCo's mono renderer averages both eyes; export that same camera.
            gl = mujoco.mjv_averageCamera(ren.scene.camera[0], ren.scene.camera[1])
            pos, fwd, up = np.array(gl.pos), np.array(gl.forward), np.array(gl.up)
            hp, hm = d.xpos[hid].copy(), d.xmat[hid].reshape(3, 3).copy()
            center = hp + hm @ np.array([0, 0, .16])
            zc = float((center - pos) @ fwd)
            mask = ((depth < zc - .13) | collar).astype(np.uint8) * 255
            cv2.imwrite(str(args.out / f'r_{i:04d}.jpg'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(args.out / f'o_{i:04d}.png'), mask)
            if camjson is None:
                camjson = dict(pos=(A @ pos).tolist(), forward=(A @ fwd).tolist(), up=(A @ up).tolist(),
                               fovy=float(m.vis.global_.fovy), size=args.size)
            heads.append(dict(p=(A @ hp).tolist(), R=(A @ hm @ A.T).reshape(-1).tolist()))
            if i % 30 == 0:
                print('frame', i, flush=True)
    (args.out / 'headpose.json').write_text(json.dumps(dict(fps=fps, T=frames, cam=camjson, head=heads)))
    print('done', args.out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stem')
    p.add_argument('--avatar', type=Path, default=Path(os.environ.get('AVATAR', str(ROOT/'model/avatar_android.xml'))))
    p.add_argument('--base-model', type=Path, default=ROOT/'model/avatar.xml')
    p.add_argument('--motion', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--size', type=int, default=720)
    p.add_argument('--distance', type=float, default=1.5)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    args.motion = args.motion or ROOT/f'data/avatar_q_{args.stem}.npz'
    args.out = args.out or ROOT/f'outputs/robot_{args.stem}'
    render(args)
