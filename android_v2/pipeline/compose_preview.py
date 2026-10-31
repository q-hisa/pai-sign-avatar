"""Combine robot RGB, VRM head and occlusion masks into a standalone preview."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image


def compose(robot_dir, head_dir, out):
    info = json.loads((robot_dir/'headpose.json').read_text())
    frames = out.parent / (out.stem+'_frames')
    frames.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(info['T']):
        r = np.asarray(Image.open(robot_dir/f'r_{i:04d}.jpg').convert('RGB'), dtype=float)
        h = np.asarray(Image.open(head_dir/f'h_{i:04d}.png').convert('RGBA'), dtype=float)
        mask = np.asarray(Image.open(robot_dir/f'o_{i:04d}.png').convert('L'), dtype=float)/255
        if r.shape[:2] != h.shape[:2] or r.shape[:2] != mask.shape:
            raise ValueError(f'Frame {i} has incompatible dimensions.')
        a = h[...,3:4]/255 * (1-mask[...,None])
        result = np.clip(r*(1-a)+h[...,:3]*a, 0, 255).astype(np.uint8)
        Image.fromarray(result).save(frames/f'c_{i:04d}.png')
    if info['T'] == 1:
        Image.open(frames/'c_0000.png').save(out.with_suffix('.png'))
    else:
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(info['fps']),
                        '-i',str(frames/'c_%04d.png'),'-frames:v',str(info['T']),'-c:v','libx264',
                        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)], check=True)
    print('composited', info['T'], 'frames', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['robot_dir','head_dir','out']:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    compose(a.robot_dir, a.head_dir, a.out)
