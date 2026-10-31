"""robot frames + head overlay (RGBA) + occlusion mask (+ skeleton video panel) -> mp4
usage: python composite.py <robot_dir> <head_dir> <hands_video> <out.mp4>"""
import sys, glob, subprocess, os
import numpy as np
from PIL import Image
rdir, hdir, hands, out = sys.argv[1:5]
tmp = out + "_frames"; os.makedirs(tmp, exist_ok=True)
n = len(glob.glob(f"{hdir}/h_*.png"))
for i in range(n):
    r = np.asarray(Image.open(f"{rdir}/r_{i:04d}.jpg").convert("RGB")).astype(np.float32)
    h = np.asarray(Image.open(f"{hdir}/h_{i:04d}.png").convert("RGBA")).astype(np.float32)
    o = np.asarray(Image.open(f"{rdir}/o_{i:04d}.png")).astype(np.float32) / 255.0
    a = (h[..., 3:4] / 255.0) * (1.0 - o[..., None])
    res = r * (1 - a) + h[..., :3] * a
    Image.fromarray(res.clip(0, 255).astype(np.uint8)).save(f"{tmp}/c_{i:04d}.jpg", quality=94)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", hands, "-framerate", "29.97", "-i", f"{tmp}/c_%04d.jpg",
  "-filter_complex", "[0:v]scale=-2:720[l];[l][1:v]hstack=inputs=2", "-r", "29.97", "-pix_fmt", "yuv420p", "-shortest", out], check=True)
print("wrote", out)
