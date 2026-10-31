"""avatar.xml -> avatar_android.xml : slender white android body (smooth shell, dark joints, blue accent lines, shoulder lenses).
The head is an empty body at the neck pivot; the character head (VRM, with its own neck) is composited on top in post."""
import re
src = open("models/avatar/avatar.xml").read()
NC = 'contype="0" conaffinity="0"'
src = src.replace('<material name="body" rgba="0.35 0.45 0.65 1"/>',
  '''<material name="body" rgba="0.17 0.18 0.21 1" specular="0.6" shininess="0.6"/>
    <material name="armor" rgba="0.96 0.96 0.98 1" specular="0.5" shininess="0.5"/>
    <material name="dark" rgba="0.17 0.18 0.21 1" specular="0.6" shininess="0.6"/>
    <material name="glass" rgba="0.08 0.09 0.12 1" specular="0.9" shininess="0.9"/>
    <material name="blue" rgba="0.15 0.45 1 1" emission="0.9"/>''')
src = src.replace('<material name="skin" rgba="0.85 0.7 0.6 1"/>', '<material name="skin" rgba="0.17 0.18 0.21 1"/>')
src = src.replace('<headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6"/>', '<headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/>')
src = src.replace('<texture name="grid"', '<texture name="sky" type="skybox" builtin="gradient" rgb1="0.97 0.97 0.99" rgb2="0.8 0.83 0.9" width="256" height="256"/>\n    <texture name="grid"')
src = src.replace('rgb1="0.2 0.22 0.25" rgb2="0.25 0.27 0.3"', 'rgb1="0.92 0.93 0.95" rgb2="0.87 0.89 0.93"')

torso = f'''<geom type="ellipsoid" size="0.15 0.095 0.07" pos="0 0 -0.24" material="armor"/>
      <geom type="ellipsoid" size="0.098 0.07 0.15" pos="0 0 -0.1" material="dark"/>
      <geom type="capsule" fromto="0 0.03 -0.17 0 0.03 -0.03" size="0.06" material="armor" {NC}/>
      <geom type="box" size="0.004 0.03 0.09" pos="-0.1 0.02 -0.1" material="blue" {NC}/>
      <geom type="box" size="0.004 0.03 0.09" pos="0.1 0.02 -0.1" material="blue" {NC}/>
      <geom type="ellipsoid" size="0.18 0.095 0.125" pos="0 0 0.1" material="armor"/>
      <geom type="ellipsoid" size="0.058 0.05 0.055" pos="-0.06 0.05 0.075" material="armor" {NC}/>
      <geom type="ellipsoid" size="0.058 0.05 0.055" pos="0.06 0.05 0.075" material="armor" {NC}/>
      <geom type="ellipsoid" size="0.235 0.085 0.06" pos="0 -0.005 0.2" material="armor" {NC}/>
      <geom type="box" size="0.04 0.004 0.005" pos="0 0.098 0.2" material="blue" {NC}/>
      <geom name="collar" type="cylinder" fromto="0 0 0.235 0 0 0.3" size="0.05" material="armor"/>
      <geom name="collar_ring" type="cylinder" fromto="0 0 0.292 0 0 0.302" size="0.053" material="dark" {NC}/>'''
src = src.replace('<geom type="box" size="0.2 0.1 0.27" material="body"/>\n      <geom type="capsule" fromto="0 0 0.27 0 0 0.36" size="0.045" material="skin"/>', torso)
head = '''      <body name="head" pos="0 0 0.30">
        <joint name="neck_yaw" axis="0 0 1" range="-1.2 1.2"/>
        <joint name="neck_pitch" axis="1 0 0" range="-0.6 0.6"/>
        <joint name="neck_roll" axis="0 1 0" range="-0.5 0.5"/>
        <geom type="sphere" size="0.01" group="3" contype="0" conaffinity="0"/>
      </body>
'''
src = re.sub(r'      <body name="head" pos=.*?\n      </body>\n(?=\n      <!-- 右腕)', lambda m: head, src, flags=re.S)
for sd, sx in (("r", 1), ("l", -1)):
    pat = re.compile(r'(<body name="%s_shoulder"[^>]*>.*?)<geom type="capsule" fromto="0 0 0 0 0 -0.30" size="0.04" material="body"/>' % sd, re.S)
    add = (f'<geom type="sphere" pos="{0.012*sx} 0 0.01" size="0.068" material="armor"/>\n        '
           f'<geom type="cylinder" fromto="{0.06*sx} 0 0.01 {0.068*sx} 0 0.01" size="0.04" material="blue" {NC}/>\n        '
           f'<geom type="cylinder" fromto="{0.066*sx} 0 0.01 {0.078*sx} 0 0.01" size="0.034" material="glass" {NC}/>\n        '
           '<geom type="capsule" fromto="0 0 -0.05 0 0 -0.08" size="0.04" material="dark"/>\n        '
           '<geom type="capsule" fromto="0 0 -0.08 0 0 -0.25" size="0.033" material="armor"/>\n        '
           f'<geom type="sphere" pos="0 0 -0.30" size="0.04" material="dark" {NC}/>')
    src, k = pat.subn(lambda m: m.group(1) + add, src)
    assert k == 1, sd
open("models/avatar/avatar_android.xml", "w").write(src)
print("ok")
