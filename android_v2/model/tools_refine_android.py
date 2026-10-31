"""Build smooth visual shells without changing the avatar's kinematics or physics."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
MESHES = HERE / 'meshes'
MESHES.mkdir(exist_ok=True)


def write_mesh(name, vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    normals = np.zeros_like(vertices)
    for a, b, c in faces:
        n = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        normals[a] += n
        normals[b] += n
        normals[c] += n
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    with (MESHES / f'{name}.obj').open('w') as f:
        f.write(f'# Smooth visual shell {name}; coordinates in metres\ns 1\n')
        for v in vertices:
            f.write('v ' + ' '.join(f'{x:.9f}' for x in v) + '\n')
        for n in normals:
            f.write('vn ' + ' '.join(f'{x:.9f}' for x in n) + '\n')
        for face in faces:
            f.write('f ' + ' '.join(f'{k+1}//{k+1}' for k in face) + '\n')


def loft(name, profile, exponent=2.5, segments=64, steps=5):
    """Closed loft with rounded rectangular sections and interpolated profiles."""
    p = np.asarray(profile, float)
    rings = []
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        # Hermite tangents in z avoid the polygonal silhouette of linear lofts.
        prev, nxt = p[max(0, i - 1)], p[min(len(p) - 1, i + 2)]
        ta = (b[1:] - prev[1:]) / (b[0] - prev[0])
        tb = (nxt[1:] - a[1:]) / (nxt[0] - a[0])
        for j in range(steps):
            t = j / steps
            z = a[0] * (1 - t) + b[0] * t
            vals = ((2*t**3 - 3*t**2 + 1)*a[1:]
                    + (t**3 - 2*t**2 + t)*(b[0]-a[0])*ta
                    + (-2*t**3 + 3*t**2)*b[1:]
                    + (t**3 - t**2)*(b[0]-a[0])*tb)
            rings.append([z, *vals])
    rings.append(p[-1])
    vertices = []
    for z, rx, ry, cy in rings:
        for j in range(segments):
            t = 2 * math.pi * j / segments
            cs, sn = math.cos(t), math.sin(t)
            vertices.append([max(rx, .0001)*math.copysign(abs(cs)**(2/exponent), cs),
                             cy+max(ry, .0001)*math.copysign(abs(sn)**(2/exponent), sn), z])
    faces = []
    for i in range(len(rings) - 1):
        for j in range(segments):
            a = i*segments+j
            b = i*segments+(j+1)%segments
            c = (i+1)*segments+(j+1)%segments
            d = (i+1)*segments+j
            faces.extend([(a, b, c), (a, c, d)])
    for ring_no, reverse in [(0, True), (len(rings)-1, False)]:
        z, rx, ry, cy = rings[ring_no]
        center = len(vertices)
        vertices.append([0, cy, z])
        for j in range(segments):
            a, b = ring_no*segments+j, ring_no*segments+(j+1)%segments
            faces.append((center, b, a) if reverse else (center, a, b))
    write_mesh(name, vertices, faces)


def torus(name, major, minor, z=0):
    vs, fs = [], []
    for i in range(64):
        a = i*2*math.pi/64
        for j in range(12):
            b = j*2*math.pi/12
            vs.append([(major+minor*math.cos(b))*math.cos(a),
                       (major+minor*math.cos(b))*math.sin(a), z+minor*math.sin(b)])
    for i in range(64):
        for j in range(12):
            a, b = i*12+j, ((i+1)%64)*12+j
            c, d = ((i+1)%64)*12+(j+1)%12, i*12+(j+1)%12
            fs.extend([(a,b,c), (a,c,d)])
    write_mesh(name, vs, fs)


loft('chest_shell', [
    [-.026,.128,.066,.012],[-.017,.151,.078,.012],[.030,.183,.092,.009],
    [.092,.207,.103,.006],[.165,.215,.094,.003],[.208,.189,.079,0],
    [.232,.120,.060,0],[.243,.073,.046,0]], exponent=2.65)
loft('abdomen_core', [[-.235,.080,.051,0],[-.215,.093,.060,0],
                     [-.060,.114,.068,0],[-.018,.131,.072,0]], exponent=2.4)
loft('abdomen_panel', [[-.221,.044,.008,.055],[-.211,.066,.011,.057],
                      [-.15,.075,.012,.064],[-.06,.074,.013,.073],
                      [-.035,.067,.010,.077]], exponent=3.2)
loft('waist_shell', [[-.281,.113,.059,0],[-.275,.147,.075,0],
                    [-.254,.161,.080,0],[-.232,.145,.071,0],
                    [-.215,.096,.053,0]], exponent=2.8)
loft('collar_shell', [[.233,.057,.052,0],[.242,.057,.052,0],
                     [.286,.049,.047,0],[.297,.048,.046,0]], exponent=2.35)
torus('collar_lip', .048, .003, .298)
loft('upper_arm_shell', [[-.268,.025,.027,0],[-.260,.032,.034,0],
                        [-.205,.035,.037,0],[-.103,.039,.040,0],
                        [-.079,.033,.035,0]], exponent=2.4)
loft('shoulder_shell', [[-.059,.024,.035,0],[-.043,.056,.052,0],
                      [.005,.069,.062,0],[.039,.059,.052,0],
                      [.056,.022,.031,0]], exponent=2.15)
loft('forearm_shell', [[.004,.025,.026,0],[.015,.038,.038,0],
                      [.045,.044,.041,0],[.095,.038,.036,0],
                      [.16,.030,.029,0],[.196,.024,.025,0],
                      [.204,.022,.023,0]], exponent=2.45)
loft('palm_shell', [[.000,.025,.014,0],[.010,.036,.016,0],
                   [.040,.039,.017,0],[.075,.040,.015,0],
                   [.089,.035,.013,0]], exponent=3.2)
for name, length, radius in [('finger_proximal', .043, .0088),
                              ('finger_middle', .024, .0082),
                              ('finger_tip', .026, .0082),
                              ('thumb_proximal', .036, .010),
                              ('thumb_middle', .030, .0095),
                              ('thumb_tip', .029, .0095)]:
    loft(name, [[.003,radius*.75,radius*.70,0], [.007,radius,radius*.85,0],
                [length-.008,radius*.94,radius*.82,0],
                [length-.003,radius*.65,radius*.64,0],
                [length-.001,radius*.25,radius*.3,0]], exponent=2.6, segments=32, steps=3)

tree = ET.parse(HERE / 'avatar_android_original.xml')
root = tree.getroot()
asset = root.find('asset')
for f in sorted(MESHES.glob('*.obj')):
    ET.SubElement(asset, 'mesh', name=f.stem, file=f'meshes/{f.name}', smoothnormal='true')
for mat in asset.findall('material'):
    name = mat.get('name')
    if name in ('body','dark','skin'):
        mat.set('rgba', '.13 .145 .17 1')
        mat.set('specular', '.35')
        mat.set('shininess', '.4')
    elif name == 'armor':
        mat.set('rgba', '.91 .94 .96 1')
        mat.set('specular', '.4')
        mat.set('shininess', '.4')
    elif name == 'blue':
        mat.set('rgba', '.055 .32 .78 1')
        mat.set('emission', '.3')
    elif name == 'glass':
        mat.set('rgba', '.065 .09 .12 1')
for tex in asset.findall('texture'):
    if tex.get('name') == 'grid':
        tex.set('rgb1', '.88 .91 .94')
        tex.set('rgb2', '.88 .91 .94')
root.find('visual/headlight').set('ambient', '.4 .4 .4')
root.find('visual/headlight').set('diffuse', '.65 .65 .65')


def add(body, name, mesh=None, material='armor', **attrs):
    attributes = dict(name=name, material=material, contype='0', conaffinity='0',
                      mass='0', group='2')
    if mesh:
        attributes.update(type='mesh', mesh=mesh)
    attributes.update({k: str(v) for k, v in attrs.items()})
    return ET.SubElement(body, 'geom', attributes)


def hide_originals(body):
    for i, g in enumerate(body.findall('geom')):
        # Keep original collision shapes and inferred inertia; only visibility changes.
        if g.get('name') in ('collar', 'collar_ring'):
            g.set('name', 'physics_' + g.get('name'))
        g.set('rgba', '0 0 0 0')
        g.set('group', '3')


torso = root.find(".//body[@name='torso']")
hide_originals(torso)
for mesh in ['chest_shell','abdomen_panel','waist_shell']:
    add(torso, mesh, mesh)
add(torso, 'abdomen_core', 'abdomen_core', 'dark')
add(torso, 'collar', 'collar_shell')
add(torso, 'collar_ring', 'collar_lip', 'dark')
add(torso, 'chest_indicator', material='blue', type='capsule',
    fromto='-.032 .101 .144 .032 .101 .144', size='.0025')
for side in (-1, 1):
    x = side*.102
    add(torso, f'waist_accent_{side}', material='blue', type='capsule',
        fromto=f'{x} .046 -.192 {side*.123} .047 -.060', size='.0023')
    sd = 'r' if side == 1 else 'l'
    body = root.find(f".//body[@name='{sd}_shoulder']")
    hide_originals(body)
    add(body, f'{sd}_shoulder_shell', 'shoulder_shell', pos=f'{side*.008} 0 .006')
    add(body, f'{sd}_upper_arm_shell', 'upper_arm_shell')
    add(body, f'{sd}_shoulder_joint', material='dark', type='sphere', size='.049')
    add(body, f'{sd}_elbow_joint', material='dark', type='sphere', size='.030', pos='0 0 -.294')
    add(body, f'{sd}_arm_socket', material='dark', type='capsule',
        fromto='0 0 -.052 0 0 -.088', size='.027')
    add(body, f'{sd}_shoulder_light', material='blue', type='cylinder',
        fromto=f'{side*.071} 0 .010 {side*.074} 0 .010', size='.027')
    add(body, f'{sd}_shoulder_lens', material='glass', type='cylinder',
        fromto=f'{side*.074} 0 .010 {side*.076} 0 .010', size='.023')

# Use custom visual hand models; all joints, bodies, colliders and inertials are retained.
for hand, prefix in [('right', 'rh'), ('left', 'lh')]:
    original = HERE.parent / 'mujoco_menagerie/shadow_hand' / f'{hand}_hand.xml'
    handtree = ET.parse(original)
    hr = handtree.getroot()
    hr.find('compiler').set('meshdir', '../mujoco_menagerie/shadow_hand/assets')
    ha = hr.find('asset')
    for mesh in ['forearm_shell','palm_shell','finger_proximal','finger_middle',
                 'finger_tip','thumb_proximal','thumb_middle','thumb_tip']:
        ET.SubElement(ha, 'mesh', name='shell_'+mesh, file=f'../../../model/meshes/{mesh}.obj',
                      scale='1 1 1', smoothnormal='true')
    for name, rgba in [('hand_armor','.91 .94 .96 1'), ('hand_joint','.13 .145 .17 1')]:
        ET.SubElement(ha, 'material', name=name, rgba=rgba, specular='.35', shininess='.4')
    for b in hr.findall('.//worldbody//body'):
        bn = b.get('name', '')
        for g in b.findall('geom'):
            if g.get('class') == 'plastic_visual':
                g.set('rgba','0 0 0 0')
        if bn.endswith('_forearm'):
            add(b, prefix+'_forearm_shell', 'shell_forearm_shell', 'hand_armor')
        elif bn.endswith('_palm'):
            add(b, prefix+'_palm_shell', 'shell_palm_shell', 'hand_armor', pos='.003 0 0')
        elif bn.endswith('_wrist'):
            for g in b.findall('geom'):
                if g.get('class') == 'plastic_visual':
                    g.set('rgba', '.13 .145 .17 1')
        elif 'knuckle' in bn or 'metacarpal' in bn:
            for g in b.findall('geom'):
                if g.get('class') == 'plastic_visual':
                    g.set('rgba','.13 .145 .17 1')
        elif any(bn.endswith(s) for s in ['proximal','middle','distal']):
            thumb = '_th' in bn
            suffix = ('proximal' if bn.endswith('proximal') else
                      'middle' if bn.endswith('middle') else 'tip')
            mesh = ('thumb_' if thumb else 'finger_') + suffix
            # The thumb's distal body is rotated 90 degrees about z; the shell is symmetric.
            add(b, bn+'_shell', 'shell_'+mesh, 'hand_armor')
            add(b, bn+'_joint', material='hand_joint', type='sphere', size='.0075')
    target = HERE / f'{hand}_hand_visual.xml'
    ET.indent(handtree, space='  ')
    handtree.write(target, encoding='utf-8', xml_declaration=True)
    asset.find(f"model[@name='{'rhand' if hand == 'right' else 'lhand'}']").set('file', target.name)

ET.indent(tree, space='  ')
tree.write(HERE / 'avatar_android.xml', encoding='utf-8', xml_declaration=True)
print('Built avatar_android.xml and visual meshes; original model retained.')
