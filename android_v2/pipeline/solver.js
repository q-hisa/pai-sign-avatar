// MediaPipe-derived motion (VRM coordinates) -> VRM humanoid bone rotations + expressions, rendered offscreen.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const I = new THREE.Quaternion();
let vrm, motion, rest = {}, renderer, renderer2, scene, cam, cam2, fps = 30;
let lastHand = { left: null, right: null }, meanShoulder = null, headRef = null;
let handQ = { left: null, right: null };
let faceOnly = false;

const medCache = {};
function med(n) {
  if (!(n in medCache)) { const a = motion.blend[n].slice().sort((x, y) => x - y); medCache[n] = a[a.length >> 1]; }
  return medCache[n];
}
function node(name) { return vrm.humanoid.getNormalizedBoneNode(name); }
function setLocal(name, q) { const n = node(name); if (n) n.quaternion.copy(q); }
function swing(restDirWorld, dir) { return new THREE.Quaternion().setFromUnitVectors(restDirWorld.clone().normalize(), dir.clone().normalize()); }
function restDir(a, b) { return rest[b].clone().sub(rest[a]).normalize(); }
function basisQ(d, a, n) { const m = new THREE.Matrix4().makeBasis(d, a, n); return new THREE.Quaternion().setFromRotationMatrix(m); }
function frameQ(w, mid, idx, lit) {
  const d = mid.clone().sub(w).normalize();
  let a = lit.clone().sub(idx); a.sub(d.clone().multiplyScalar(a.dot(d))).normalize();
  return basisQ(d, a, new THREE.Vector3().crossVectors(d, a));
}
// drive a child bone chain: each bone follows the direction of consecutive landmarks, inheriting the parent's roll
function chain(bones, lms, parentQ, restDirs) {
  let pq = parentQ.clone();
  bones.forEach((b, i) => {
    const dir = lms[i + 1].clone().sub(lms[i]);
    const rd = restDirs[i].clone().applyQuaternion(pq);
    const Qw = swing(rd, dir).multiply(pq);
    setLocal(b, pq.clone().invert().multiply(Qw));
    pq = Qw;
  });
}

function solveArm(side, P, hand, chestQ) {
  const L = side === 'left';
  const iSh = L ? 11 : 12, iEl = L ? 13 : 14, iWr = L ? 15 : 16;
  const up = side + 'UpperArm', low = side + 'LowerArm', hnd = side + 'Hand';
  const Qu = swing(restDir(up, low).applyQuaternion(chestQ), P[iEl].clone().sub(P[iSh])).multiply(chestQ);
  setLocal(up, chestQ.clone().invert().multiply(Qu));
  const restL = restDir(low, hnd);
  let Ql = swing(restL.clone().applyQuaternion(Qu), P[iWr].clone().sub(P[iEl])).multiply(Qu);
  let Qh;
  if (hand) {
    const lm = hand.map(V);
    const cap = (s) => side.charAt(0).toUpperCase() + side.slice(1) + s;
    const Qr = frameQ(rest[hnd], rest[side + 'MiddleProximal'], rest[side + 'IndexProximal'], rest[side + 'LittleProximal']);
    const Qt = frameQ(lm[0], lm[9], lm[5], lm[17]);
    Qh = Qt.multiply(Qr.invert());
    handQ[side] = Qh.clone(); lastHand[side] = hand;
  } else if (handQ[side]) {
    Qh = handQ[side].clone();
  } else {
    Qh = Ql.clone();
  }
  // share the forearm roll between lowerArm and hand
  const rel = Ql.clone().invert().multiply(Qh);
  const ax = restL;
  const pr = rel.x * ax.x + rel.y * ax.y + rel.z * ax.z;
  const tw = new THREE.Quaternion(ax.x * pr, ax.y * pr, ax.z * pr, rel.w).normalize();
  const half = new THREE.Quaternion().slerp(tw, 0.5);
  const Ql2 = Ql.clone().multiply(half);
  setLocal(low, Qu.clone().invert().multiply(Ql2));
  setLocal(hnd, Ql2.clone().invert().multiply(Qh));
  // fingers
  const h = hand || lastHand[side];
  if (h) {
    const lm = h.map(V);
    const S = side;
    const F = { Index: [5, 6, 7, 8], Middle: [9, 10, 11, 12], Ring: [13, 14, 15, 16], Little: [17, 18, 19, 20] };
    for (const [nm, ix] of Object.entries(F)) {
      const b = [S + nm + 'Proximal', S + nm + 'Intermediate', S + nm + 'Distal'];
      const r1 = restDir(b[0], b[1]), r2 = restDir(b[1], b[2]);
      chain(b, ix.map((k) => lm[k]), Qh, [r1, r2, r2]);
    }
    const tb = [S + 'ThumbMetacarpal', S + 'ThumbProximal', S + 'ThumbDistal'];
    const t1 = restDir(tb[0], tb[1]), t2 = restDir(tb[1], tb[2]);
    chain(tb, [1, 2, 3, 4].map((k) => lm[k]), Qh, [t1, t2, t2]);
  }
  return Qh;
}

function solveFrame(i) {
  const P = motion.pose[i].map(V);
  if (faceOnly) return solveFace(i);
  // chest: follow the shoulder line (relative to the mean line), damped
  const sh = P[11].clone().sub(P[12]).normalize();
  const dq = new THREE.Quaternion().setFromUnitVectors(meanShoulder, sh);
  const chestQ = new THREE.Quaternion().slerp(dq, 0.6);
  const chestName = node('upperChest') ? 'upperChest' : 'chest';
  setLocal(chestName, chestQ);
  solveArm('left', P, motion.handL[i], chestQ);
  solveArm('right', P, motion.handR[i], chestQ);
  // head from ears + nose
  const x = P[7].clone().sub(P[8]).normalize();
  const mid = P[7].clone().add(P[8]).multiplyScalar(0.5);
  let f = P[0].clone().sub(mid); f.sub(x.clone().multiplyScalar(f.dot(x))); const z = f.normalize();
  const y = new THREE.Vector3().crossVectors(z, x);
  const Rt = basisQ(x, y, z);
  const delta = Rt.multiply(headRef.clone().invert());
  const neckW = new THREE.Quaternion().slerp(delta, 0.45);
  setLocal('neck', chestQ.clone().invert().multiply(neckW));
  setLocal('head', neckW.clone().invert().multiply(delta));
  solveFace(i);
}

function solveFace(i) {
  const g = (n) => motion.blend[n][i];
  const gr = (n) => motion.blend[n][i] - med(n);
  const em = vrm.expressionManager;
  // 瞬きは左右の平均で両目同時に（左右の検出差でウィンクのように見えるのを防ぐ）
  const blink = clamp(((g('eyeBlinkLeft') + g('eyeBlinkRight')) / 2 - 0.3) * 1.8);
  em.setValue('blinkLeft', blink);
  em.setValue('blinkRight', blink);
  em.setValue('aa', clamp(g('jawOpen') * 1.1));
  em.setValue('oh', clamp(g('mouthFunnel') * 1.2));
  em.setValue('ou', clamp(g('mouthPucker') * 1.3));
  em.setValue('ee', clamp((g('mouthStretchLeft') + g('mouthStretchRight')) * 0.7));
  em.setValue('happy', clamp((g('mouthSmileLeft') + g('mouthSmileRight')) * 0.6));
  em.setValue('sad', clamp((g('mouthFrownLeft') + g('mouthFrownRight')) * 0.6 + g('browInnerUp') * 0.3));
  em.setValue('angry', clamp((g('browDownLeft') + g('browDownRight')) * 0.5));
  em.setValue('surprised', clamp(g('browInnerUp') * 0.5 + (g('eyeWideLeft') + g('eyeWideRight')) * 0.4));
  // gaze: 左右の目の値は食い違うことがあり（寄り目になる）、そのまま使うと黒目が中央からずれる。
  // 両目の平均を使い、小さな動きは無視（正面なら黒目は中央）、横を見る量には上限をつける。
  const eyeL = node('leftEye'), eyeR = node('rightEye');
  const dz = (v, t) => Math.sign(v) * Math.max(0, Math.abs(v) - t);
  const dirX = 0.5 * ((gr('eyeLookOutLeft') - gr('eyeLookInLeft')) - (gr('eyeLookOutRight') - gr('eyeLookInRight')));
  const dirY = 0.25 * ((gr('eyeLookUpLeft') + gr('eyeLookUpRight')) - (gr('eyeLookDownLeft') + gr('eyeLookDownRight')));
  const yaw = clamp(0.35 * dz(dirX, 0.12), -0.14, 0.14);
  const pit = clamp(-0.5 * dz(dirY, 0.08), -0.08, 0.08);
  if (eyeL) eyeL.quaternion.setFromEuler(new THREE.Euler(pit, yaw, 0));
  if (eyeR) eyeR.quaternion.setFromEuler(new THREE.Euler(pit, yaw, 0));
}

window.loadAll = async (modelUrl, motionUrl, opt = {}) => {
  motion = await (await fetch(motionUrl)).json();
  fps = motion.fps;
  faceOnly = !!opt.faceOnly;
  const W = opt.size || 720, FW = opt.faceSize || 256;
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, W); renderer.setPixelRatio(1);
  renderer2 = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer2.setSize(FW, FW); renderer2.setPixelRatio(1);
  document.body.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(opt.bg || 0x87d4f0);
  const dl = new THREE.DirectionalLight(0xffffff, Math.PI); dl.position.set(0.6, 1.2, 1.6); scene.add(dl);
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const loader = new GLTFLoader();
  loader.register((p) => new VRMLoaderPlugin(p));
  const gltf = await loader.loadAsync(modelUrl);
  vrm = gltf.userData.vrm;
  VRMUtils.rotateVRM0(vrm);
  scene.add(vrm.scene);
  vrm.scene.updateMatrixWorld(true);
  for (const name of Object.keys(vrm.humanoid.humanBones)) {
    const n = node(name); if (n) rest[name] = n.getWorldPosition(new THREE.Vector3());
  }
  // calibration: mean shoulder line and mean head frame
  const sh = new THREE.Vector3(); let hq = null;
  const accum = new THREE.Vector4();
  for (let i = 0; i < motion.T; i++) {
    const P = motion.pose[i].map(V);
    sh.add(P[11].clone().sub(P[12]).normalize());
    const x = P[7].clone().sub(P[8]).normalize();
    const mid = P[7].clone().add(P[8]).multiplyScalar(0.5);
    let f = P[0].clone().sub(mid); f.sub(x.clone().multiplyScalar(f.dot(x))); const z = f.normalize();
    const q = basisQ(x, new THREE.Vector3().crossVectors(z, x), z);
    if (!hq) hq = q.clone();
    const s = q.dot(hq) < 0 ? -1 : 1;
    accum.add(new THREE.Vector4(q.x * s, q.y * s, q.z * s, q.w * s));
  }
  meanShoulder = sh.normalize();
  headRef = new THREE.Quaternion(accum.x, accum.y, accum.z, accum.w).normalize();
  // cameras
  const headY = rest.head.y;
  const fov = 28;
  const dist = (vis) => vis / (2 * Math.tan(THREE.MathUtils.degToRad(fov / 2)));
  const vis = opt.visible || 1.15;
  cam = new THREE.PerspectiveCamera(fov, 1, 0.05, 20);
  cam.position.set(0, headY - 0.3, dist(vis)); cam.lookAt(0, headY - 0.3, 0);
  cam2 = new THREE.PerspectiveCamera(fov, 1, 0.05, 20);
  const fv = opt.faceVisible || 0.42, fy = headY + (opt.faceY ?? 0.08);
  cam2.position.set(0, fy, dist(fv)); cam2.lookAt(0, fy, 0);
  for (let k = 0; k < 40; k++) vrm.update(1 / fps);
  return { T: motion.T, fps, headY };
};

window.renderFrame = (i) => {
  solveFrame(i);
  vrm.update(1 / fps);
  renderer.render(scene, cam);
  renderer2.render(scene, cam2);
  return [renderer.domElement.toDataURL('image/jpeg', 0.93), renderer2.domElement.toDataURL('image/jpeg', 0.93)];
};
window.__ready = true;


// ---- overlay mode: character head (face + hair) placed on a head pose coming from the robot simulation
let hp = null, overlayCam = null, overlayRen = null, headScale = 1.0, neckOffset = -0.015;
window.loadOverlay = async (modelUrl, motionUrl, poseUrl, opt = {}) => {
  motion = await (await fetch(motionUrl)).json();
  hp = await (await fetch(poseUrl)).json();
  if (motion.T < hp.T || Math.abs(motion.fps - hp.fps) > 0.01) {
    throw new Error('Head poses and facial motion must have matching FPS and enough frames.');
  }
  fps = motion.fps; headScale = opt.scale || 1.0;
  neckOffset = opt.neckOffset ?? -0.015;
  const S = hp.cam.size;
  overlayRen = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  overlayRen.setSize(S, S); overlayRen.setPixelRatio(1); overlayRen.setClearColor(0x000000, 0);
  document.body.appendChild(overlayRen.domElement);
  scene = new THREE.Scene();
  const c = hp.cam;
  overlayCam = new THREE.PerspectiveCamera(c.fovy, 1, 0.05, 20);
  overlayCam.position.fromArray(c.pos);
  overlayCam.up.fromArray(c.up);
  overlayCam.lookAt(new THREE.Vector3().fromArray(c.pos).add(new THREE.Vector3().fromArray(c.forward)));
  const dl = new THREE.DirectionalLight(0xffffff, Math.PI * (opt.light || 1.0)); dl.position.set(0.7, 1.4, 2.2); scene.add(dl);
  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const loader = new GLTFLoader();
  loader.register((p) => new VRMLoaderPlugin(p));
  const gltf = await loader.loadAsync(modelUrl);
  vrm = gltf.userData.vrm;
  VRMUtils.rotateVRM0(vrm);
  scene.add(vrm.scene);
  vrm.scene.traverse((o) => {
    if (!o.isMesh) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    mats.forEach((mt) => { if (/Bottoms|Shoes|Tops|Body_00_SKIN/.test(mt.name)) mt.visible = false; });
  });
  vrm.scene.updateMatrixWorld(true);
  const hb = vrm.humanoid.getNormalizedBoneNode('head');
  rest.head = hb.getWorldPosition(new THREE.Vector3());
  rest.neck = vrm.humanoid.getNormalizedBoneNode('neck').getWorldPosition(new THREE.Vector3());
  trimMeshes(opt.hairCut !== undefined ? rest.head.y + opt.hairCut : -1e9, rest.neck.y - 0.03);
  // Replace the cropped body-skin strip with a clean neck inside the robot collar.
  const neckSkin = new THREE.Mesh(new THREE.CylinderGeometry(0.021, 0.026, 0.060, 32),
    new THREE.MeshStandardMaterial({color: 0xffdfcd, roughness: 0.85}));
  neckSkin.position.set(rest.neck.x, rest.neck.y + 0.030, rest.neck.z);
  vrm.scene.add(neckSkin);
  faceOnly = true;
  vrm.scene.matrixAutoUpdate = false;
  for (let k = 0; k < 40; k++) { placeHead(0); vrm.update(1 / fps); }
  return { T: hp.T, fps: hp.fps };
};
function placeHead(i) {
  const h = hp.head[i];
  const R = new THREE.Matrix4().set(h.R[0], h.R[1], h.R[2], 0, h.R[3], h.R[4], h.R[5], 0, h.R[6], h.R[7], h.R[8], 0, 0, 0, 0, 1);
  const p = new THREE.Vector3().fromArray(h.p);
  const sc = headScale;
  // root = T(p) * R * S * T(-restHead)
  const m = new THREE.Matrix4().makeTranslation(p.x, p.y, p.z)
    .multiply(R).multiply(new THREE.Matrix4().makeScale(sc, sc, sc))
    .multiply(new THREE.Matrix4().makeTranslation(-rest.neck.x, -rest.neck.y + neckOffset / sc, -rest.neck.z));
  vrm.scene.matrix.copy(m); vrm.scene.matrixWorldNeedsUpdate = true; vrm.scene.updateMatrixWorld(true);
}
window.renderOverlay = (i) => {
  placeHead(i);
  solveFace(i);
  vrm.update(1 / fps);
  overlayRen.render(scene, overlayCam);
  return overlayRen.domElement.toDataURL('image/png');
};


// trim meshes in rest pose: hair shortened below hairCutY; body skin kept only as the neck (above neckY, near the midline)
function trimMeshes(hairCutY, neckY) {
  vrm.scene.traverse((o) => {
    if (!o.isMesh || !o.geometry || !o.geometry.index) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    const kind = mats.map((mt) => (/HAIR/.test(mt.name) ? 'hair' : /Body_00_SKIN/.test(mt.name) ? 'neck' : 'other'));
    if (kind.every((k) => k === 'other')) return;
    const g = o.geometry, idx = g.index.array, pos = g.attributes.position;
    const groups = g.groups.length ? g.groups : [{ start: 0, count: idx.length, materialIndex: 0 }];
    const out = [], ng = [];
    for (const gr of groups) {
      const start = out.length, k = kind[gr.materialIndex];
      for (let t = gr.start; t < gr.start + gr.count; t += 3) {
        const a = idx[t], b = idx[t + 1], c = idx[t + 2];
        const cy = (pos.getY(a) + pos.getY(b) + pos.getY(c)) / 3;
        const cx = (pos.getX(a) + pos.getX(b) + pos.getX(c)) / 3;
        const keep = k === 'hair' ? true : k === 'neck' ? (cy >= neckY && Math.abs(cx) < 0.034) : true;
        if (keep) out.push(a, b, c);
      }
      ng.push({ start, count: out.length - start, materialIndex: gr.materialIndex });
    }
    const Arr = idx.constructor;
    g.setIndex(new THREE.BufferAttribute(new Arr(out), 1));
    g.clearGroups(); ng.forEach((x) => g.addGroup(x.start, x.count, x.materialIndex));
    if (kind.includes('hair') && Number.isFinite(hairCutY)) {
      o.geometry = clipHairGeometry(g, kind, hairCutY);
    }
  });
}

// Clip triangles at the bob hem rather than deleting them by their centre.
// All morph targets and skinning weights follow the new edge vertices.
function clipHairGeometry(g, kinds, cutY) {
  const attrs = Object.keys(g.attributes);
  const morphs = Object.entries(g.morphAttributes).flatMap(([name, values]) =>
    values.map((attribute, i) => ({name, i, attribute, key: `morph:${name}:${i}`})));
  const read = (i) => {
    const vertex = {};
    for (const name of attrs) {
      const a = g.attributes[name];
      vertex[name] = Array.from(a.array.slice(i*a.itemSize, (i+1)*a.itemSize));
    }
    for (const m of morphs) {
      vertex[m.key] = Array.from(m.attribute.array.slice(i*m.attribute.itemSize, (i+1)*m.attribute.itemSize));
    }
    return vertex;
  };
  const interpolate = (a, b, t) => {
    const v = {};
    for (const key of Object.keys(a)) v[key] = a[key].map((x, i) => x*(1-t)+b[key][i]*t);
    if (a.skinIndex && a.skinWeight) {
      const weights = new Map();
      for (const [vertex, factor] of [[a,1-t],[b,t]]) {
        vertex.skinIndex.forEach((id,i) => weights.set(id,(weights.get(id)||0)+vertex.skinWeight[i]*factor));
      }
      const top = [...weights.entries()].sort((x,y) => y[1]-x[1]).slice(0,4);
      const sum = top.reduce((s,x) => s+x[1],0) || 1;
      v.skinIndex = Array.from({length:4},(_,i) => top[i]?.[0] || 0);
      v.skinWeight = Array.from({length:4},(_,i) => (top[i]?.[1] || 0)/sum);
    }
    if (v.normal) {
      const length = Math.hypot(...v.normal) || 1;
      v.normal = v.normal.map(x => x/length);
    }
    v.position[1] = cutY;
    return v;
  };
  const output = Object.fromEntries([...attrs,...morphs.map(m => m.key)].map(name => [name,[]]));
  const result = new THREE.BufferGeometry();
  let count = 0;
  for (const group of g.groups) {
    const start = count;
    for (let i=group.start; i<group.start+group.count; i+=3) {
      let polygon = [0,1,2].map(k => read(g.index.array[i+k]));
      if (kinds[group.materialIndex] === 'hair') {
        const clipped = [];
        for (let j=0; j<polygon.length; j++) {
          const a = polygon[j], b = polygon[(j+1)%polygon.length];
          const ia = a.position[1] >= cutY, ib = b.position[1] >= cutY;
          if (ia) clipped.push(a);
          if (ia !== ib) clipped.push(interpolate(a,b,(cutY-a.position[1])/(b.position[1]-a.position[1])));
        }
        polygon = clipped;
      }
      for (let k=1; k<polygon.length-1; k++) {
        for (const vertex of [polygon[0],polygon[k],polygon[k+1]]) {
          for (const name of Object.keys(output)) output[name].push(...vertex[name]);
          count++;
        }
      }
    }
    result.addGroup(start,count-start,group.materialIndex);
  }
  for (const name of attrs) {
    const a = g.attributes[name];
    result.setAttribute(name,new THREE.BufferAttribute(new a.array.constructor(output[name]),a.itemSize,a.normalized));
  }
  for (const m of morphs) {
    result.morphAttributes[m.name] ||= [];
    result.morphAttributes[m.name][m.i] = new THREE.BufferAttribute(new m.attribute.array.constructor(output[m.key]),m.attribute.itemSize,m.attribute.normalized);
  }
  result.morphTargetsRelative = g.morphTargetsRelative;
  result.computeBoundingBox(); result.computeBoundingSphere();
  return result;
}
