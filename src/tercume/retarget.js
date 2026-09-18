/**
 * Landmark → VRM humanoid retargeting (riyazi hissə, DOM-suz).
 *
 * Koordinatlar "bədən məkanında"dır: avatar +Z-yə (kameraya) baxır, +Y yuxarı,
 * avatarın sağ əli −X tərəfdədir. three-vrm-in normallaşdırılmış skeletində
 * istirahət (T-poza) dönmələri vahiddir, ona görə sümüyün lokal dönməsi =
 * valideynin toplam dönməsi⁻¹ × öz toplam dönməsi.
 *
 *   handTarget   — MediaPipe dünya landmark-ları (metr) → əlin hədəf dönməsi R
 *   solveArm     — ovuc mərkəzi işarə məkanına düşsün deyə iki-sümük IK. Dirsəyin
 *                  fırlanması elə seçilir ki, bilək bükülməsi, önqol burulması və
 *                  dirsək mövqeyi anatomik həddlərdə qalsın
 *   solveFingers — MCP 2 sərbəstlik (bükülmə + yana açılma), PIP/DIP menteşə,
 *                  baş barmaq istiqamət izləmə; hamısı oynaq həddləri ilə
 */
import * as THREE from 'three';

const DEG = Math.PI / 180;
const clamp = THREE.MathUtils.clamp;
const vec = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const over = (x, lo, hi) => (x < lo ? lo - x : x > hi ? x - hi : 0);

export const FINGER_CHAINS = {
  Thumb: { lm: [1, 2, 3, 4], bones: ['ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal'] },
  Index: { lm: [5, 6, 7, 8], bones: ['IndexProximal', 'IndexIntermediate', 'IndexDistal'] },
  Middle: { lm: [9, 10, 11, 12], bones: ['MiddleProximal', 'MiddleIntermediate', 'MiddleDistal'] },
  Ring: { lm: [13, 14, 15, 16], bones: ['RingProximal', 'RingIntermediate', 'RingDistal'] },
  Little: { lm: [17, 18, 19, 20], bones: ['LittleProximal', 'LittleIntermediate', 'LittleDistal'] },
};

/** Oynaq həddləri (radian) — klinik hərəkət diapazonları, bir az daraldılıb. */
export const LIMITS = {
  wristFlex: [-65 * DEG, 90 * DEG],   // mənfi = arxaya (ekstensiya), müsbət = ovuca doğru
  wristRadial: [-35 * DEG, 22 * DEG], // mənfi = çeçələ tərəfə (ulnar)
  forearmTwist: 95 * DEG,             // neytraldan (ovuc bədənə baxır) hər iki tərəfə
  wristTwist: 15 * DEG,               // biləyin özü demək olar burulmur
  mcpFlex: [-25 * DEG, 95 * DEG],
  mcpSpread: 22 * DEG,
  pip: [-5 * DEG, 110 * DEG],
  dip: [-5 * DEG, 85 * DEG],
  thumb: [70 * DEG, 65 * DEG, 85 * DEG], // baş barmaq sümükləri: maks. yayınma
};

/* ───────────────────────── kvaternion köməkçiləri ───────────────────────── */

function basis(a, b) {
  const x = a.clone().normalize();
  const y = b.clone().addScaledVector(x, -b.dot(x)).normalize();
  return new THREE.Matrix4().makeBasis(x, y, vec().crossVectors(x, y));
}

/** (a0, b0) çərçivəsini (a, b) çərçivəsinə aparan dönmə. */
export function frameRotation(a0, b0, a, b) {
  const m = basis(a, b).multiply(basis(a0, b0).transpose());
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

/** Vahid `axis` ətrafında burulma bucağı, (−π, π]. Swing·twist və twist·swing üçün eynidir. */
function twistAngle(q, axis) {
  const p = q.x * axis.x + q.y * axis.y + q.z * axis.z;
  let a = 2 * Math.atan2(p, q.w);
  if (a > Math.PI) a -= 2 * Math.PI;
  else if (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

/** Kvaternion → dönmə vektoru (ox × bucaq), qısa yol. */
export function quatToVec(q, out = vec()) {
  let { x, y, z, w } = q;
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
  const s = Math.hypot(x, y, z);
  if (s < 1e-9) return out.set(2 * x, 2 * y, 2 * z);
  const k = (2 * Math.atan2(s, w)) / s;
  return out.set(x * k, y * k, z * k);
}

const _axis = vec();
/** Dönmə vektoru → kvaternion. */
export function vecToQuat(v, out = new THREE.Quaternion()) {
  const ang = v.length();
  if (ang < 1e-9) return out.set(v.x / 2, v.y / 2, v.z / 2, 1).normalize();
  return out.setFromAxisAngle(_axis.copy(v).divideScalar(ang), ang);
}

const orthTo = (v, axis) => v.clone().addScaledVector(axis, -v.dot(axis)).normalize();

/* ───────────────────────── skelet ölçüləri ───────────────────────── */

/**
 * Bir tərəfin istirahət həndəsəsi. Bütün normallaşdırılmış dönmələr vahid
 * olanda (yükləmədən dərhal sonra) çağırılmalıdır.
 */
export function measureRig(humanoid, side) {
  const pos = (n) => humanoid.getNormalizedBoneNode(side + n).getWorldPosition(vec());
  const S = pos('UpperArm'), E = pos('LowerArm'), W = pos('Hand');
  const u0 = E.clone().sub(S).normalize();
  const f0 = W.clone().sub(E).normalize();
  const idx = pos('IndexProximal'), mid = pos('MiddleProximal'), lit = pos('LittleProximal');
  const d0 = mid.clone().sub(W).normalize();
  const s0 = idx.clone().sub(lit).normalize();
  // VRM T-pozasında ovuclar aşağı baxır (barmaq bükülməsi ilə yoxlanılıb: ovuc tərəfi −Y)
  const n0 = vec(0, -1, 0);

  const fingers = {};
  for (const [name, { bones }] of Object.entries(FINGER_CHAINS)) {
    const p = bones.map(pos);
    const dirs = [p[1].clone().sub(p[0]).normalize(), p[2].clone().sub(p[1]).normalize()];
    dirs.push(dirs[1].clone());   // uc sümüyü yoxdur — distal ortadakının davamıdır
    fingers[name] = { pos: p, dirs };
  }

  return {
    side,
    out: side === 'left' ? 1 : -1,      // çölə baxan x işarəsi
    S, E, W, L1: S.distanceTo(E), L2: E.distanceTo(W),
    u0, f0,
    h0: vec().crossVectors(u0, vec(0, 0, 1)).normalize(),  // dirsək menteşəsi (önə bükülür)
    d0, s0, n0,
    // ovuc normalı = chir · (barmaq istiqaməti × çeçələ→şəhadət); sol/sağ əldə işarə tərsdir
    chir: Math.sign(vec().crossVectors(d0, s0).dot(n0)) || 1,
    fingers,
    palmOff: idx.clone().add(mid).add(lit).add(W).multiplyScalar(0.25).sub(W),
    aFlex: vec().crossVectors(f0, n0).normalize(),              // bu ox ətrafında + = ovuca doğru
    aRad: vec().crossVectors(f0, orthTo(s0, f0)).normalize(),   // + = baş barmaq tərəfə
    head: humanoid.getNormalizedBoneNode('head').getWorldPosition(vec()),
  };
}

/* ───────────────────────── əl hədəfi ───────────────────────── */

function palmFrame(rig, P) {
  const d = P[9].clone().sub(P[0]).normalize();
  const s = P[5].clone().sub(P[17]).normalize();
  const n = vec().crossVectors(d, s).multiplyScalar(rig.chir).normalize();
  const R = frameRotation(rig.d0, rig.n0, d, n);
  return { d, s, n, R, Rinv: R.clone().invert() };
}

/**
 * @param {object} rig measureRig nəticəsi
 * @param {object} src {world: 21×[x,y,z] metr | null, norm: 21×{x,y,z}, size: [W,H], mirror}
 *   MediaPipe oxları: x sağa, y aşağı, z kameradan uzağa → bədən məkanı (x, −y, −z).
 *   mirror: x güzgülənir (əl həndəsəsi avatarın tərəfinə çevrilir).
 *
 * İki mənbə birləşdirilir: görünən oriyentasiya şəkil koordinatlarından
 * (x/y dəqiq proyeksiyadır; en/hündürlük nisbəti düzəldilir), barmaq forması
 * isə metrik dünya koordinatlarından (perspektiv və nisbət təhrifi yoxdur).
 * İkisi arasındakı fərq bəzi hərflərdə 30°-yə çatır (ölçülüb).
 */
export function handTarget(rig, { world, norm, size, mirror = false, flipDepth = false }) {
  const sx = mirror ? -1 : 1;
  const sz = flipDepth ? 1 : -1;
  const a = size ? size[0] / size[1] : 1;
  const P = norm.map((p) => vec(sx * p.x * a, -p.y, sz * p.z * a));
  const view = palmFrame(rig, P);
  const shape = world ? world.map(([x, y, z]) => vec(sx * x, -y, sz * z)) : P;
  const shapeFrame = world ? palmFrame(rig, shape) : view;
  return { ...view, P, shape, shapeRinv: shapeFrame.Rinv };
}

const _hands = new WeakMap();
const CURL_PAIRS = [[5, 8], [9, 12], [13, 16], [17, 20], [6, 8], [10, 12], [14, 16], [18, 20]];

/**
 * Bükülmənin işarəsi (21×[x,y,z]): barmaqlar yalnız ovuca doğru bükülə bilər,
 * ona görə sağ-əl qaydası ilə tapılan ovuc normalına görə müsbət → sağ əl.
 */
function curlEvidence(pts) {
  const V = (i) => vec(...pts[i]);
  const d = V(9).sub(V(0)).normalize();
  const s = V(5).sub(V(17)).normalize();
  const nRight = vec().crossVectors(s, d).normalize();
  const size = V(9).distanceTo(V(0)) || 1;
  let e = (2 * V(4).sub(V(1)).dot(nRight)) / size;
  for (const [m, t] of CURL_PAIRS) e += V(t).sub(V(m)).dot(nRight) / size;
  return e;
}

/**
 * Pozadakı əlin tərəfi və hər kadr üçün dərinlik düzəlişi.
 *
 * MediaPipe dərinliyi (z) bəzən güzgü həlli kimi qiymətləndirir: o kadrda həm
 * etiket, həm 3D həndəsə tərs əli göstərir ("k"-nın 12 kadrından 2-si, "y"-nin
 * 3-ü — ölçülüb). Çəkilişdə əl dəyişmir, ona görə tərəf bütün kadrlar üzrə bir
 * dəfə qərarlaşdırılır; uyğun gəlməyən kadrda z işarəsi çevrilir — 2D proyeksiya
 * eyni qalır, forma düzgün tərəfə keçir. Düz, açıq əldə bükülmə zəifdir —
 * onda MediaPipe etiketlərinin çoxluğu götürülür.
 * @returns {{left: boolean, flip: boolean[]}}
 */
export function poseHands(pose) {
  if (!pose) return { left: false, flip: [] };
  if (_hands.has(pose)) return _hands.get(pose);
  const pts = pose.world ?? pose.frames.map((f) => f.map((p) => [p.x, p.y, p.z]));
  const ev = pts.map(curlEvidence);
  const total = ev.reduce((a, b) => a + b, 0);
  let left;
  if (Math.abs(total) / ev.length > 0.3) {
    left = total < 0;
  } else {
    const labels = pose.handedness ?? [];
    left = labels.filter((l) => l === 'Left').length * 2 > labels.length;
  }
  const res = { left, flip: ev.map((e) => (left ? e > 0 : e < 0)) };
  _hands.set(pose, res);
  return res;
}

/**
 * MediaPipe-in dərinliyi güzgü həll etməsinə dəlil (müsbət — bükülmə əlin tərəfinə
 * uymur). Düz əldə və kameraya baxan barmaqlarda zəifdir, ona görə söz işarələrində
 * yalnız köməkçi xal kimi işlədilir.
 */
export function mirrorEvidence(world, side) {
  const e = curlEvidence(world);
  return side === 'left' ? e : -e;
}

/** Kadr üçün handTarget girişi: avatarın tərəfinə güzgü və dərinlik düzəlişi daxil. */
export function frameSource(pose, frame, side, landmarks = pose?.frames?.[frame]) {
  const hands = poseHands(pose);
  return {
    world: pose?.world?.[frame], norm: landmarks, size: pose?.size?.[frame],
    mirror: hands.left !== (side === 'left'),
    flipDepth: hands.flip[frame] ?? false,
  };
}

/* ───────────────────────── qol ───────────────────────── */

export function evalArm(rig, R, T, swivel, elbow = null) {
  const toT = T.clone().sub(rig.S);
  const dist = toT.length();
  if (dist < 1e-6) return null;
  const t = toT.divideScalar(dist);
  const D = clamp(dist, Math.abs(rig.L1 - rig.L2) + 0.03, (rig.L1 + rig.L2) * 0.995);
  const cosA = clamp((rig.L1 ** 2 + D * D - rig.L2 ** 2) / (2 * rig.L1 * D), -1, 1);
  const sinA = Math.sqrt(1 - cosA * cosA);

  // Dirsəyin təbii istiqaməti: aşağı, bir az çölə və geri; swivel onu t oxu ətrafında fırladır
  const pole = vec(0.35 * rig.out, -1, -0.15);
  pole.addScaledVector(t, -pole.dot(t));
  if (pole.lengthSq() < 1e-6) pole.set(0, 0, -1).addScaledVector(t, t.z);
  pole.normalize();
  const bi = vec().crossVectors(t, pole);
  const p = pole.multiplyScalar(Math.cos(swivel)).addScaledVector(bi, Math.sin(swivel));

  const E = rig.S.clone().addScaledVector(t, rig.L1 * cosA).addScaledVector(p, rig.L1 * sinA);
  const Wr = rig.S.clone().addScaledVector(t, D);
  const u = E.clone().sub(rig.S).normalize();
  const f = Wr.clone().sub(E).normalize();
  const h = vec().crossVectors(u, f);
  if (h.lengthSq() < 1e-8) return null;
  h.normalize();

  // Menteşə oxu hər iki seqmentdə eynidir → dirsək yalnız anatomik tərəfə bükülür
  const Qu = frameRotation(rig.u0, rig.h0, u, h);
  const Ql0 = frameRotation(rig.f0, rig.h0, f, h);
  // Əlin tələb etdiyi burulmanı önqol daşıyır (bilək burulmur)
  const tau = clamp(twistAngle(Ql0.clone().invert().multiply(R), rig.f0),
    -LIMITS.forearmTwist, LIMITS.forearmTwist);
  const Ql = new THREE.Quaternion().setFromAxisAngle(f, tau).multiply(Ql0);
  const rel = Ql.clone().invert().multiply(R);
  const twist = twistAngle(rel, rig.f0);
  const swing = rel.clone().multiply(new THREE.Quaternion().setFromAxisAngle(rig.f0, -twist));
  const w = quatToVec(swing);
  const flex = w.dot(rig.aFlex), radial = w.dot(rig.aRad);

  const rE = E.clone().sub(rig.S);
  // hard: oynaq həddini aşma (fiziki mümkünsüz); qalanı rahatlıq və görünüş üçündür
  const hard = over(flex, ...LIMITS.wristFlex) ** 2 + over(radial, ...LIMITS.wristRadial) ** 2 +
    over(twist, -LIMITS.wristTwist, LIMITS.wristTwist) ** 2;
  // Söz işarəsində dirsəyin yeri siqnalçıdan məlumdur (dərinlik daha az etibarlıdır);
  // hərflərdə isə təbii zona ilə məhdudlaşdırılır
  const elbowCost = elbow
    ? 40 * ((E.x - elbow.x) ** 2 + (E.y - elbow.y) ** 2 + 0.4 * (E.z - elbow.z) ** 2)
    : 40 * over(rE.y, -0.30, -0.08) ** 2 +          // dirsək çiyindən aşağıda, "qanad" kimi qalxmasın
      40 * over(rE.x * rig.out, -0.02, 0.20) ** 2 + // bədənin içinə girməsin, "qanad" açılmasın
      20 * over(rE.z, -0.14, 0.20) ** 2;
  const cost =
    40 * hard +
    0.35 * (flex * flex + 1.5 * radial * radial) + 0.12 * tau * tau +   // neytrala yaxın
    elbowCost +
    30 * (dist - D) ** 2;                          // çatmırsa
  return { cost, hard, Qu, Ql, flex, radial, twist, tau, E, W: Wr, swivel };
}

/**
 * İşarə məkanı — ovuc mərkəzinin hədəfi əlin istiqamətindən asılıdır.
 * Barmaqlar yuxarı: əl çiyin önündə, çənə səviyyəsinə yaxın. Barmaqlar aşağı:
 * əl sinə səviyyəsində və öndə sallanır, dirsək bədənə yaxın qalır, bilək bükülür.
 * Önə (kameraya) baxan əl daha irəlidə olur. Başlanğıc nöqtələr bütün hərflər
 * üzrə geniş axtarışla seçilib; dəqiq yeri solveArm ±6 sm daxilində tapır.
 */
export function signAnchor(rig, handDir) {
  const down = clamp((1 - handDir.y) / 2, 0, 1);   // 0 = yuxarı, 1 = aşağı
  const fwd = clamp(handDir.z, 0, 1);
  const lerp = THREE.MathUtils.lerp;
  return vec(
    rig.S.x + rig.out * lerp(0.045, 0, down),
    rig.S.y + lerp(0.05, -0.18, down),
    rig.S.z + lerp(0.21, 0.32, down) + 0.10 * fwd * (1 - down),
  );
}

/**
 * Ovuc mərkəzi `anchor` ətrafına düşsün, əl R ilə dönsün. Dirsək fırlanması
 * (±90°) və əlin ±6 sm yerdəyişməsi arasında ən təbii həll iki mərhələdə
 * (kobud → dəqiq) axtarılır. `prev` — əvvəlki kadrın həlli: dinamik hərfdə
 * dirsək və önqol burulması kadrdan kadra sıçramasın.
 * Söz işarəsində `elbow` (siqnalçının dirsəyi) verilir, ovucun yeri datadan
 * gəldiyi üçün sürüşmə baha olur (`shiftCost`).
 * @returns {{upper, lower, hand, info}} lokal kvaternionlar
 */
export function solveArm(rig, R, anchor, prev = null, { elbow = null, shiftCost = 15 } = {}) {
  const W0 = anchor.clone().sub(rig.palmOff.clone().applyQuaternion(R));
  let best = null;
  const consider = (shift, sw) => {
    const c = evalArm(rig, R, W0.clone().add(shift), sw, elbow);
    if (!c) return;
    c.cost += shiftCost * shift.lengthSq() + 0.02 * sw * sw;
    if (prev) c.cost += 25 * c.E.distanceToSquared(prev.E) + 0.5 * (c.tau - prev.tau) ** 2;
    c.shift = shift;
    if (!best || c.cost < best.cost) best = c;
  };
  // Şəbəkə tərəfə nisbidir (x və swivel `out` ilə): sol və sağ əl eyni namizədləri eyni
  // sırada görür, bərabər dəyərli həllər arasında seçim də güzgü olur
  const o = rig.out;
  const grid = (step, fn) => {
    for (const dx of [-step, 0, step]) for (const dy of [-step, 0, step]) for (const dz of [-step, 0, step]) fn(vec(dx * o, dy, dz));
  };
  grid(0.04, (s) => { for (let k = -9; k <= 9; k++) consider(s, k * 10 * DEG * o); });
  if (!best) return null;
  const coarse = best;
  grid(0.02, (s) => {
    const shift = s.add(coarse.shift);   // hər dönmə bucağı eyni sürüşmə ilə yoxlanır
    for (let k = -4; k <= 4; k++) consider(shift, coarse.swivel + k * 2.5 * DEG * o);
  });

  // Ən yaxşı həll də həddi aşırsa, aşan hissə kəsilir — oynaq qırılmış görünməsin
  const flex = clamp(best.flex, ...LIMITS.wristFlex);
  const radial = clamp(best.radial, ...LIMITS.wristRadial);
  const twist = clamp(best.twist, -LIMITS.wristTwist, LIMITS.wristTwist);
  const hand = vecToQuat(rig.aFlex.clone().multiplyScalar(flex).addScaledVector(rig.aRad, radial))
    .multiply(new THREE.Quaternion().setFromAxisAngle(rig.f0, twist));
  return {
    upper: best.Qu.clone(),
    lower: best.Qu.clone().invert().multiply(best.Ql),
    hand,
    info: { ...best, flexC: flex, radialC: radial, twistC: twist },
  };
}

/** Qollar aşağı, dirsək azca bükülü, ovuc budlara baxır. */
export function restArm(rig) {
  const u = vec(0.16 * rig.out, -1, 0.02).normalize();
  const f = vec(0.07 * rig.out, -1, 0.30).normalize();
  const h = vec().crossVectors(u, f).normalize();
  const Qu = frameRotation(rig.u0, rig.h0, u, h);
  const Ql = frameRotation(rig.f0, rig.h0, f, h);
  return {
    upper: Qu,
    lower: Qu.clone().invert().multiply(Ql),
    hand: new THREE.Quaternion().setFromAxisAngle(rig.aFlex, 10 * DEG),
  };
}

/** Hazır vəziyyət (sözlər arası): barmaqlar yuxarı, ovuc önə və bir az içəri. */
export function readyHand(rig) {
  const d = vec(0.08 * rig.out, 1, 0.12).normalize();
  const n = vec(-0.45 * rig.out, 0, 0.9).normalize();
  return frameRotation(rig.d0, rig.n0, d, n);
}

/* ───────────────────────── barmaqlar ───────────────────────── */

/**
 * Bir barmaq zənciri. `angles(k, f, n)` k-cı oynağın bucağını qaytarır
 * (k=0: {spread, flex}); f — sümüyün hazırkı istiqaməti, n — ovuc tərəfi.
 */
function buildChain(fr, n0, angles) {
  const out = [];
  let M = new THREE.Quaternion();
  for (let k = 0; k < 3; k++) {
    const rf = fr.dirs[k], rn = orthTo(n0, rf);
    const gf = rf.clone().applyQuaternion(M), gn = rn.clone().applyQuaternion(M);
    let F, N;
    if (k === 0) {
      const { spread, flex } = angles(0, gf, gn);
      const gl = vec().crossVectors(gf, gn);
      const f1 = gf.clone().multiplyScalar(Math.cos(spread)).addScaledVector(gl, Math.sin(spread));
      F = f1.clone().multiplyScalar(Math.cos(flex)).addScaledVector(gn, Math.sin(flex));
      N = gn.clone().multiplyScalar(Math.cos(flex)).addScaledVector(f1, -Math.sin(flex));
    } else {
      const th = angles(k, gf, gn);
      F = gf.clone().multiplyScalar(Math.cos(th)).addScaledVector(gn, Math.sin(th));
      N = gn.clone().multiplyScalar(Math.cos(th)).addScaledVector(gf, -Math.sin(th));
    }
    const Mk = frameRotation(rf, rn, F, N);
    out.push(M.clone().invert().multiply(Mk));
    M = Mk;
  }
  return out;
}

/** Baş barmaq: hər sümük hədəf istiqamətə qısa yolla döner (yayınma həddi ilə). */
function thumbChain(fr, t) {
  const out = [];
  let M = new THREE.Quaternion();
  for (let k = 0; k < 3; k++) {
    const cur = fr.dirs[k].clone().applyQuaternion(M);
    const q = new THREE.Quaternion().setFromUnitVectors(cur, t[k]);
    const ang = 2 * Math.acos(clamp(Math.abs(q.w), 0, 1));
    if (ang > LIMITS.thumb[k]) q.slerp(new THREE.Quaternion(), 1 - LIMITS.thumb[k] / ang);
    const Mk = q.multiply(M);
    out.push(M.clone().invert().multiply(Mk));
    M = Mk;
  }
  return out;
}

/** @returns {Object<string, THREE.Quaternion>} qısa sümük adı → lokal dönmə */
export function solveFingers(rig, P, Rinv) {
  const res = {};
  const seg = (i, j) => P[j].clone().sub(P[i]).applyQuaternion(Rinv).normalize();
  for (const [name, { lm, bones }] of Object.entries(FINGER_CHAINS)) {
    const fr = rig.fingers[name];
    const t = [seg(lm[0], lm[1]), seg(lm[1], lm[2]), seg(lm[2], lm[3])];
    const locals = name === 'Thumb' ? thumbChain(fr, t) : buildChain(fr, rig.n0, (k, f, n) => {
      if (k === 0) {
        const l = vec().crossVectors(f, n);
        const tp = t[0].clone().addScaledVector(n, -t[0].dot(n));
        // çox bükülmüş barmaqda ovuc müstəvisindəki proyeksiya kiçikdir → yana açılma etibarsız
        const spread = clamp(Math.atan2(tp.dot(l), tp.dot(f)) * THREE.MathUtils.smoothstep(tp.length(), 0.2, 0.6),
          -LIMITS.mcpSpread, LIMITS.mcpSpread);
        const f1 = f.clone().multiplyScalar(Math.cos(spread)).addScaledVector(l, Math.sin(spread));
        return { spread, flex: clamp(Math.atan2(t[0].dot(n), t[0].dot(f1)), ...LIMITS.mcpFlex) };
      }
      return clamp(Math.atan2(t[k].dot(n), t[k].dot(f)), ...(k === 1 ? LIMITS.pip : LIMITS.dip));
    });
    bones.forEach((b, i) => { res[b] = locals[i]; });
  }
  return res;
}

const RELAX = { Index: [8, 14, 8], Middle: [10, 18, 10], Ring: [13, 22, 12], Little: [16, 26, 14] };

/** Rahat, azca bükülmüş əl (tam düz barmaq təbii görünmür). */
export function relaxedFingers(rig) {
  const res = {};
  for (const [name, { bones }] of Object.entries(FINGER_CHAINS)) {
    const fr = rig.fingers[name];
    const locals = name === 'Thumb'
      ? thumbChain(fr, fr.dirs.map((d, k) => d.clone().addScaledVector(rig.n0, 0.12 + 0.08 * k).normalize()))
      : buildChain(fr, rig.n0, (k) => (k === 0
        ? { spread: 0, flex: RELAX[name][0] * DEG }
        : RELAX[name][k] * DEG));
    bones.forEach((b, i) => { res[b] = locals[i]; });
  }
  return res;
}

/* ───────────────────────── dinamik hərflər ───────────────────────── */

const _traj = new WeakMap();
const MAX_TRAVEL = 0.12;

/**
 * Dinamik hərfdə biləyin kadr boyu yerdəyişməsi (metr, bədən məkanı, datanın
 * öz tərəfində). Piksel → metr miqyası əlin dünya uzunluğundan alınır.
 */
export function trajectory(pose) {
  if (!pose || pose.frames.length < 2 || !pose.world || !pose.size) return null;
  if (_traj.has(pose)) return _traj.get(pose);
  const px = pose.frames.map((f, i) => {
    const [W, H] = pose.size[i];
    return { x: f[0].x * W, y: f[0].y * H,
             len: Math.hypot((f[9].x - f[0].x) * W, (f[9].y - f[0].y) * H) };
  });
  const scales = pose.world.map((w, i) =>
    Math.hypot(w[9][0] - w[0][0], w[9][1] - w[0][1], w[9][2] - w[0][2]) / Math.max(px[i].len, 1));
  scales.sort((a, b) => a - b);
  const scale = scales[scales.length >> 1];   // median: qısalmış kadrlar miqyası şişirtməsin
  const cx = px.reduce((s, p) => s + p.x, 0) / px.length;
  const cy = px.reduce((s, p) => s + p.y, 0) / px.length;
  const out = px.map((p) => {
    const v = vec((p.x - cx) * scale, -(p.y - cy) * scale, 0);
    return v.length() > MAX_TRAVEL ? v.setLength(MAX_TRAVEL) : v;
  });
  _traj.set(pose, out);
  return out;
}
