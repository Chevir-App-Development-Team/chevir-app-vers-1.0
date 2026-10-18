/**
 * Söz işarələrində iki əlin bir-birinə girməsinin ölçülməsi (VRM avatarı + cyber skeleton).
 *
 * Brauzerdə, /tercume/ səhifəsində avatar yüklənəndən sonra işləyir:
 *   const m = await import('/tools/measure_hands.js');
 *   const rep = await m.measureWords(window.__chevir);
 * tools/measure_hands.mjs bunu puppeteer ilə komanda sətrindən işə salır.
 *
 * Əl modeli avatarın mesh-indən ölçülür (T-pozada), toqquşma həllindəki sabitlərdən asılı
 * deyil: hər barmaq sümüyü kapsuldur, radiusu həmin sümüyə bağlı vertex-lərin oxdan median
 * məsafəsidir. Barmağın son sümüyü (distal → uc) modelin uc sümüyünə qədər gedir. Ovuc —
 * bilək → barmaq kökləri yelpiyi + barmaq kökləri xətti, radiusu ovucun yarım qalınlığı.
 * Nöqtələr MediaPipe-in 21 landmark-ı ilə eyni sıradadır, ona görə siqnalçının datası da
 * eyni kapsullarla ölçülür.
 *
 * Kadr üzrə ölçülər (metr):
 *   gap      — iki əlin (barmaq + ovuc) səthləri arasındakı ən kiçik məsafə; mənfi = girmə
 *   axis     — eyni, radiussuz: sümük oxları arasındakı ən kiçik məsafə
 *   arm      — əl ilə qarşı tərəfin saidi arasında səth məsafəsi
 *   signer   — siqnalçının datasında `gap` (avatarın əl ölçüsünə miqyaslanıb, eyni radiuslar)
 *   skel     — cyber skeleton-un göstərdiyi nöqtələrdə ox məsafəsi (siqnalçının metri);
 *              skelRaw — t2s.js əlləri ayırmazdan əvvəlki data
 * Oynatmada (yaylarla, 60 Hz, istirahətdən başlayaraq) hər yeniləmədə `gap` və ovucun bir
 * yeniləmədəki yerdəyişməsi (`step`, qoruyucu əlləri sıçratmasın). `guard: false` — vrm.js-in
 * oynatma qoruyucusu söndürülür, yalnız həllin özü ölçülür.
 * Mövcud keyfiyyət ölçüləri də yığılır: ovucun hədəfə məsafəsi, bədənə girmə, kadrdan çıxma,
 * bilək həddinin aşılması.
 */
import * as THREE from 'three';
import * as T2S from '../src/tercume/t2s.js';

const SIDES = ['left', 'right'];
const FINGERS = {
  Thumb: ['Metacarpal', 'Proximal', 'Distal'],
  Index: ['Proximal', 'Intermediate', 'Distal'],
  Middle: ['Proximal', 'Intermediate', 'Distal'],
  Ring: ['Proximal', 'Intermediate', 'Distal'],
  Little: ['Proximal', 'Intermediate', 'Distal'],
};
const NAMES = Object.keys(FINGERS);
// Sümük parçaları landmark indeksləri ilə: [a, b, hissə]
const SEGS = [];
NAMES.forEach((n, i) => {
  const b = 1 + 4 * i;
  SEGS.push([b, b + 1, `${n}0`], [b + 1, b + 2, `${n}1`], [b + 2, b + 3, `${n}2`]);
});
SEGS.push([0, 5, 'ovuc'], [0, 9, 'ovuc'], [0, 13, 'ovuc'], [0, 17, 'ovuc'], [5, 17, 'ovuc'], [0, 1, 'Thumb0']);

const q = (arr, t) => {
  const s = [...arr].sort((a, b) => a - b);
  return s.length ? s[Math.min(s.length - 1, Math.floor(t * s.length))] : NaN;
};

/** İki parça arasındakı ən kiçik məsafə (allokasiyasız, massivlər [x,y,z]). */
function segDist(p1, p2, p3, p4) {
  const ux = p2[0] - p1[0], uy = p2[1] - p1[1], uz = p2[2] - p1[2];
  const vx = p4[0] - p3[0], vy = p4[1] - p3[1], vz = p4[2] - p3[2];
  const wx = p1[0] - p3[0], wy = p1[1] - p3[1], wz = p1[2] - p3[2];
  const a = ux * ux + uy * uy + uz * uz, b = ux * vx + uy * vy + uz * vz, c = vx * vx + vy * vy + vz * vz;
  const d = ux * wx + uy * wy + uz * wz, e = vx * wx + vy * wy + vz * wz;
  const den = a * c - b * b;
  let s = 0, t = 0;
  if (den > 1e-12) s = Math.min(1, Math.max(0, (b * e - c * d) / den));
  t = c > 1e-12 ? (b * s + e) / c : 0;
  if (t < 0) { t = 0; s = a > 1e-12 ? Math.min(1, Math.max(0, -d / a)) : 0; }
  else if (t > 1) { t = 1; s = a > 1e-12 ? Math.min(1, Math.max(0, (b - d) / a)) : 0; }
  const dx = wx + s * ux - t * vx, dy = wy + s * uy - t * vy, dz = wz + s * uz - t * vz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** İki əlin 21 nöqtəsi arasında ən kiçik səth (gap) və ox (axis) məsafəsi. */
export function handGap(A, B, radius) {
  let gap = Infinity, axis = Infinity, pair = null;
  for (const [a0, a1, na] of SEGS) {
    for (const [b0, b1, nb] of SEGS) {
      const d = segDist(A[a0], A[a1], B[b0], B[b1]);
      if (d < axis) axis = d;
      const g = d - radius[na] - radius[nb];
      if (g < gap) { gap = g; pair = `${na}/${nb}`; }
    }
  }
  return { gap, axis, pair };
}

/**
 * Avatarın əl modeli T-pozada: barmaq ucunun distal sümükdə lokal yeri və hissələrin
 * radiusları (iki əlin ortası). Normallaşdırılmış dönmələr müvəqqəti vahid edilir.
 */
export function handModel(avatar) {
  const { vrm } = avatar;
  const h = vrm.humanoid;
  const saved = [];
  for (const n of Object.keys(h.humanBones)) {
    const node = h.getNormalizedBoneNode(n);
    if (node) { saved.push([node, node.quaternion.clone()]); node.quaternion.identity(); }
  }
  h.update();
  vrm.scene.updateMatrixWorld(true);
  const W = (o) => o.getWorldPosition(new THREE.Vector3());

  // Hər xam sümüyə üstün çəki ilə bağlı vertex-lər (T-pozada dünya koordinatı)
  const verts = new Map();
  vrm.scene.traverse((m) => {
    if (!m.isSkinnedMesh) return;
    const pos = m.geometry.attributes.position, si = m.geometry.attributes.skinIndex, sw = m.geometry.attributes.skinWeight;
    if (!pos || !si || !sw) return;
    for (let i = 0; i < pos.count; i++) {
      let bi = -1, best = 0;
      for (let k = 0; k < 4; k++) {
        const w = sw.getComponent(i, k);
        if (w > best) { best = w; bi = si.getComponent(i, k); }
      }
      const bone = m.skeleton.bones[bi];
      if (best < 0.5 || !bone) continue;
      const p = new THREE.Vector3().fromBufferAttribute(pos, i);
      m.applyBoneTransform(i, p);
      p.applyMatrix4(m.matrixWorld);
      if (!verts.has(bone)) verts.set(bone, []);
      verts.get(bone).push(p);
    }
  });
  const axisDist = (a, b, p) => {
    const ab = b.clone().sub(a), t = Math.min(1, Math.max(0, p.clone().sub(a).dot(ab) / ab.lengthSq()));
    return p.distanceTo(a.clone().addScaledVector(ab, t));
  };
  const radiusOf = (rawBone, a, b) => q((verts.get(rawBone) ?? []).map((p) => axisDist(a, b, p)), 0.5);

  const tips = {}, acc = {};
  const add = (k, v) => { if (Number.isFinite(v)) (acc[k] ??= []).push(v); };
  for (const side of SIDES) {
    for (const [name, bones] of Object.entries(FINGERS)) {
      const raw = bones.map((b) => h.getRawBoneNode(side + name + b));
      const norm = bones.map((b) => h.getNormalizedBoneNode(side + name + b));
      const P = raw.map(W);
      // Uc: modelin uc sümüyü; yoxdursa distal ortadakının davamıdır (vertex-lərin ən uzaq proyeksiyası)
      let tip = raw[2].children.find((c) => c.isBone) ? W(raw[2].children.find((c) => c.isBone)) : null;
      if (!tip) {
        const dir = P[2].clone().sub(P[1]).normalize();
        const far = Math.max(0.012, ...(verts.get(raw[2]) ?? []).map((p) => p.clone().sub(P[2]).dot(dir)));
        tip = P[2].clone().addScaledVector(dir, far - 0.004);
      }
      tips[side + name] = norm[2].worldToLocal(tip.clone());
      const chain = [...P, tip];
      for (let k = 0; k < 3; k++) add(`${name}${k}`, radiusOf(raw[k], chain[k], chain[k + 1]));
    }
    // Ovucun yarım qalınlığı: ovucun ortasındakı zolaqda vertex-lərin ovuc müstəvisinə məsafəsi
    const hand = h.getRawBoneNode(side + 'Hand');
    const w0 = W(hand), mid = W(h.getRawBoneNode(side + 'MiddleProximal'));
    const idx = W(h.getRawBoneNode(side + 'IndexProximal')), lit = W(h.getRawBoneNode(side + 'LittleProximal'));
    const d = mid.clone().sub(w0), len = d.length();
    d.normalize();
    const n = d.clone().cross(idx.clone().sub(lit)).normalize();
    const band = (verts.get(hand) ?? []).filter((p) => {
      const t = p.clone().sub(w0).dot(d) / len;
      return t > 0.45 && t < 0.9;
    });
    add('ovuc', q(band.map((p) => Math.abs(p.clone().sub(w0).dot(n))), 0.5));
    const lower = h.getRawBoneNode(side + 'LowerArm');
    add('said', radiusOf(lower, W(lower), w0));
  }
  for (const [node, quat] of saved) node.quaternion.copy(quat);
  h.update();
  vrm.scene.updateMatrixWorld(true);

  const radius = {};
  for (const [k, v] of Object.entries(acc)) radius[k] = v.reduce((a, b) => a + b, 0) / v.length;
  return { tips, radius };
}

/** Avatarın bir əlinin 21 nöqtəsi (dünya koordinatı, [x,y,z]) — skelet yenilənmiş olmalıdır. */
function avatarHand(avatar, model, side) {
  const at = (n) => avatar.bone(side + n).getWorldPosition(new THREE.Vector3());
  const out = [at('Hand').toArray()];
  for (const [name, bones] of Object.entries(FINGERS)) {
    for (const b of bones) out.push(at(name + b).toArray());
    out.push(avatar.bone(side + name + bones[2]).localToWorld(model.tips[side + name].clone()).toArray());
  }
  return out;
}

/** Siqnalçının əli pozanın koordinatlarında: bilək + əl landmark-larının biləyə nisbəti. */
function signerHand(fr, key, wrist, flip, zScale = 1) {
  const hand = fr[key];
  if (!hand || !fr.b?.[wrist]) return null;
  const base = fr.b[wrist];
  return hand.w.map(([x, y, z]) => [
    base[0] + x - hand.w[0][0],
    base[1] + y - hand.w[0][1],
    (base[2] + (flip ? -1 : 1) * (z - hand.w[0][2])) * zScale,
  ]);
}

function palmRadius(w) {
  const off = [0, 1, 2].map((k) => (w[5][k] + w[9][k] + w[17][k] - 3 * w[0][k]) / 4);
  return Math.hypot(...off);
}

/** vrm.js-dəki #wordScale ilə eyni: avatarın ovucu / siqnalçının ovucu (söz üzrə median). */
function wordScale(avatar, word) {
  const target = avatar.rigs.left.palmOff.length();
  const r = [];
  for (const fr of word.frames) for (const h of [fr.L, fr.R]) {
    const pr = h && palmRadius(h.w);
    if (pr > 1e-3) r.push(target / pr);
  }
  return r.length ? THREE.MathUtils.clamp(q(r, 0.5), 0.7, 1.3) : 1;
}

/** Nöqtənin bədən səthinə məsafəsi (vrm.js #solidDistance ilə eyni; mənfi = içəridə). */
function solidDistance(s, p) {
  let dist = Infinity;
  if (s.head) dist = p.distanceTo(s.head.center) - s.head.radius;
  if (p.y >= s.yMin && p.y <= s.yMax) {
    let sl = s.slices[0];
    for (const c of s.slices) if (Math.abs(c.y - p.y) < Math.abs(sl.y - p.y)) sl = c;
    const u = (p.x - sl.cx) / sl.rx, w = (p.z - sl.cz) / sl.rz;
    const k = Math.hypot(u, w) || 1e-6;
    dist = Math.min(dist, (k - 1) * Math.hypot((u / k) * sl.rx, (w / k) * sl.rz));
  }
  return dist;
}

/** Əlin (barmaq + ovuc) və saidin bədənə ən dərin girməsi (metr, müsbət = içəridə). */
function bodyDepth(avatar, radius, side, pts) {
  const s = avatar.solid;
  if (!s) return { hand: 0, arm: 0 };
  const p = new THREE.Vector3();
  let hand = -Infinity, arm = -Infinity;
  for (const [a, b, n] of SEGS) {
    for (let i = 0; i <= 4; i++) {
      p.fromArray(pts[a]).lerp(new THREE.Vector3().fromArray(pts[b]), i / 4);
      hand = Math.max(hand, radius[n] - solidDistance(s, p));
    }
  }
  const e = avatar.bone(side + 'LowerArm').getWorldPosition(new THREE.Vector3());
  const w = new THREE.Vector3().fromArray(pts[0]);
  for (let i = 0; i <= 10; i++) arm = Math.max(arm, radius.said - solidDistance(s, p.copy(e).lerp(w, i / 10)));
  return { hand, arm };
}

/** Əlin nöqtələrindən ən çox kadrdan çıxanı: NDC-də max(|x|, |y|) (1-dən böyük = kənarda). */
function frameOut(avatar, cam, pts) {
  let worst = 0;
  const v = new THREE.Vector3();
  for (const p of pts) {
    v.fromArray(p).project(cam);
    worst = Math.max(worst, Math.abs(v.x), Math.abs(v.y));
  }
  return worst;
}

/** Ölçmə üçün avatarı hazırlayır: canlılıq söndürülür, zaman addımı xaricdən verilir. */
function takeOver(avatar, guard) {
  const keep = { paused: avatar.paused, life: avatar.life, guard: avatar.guard };
  avatar.paused = true;
  avatar.life = false;
  if (guard != null) avatar.guard = guard;   // oynatma qoruyucusu (vrm.js) — müqayisə üçün söndürülə bilər
  avatar.update(0);                   // canlılıq ofsetləri sıfırlansın
  return () => Object.assign(avatar, keep);
}

function toRest(avatar) {
  avatar.clearHandPose();
  for (let i = 0; i < 90; i++) avatar.update(1 / 60);   // 1.5 s: PAUSE_TO_REST keçir
  avatar.snap();
}

/**
 * Bir söz: hər kadrın həlli birbaşa (snap) və oynatma kimi (yaylarla, 60 Hz) ölçülür.
 * @returns {{frames: object[], play: object[]}}
 */
export function measureWord(avatar, model, word, { playback = true } = {}) {
  const { radius } = model;
  const scale = wordScale(avatar, word);
  const cam = avatar.camera.clone();
  const frames = [];
  const measureNow = () => {
    avatar.scene.updateMatrixWorld(true);
    const L = avatar.last?.left ? avatarHand(avatar, model, 'left') : null;
    const R = avatar.last?.right ? avatarHand(avatar, model, 'right') : null;
    return { L, R };
  };
  for (let f = 0; f < word.frames.length; f++) {
    avatar.setWordPose(word, f);
    avatar.snap();
    const { L, R } = measureNow();
    const sol = avatar.last;
    const row = { f };
    if (L && R) {
      Object.assign(row, handGap(L, R, radius));
      // Əl ↔ qarşı said
      const arm = (side, P) => {
        const e = avatar.bone(side + 'LowerArm').getWorldPosition(new THREE.Vector3()).toArray();
        let g = Infinity;
        for (const [a, b, n] of SEGS) g = Math.min(g, segDist(P[a], P[b], e, side === 'left' ? L[0] : R[0]) - radius[n] - radius.said);
        return g;
      };
      row.arm = Math.min(arm('left', R), arm('right', L));
      // Siqnalçının datası: avatarın seçdiyi kadr və dərinlik variantı ilə
      const fl = word.frames[sol.left.frame] ?? word.frames[f], frr = word.frames[sol.right.frame] ?? word.frames[f];
      const sL = signerHand(fl, 'L', 5, sol.left.flip), sR = signerHand(frr, 'R', 6, sol.right.flip);
      if (sL && sR) {
        const sc = (P) => P.map((p) => p.map((v) => v * scale));
        const sg = handGap(sc(sL), sc(sR), radius);
        row.signer = sg.gap;
        row.signerPair = sg.pair;
        // Skeleton: t2s.js-in göndərdiyi nöqtələr (əllər ayrıldıqdan sonra), dərinlik 0.8 ilə
        // sıxılır (skeleton.js). skelRaw — ayırmadan əvvəlki data.
        const zero = Object.fromEntries(Object.keys(radius).map((k) => [k, 0]));
        const sk = [...sL, ...sR].map(([x, y, z]) => ({ x, y, z }));
        T2S.separateHands?.(sk);
        const z8 = (P) => P.map((p) => [p.x, p.y, p.z * 0.8]);
        row.skel = handGap(z8(sk.slice(0, 21)), z8(sk.slice(21)), zero).axis;
        row.skelRaw = handGap(sL.map(([x, y, z]) => [x, y, z * 0.8]), sR.map(([x, y, z]) => [x, y, z * 0.8]), zero).axis;
      }
    }
    // Mövcud keyfiyyət ölçüləri (hər iki əl üçün)
    cam.position.copy(avatar.camGoal.pos);
    cam.lookAt(avatar.camGoal.target);
    cam.updateMatrixWorld(true);
    for (const [side, P] of [['left', L], ['right', R]]) {
      if (!P) continue;
      const c = sol[side];
      const palm = new THREE.Vector3();
      for (const j of [0, 5, 9, 17]) palm.add(new THREE.Vector3().fromArray(P[j]));
      palm.multiplyScalar(0.25);
      const body = bodyDepth(avatar, radius, side, P);
      row[side] = {
        reach: palm.distanceTo(c.anchor),                     // IK hədəfə çatırmı
        place: c.goal ? palm.distanceTo(c.goal) : null,       // datadan gələn yerə məsafə
        hand: body.hand, arm: body.arm,
        out: frameOut(avatar, cam, P),
        wrist: Math.sqrt(Math.max(0, c.arm.info.hard)) * 180 / Math.PI,
      };
    }
    frames.push(row);
  }

  // Oynatma: t2s.js kimi — istirahətdən başlayır, ilk kadra lead-in əlavə olunur
  const play = [];
  if (playback) {
    toRest(avatar);
    const dt = 1 / word.fps;
    const hold = (0.15 + 0.12);
    const total = word.frames.length;
    // Hamarlıq: ovuc mərkəzinin bir yeniləmədə (1/60 s) ən böyük yerdəyişməsi — qoruyucu
    // əlləri sıçratmamalıdır
    let prev = null;
    const palmOf = (P) => [0, 1, 2].map((k) => (P[0][k] + P[5][k] + P[9][k] + P[17][k]) / 4);
    const run = (f, dur) => {
      const n = Math.max(1, Math.round(dur * 60));
      for (let i = 0; i < n; i++) {
        avatar.update(dur / n);
        const { L, R } = measureNow();
        const now = [L, R].map((P) => P && palmOf(P));
        const step = prev ? Math.max(0, ...now.map((p, k) => (p && prev[k] ? Math.hypot(...p.map((v, j) => v - prev[k][j])) : 0))) : 0;
        prev = now;
        if (L && R) play.push({ f, step, ...handGap(L, R, radius) });
      }
    };
    for (let f = 0; f < total; f++) {
      avatar.setWordPose(word, f);
      run(f, f === 0 ? dt + 0.38 : dt);
    }
    run(total - 1, hold);
  }
  return { frames, play, scale };
}

/** Sözün xülasəsi: ən pis kadr və hədlərdən keçən kadrların sayı. */
export function summarize(id, text, res) {
  const two = res.frames.filter((r) => Number.isFinite(r.gap));
  const worst = two.reduce((a, r) => (!a || r.gap < a.gap ? r : a), null);
  const sides = res.frames.flatMap((r) => SIDES.map((s) => r[s]).filter(Boolean));
  const max = (arr) => (arr.length ? Math.max(...arr) : null);
  return {
    id, text,
    frames: res.frames.length,
    twoHanded: two.length,
    gap: worst ? worst.gap : null,
    gapFrame: worst?.f ?? null,
    pair: worst?.pair ?? null,
    axis: worst?.axis ?? null,
    signer: worst?.signer ?? null,
    skel: two.length ? Math.min(...two.map((r) => r.skel ?? Infinity)) : null,
    skelRaw: two.length ? Math.min(...two.map((r) => r.skelRaw ?? Infinity)) : null,
    deep: two.filter((r) => r.gap < -0.005).length,           // >5 mm girmə olan kadrlar
    armGap: two.length ? Math.min(...two.map((r) => r.arm)) : null,
    playGap: res.play.length ? Math.min(...res.play.map((r) => r.gap)) : null,
    playStep: res.play.length ? Math.max(...res.play.map((r) => r.step)) : null,
    reach: sides.map((s) => s.reach),
    place: sides.map((s) => s.place).filter((v) => v != null),
    bodyHand: max(sides.map((s) => s.hand)),
    bodyArm: max(sides.map((s) => s.arm)),
    out: max(sides.map((s) => s.out)),
    wrist: max(sides.map((s) => s.wrist)),
  };
}

/** Bütün lüğət (və ya `ids`): söz-söz xülasə + ümumi göstəricilər. */
export async function measureWords(state, { ids = null, playback = true, keepFrames = [], guard = null } = {}) {
  const avatar = state.avatar;
  const model = handModel(avatar);
  const release = takeOver(avatar, guard);
  const rows = [], detail = {};
  const t0 = performance.now();
  try {
    for (const e of state.lexicon.entries) {
      if (ids && !ids.includes(e.id)) continue;
      const word = await state.lexicon.get(e.id);
      const res = measureWord(avatar, model, word, { playback });
      rows.push(summarize(e.id, e.text, res));
      if (keepFrames.includes(e.id)) detail[e.id] = res;
    }
  } finally {
    release();
  }
  return { model: { radius: model.radius }, rows, detail, totals: totals(rows), ms: Math.round(performance.now() - t0) };
}

export function totals(rows) {
  const two = rows.filter((r) => r.twoHanded > 0);
  const reach = rows.flatMap((r) => r.reach), place = rows.flatMap((r) => r.place);
  const cm = (v) => Math.round(v * 1000) / 10;
  const count = (fn) => rows.filter(fn).length;
  return {
    words: rows.length,
    twoHanded: two.length,
    handInHand: count((r) => r.gap != null && r.gap < -0.005),          // >5 mm
    handInHand10: count((r) => r.gap != null && r.gap < -0.010),        // >1 sm
    handInHandPlay: count((r) => r.playGap != null && r.playGap < -0.005),
    deepFrames: rows.reduce((a, r) => a + r.deep, 0),
    playStepMaxCm: cm(Math.max(...rows.map((r) => r.playStep ?? 0))),
    playStepP90Cm: cm(q(rows.filter((r) => r.playStep != null).map((r) => r.playStep), 0.9)),
    handInArm: count((r) => r.armGap != null && r.armGap < -0.005),
    skelCrossing: count((r) => r.skel != null && r.skel < 0.01),       // skeletdə sümük oxları <1 sm
    worstGapCm: cm(Math.min(...two.map((r) => r.gap))),
    reachMedianCm: cm(q(reach, 0.5)), reachP90Cm: cm(q(reach, 0.9)),
    placeMedianCm: cm(q(place, 0.5)), placeP90Cm: cm(q(place, 0.9)),
    bodyHandMaxCm: cm(Math.max(...rows.map((r) => r.bodyHand ?? -1))),
    bodyHandWords: count((r) => r.bodyHand > 0.005),
    bodyArmMaxCm: cm(Math.max(...rows.map((r) => r.bodyArm ?? -1))),
    bodyArmWords: count((r) => r.bodyArm > 0.005),
    outOfFrame: count((r) => r.out > 1),
    wristWords: count((r) => r.wrist > 5),                             // biləyin həddini >5° aşan
    wristCeil: count((r) => r.wrist > 10.7),                           // vrm.js HARD_CEIL-dən çox
    wristMaxDeg: Math.round(Math.max(...rows.map((r) => r.wrist ?? 0)) * 10) / 10,
  };
}
