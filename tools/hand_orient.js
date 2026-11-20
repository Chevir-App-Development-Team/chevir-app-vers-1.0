// tools/hand_orient.mjs-in səhifə hissəsi: hər söz kadrı avatara qoyulur, əlin oriyentasiyası ölçülür (E37).
import * as THREE from 'three';
import * as MH from './measure_hands.js';

/** Hər söz və kadr üçün: hədəf, world və avatarın əli arasında bucaqlar; avatarın rig-inin oynaq xətti konsola. */
export async function probe(avatar, lexicon, ids) {
  avatar.paused = true; avatar.life = false;
  const model = MH.handModel(avatar);
  for (const side of ['left', 'right']) {
    const r = avatar.rigs[side];
    const nk = new THREE.Vector3().crossVectors(r.d0, r.s0).normalize();
    if (nk.dot(r.n0) < 0) nk.negate();
    console.log(`rig ${side}: d0 ${r.d0.toArray().map((v) => v.toFixed(3))} s0 ${r.s0.toArray().map((v) => v.toFixed(3))} knuckle normal ${nk.toArray().map((v) => v.toFixed(3))} vs n0 ${(Math.acos(Math.min(1, nk.dot(r.n0))) * 180 / Math.PI).toFixed(1)}°`);
  }
  const V = (p) => new THREE.Vector3(...p);
  const frameOf = (P, chir) => {
    const d = V(P[9]).sub(V(P[0])).normalize();
    const s = V(P[5]).sub(V(P[17])).normalize();
    const n = new THREE.Vector3().crossVectors(d, s).multiplyScalar(chir).normalize();
    return { d, n };
  };
  const ang = (a, b) => Math.acos(Math.max(-1, Math.min(1, a.dot(b)))) * 180 / Math.PI;
  const out = [];
  for (const id of ids) {
    const word = await lexicon.get(id).catch(() => null);
    if (!word) { out.push({ id, missing: true }); continue; }
    const sol = avatar.wordSolution(word);
    const frames = [];
    for (let f = 0; f < word.frames.length; f++) {
      avatar.setWordPose(word, f);
      avatar.snap();
      avatar.vrm.scene.updateMatrixWorld(true);
      const row = { f };
      for (const side of ['left', 'right']) {
        const c = sol[f][side];
        const key = side === 'left' ? 'L' : 'R';
        if (!c) { row[side] = { up: false, data: !!word.frames[f][key] }; continue; }
        const rig = avatar.rigs[side];
        const ach = frameOf(MH.avatarHand(avatar, model, side), rig.chir);
        const view = { d: c.target.d, n: c.target.n };
        const wld = frameOf(c.target.shape.map((p) => p.toArray()), rig.chir);
        row[side] = {
          up: true, hard: +c.arm.info.hard.toFixed(3), flip: c.flip,
          achView: [ang(ach.d, view.d), ang(ach.n, view.n)].map((v) => +v.toFixed(1)),
          achWorld: [ang(ach.d, wld.d), ang(ach.n, wld.n)].map((v) => +v.toFixed(1)),
          viewWorld: [ang(view.d, wld.d), ang(view.n, wld.n)].map((v) => +v.toFixed(1)),
          d: ach.d.toArray().map((v) => +v.toFixed(2)), td: view.d.toArray().map((v) => +v.toFixed(2)), wd: wld.d.toArray().map((v) => +v.toFixed(2)),
          n: ach.n.toArray().map((v) => +v.toFixed(2)), tn: view.n.toArray().map((v) => +v.toFixed(2)), wn: wld.n.toArray().map((v) => +v.toFixed(2)),
        };
      }
      frames.push(row);
    }
    out.push({ id, hands: word.hands, frames });
  }
  return out;
}
