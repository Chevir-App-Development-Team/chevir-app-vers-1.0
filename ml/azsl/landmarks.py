"""MediaPipe landmarks per frame, with the same models and settings as the browser.

hand_landmarker.task below is byte-identical to public/tercume/assets/hand_landmarker.task
(same md5), and the hand options match src/tercume/hands.js. Videos run in VIDEO mode
(tracking), like the live camera, so training data has the same jitter and misses
the model will see in the browser. Pose and face use the float16/1 releases that the
browser will load when they are added there.

Per frame:
  hand        (2, 21, 3)  image coords x, y in [0, 1], z relative — slot 0 = signer's RIGHT hand
  hand_world  (2, 21, 3)  metric 3D hand (MediaPipe world landmarks)
  hand_score  (2,)        handedness score of the detection in that slot
  pose        (33, 4)     image coords + visibility
  pose_world  (33, 3)
  face        (102, 3)    lips, eyes, brows, nose, oval, irises (FACE_IDX)
  blend       (52,)       face blendshapes (brows, eyes, mouth — non-manual signals)
Missing detections are NaN. Slots are anatomical: pose wrists decide which hand is which
(same rule as tools/build_words.py), so crossing hands keep their slot.
"""
import base64
import hashlib
import os
import urllib.request

import cv2
import numpy as np

_MP = 'https://storage.googleapis.com/mediapipe-models'
MODELS = {   # name: (url, base64 md5 as GCS reports it)
    'hand': (f'{_MP}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'FTGEMOo4UWcP6ZFBFqnPrQ=='),
    'pose': (f'{_MP}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task', 'BKdd33yBGsehpFIyZt19iA=='),
    'face': (f'{_MP}/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'sOcnSQehZEQE/vZrKN1thQ=='),
}
# OpenCV SFace — only to group recordings by signer for the split; embeddings never leave the split step.
SFACE_URLS = [
    'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
    'https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx',
]
SFACE_FILE = 'face_recognition_sface_2021dec.onnx'
SFACE_MD5 = 'lDaT6px939mmbce36NUqTw=='

FACE_IDX = [
    # lips (outer + inner)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    # eyes
    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
    # eyebrows
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    # nose, face oval extremes (head pose), iris centres
    1, 2, 98, 327, 10, 152, 234, 454, 468, 473,
]
FIELDS = {
    'hand': (2, 21, 3), 'hand_world': (2, 21, 3), 'hand_score': (2,),
    'pose': (33, 4), 'pose_world': (33, 3),
    'face': (len(FACE_IDX), 3), 'blend': (52,),
}
P_NOSE, P_L_SH, P_R_SH, P_L_WR, P_R_WR = 0, 11, 12, 15, 16
EMB_DIM = 128


def _md5_b64(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def ensure_models(model_dir: str, sface: bool = False) -> dict:
    """Download the .task files (and SFace) once, verify md5. Returns name → local path."""
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for name, (url, md5) in MODELS.items():
        p = os.path.join(model_dir, os.path.basename(url))
        if not os.path.exists(p) or _md5_b64(p) != md5:
            urllib.request.urlretrieve(url, p)
        got = _md5_b64(p)
        if got != md5:
            raise IOError(f'{name}: md5 {got} != {md5} — model release changed, landmarks would not match the browser')
        paths[name] = p
    if sface:
        p = os.path.join(model_dir, SFACE_FILE)
        for url in SFACE_URLS:
            if os.path.exists(p) and _md5_b64(p) == SFACE_MD5:
                break
            try:
                urllib.request.urlretrieve(url, p)
            except OSError as e:
                print(f'SFace download failed from {url}: {e}')
        if not os.path.exists(p) or _md5_b64(p) != SFACE_MD5:
            raise IOError('SFace model missing or md5 mismatch')
        paths['sface'] = p
    return paths


def empty(n: int) -> dict:
    return {k: np.full((n, *s), np.nan, np.float32) for k, s in FIELDS.items()}


def assign_hands(wrists_xy, pose_wrists, aspect):
    """Detected hands → [right slot, left slot] (indices into the detections or None).

    Pose wrists decide, so crossed hands keep their side. Without a pose the signer's
    right hand is on the image's left (the recordings are not mirrored). Same rule as
    tools/build_words.py `_assign`.
    """
    slots = [None, None]
    if not wrists_xy:
        return slots
    if pose_wrists is None:
        for s, k in enumerate(sorted(range(len(wrists_xy)), key=lambda k: wrists_xy[k][0])[:2]):
            slots[s] = k
        return slots
    (lx, ly), (rx, ry) = pose_wrists

    def d(k, px, py):
        return np.hypot((wrists_xy[k][0] - px) * aspect, wrists_xy[k][1] - py)

    if len(wrists_xy) >= 2:
        return [0, 1] if d(0, rx, ry) + d(1, lx, ly) <= d(0, lx, ly) + d(1, rx, ry) else [1, 0]
    slots[0 if d(0, rx, ry) <= d(0, lx, ly) else 1] = 0
    return slots


class Extractor:
    """One per worker process. VIDEO-mode landmarkers are recreated per video (tracking state
    must not leak between clips); IMAGE-mode ones are created once and reused."""

    def __init__(self, paths: dict, face: bool = True, embed: bool = False):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
        self.mp, self.V, self.Base = mp, vision, BaseOptions
        self.paths, self.face_on = paths, face
        self.recognizer = cv2.FaceRecognizerSF.create(paths['sface'], '') if embed else None
        self._image_lms = None

    def _create(self, video: bool):
        V, B = self.V, self.Base
        mode = V.RunningMode.VIDEO if video else V.RunningMode.IMAGE
        hand = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
            base_options=B(model_asset_path=self.paths['hand']), running_mode=mode, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5))
        pose = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(
            base_options=B(model_asset_path=self.paths['pose']), running_mode=mode, num_poses=1))
        face = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
            base_options=B(model_asset_path=self.paths['face']), running_mode=mode, num_faces=1,
            output_face_blendshapes=True)) if self.face_on else None
        return hand, pose, face

    @staticmethod
    def _close(lms):
        for lm in lms:
            if lm is not None:
                lm.close()

    def _frame(self, out, t, rgb, lms, ts):
        """Fills row t of `out`; returns the 478 face landmarks (or None) for embedding."""
        hand, pose, face = lms
        im = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if ts is None:
            hr, pr = hand.detect(im), pose.detect(im)
            fr = face.detect(im) if face else None
        else:
            hr, pr = hand.detect_for_video(im, ts), pose.detect_for_video(im, ts)
            fr = face.detect_for_video(im, ts) if face else None
        H, W = rgb.shape[:2]

        wrists = None
        if pr.pose_landmarks:
            pl = pr.pose_landmarks[0]
            out['pose'][t] = [[p.x, p.y, p.z, p.visibility] for p in pl]
            out['pose_world'][t] = [[p.x, p.y, p.z] for p in pr.pose_world_landmarks[0]]
            wrists = (pl[P_L_WR].x, pl[P_L_WR].y), (pl[P_R_WR].x, pl[P_R_WR].y)

        found = hr.hand_landmarks or []
        slots = assign_hands([(h[0].x, h[0].y) for h in found], wrists, W / H)
        for s, k in enumerate(slots):
            if k is None:
                continue
            out['hand'][t, s] = [[p.x, p.y, p.z] for p in found[k]]
            out['hand_world'][t, s] = [[p.x, p.y, p.z] for p in hr.hand_world_landmarks[k]]
            out['hand_score'][t, s] = hr.handedness[k][0].score

        if fr is not None and fr.face_landmarks:
            fl = fr.face_landmarks[0]
            out['face'][t] = [[fl[i].x, fl[i].y, fl[i].z] for i in FACE_IDX]
            if fr.face_blendshapes:
                bs = [c.score for c in fr.face_blendshapes[0]][:FIELDS['blend'][0]]
                out['blend'][t, :len(bs)] = bs
            return fl
        return None

    def video(self, path: str, stride: int = 1, embed_at=(0.25, 0.5, 0.75)):
        """→ (arrays, meta, face) for a video file; face = {'emb', 'thumb'} or None."""
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        rows, picks = [], []
        targets = sorted({int(f * n_hint) for f in embed_at}) if (self.recognizer and n_hint) else []
        lms = self._create(video=True)
        try:
            i = -1
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                i += 1
                if i % stride:
                    continue
                row = empty(1)
                fl = self._frame(row, 0, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), lms, int(round(i * 1000.0 / fps)))
                rows.append(row)
                if fl is not None and targets and i >= targets[0]:
                    picks.append((bgr, fl))
                    targets.pop(0)
        finally:
            cap.release()
            self._close(lms)
        arrays = {k: np.concatenate([r[k] for r in rows]) if rows else np.zeros((0, *s), np.float32)
                  for k, s in FIELDS.items()}
        meta = summarize(arrays, W, H)
        meta.update(fps=round(fps / stride, 3), width=W, height=H, n_frames=len(rows))
        return arrays, meta, self._embed(picks)

    def image(self, data: bytes):
        """Single image (fingerspelling) → (arrays with T=1, meta)."""
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('could not decode image')
        if self._image_lms is None:
            self._image_lms = self._create(video=False)
        out = empty(1)
        self._frame(out, 0, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self._image_lms, None)
        H, W = bgr.shape[:2]
        meta = summarize(out, W, H)
        meta.update(fps=0.0, width=W, height=H, n_frames=1)
        return out, meta

    def _embed(self, picks):
        """Mean SFace embedding over a few frames + one aligned 112×112 crop for eyeballing clusters."""
        if not self.recognizer or not picks:
            return None
        feats, thumb = [], None
        for bgr, fl in picks:
            H, W = bgr.shape[:2]
            pt = lambda i: (fl[i].x * W, fl[i].y * H)
            xs, ys = [p.x * W for p in fl], [p.y * H for p in fl]
            (e1, e2), (m1, m2) = sorted([pt(468), pt(473)]), sorted([pt(61), pt(291)])
            row = np.array([[min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys),
                             *e1, *e2, *pt(1), *m1, *m2, 1.0]], np.float32)
            crop = self.recognizer.alignCrop(bgr, row)
            f = self.recognizer.feature(crop).ravel().astype(np.float32)
            feats.append(f / (np.linalg.norm(f) + 1e-9))
            if thumb is None:
                thumb = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        emb = np.mean(feats, axis=0)
        return {'emb': emb / (np.linalg.norm(emb) + 1e-9), 'thumb': thumb}


def summarize(a: dict, W: int, H: int) -> dict:
    """Detection rates + a frontal-camera check (nose offset from shoulder centre / shoulder width)."""
    T = a['hand'].shape[0]
    if not T:
        return dict(hand_rate=0.0, two_hand_rate=0.0, pose_rate=0.0, face_rate=0.0, frontal=float('nan'))
    hands = ~np.isnan(a['hand'][:, :, 0, 0])
    pose_ok = ~np.isnan(a['pose'][:, 0, 0])
    frontal = float('nan')
    if pose_ok.any():
        p = a['pose'][pose_ok]
        sw = np.abs(p[:, P_L_SH, 0] - p[:, P_R_SH, 0]) * W
        off = np.abs(p[:, P_NOSE, 0] - (p[:, P_L_SH, 0] + p[:, P_R_SH, 0]) / 2) * W
        frontal = float(np.median(off / np.maximum(sw, 1.0)))
    return dict(
        hand_rate=round(float(hands.any(1).mean()), 4),
        two_hand_rate=round(float(hands.all(1).mean()), 4),
        pose_rate=round(float(pose_ok.mean()), 4),
        face_rate=round(float((~np.isnan(a['face'][:, 0, 0])).mean()), 4),
        frontal=round(frontal, 4),
    )
