"""
Üz kanalı (roadmap R3): cümlə sualdırmı (bəli/xeyr, sual sözü ilə), inkardırmı — əllərdən yox, üzdən və başdan.

Giriş tanıma müqaviləsinin kompakt kadrlarıdır (features.reduce: face 102×2, blend 52, pose 9×3, hand 2×21×3),
30 fps. Brauzer əkizi: src/tercume/nonmanual.js (tools/verify_nonmanual.mjs eyni rəqəmləri tələb edir).

1. Kadr (frame_features): baş pozası (yan dönmə, əyilmə, yanlara əyilmə, çiyinlərə görə yeri və ölçüsü), qaşların
   gözdən hündürlüyü, gözün, ağzın açıqlığı — başın öz oxlarında, göz arası məsafə vahidi ilə — və 51 blendshape.
2. Neytral üz (neutral): adamın öz üzü, əllər aşağıdaykən (cümlədən əvvəl və sonra); kadrlardan çıxılır.
3. Cümlə (clip_features): əllərin qalxdığı aralıqda orta, std, maksimum, minimum, son üçdə birin ortası, başın
   hərəkət enerjisi (yellənmə — inkar, tərpənmə — baş əymə).
4. Model: standartlaşdırma + logistik reqressiya (sual və inkar ayrı), çəkilər JSON-da.
"""
import re

import numpy as np

from .features import BLEND_NAMES, POSE_IDX
from .landmarks import FACE_IDX

NM_VERSION = 1
FPS = 30
_F = {m: k for k, m in enumerate(FACE_IDX)}
# göz küncləri, qapaqlar, qaşlar (daxili, orta, xarici), burun ucu, alın, çənə, yanaqlar, dodaqlar
EYE_R, EYE_L = (33, 133), (263, 362)
LID_R, LID_L = (159, 145), (386, 374)            # yuxarı, aşağı
BROW_R, BROW_L = (107, 105, 70), (336, 334, 300)
NOSE, FOREHEAD, CHIN, CHEEK_R, CHEEK_L = 1, 10, 152, 234, 454
LIP_IN, LIP_OUT, MOUTH_C = (13, 14), (0, 17), (61, 291)
SH_L, SH_R = POSE_IDX.index(11), POSE_IDX.index(12)

GEOM = ['roll', 'yaw', 'yaw_cheek', 'pitch', 'pitch_chin', 'head_x', 'head_y', 'head_size',
        'brow_in_r', 'brow_in_l', 'brow_mid_r', 'brow_mid_l', 'brow_out_r', 'brow_out_l',
        'eye_open_r', 'eye_open_l', 'mouth_open', 'mouth_width', 'mouth_corner']
BLEND = BLEND_NAMES[1:]                             # _neutral həmişə 0-dır
FRAME_NAMES = GEOM + BLEND
N_FRAME = len(FRAME_NAMES)
# başın hərəkəti (sürət) bu sütunlardan: yaw, pitch, roll
MOTION = [FRAME_NAMES.index(n) for n in ('yaw', 'pitch', 'roll')]
# neytraldan çıxılmayan sütunlar yoxdur: hamısı adamın öz neytralına görədir
GROUPS = {
    'head': [FRAME_NAMES.index(n) for n in GEOM[:8]],
    'upper': [FRAME_NAMES.index(n) for n in GEOM[8:16]] + [FRAME_NAMES.index(n) for n in BLEND
                                                           if n.startswith(('brow', 'eye', 'cheekSquint', 'noseSneer'))],
    'mouth': [FRAME_NAMES.index(n) for n in GEOM[16:]] + [FRAME_NAMES.index(n) for n in BLEND
                                                         if n.startswith(('mouth', 'jaw', 'cheekPuff'))],
}
assert sorted(sum(GROUPS.values(), [])) == list(range(N_FRAME))


def _pt(face, m):
    return face[:, _F[m], :]


def frame_features(raw: dict, aspect: float) -> np.ndarray:
    """Kompakt kadrlar → T×N_FRAME (üz yoxdursa sətir NaN). x əvvəlcə en/hündürlük nisbətinə vurulur."""
    face = raw['face'][..., :2].astype(np.float64).copy()
    face[..., 0] *= aspect
    pose = raw['pose'][..., :2].astype(np.float64).copy()
    pose[..., 0] *= aspect
    T = face.shape[0]
    er = (_pt(face, EYE_R[0]) + _pt(face, EYE_R[1])) / 2
    el = (_pt(face, EYE_L[0]) + _pt(face, EYE_L[1])) / 2
    d = el - er
    io = np.hypot(d[:, 0], d[:, 1])
    io = np.where(io > 1e-6, io, np.nan)
    ex = d / io[:, None]                                  # başın x oxu (iştirakçının sağından soluna)
    ey = np.stack([-ex[:, 1], ex[:, 0]], 1)               # aşağı
    mid = (er + el) / 2

    def u(p, o=mid):
        return ((p - o) * ex).sum(1) / io

    def v(p, o=mid):
        return ((p - o) * ey).sum(1) / io

    nose = _pt(face, NOSE)
    dr = np.hypot(*(nose - _pt(face, CHEEK_R)).T)
    dl = np.hypot(*(nose - _pt(face, CHEEK_L)).T)
    sh_mid = (pose[:, SH_L] + pose[:, SH_R]) / 2
    sw = np.hypot(*(pose[:, SH_L] - pose[:, SH_R]).T)
    sw = np.where(sw > 1e-6, sw, np.nan)
    g = [
        np.arctan2(ex[:, 1], ex[:, 0]),
        u(nose),
        (dr - dl) / (dr + dl),
        v(nose),
        v(_pt(face, CHIN)),
        (nose[:, 0] - sh_mid[:, 0]) / sw,
        (nose[:, 1] - sh_mid[:, 1]) / sw,
        io / sw,
    ]
    for k in range(3):                                    # qaş: gözün mərkəzindən yuxarı (müsbət)
        g += [-v(_pt(face, BROW_R[k]), er), -v(_pt(face, BROW_L[k]), el)]
    g += [v(_pt(face, LID_R[1]), _pt(face, LID_R[0])), v(_pt(face, LID_L[1]), _pt(face, LID_L[0]))]
    g += [v(_pt(face, LIP_IN[1]), _pt(face, LIP_IN[0])),
          np.hypot(*(_pt(face, MOUTH_C[0]) - _pt(face, MOUTH_C[1])).T) / io,
          v((_pt(face, MOUTH_C[0]) + _pt(face, MOUTH_C[1])) / 2, (_pt(face, LIP_OUT[0]) + _pt(face, LIP_OUT[1])) / 2)]
    x = np.concatenate([np.stack(g, 1), raw['blend'][:, 1:].astype(np.float64)], 1)
    x[~np.isfinite(face[:, 0, 0]) | ~np.isfinite(x).all(1)] = np.nan
    return x.astype(np.float32)


def hands_up(raw: dict, aspect: float) -> np.ndarray:
    """T bool: ən azı bir əl görünür və biləyi çiyin xəttindən bir çiyin enindən az aşağıdadır (dizdəki, kadrın
    altındakı əl qalxmış sayılmır). Poza yoxdursa əlin görünməsi kifayətdir."""
    w = raw['hand'][:, :, 0, :2].astype(np.float64)
    seen = np.isfinite(w[..., 0])
    sh = raw['pose'][:, SH_L:SH_R + 1, :2].astype(np.float64)
    sy = sh[:, :, 1].mean(1)
    sw = np.hypot((sh[:, 0, 0] - sh[:, 1, 0]) * aspect, sh[:, 0, 1] - sh[:, 1, 1])
    ok = np.isfinite(sy) & np.isfinite(sw) & (sw > 1e-6)
    with np.errstate(invalid='ignore'):
        high = (w[..., 1] - sy[:, None]) < np.where(ok, sw, np.inf)[:, None]
    return (seen & (high | ~ok[:, None])).any(1)


def active_range(up: np.ndarray, pad: int = 3):
    """Əllərin ilk qalxdığı və son endiyi kadr: [a, b). Əl yoxdursa bütün klip."""
    k = np.flatnonzero(up)
    if len(k) == 0:
        return 0, len(up)
    return max(0, k[0] - pad), min(len(up), k[-1] + 1 + pad)


def neutral(x: np.ndarray, up: np.ndarray, mode: str = 'rest', min_rest: int = 5) -> np.ndarray:
    """Adamın neytral üzü (N_FRAME): 'rest' — əllər aşağıdakı kadrların medianı (azdırsa bütün klipin medianı),
    'clip' — bütün klipin medianı, 'none' — sıfır."""
    if mode == 'none':
        return np.zeros(x.shape[1], np.float32)
    ok = np.isfinite(x[:, 0])
    rest = ok & ~up
    sel = rest if mode == 'rest' and rest.sum() >= min_rest else ok
    if not sel.any():
        return np.zeros(x.shape[1], np.float32)
    return np.median(x[sel], axis=0).astype(np.float32)


STATS = ['mean', 'std', 'max', 'min', 'last', 'first']


def clip_features(x: np.ndarray, up: np.ndarray, base: np.ndarray, cols=None) -> np.ndarray:
    """T×N_FRAME + neytral → bir cümlənin vektoru: STATS × sütun + başın hərəkəti (3 × 2) + üzün görünmə payı."""
    cols = np.arange(N_FRAME) if cols is None else np.asarray(cols)
    a, b = active_range(up)
    seg = x[a:b] - base
    ok = np.isfinite(seg[:, 0])
    n = len(cols)
    if ok.sum() < 3:
        return np.concatenate([np.zeros(n * len(STATS) + 6, np.float32), [0.0]]).astype(np.float32)
    s = seg[ok][:, cols]
    L = len(s)
    k = max(1, L // 3)
    st = [s.mean(0), s.std(0), s.max(0), s.min(0), s[-k:].mean(0), s[:k].mean(0)]
    # başın hərəkəti: ardıcıl görünən kadrlar arası fərq (FPS-də), orta mütləq sürət və istiqamət dəyişmə tezliyi
    m = seg[:, MOTION]
    dv = m[1:] - m[:-1]
    dv = dv[np.isfinite(dv).all(1)]
    if len(dv) >= 2:
        speed = np.abs(dv).mean(0) * FPS
        sgn = np.sign(dv)
        flips = ((sgn[1:] * sgn[:-1]) < 0).mean(0)
    else:
        speed = np.zeros(3)
        flips = np.zeros(3)
    return np.concatenate(st + [speed, flips, [ok.mean()]]).astype(np.float32)


def clip_names(cols=None):
    cols = np.arange(N_FRAME) if cols is None else np.asarray(cols)
    names = [f'{s}:{FRAME_NAMES[c]}' for s in STATS for c in cols]
    return names + [f'speed:{FRAME_NAMES[c]}' for c in MOTION] + [f'flips:{FRAME_NAMES[c]}' for c in MOTION] + ['face_rate']


def to_fps(raw: dict, fps: float, target: float = FPS) -> dict:
    """Kadrları `target` fps-ə gətirir (ən yaxın kadr). 30 fps-dirsə dəyişmir."""
    T = raw['hand'].shape[0]
    if abs(fps - target) < 0.5 or T == 0:
        return raw
    n = max(1, int(round(T * target / fps)))
    idx = np.clip(np.round(np.arange(n) * fps / target).astype(int), 0, T - 1)
    return {k: v[idx] for k, v in raw.items()}


# ---------------------------------------------------------------- zəif etiketlər mətndən

WH = ['nə', 'nədir', 'nəyə', 'nəyi', 'nəyin', 'nədən', 'niyə', 'necə', 'necədir', 'necəsən', 'necəsiniz', 'hara',
      'harada', 'harda', 'haraya', 'haradan', 'hardan', 'hardadır', 'haradır', 'kim', 'kimə', 'kimin',
      'kimdir', 'kimindir', 'neçə', 'neçəyə', 'neçəyədir', 'neçənci', 'hansı', 'hansında']
# hissəcik -mı (varmı, bilərsinizmi, olarmı, bilərəmmi): samitdən sonra mı / mi / mu / mü. Saitdən sonra isə
# təsirlik haldır (özümü, işimi, babamı)
_MI = re.compile(r'\w+[bcçdfgğhjkqlmnprsştvxz]m[ıiuü]$')
_NOT_MI = {'rəsmi', 'həcmi', 'ismi', 'əzmi', 'kimi'}
NEG_WORDS = {'yox', 'yoxdur', 'yoxdu', 'deyil', 'deyiləm', 'deyilsən', 'deyilik', 'deyilsiniz', 'deyildir', 'deyildi',
             'heç', 'xeyr', 'olmaz'}
# felin inkarı (sevmirəm, işləmir, verməyib, geyinməmişəm, düşmədim, olmayın, gəlməyəcək, gəlməz); məsdər (olmaq) inkar deyil
_NEG_VERB = re.compile(r'\w+(?:m[ıiuü]r|m[aə]y[ıi]b|m[aə]m[ıi]ş|m[aə]d[ıi]|m[aə]y[ıi]n|m[aə]y[aə]c|m[aə]z)\w*$')
_NOT_NEG = {'damır', 'kəmər', 'nümayiş', 'həmişə', 'təmir', 'təmiri', 'gecikməyinizin', 'anadangəlmədir'}
# AzSLD-nin qlosunda inkar: YOX, DEYİL, HEÇ, inkar feli (SEVMİR, BİLMƏMƏK, OLMAMAQ, ANLAMAMAQ)
_NEG_GLOSS = re.compile(r'^(yox|yoxdur|deyil|heç)$|m[ıiuü]r$|m[aə]m[aə][kq]$')


def _words(text: str):
    t = text.replace('I', 'ı').replace('İ', 'i').lower()
    return re.findall(r'[a-zçəğıöşü]+', t)


def weak_labels(text: str, mark: bool = True, glosses: str = ''):
    """Mətn → (sual 0/1, növ 'wh' / 'yn' / '', inkar 0/1). mark: mətndə '?' etibarlıdır (AzSLD); JestDili.az-da
    '?' yoxdur, sual sözü və -mı hissəciyi ilə tapılır (siyahı əl ilə yoxlanıb, E31). glosses: AzSLD-nin qlosu
    (inkar işarəsi mətndə görünməyə bilər: işsizdir = İŞ YOX)."""
    ws = _words(text)
    wh = any(w in WH for w in ws) or 'nə vaxt' in ' '.join(ws) or 'nə zaman' in ' '.join(ws)
    mi = any(_MI.match(w) and w not in _NOT_MI for w in ws)
    q = ('?' in text) if mark else (wh or mi)
    qtype = ('wh' if wh else 'yn') if q else ''
    neg = (any(w in NEG_WORDS for w in ws) or any(_NEG_VERB.match(w) and w not in _NOT_NEG for w in ws)
           or any(_NEG_GLOSS.search(w) and w not in _NOT_NEG and w != 'çatmamaq' for w in _words(glosses)))
    return int(q), qtype, int(neg)

