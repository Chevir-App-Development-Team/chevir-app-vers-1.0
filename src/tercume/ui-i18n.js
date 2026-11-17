/**
 * s2t.js/s2w.js/s2s.js are off-limits (ML/recognition code owned by another team) so their
 * literal onStatus() message strings can't be edited at the source. This module recognizes the
 * known AZ strings/templates they emit and maps them to translated tercume.status.fwd.* keys
 * when the page is EN. On AZ, or for any message that doesn't match a known pattern, the input
 * is returned unchanged - AZ behavior is untouched, and an unrecognized message simply falls
 * back to its original (Azerbaijani) text instead of breaking.
 */
import { t, currentLang } from '../lib/i18n.js'

const EXACT = {
  'MediaPipe yüklənir…': 'tercume.status.fwd.mediapipeLoading',
  'Hazır': 'tercume.status.fwd.ready',
  'Kamera işləyir · hərfi 1 saniyə sabit saxla': 'tercume.status.fwd.cameraRunningLetters',
  'Video emal olunur': 'tercume.status.fwd.videoProcessing',
  'Video bitdi': 'tercume.status.fwd.videoEnded',
  'Dayandırıldı': 'tercume.status.fwd.stopped',
  'Söz modeli yüklənir…': 'tercume.status.fwd.wordModelLoading',
  'MediaPipe yüklənir (əllər, bədən, üz)…': 'tercume.status.fwd.mediapipeLoadingFace',
  'MediaPipe yüklənir (əllər, bədən)…': 'tercume.status.fwd.mediapipeLoadingNoFace',
  'Kamera işləyir · sözü göstər, sonra əlləri endir': 'tercume.status.fwd.cameraRunningWords',
  'Cümlə modeli yüklənir…': 'tercume.status.fwd.sentenceModelLoading',
  'Kamera işləyir · cümləni fasiləsiz göstər, sonda əlləri endir': 'tercume.status.fwd.cameraRunningSentence',
}

const PATTERNS = [
  { re: /^Hazır · (\d+) söz, davamlı$/, key: 'tercume.status.fwd.readyContinuous', params: (m) => ({ n: m[1] }) },
  { re: /^Hazır · (\d+) söz$/, key: 'tercume.status.fwd.readyWords', params: (m) => ({ n: m[1] }) },
  { re: /^Tanıma xətası: (.+)$/s, prefixKey: 'tercume.status.fwd.recognitionErrorPrefix', rest: (m) => m[1] },
]

export function translateStatusMessage(text) {
  if (currentLang !== 'en' || typeof text !== 'string') return text
  if (text in EXACT) return t(EXACT[text])
  for (const p of PATTERNS) {
    const m = text.match(p.re)
    if (!m) continue
    if (p.prefixKey) return `${t(p.prefixKey)} ${p.rest(m)}`
    return t(p.key, p.params(m))
  }
  return text
}
