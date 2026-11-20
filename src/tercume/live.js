/**
 * Canlı göstərişin qaydaları (/danis/, s2s.js): yalnız ekranda nə vaxt göstərmək və cümləni nə vaxt bağlamaq.
 * Python-un decode() qaydalarına toxunmur. tools/sim_danis.mjs bunları yazılar üzərində ölçür (E35).
 */

/**
 * Hələ yekun olmayan (boz) söz: SentenceStream.peek() onu neçə ardıcıl pəncərədə (seen) verib, orta ehtimalı (p),
 * kadr uzunluğu (len). fastP: bu ehtimaldan yuxarı söz ilk pəncərədəcə göstərilir.
 * E35, JestDili.az-ın 500 cümləsi brauzerin yolu ilə: hamısını göstərəndə (E28) dəqiqədə 9.5 yalançı boz söz
 * (görünüb yox olan), "2 pəncərə və ya p ≥ 0.7" ilə 3.2; doğru sözün bozu orta hesabla 0.16 s gec gəlir.
 */
export const PENDING_RULE = { seen: 2, p: 0, len: 0, fastP: 0.7 };

export function showPending(w, seen, { seen: need = 1, p: minP = 0, len = 0, fastP = null } = PENDING_RULE) {
  if (fastP != null && w.p >= fastP) return true;
  return seen >= need && w.p >= minP && w.len >= len;
}

/**
 * Cümlə son yekun sözdən bu qədər bərkimiş kadr (30 fps) sonra bağlanır; bərkimə canlıdan 24 kadr geri qaldığı üçün
 * bu, son işarədən ~2.8-3.1 s sonradır (E35, canlı kamera). JestDili.az cümlələrində sözlər arası fasilənin p99-u
 * 44 kadrdır: 36 (~2 s) cümlələrin 12 %-ni ortadan bölür, 48 — 3.6 %, 60 — 0.6 %.
 */
export const SENTENCE_GAP = 60;

/**
 * Söz (boz da, ağ da) yalnız kadrlarının bu qədərində ən azı bir əl görünəndə göstərilir. Canlı sınaqda (E35) əllər
 * kadrdan aşağıda olanda qolların hərəkətindən 0 % əlli sözlər çıxırdı (BEL, GƏLİB); işarə edən insanın öz sözlərində
 * ən az 42 %, JestDili.az-ın 1,894 sözündə ən az 33 % (yalnız biri 70 %-dən az).
 */
export const MIN_HANDS = 0.25;

/**
 * Kamera kadrları bu qədər (ms) gəlməyəndə açıq cümlə bağlanır və axın təzədən başlayır. Chrome örtülmüş və ya gizli
 * səhifədə kadr vermir; canlı sınaqda (E35) cümlə belə 39 s açıq qalıb qayıdanda bağlandı. Səhifə gizlənəndə
 * (visibilitychange) cümlə dərhal bağlanır.
 */
export const FRAME_GAP_MS = 1000;
