import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

// Hər keçid üçün pin-scroll payı (viewport hündürlüyünün faizi). Sərgidə
// telefonlarda açılacağı üçün 100%-dən aşağı, 80% ilə başlanılıb - sətir
// başına tam ekran hündürlüyü scroll uzun hiss oluna bilər. Lazım gələrsə
// tənzimləmək üçün tək dəyişəndir.
const SCROLL_PER_TRANSITION_PCT = 80

// Solğun/gizli sətirlər arasında əlavə nəfəs boşluğu (piksel) - sətir
// blokunun öz hündürlüyünün üstünə gəlir.
const GAP_PX = 24

// gsap.matchMedia().add()-a YALNIZ bir şərt (reduceMotion) verilsə və o
// hazırda false-dursa, heç bir sorğu "uyğun" olmadığı üçün callback
// ÜMUMİYYƏTLƏ çağırılmır (sınaqla təsdiqlənib). Digər fayllardakı
// BREAKPOINTS-in problemi olmamasının səbəbi isDesktop/isMobile cütünün
// həmişə ən azı birinin true olmasıdır - eyni prinsiplə tamamlayıcı ikinci
// şərt əlavə olunur ki, iki haldan biri həmişə true olsun.
const MOTION_CONDITIONS = {
  reduceMotion: '(prefers-reduced-motion: reduce)',
  motionOk: '(prefers-reduced-motion: no-preference)',
}

// Bəyanat və Təsir bölmələri paylaşır: hər an EKRANDA MAKSİMUM İKİ sətir
// olur - aktiv (mərkəzdə, tam görünən) və ondan bir əvvəlki (yuxarıda,
// solğun). Bundan əvvəlki bütün sətirlər tamamilə görünməz və daha da
// yuxarı itələnib ki, solğun sətirlə üst-üstə düşməsin. Pin+scrub olduğu
// üçün geri scroll təbii şəkildə tərsinə işləyir.
export function initLineCarousel(selector) {
  const section = document.querySelector(selector)
  if (!section) return

  const lines = Array.from(section.querySelectorAll('.line-carousel__line'))
  if (lines.length < 2) return

  // Sətir bloklarının faktiki hündürlüyünə görə - 40px kimi sabit dəyər
  // 2-3 sətirlik bloklar üçün kifayət etmir, üst-üstə düşməyə səbəb olurdu.
  // Grid-stack olduğu üçün (align-self: center, stretch yoxdur) hər sətrin
  // öz getBoundingClientRect().height-i real məzmun hündürlüyünü verir.
  // Funksiya kimi saxlanılır ki, resize-da (invalidateOnRefresh ilə)
  // yenidən hesablana bilsin.
  function getOffset() {
    const maxHeight = Math.max(...lines.map((line) => line.getBoundingClientRect().height))
    return maxHeight + GAP_PX
  }

  const mm = gsap.matchMedia()

  mm.add(MOTION_CONDITIONS, (context) => {
    if (context.conditions.reduceMotion) {
      gsap.set(lines, { clearProps: 'all' })
      return
    }

    gsap.set(lines[0], { opacity: 1, y: 0 })
    gsap.set(lines.slice(1), { opacity: 0, y: () => getOffset() })

    const transitions = lines.length - 1
    const tl = gsap.timeline({
      defaults: { ease: 'none', duration: 0.6 },
      scrollTrigger: {
        trigger: section,
        start: 'center center',
        end: `+=${transitions * SCROLL_PER_TRANSITION_PCT}%`,
        pin: true,
        scrub: 1,
        // Sətir hündürlüyünə bağlı y dəyərləri funksiya kimi yazılıb -
        // invalidateOnRefresh sayəsində resize-da (ScrollTrigger.refresh,
        // avtomatik debounce ilə) yenidən hesablanır.
        invalidateOnRefresh: true,
      },
    })

    for (let i = 0; i < transitions; i++) {
      // 1.0 vahidlik "slot"un ortasında (0.2-dən başlayır) - əvvəlində və
      // sonunda kiçik dayanma payı qalır ki, hər cümlə bir az sabit görünsün.
      const start = i + 0.2

      // Növbəti sətir aşağıdan gəlib mərkəzə oturur - aktiv olur.
      tl.fromTo(
        lines[i + 1],
        { opacity: 0, y: () => getOffset() },
        { opacity: 1, y: 0 },
        start
      )
      // Hazırkı aktiv sətir solğunlaşıb yuxarı qalxır - "bir əvvəlki" olur.
      tl.to(lines[i], { opacity: 0.25, y: () => -getOffset() }, start)
      // Əvvəlki "bir əvvəlki" sətir varsa, o tamamilə görünməz olub daha da
      // yuxarı itələnir - solğun sətirlə üst-üstə düşməsin.
      if (i >= 1) {
        tl.to(lines[i - 1], { opacity: 0, y: () => -2 * getOffset() }, start)
      }
    }

    return () => tl.scrollTrigger?.kill()
  })

  document.fonts?.ready?.then(() => ScrollTrigger.refresh())
}
