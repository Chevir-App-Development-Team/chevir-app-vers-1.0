import gsap from 'gsap'
import { prefersReducedMotion } from '../lib/gsapSetup.js'
import { HAND_POSE_OPEN, HAND_CONNECTIONS } from '../lib/handPoses.js'
import { initLineCarousel } from '../lib/lineCarousel.js'

// public/ altındakı asset-lərə mütləq "/..." yolla deyil, BASE_URL ilə
// istinad olunur (kök yolda "/" verir, amma alt qovluqda yayımlansa da işləyər).
const BASE_URL = import.meta.env.BASE_URL

// logo.svg-nin TAM forması (iki qövs, iki ox ucu, mərkəzi nöqtə) - fon
// dekoru kimi təkrar istifadə, boz və fırlanan.
const ARC_SIGNED_D =
  'M50.5 88.5C45.3784 88.5 40.307 87.4912 35.5753 85.5313C30.8436 83.5714 26.5443 80.6986 22.9228 77.0772C19.3013 73.4557 16.4286 69.1563 14.4687 64.4247C12.5088 59.693 11.5 54.6215 11.5 49.5'
const ARC_SPOKEN_D =
  'M49.5 10.5C54.6216 10.5 59.693 11.5088 64.4247 13.4687C69.1564 15.4286 73.4557 18.3014 77.0772 21.9228C80.6986 25.5443 83.5714 29.8436 85.5313 34.5753C87.4912 39.307 88.5 44.3784 88.5 49.5'
const HEAD_SIGNED_D = 'M11.5 34L21.4593 49.75H1.54071L11.5 34Z'
const HEAD_SPOKEN_D = 'M88.5 65L78.5407 49.25L98.4593 49.25L88.5 65Z'

// public/wordmark.svg-nin dəyişməz kopyası (hero.js-in özünün eyni prinsiplə
// WORDMARK_MARKUP-u kimi) - currentColor ilə əlaqə/footer bölməsində böyük,
// fon elementi kimi (aşağı opasitiya, bulanıq). viewBox 0 0 81 24 (əsl fayl
// ilə eyni).
const CONTACT_WORDMARK_MARKUP = `
  <path d="M72.418 13.7919C72.418 8.92787 75.49 6.46387 80.354 6.46387V9.98387C77.794 9.98387 76.258 11.2319 76.258 13.7919V22.7519H72.418V13.7919Z" fill="currentColor"/>
  <path d="M66.6309 2.112C66.6309 0.959999 67.5589 0 68.7429 0C69.9269 0 70.8549 0.959999 70.8549 2.112C70.8549 3.328 69.9269 4.256 68.7429 4.256C67.5589 4.256 66.6309 3.328 66.6309 2.112ZM66.8229 22.752V6.752H70.6629V22.752H66.8229Z" fill="currentColor"/>
  <path d="M55.8488 22.752L49.8008 6.75195H53.9288L57.8328 17.6L61.7368 6.75195H65.8648L59.8168 22.752H55.8488Z" fill="currentColor"/>
  <path d="M38.0373 12.5118H46.0692C45.3972 10.9438 43.9573 10.0478 42.0373 10.0478C40.1493 10.0478 38.7093 10.9438 38.0373 12.5118ZM42.0373 19.5198C43.6372 19.5198 44.8853 18.9118 45.6533 17.7918H49.8452C48.7892 21.0238 45.9412 23.0398 42.0692 23.0398C37.1092 23.0398 33.7812 19.7118 33.7812 14.7518C33.7812 9.82383 37.1092 6.52783 42.0692 6.52783C47.0292 6.52783 50.3252 9.82383 50.3252 14.7518C50.3252 15.1998 50.2932 15.6158 50.2612 16.0318H37.7492C38.1652 18.2398 39.7653 19.5198 42.0373 19.5198Z" fill="currentColor"/>
  <path d="M21.4455 0.352051V7.87205C22.3735 7.00805 23.6855 6.46405 25.4775 6.46405C30.2455 6.46405 32.3255 10.4321 32.3255 13.2161V22.7521H28.4855V13.3121C28.4855 11.0401 26.8215 10.0161 24.9655 10.0161C23.1095 10.0161 21.4455 11.0401 21.4455 13.3121V22.7521H17.6055V0.352051H21.4455Z" fill="currentColor"/>
  <path d="M3.84 14.7838C3.84 17.7598 5.6 19.5198 8.256 19.5198C10.208 19.5198 11.648 18.5918 12.32 16.9598H16.32C15.456 20.6718 12.48 23.0398 8.288 23.0398C3.328 23.0398 0 19.7118 0 14.7518C0 9.82383 3.328 6.52783 8.288 6.52783C12.448 6.52783 15.456 8.86383 16.288 12.5118H12.288C11.616 10.9438 10.176 10.0478 8.256 10.0478C5.6 10.0478 3.84 11.8078 3.84 14.7838Z" fill="currentColor"/>
`

// side: 'left' | 'right' - dekor ekran kənarına yapışır (bax content.css
// .section-bg--left/--right).
function logoBackgroundMarkup(side) {
  return `
  <div class="section-bg section-bg--logo section-bg--${side}" aria-hidden="true">
    <svg class="section-bg__svg" viewBox="0 0 100 100">
      <g class="section-bg__logo-group">
        <path class="section-bg__logo-arc" d="${ARC_SIGNED_D}" />
        <path class="section-bg__logo-arc" d="${ARC_SPOKEN_D}" />
        <path class="section-bg__logo-head" d="${HEAD_SIGNED_D}" />
        <path class="section-bg__logo-head" d="${HEAD_SPOKEN_D}" />
        <circle class="section-bg__logo-dot" cx="50" cy="50" r="4" />
      </g>
    </svg>
  </div>`
}

// 21-nöqtəli əl skeletini (lib/handPoses.js) mərkəzdən aralı salıb sönük
// nöqtə/xətt buludu kimi fon dekoruna çevirir. z oxu 2D fon üçün nəzərə alınmır.
function dotsBackgroundMarkup(side) {
  const disperse = 2.4
  const scale = 60
  const offsetX = 100
  const offsetY = 20
  const points = HAND_POSE_OPEN.map(([x, y]) => [
    x * disperse * scale + offsetX,
    y * disperse * scale + offsetY,
  ])
  const lines = HAND_CONNECTIONS.map(([a, b]) => {
    const [x1, y1] = points[a]
    const [x2, y2] = points[b]
    return `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" />`
  }).join('')
  const dots = points
    .map(([x, y]) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.4" />`)
    .join('')
  return `
  <div class="section-bg section-bg--dots section-bg--${side}" aria-hidden="true">
    <svg class="section-bg__svg" viewBox="0 0 200 200">
      <g class="section-bg__cloud">${lines}${dots}</g>
    </svg>
  </div>`
}

export const contentMarkup = `
<section class="problem" aria-labelledby="problem-heading">
  ${logoBackgroundMarkup('left')}
  <div class="reveal">
    <div class="problem__stat">
      <p class="problem__number">13&nbsp;000</p>
      <p class="problem__stat-desc">
        Azərbaycanda eşitmə məhdudiyyətli insanların təxmini sayı.
      </p>
    </div>
    <div class="problem__content">
      <h2 id="problem-heading" class="problem__heading">
        Yazı həll deyil, maneədir.
      </h2>
      <div class="problem__body">
        <p class="problem__paragraph">
          Eşitmə məhdudiyyəti ilə doğulan insan Azərbaycan dilini heç vaxt eşitmir.
          Onu ikinci dil kimi öyrənir. Poliklinikada anket doldurmaq, bank
          müqaviləsi oxumaq - bunlar onun üçün yad dildə sənəd doldurmaqdır.
        </p>
        <p class="problem__paragraph">
          Mövcud əlçatanlıq alətləri isə yalnız bir istiqamətdə işləyir: eşidəndən
          eşitmə məhdudiyyətli şəxsə. Altyazı var. Əks istiqamət - jest dilində
          deyilənin eşidənə çatması - tamamilə boşdur.
        </p>
      </div>
    </div>
  </div>
</section>

<section class="difference" aria-labelledby="difference-heading">
  <div class="reveal">
    <h2 id="difference-heading" class="difference__heading">Fərqimiz nədir</h2>
    <div class="difference__grid">
      <div class="difference__item">
        <h3 class="difference__item-title">Bir skelet, iki istiqamət</h3>
        <p class="difference__item-desc">
          Tanıma və istehsal ayrı sistemlər deyil. Hər ikisi eyni skelet
          təsvirindən qidalanır - kameradan çıxarılan nöqtələr həm jesti
          oxumaq, həm də avatarı hərəkətə gətirmək üçün işlədilir. Bu, iki
          istiqamətin uyğunluğunu arxitektura səviyyəsində təmin edir.
        </p>
      </div>
      <div class="difference__item">
        <h3 class="difference__item-title">Mimika sonradan əlavə deyil</h3>
        <p class="difference__item-desc">
          Jest dilində inkar, sual və şərt üz və baş hərəkətləri ilə ötürülür.
          Sahənin aparıcı kommersiya oyunçusu bu elementlərin səlis
          istehsalını həll olunmamış problem kimi elan edir. Bizdə bu,
          sonradan gələcək yaxşılaşdırma deyil - arxitekturanın başlanğıc
          tələbidir.
        </p>
      </div>
      <div class="difference__item">
        <h3 class="difference__item-title">İcma ilə yoxlama</h3>
        <p class="difference__item-desc">
          Model çıxışları laboratoriya metrikası ilə deyil, ana dili jest
          dili olan insanların anlama səviyyəsi ilə yoxlanılır. Nəzəri
          olaraq uğurlu görünən, praktikada isə anlaşılmayan nəticələrin
          qarşısı belə alınır.
        </p>
      </div>
    </div>
  </div>
</section>

<section class="values" aria-labelledby="values-heading">
  <div class="reveal">
    <p id="values-heading" class="values__statement">
      Biz olmadan bizim haqqımızda heç nə.
    </p>
    <p class="values__slogan">Eşitmə məhdudiyyətli icmanın şüarı.</p>
    <p class="values__principle">
      Eşitmə məhdudiyyətli istifadəçi öz ana dilində məzmun oxumaq və yaratmaq
      üçün heç vaxt ödəniş etməyəcək. Əlçatanlıq ödəniş divarının arxasında
      ola bilməz.
    </p>
  </div>
</section>

<section class="usecases" aria-labelledby="usecases-heading">
  ${dotsBackgroundMarkup('right')}
  <div class="reveal">
    <h2 id="usecases-heading" class="usecases__heading">Harada işləyir</h2>
    <div class="usecases__grid">
      <div class="usecases__item">
        <h3 class="usecases__item-title">Dövlət xidmətləri və banklar</h3>
        <p class="usecases__item-desc">
          Ərizə qəbulu, pəncərə xidməti və çağrı mərkəzlərində birbaşa
          tərcümə.
        </p>
      </div>
      <div class="usecases__item">
        <h3 class="usecases__item-title">Səhiyyə</h3>
        <p class="usecases__item-desc">
          Həkim və xəstə arasında təcili və planlı görüşlərdə ünsiyyət.
        </p>
      </div>
      <div class="usecases__item">
        <h3 class="usecases__item-title">Təhsil</h3>
        <p class="usecases__item-desc">
          Dərs otağında və valideyn-müəllim görüşlərində real vaxtda
          tərcümə.
        </p>
      </div>
      <div class="usecases__item">
        <h3 class="usecases__item-title">Rəqəmsal platformalar</h3>
        <p class="usecases__item-desc">
          Tətbiq və veb-saytlara API ilə inteqrasiya olunan tərcümə qatı.
        </p>
      </div>
    </div>
  </div>
</section>

<section class="results" aria-labelledby="results-heading">
  ${logoBackgroundMarkup('left')}
  <div class="reveal">
    <div class="results__intro">
      <h2 id="results-heading" class="results__heading">Hazırkı nəticələr</h2>
      <p class="results__lead">
        Aşağıdakılar AzSLD (Azərbaycan Jest Dili Datasetı) üzərində öyrədilmiş
        modellərin real çıxışlarıdır.
      </p>
    </div>
    <div class="results__accuracy">
      <p class="results__accuracy-number">84,85%</p>
      <p class="results__accuracy-desc">
        AzSLD - 30 000 annotasiyalı video üzərində öyrədilmiş tanıma
        modelimizin dəqiqliyi.
      </p>
      <p class="results__accuracy-caveat">
        Bu sahədə adətən 95%-dən yuxarı rəqəmlər elan olunur. Amma həmin rəqəm
        eyni insanların videoları həm öyrətmədə, həm testdə olanda alınır.
        Model heç görmədiyi yeni bir insanı tanımalı olanda dəqiqlik 62%-ə
        düşür. Biz birinci rəqəmi ümumiyyətlə göstərmirik.
      </p>
    </div>
    <div class="results__grid">
      <figure class="results__card">
        <div class="results__video-frame">
          <video
            class="results__video"
            data-autoplay-in-view
            muted
            loop
            playsinline
            preload="metadata"
            poster="${BASE_URL}media/predict.jpg"
            aria-labelledby="results-title-0"
            aria-describedby="results-desc-0"
          >
            <source src="${BASE_URL}media/predict.webm" type="video/webm" />
            <source src="${BASE_URL}media/predict.mp4" type="video/mp4" />
          </video>
        </div>
        <figcaption>
          <h3 class="results__card-title" id="results-title-0">Proqnoz</h3>
          <p class="results__card-desc" id="results-desc-0">
            Orijinal görüntü və modelin proqnozu yan-yana müqayisə olunur.
          </p>
        </figcaption>
      </figure>
      <figure class="results__card">
        <div class="results__video-frame">
          <video
            class="results__video"
            data-autoplay-in-view
            muted
            loop
            playsinline
            preload="metadata"
            poster="${BASE_URL}media/recognise.jpg"
            aria-labelledby="results-title-1"
            aria-describedby="results-desc-1"
          >
            <source src="${BASE_URL}media/recognise.webm" type="video/webm" />
            <source src="${BASE_URL}media/recognise.mp4" type="video/mp4" />
          </video>
        </div>
        <figcaption>
          <h3 class="results__card-title" id="results-title-1">Tanınma çıxışı</h3>
          <p class="results__card-desc" id="results-desc-1">
            Tanınan işarənin mətn və səs çıxışına çevrilməsi.
          </p>
        </figcaption>
      </figure>
    </div>
  </div>
</section>

<section class="impact" id="impact" aria-label="Təsir">
  <div class="line-carousel">
    <p class="line-carousel__line">
      Azərbaycanda eşitmə məhdudiyyətli şəxslərin ünsiyyət qurması ya
      tərcüməçilərdən, ya da ailə üzvlərindən asılıdır.
    </p>
    <p class="line-carousel__line">
      Chevir bunu dəyişdirir və ana dili işarə dili olanları cəmiyyətə
      geri qazandırmağı hədəfləyir.
    </p>
  </div>
</section>

<section class="roadmap" aria-labelledby="roadmap-heading">
  ${logoBackgroundMarkup('right')}
  <div class="reveal">
    <h2 id="roadmap-heading" class="roadmap__heading">Mərhələlər</h2>
    <ol class="roadmap__list">
      <li class="roadmap__item">
        <span class="roadmap__number">01</span>
        <h3 class="roadmap__item-title">Veb platformalar üçün tərcümə API-si</h3>
        <p class="roadmap__item-desc">
          Bank, dövlət və səhiyyə sistemlərinin birbaşa inteqrasiya edə
          biləcəyi tərcümə xidməti.
        </p>
      </li>
      <li class="roadmap__item">
        <span class="roadmap__number">02</span>
        <h3 class="roadmap__item-title">Müstəqil mobil tətbiq</h3>
        <p class="roadmap__item-desc">
          Eşitmə məhdudiyyətli və eşidən istifadəçilərin birbaşa öz telefonundan istifadə edə
          biləcəyi tətbiq.
        </p>
      </li>
      <li class="roadmap__item">
        <span class="roadmap__number">03</span>
        <h3 class="roadmap__item-title">Platforma və dataset genişlənməsi</h3>
        <p class="roadmap__item-desc">
          Daha geniş işarə ehtiyatı və yeni istifadə sahələri üçün dataset
          böyüməsi.
        </p>
      </li>
    </ol>
  </div>
</section>

<section class="trust" aria-labelledby="trust-heading">
  <div class="reveal">
    <div class="trust__intro">
      <h2 id="trust-heading" class="trust__heading">Arxamızda nə dayanır</h2>
      <p class="trust__lead">
        Chevir sıfırdan başlamır - arxasında dərc olunmuş tədqiqat, açıq dataset
        və icma ilə real əməkdaşlıq var.
      </p>
    </div>
    <div class="trust__grid">
      <div class="trust__item">
        <h3 class="trust__item-title">AzSLD dataseti</h3>
        <p class="trust__item-desc">Azərbaycan jest dili üçün ilk açıq dataset.</p>
        <p class="trust__item-meta">
          Alishzade, N. &amp; Hasanov, J. (2025), Data in Brief, DOI:
          10.1016/j.dib.2024.111230
        </p>
      </div>
      <div class="trust__item">
        <h3 class="trust__item-title">Akademik baza</h3>
        <p class="trust__item-desc">
          Qarabağ Universiteti, Bakı Dövlət Universiteti, AMEA Molekulyar
          Biologiya İnstitutu, MRC LMB Cambridge.
        </p>
      </div>
      <div class="trust__item">
        <h3 class="trust__item-title">İcma əməkdaşlığı</h3>
        <p class="trust__item-desc">"Karlara Dəstək" İctimai Birliyi.</p>
      </div>
    </div>
  </div>
</section>

<section class="team" aria-labelledby="team-heading">
  ${dotsBackgroundMarkup('left')}
  <div class="reveal">
    <h2 id="team-heading" class="team__heading">Komanda</h2>
    <div class="team__grid">
      <div class="team__card">
        <h3 class="team__name">Sitara Aghayeva</h3>
        <p class="team__role">Baş İcraçı Direktor</p>
        <p class="team__bio">
          Data Science və maşın öyrənməsi üzrə yeddi, risk idarəçiliyi üzrə
          dörd il təcrübə. IU Beynəlxalq Tətbiqi Elmlər Universitetində
          magistr, beynəlxalq Risk və Süni İntellekt sertifikatı sahibi.
          Chevir-in icrasına və pilot tərəfdaşlıqlarına rəhbərlik edir.
        </p>
      </div>
      <div class="team__card">
        <h3 class="team__name">Lala Ibadullayeva</h3>
        <p class="team__role">Baş Tədqiqat Direktoru</p>
        <p class="team__bio">
          AMEA Molekulyar Biologiya İnstitutunda hesablamalı struktur
          biologiyası üzrə PhD namizədi, Bakı Dövlət Universitetində süni
          intellekt həlləri qurur. Cambridge MRC LMB-də tədqiqat təcrübəsi.
          Chevir-in hərəkət tanıma və jest tərcüməsi modellərinin dərin
          öyrənmə tədqiqatına rəhbərlik edir.
        </p>
      </div>
      <div class="team__card">
        <h3 class="team__name">Nigar Alishzade</h3>
        <p class="team__role">Baş Texnologiya Direktoru</p>
        <p class="team__bio">
          PhD namizədi, Qarabağ Universitetində kompüter elmləri müəllimi və
          "Karlara Dəstək" İctimai Birliyinin icraçı üzvü. AzSLD datasetinin
          və bir sıra jest dili tanıma məqalələrinin müəllifidir. Chevir-in
          texnologiya inkişafına və icma ilə əlaqələrinə rəhbərlik edir.
        </p>
      </div>
    </div>
  </div>
</section>

<footer class="contact" id="pilot" aria-labelledby="contact-heading">
  <div class="contact__bg" aria-hidden="true"></div>
  <div class="contact__grain" aria-hidden="true"></div>
  <svg class="contact__wordmark" viewBox="0 0 81 24" aria-hidden="true">${CONTACT_WORDMARK_MARKUP}</svg>
  <div class="reveal">
    <span class="visually-hidden">Chevir</span>
    <div class="contact__content">
      <h2 id="contact-heading" class="contact__heading">Bizə nə lazımdır</h2>
      <ul class="contact__asks">
        <li>Pilot tərəfdaşlar - dövlət qurumu, bank və ya xəstəxana.</li>
        <li>Eşitmə məhdudiyyətli icma ilə davamlı əməkdaşlıq.</li>
      </ul>
      <a class="contact__cta" href="mailto:devteam@chevirapp.com">devteam@chevirapp.com</a>
      <div class="contact__meta">
        <p class="contact__meta-copyright">© 2026 Chevir · chevirapp.com</p>
        <p class="contact__meta-slogan">Biz olmadan bizim haqqımızda heç nə.</p>
      </div>
    </div>
  </div>
</footer>
`

function setupReveal() {
  const targets = document.querySelectorAll('.reveal')
  if (!targets.length) return

  if (prefersReducedMotion) {
    targets.forEach((el) => el.classList.add('is-visible'))
    return
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible')
          observer.unobserve(entry.target)
        }
      })
    },
    { threshold: 0.15 }
  )
  targets.forEach((el) => observer.observe(el))
}

function setupResultsVideos() {
  const videos = document.querySelectorAll('.results__video[data-autoplay-in-view]')
  if (!videos.length) return

  // Videolar kiçik mənbədən gəlir; çərçivə böyüyə bilər, amma video özü öz
  // doğma piksel ölçüsündən böyüməsin (böyüdükdə bulanıqlaşır).
  const capNativeSize = (video) => {
    if (!video.videoWidth || !video.videoHeight) return
    video.style.maxWidth = `min(100%, ${video.videoWidth}px)`
    video.style.maxHeight = `min(100%, ${video.videoHeight}px)`
  }

  videos.forEach((video) => {
    if (video.readyState >= 1) capNativeSize(video)
    else video.addEventListener('loadedmetadata', () => capNativeSize(video), { once: true })
  })

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.play().catch(() => {})
        } else {
          entry.target.pause()
        }
      })
    },
    { threshold: 0.1, rootMargin: '200px' }
  )
  videos.forEach((video) => observer.observe(video))
}

// Fərqli sürətdə, scroll-scrub ilə idarə olunan fon dekorları (tam loqo
// forması / əl nöqtə buludu). Avtomatik hərəkət yoxdur - yalnız scroll ilə.
// "logo" tipi ardıcıl bölmələrdə istiqaməti növbələşir (saat / saat əksinə).
const SECTION_BG_CONFIG = [
  { selector: '.problem', type: 'logo', rotate: 150 },
  { selector: '.usecases', type: 'dots', distance: 34, rotate: -40 },
  { selector: '.results', type: 'logo', rotate: -150 },
  { selector: '.roadmap', type: 'logo', rotate: 150 },
  { selector: '.team', type: 'dots', distance: 30, rotate: 42 },
]

function setupSectionBackgrounds() {
  if (prefersReducedMotion) return

  SECTION_BG_CONFIG.forEach(({ selector, type, distance, rotate }) => {
    const section = document.querySelector(selector)
    if (!section) return

    const scrollTrigger = {
      trigger: section,
      start: 'top bottom',
      end: 'bottom top',
      scrub: 1.3,
    }

    if (type === 'logo') {
      // Mərkəzi nöqtə (viewBox 50,50) ətrafında fırlanır. CSS transform-box:
      // view-box etibarsız çıxdı (qrupu SVG-nin öz sərhədindən kənara
      // fırladıb görünməz edirdi) - svgOrigin GSAP-ın SVG-üçün xüsusi,
      // etibarlı mexanizmidir (preloader.js-də də eyni texnika işlədilib).
      const group = section.querySelector('.section-bg__logo-group')
      gsap.fromTo(
        group,
        { rotate: 0, svgOrigin: '50 50' },
        { rotate, ease: 'none', scrollTrigger }
      )
    } else {
      const cloud = section.querySelector('.section-bg__cloud')
      gsap.fromTo(
        cloud,
        { yPercent: -distance, rotate: 0 },
        { yPercent: distance, rotate, ease: 'none', scrollTrigger }
      )
    }
  })
}

// .section-bg-lər dar (max-width) mərkəzləşmiş bölmələrin içindən ekranın
// həqiqi kənarına "full-bleed" çıxır. CSS-də 100vw/-50vw ilə edilsəydi,
// klassik scroll zolaqlı brauzerlərdə (100vw zolağı daxil edir,
// clientWidth isə yox) səhifədə üfüqi scroll yaranardı - bax content.css-
// dəki şərh. Əvəzinə clientWidth-ə əsaslanan piksel dəyərləri JS ilə tətbiq
// olunur.
//
// Bu tək başına kifayət deyildi - səbəb TİMİNQ idi: bu funksiya səhifə
// yüklənən kimi işə düşür, o an hələ `html.preloading` aktivdir (base.css:
// `overflow:hidden`), yəni səhifə hələ scroll-lana bilməz və brauzer heç bir
// scroll zolağı ayırmır - clientWidth bu anda TAM pəncərə enidir. Preloader
// bitəndə `preloading` sinfi silinir, səhifə real scroll-lanan olur və
// (klassik/enlik tutan scroll zolaqlı sistemlərdə) zolaq görünür -
// clientWidth bir neçə piksel KİÇİLİR, amma artıq tətbiq olunmuş inline
// width/left köhnə (daha geniş) ölçüdə qalıb qalır. ResizeObserver
// documentElement-in content-box ölçüsündəki İSTƏNİLƏN dəyişikliyi (window
// resize-dən əlavə, məhz scroll zolağının görünüb-yox olmasını da) tutur və
// yenidən tətbiq edir.
function setupFullBleedBackgrounds() {
  const bgs = Array.from(document.querySelectorAll('.section-bg'))
  if (!bgs.length) return

  function update() {
    const vw = document.documentElement.clientWidth
    bgs.forEach((bg) => {
      const parentLeft = bg.parentElement.getBoundingClientRect().left
      bg.style.width = `${vw}px`
      bg.style.left = `${-parentLeft}px`
    })
  }

  update()
  window.addEventListener('resize', update, { passive: true })

  if (window.ResizeObserver) {
    const ro = new ResizeObserver(update)
    ro.observe(document.documentElement)
  }
}

export function initContent() {
  setupReveal()
  setupResultsVideos()
  setupSectionBackgrounds()
  setupFullBleedBackgrounds()
  // Təsir bölməsi müvəqqəti gizlədilib (bax statement.css: .impact{display:none}).
  // Geri qaytarmaq üçün aşağıdakı sətri aç.
  // initLineCarousel('.impact')
}
