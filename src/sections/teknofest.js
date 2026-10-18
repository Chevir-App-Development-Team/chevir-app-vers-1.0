import { t } from '../lib/i18n.js'

// public/ altındakı asset-lərə BASE_URL ilə istinad olunur (digər bölmələrdə
// işlədilən eyni konvensiya, bax content.js/pipeline.js).
const BASE_URL = import.meta.env.BASE_URL

// Fon şəklinin öz, həqiqi piksel ölçüsü (public/img/teknofest/teknofest-stage.webp,
// 1600x992) - img width/height atributları buradan gəlir ki, yüklənmədən
// əvvəl brauzer düzgün nisbətlə yer ayırsın (layout shift olmasın). CSS-dəki
// `width:1040px;height:auto` göstərilən ölçünü təyin edir, atributlar yalnız
// nisbət/placeholder üçündür.
const STAGE_IMG_W = 1600
const STAGE_IMG_H = 992
const TEAM_IMG_W = 1200
const TEAM_IMG_H = 900
const TEAM2_IMG_W = 1200
const TEAM2_IMG_H = 800

export const teknofestMarkup = `
<section class="teknofest" aria-labelledby="teknofest-heading">
  <div class="teknofest__bg" aria-hidden="true">
    <img
      class="teknofest__bg-img"
      src="${BASE_URL}img/teknofest/teknofest-stage.webp"
      alt="${t('home.teknofest.img.stage.alt')}"
      width="${STAGE_IMG_W}"
      height="${STAGE_IMG_H}"
      loading="lazy"
      aria-hidden="true"
    />
    <div class="teknofest__bg-overlay"></div>
  </div>
  <div class="reveal teknofest__inner">
    <div class="teknofest__text">
      <h2 id="teknofest-heading" class="teknofest__heading">
        <span class="teknofest__heading-line">${t('home.teknofest.title.line1')}</span>
        <span class="teknofest__heading-line">${t('home.teknofest.title.line2')}</span>
        <strong class="teknofest__heading-line teknofest__heading-line--bold">${t('home.teknofest.title.line3')}</strong>
      </h2>
      <p class="teknofest__paragraph">${t('home.teknofest.text')}</p>
    </div>
    <div class="teknofest__stats">
      <div class="teknofest__stat">
        <span class="teknofest__stat-value">${t('home.teknofest.stat.1.value')}</span>
        <span class="teknofest__stat-label">${t('home.teknofest.stat.1.label')}</span>
      </div>
      <div class="teknofest__stat">
        <span class="teknofest__stat-value">${t('home.teknofest.stat.2.value')}</span>
        <span class="teknofest__stat-label">${t('home.teknofest.stat.2.label')}</span>
      </div>
      <div class="teknofest__stat">
        <span class="teknofest__stat-value">${t('home.teknofest.stat.3.value')}</span>
        <span class="teknofest__stat-label">${t('home.teknofest.stat.3.label')}</span>
      </div>
    </div>
    <div class="teknofest__cards">
      <div class="teknofest__card teknofest__card--team2">
        <img
          src="${BASE_URL}img/teknofest/teknofest-team2.webp"
          alt="${t('home.teknofest.img.team2.alt')}"
          width="${TEAM2_IMG_W}"
          height="${TEAM2_IMG_H}"
          loading="lazy"
        />
      </div>
      <div class="teknofest__card teknofest__card--team">
        <img
          src="${BASE_URL}img/teknofest/teknofest-team.webp"
          alt="${t('home.teknofest.img.team.alt')}"
          width="${TEAM_IMG_W}"
          height="${TEAM_IMG_H}"
          loading="lazy"
        />
      </div>
    </div>
    <p class="teknofest__date">${t('home.teknofest.date')}</p>
  </div>
</section>
`
