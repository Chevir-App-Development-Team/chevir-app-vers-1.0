import { t } from '../lib/i18n.js'

// public/ altındakı asset-lərə BASE_URL ilə istinad olunur (bax teknofest.js).
const BASE_URL = import.meta.env.BASE_URL

// Şəklin öz, həqiqi piksel ölçüsü (public/img/bakuid/seeder-top-startups.png,
// 390x487) - layout shift olmasın deyə atribut kimi qoyulur.
const IMG_W = 390
const IMG_H = 487

export const bakuidMarkup = `
<section class="bakuid" aria-labelledby="bakuid-heading">
  <div class="bakuid__glow" aria-hidden="true"></div>
  <div class="reveal bakuid__inner">
    <div class="bakuid__text">
      <h2 id="bakuid-heading" class="bakuid__heading">
        <span class="bakuid__heading-line">${t('home.bakuid.title.line1')}</span>
        <strong class="bakuid__heading-line bakuid__heading-line--bold">${t('home.bakuid.title.line2')}</strong>
      </h2>
      <p class="bakuid__paragraph">${t('home.bakuid.text')}</p>
      <span class="bakuid__badge">${t('home.bakuid.badge')}</span>
    </div>
    <div class="bakuid__shot">
      <img
        src="${BASE_URL}img/bakuid/seeder-top-startups.png"
        alt="${t('home.bakuid.img.alt')}"
        width="${IMG_W}"
        height="${IMG_H}"
        loading="lazy"
      />
    </div>
  </div>
</section>
`
