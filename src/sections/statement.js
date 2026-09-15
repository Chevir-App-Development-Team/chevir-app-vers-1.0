import { initLineCarousel } from '../lib/lineCarousel.js'

export const statementMarkup = `
<section class="statement" id="statement" aria-label="Missiyamız">
  <div class="line-carousel">
    <p class="line-carousel__line">
      Chevir işarə dilini mətnə, mətni isə işarə dilinə çevirən süni
      intellekt əsaslı proqram həllidir.
    </p>
    <p class="line-carousel__line">
      Azərbaycanda 13 000 nəfər eşitmə məhdudiyyətli şəxs olduğunu nəzərə
      alaraq,
    </p>
    <p class="line-carousel__line">
      onların Azərbaycan İşarə dilində ünsiyyət qurmasını asanlaşdırmaq
      məqsədilə yaradılıb.
    </p>
  </div>
</section>
`

export function initStatement() {
  initLineCarousel('.statement')
}
