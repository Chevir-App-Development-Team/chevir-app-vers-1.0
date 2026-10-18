import { t } from '../lib/i18n.js'

export const statementMarkup = `
<section class="statement" id="statement" aria-label="${t('home.statement.ariaLabel')}">
  <div class="reveal statement__content">
    <p class="statement__line">
      ${t('home.statement.line1')}
    </p>
    <p class="statement__line">
      ${t('home.statement.line2')}
    </p>
    <p class="statement__line">
      ${t('home.statement.line3')}
    </p>
  </div>
</section>
`
