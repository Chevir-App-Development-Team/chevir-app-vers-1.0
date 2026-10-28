// Masaüstü "fit" düzülüşü (Variant A): yalnız CSS sinif/aria-selected toggle edir,
// heç bir DOM elementini daşımır/dublikatlaşdırmır, s2t/t2s/ML datasına toxunmur.
const FIT_QUERY = '(min-width: 1100px) and (min-height: 560px)';

function setNavbarHeightVar() {
  const navbar = document.querySelector('[data-navbar]');
  if (!navbar) return;
  const ro = new ResizeObserver((entries) => {
    const h = entries[0].contentRect.height;
    document.documentElement.style.setProperty('--navbar-h-real', `${h}px`);
  });
  ro.observe(navbar);
}

// .panel.is-active dəyişəndə topbar-dakı uyğun idarə qrupunu göstərir (#topbar-ctrl-s2t/t2s).
function initTopbarControls() {
  const groups = {
    s2t: document.querySelector('#topbar-ctrl-s2t'),
    t2s: document.querySelector('#topbar-ctrl-t2s'),
  };
  const sync = () => {
    const active = document.querySelector('.panel.is-active');
    const name = active?.id.replace('tab-', '');
    for (const [key, el] of Object.entries(groups)) {
      if (el) el.hidden = key !== name;
    }
  };
  sync();
  document.querySelectorAll('.panel').forEach((panel) => {
    new MutationObserver(sync).observe(panel, { attributes: true, attributeFilter: ['class'] });
  });
}

// Sağ panelin yeni tablistini idarə edir (klik + ox düymələri), aria-selected/is-active toggle edir.
function initFitTabs(root) {
  const tabs = [...root.querySelectorAll('.fit-tab')];
  const panels = [...root.querySelectorAll('.fit-tabpanel')];
  if (!tabs.length) return;

  const activate = (name) => {
    tabs.forEach((t) => {
      const on = t.dataset.fitTab === name;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
    });
    panels.forEach((p) => p.classList.toggle('is-active', p.dataset.fitTabpanel === name));
  };

  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => activate(tab.dataset.fitTab));
    tab.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const dir = e.key === 'ArrowRight' ? 1 : -1;
      const next = tabs[(i + dir + tabs.length) % tabs.length];
      next.focus();
      activate(next.dataset.fitTab);
    });
  });

  // Aktiv tab-ın altında duran data-kind elementi gizlənsə (rejim dəyişəndə),
  // o tabı da gizlədir və "proqnoz"/"lugat" (hər iki panelin ilk tabı) tabına qaytarır.
  const fallback = tabs[0].dataset.fitTab;
  panels.forEach((panel) => {
    if (!panel.hasAttribute('data-kind')) return;
    const tabBtn = tabs.find((t) => t.dataset.fitTab === panel.dataset.fitTabpanel);
    const sync = () => {
      if (tabBtn) tabBtn.hidden = panel.hidden;
      if (panel.hidden && panel.classList.contains('is-active')) activate(fallback);
    };
    sync();
    new MutationObserver(sync).observe(panel, { attributes: true, attributeFilter: ['hidden'] });
  });

  activate(tabs.find((t) => t.classList.contains('is-active'))?.dataset.fitTab ?? fallback);
}

// #spelled yalnız CSS ilə üfüqi sürüşən cərgəyə çevrilib (fit rejimində);
// app.js heç bir scroll/scrollIntoView etmir (yoxlandı), ona görə aktiv
// tokeni (.now) mərkəzdə saxlamaq üçün sırf vizual scroll buradan idarə
// olunur - heç bir data oxunmur/yazılmır, yalnız artıq mövcud .now sinfi
// izlənir və DOM-un öz scrollLeft-i/padding-i dəyişdirilir.
function initSpelledAutoScroll() {
  const spelled = document.querySelector('#spelled');
  if (!spelled) return;
  const centerActive = () => {
    const items = spelled.querySelectorAll('i');
    if (!items.length) { spelled.style.paddingRight = ''; return; }
    // Təbii məzmun eni - getBoundingClientRect ilə (özümüzün qoyduğumuz
    // paddingRight-dan asılı olmayan, etibarlı ölçü), hələ padding
    // əlavə edilmədən öncəki vəziyyəti əks etdirir.
    const first = items[0].getBoundingClientRect();
    const last = items[items.length - 1].getBoundingClientRect();
    const contentWidth = last.right - first.left;
    const overflowing = contentWidth > spelled.clientWidth;

    // Padding YALNIZ daşanda əlavə olunur - sabit/həmişə-aktiv padding qısa
    // mətndə tokenləri ortada "üzdürüb" #topbar-ctrl-t2s-i yeni sətrə salırdı.
    spelled.style.paddingRight = overflowing ? `${Math.round(spelled.clientWidth / 2)}px` : '';

    if (!overflowing) {
      spelled.scrollLeft = 0;
      return;
    }
    const now = spelled.querySelector('i.now');
    if (!now) return;
    // offsetLeft mövqelənmiş əcdada görədir (burada #spelled deyil), ona görə
    // getBoundingClientRect ilə ekran koordinatına görə hesablanır - CSS
    // kontekstindən asılı olmayan, etibarlı üsul.
    const nowRect = now.getBoundingClientRect();
    const spelledRect = spelled.getBoundingClientRect();
    const delta = (nowRect.left + nowRect.width / 2) - (spelledRect.left + spelledRect.width / 2);
    const target = spelled.scrollLeft + delta;
    const max = spelled.scrollWidth - spelled.clientWidth;
    spelled.scrollTo({ left: Math.min(Math.max(0, target), max), behavior: 'smooth' });
  };
  // childList: mətn dəyişəndə #spelled-in bütün tərkibi innerHTML ilə
  // əvəz olunur (sinif mutasiyası deyil) - bunu da tutmaq lazımdır ki,
  // yeni mətn yazılanda padding/scroll dərhal yenidən hesablansın.
  new MutationObserver(centerActive).observe(spelled, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ['class'],
  });
}

function initFit() {
  const mq = window.matchMedia(FIT_QUERY);
  const apply = (matches) => document.body.classList.toggle('is-fit', matches);
  apply(mq.matches);
  mq.addEventListener('change', (e) => {
    apply(e.matches);
    // CSS dəyişikliyindən sonra skeleton.js/vrm.js öz mövcud `resize` listener-ləri
    // ilə canvas ölçüsünü yeniləsin - heç bir fayla toxunmadan.
    window.dispatchEvent(new Event('resize'));
  });
}

function boot() {
  setNavbarHeightVar();
  initTopbarControls();
  document.querySelectorAll('#tab-s2t .fit-right, #tab-t2s .fit-right').forEach(initFitTabs);
  initSpelledAutoScroll();
  initFit();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
