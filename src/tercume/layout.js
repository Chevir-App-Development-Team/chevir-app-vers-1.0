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
  initFit();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
