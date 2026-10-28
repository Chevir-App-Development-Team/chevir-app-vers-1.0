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
  const tabpanels = root.querySelector('.fit-tabpanels');
  if (!tabs.length) return;

  // Yalnız HƏQİQƏTƏN daşıyanda (tab keçidindən/məzmun dəyişməsindən sonra)
  // alt kənar fade-i göstərir - ResizeObserver tək özü kifayət deyil, çünki
  // .fit-tabpanels-in öz qutusu tab keçidində dəyişmir (yalnız içindəki
  // hansı panelin görünməsi dəyişir), ona görə hər activate() çağırışında
  // əlavə yoxlanılır.
  const syncOverflow = () => {
    if (tabpanels) tabpanels.classList.toggle('has-overflow', tabpanels.scrollHeight > tabpanels.clientHeight + 1);
  };

  const activate = (name) => {
    tabs.forEach((t) => {
      const on = t.dataset.fitTab === name;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
    });
    panels.forEach((p) => p.classList.toggle('is-active', p.dataset.fitTabpanel === name));
    syncOverflow();
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

  // Eyni tab aktiv qalarkən məzmun dəyişsə (söz axtarışı, proqnoz/kandidat
  // siyahısı yenilənsə) da daşma vəziyyətini yenidən yoxlayır.
  if (tabpanels) new ResizeObserver(syncOverflow).observe(tabpanels);
  panels.forEach((p) => new MutationObserver(syncOverflow).observe(p, { childList: true, subtree: true }));
}

// #spelled yalnız CSS ilə üfüqi sürüşən cərgəyə çevrilib (fit rejimində);
// app.js heç bir scroll/scrollIntoView etmir (yoxlandı), ona görə aktiv
// tokeni (.now) mərkəzdə saxlamaq üçün sırf vizual scroll buradan idarə
// olunur - heç bir data oxunmur/yazılmır, yalnız artıq mövcud .now sinfi
// izlənir və DOM-un öz scrollLeft-i/padding-i dəyişdirilir.
//
// Sabit "oynatma nöqtəsi": CSS-də #spelled-ə padding-left:50% qoyulub
// (zolağın sol yarısı həmişə boş, ilk token mərkəzdən başlayır). Sığma
// qərarı YALNIZ məzmun dəyişəndə (childList) bir dəfə verilir və `overflowing`
// dəyişənində yaddaşda saxlanılır - aktiv token dəyişəndə (sadəcə class
// mutasiyası) təkrar hesablanmır, ona görə oynatma zamanı rejim titrəməsi
// olmur.
function initSpelledAutoScroll() {
  const spelled = document.querySelector('#spelled');
  if (!spelled) return;
  let overflowing = false;

  const recomputeMode = () => {
    const items = spelled.querySelectorAll('i');
    if (!items.length) {
      overflowing = false;
      spelled.style.paddingRight = '';
      spelled.scrollLeft = 0;
      return;
    }
    // Təbii məzmun eni - getBoundingClientRect ilə (padding-left:50%-dən
    // sonrakı mövqeyə görə deyil, sırf tokenlərin öz aralarındakı enə görə).
    const first = items[0].getBoundingClientRect();
    const last = items[items.length - 1].getBoundingClientRect();
    const contentWidth = last.right - first.left;
    // padding-left:50% artıq zolağın yarısını tutduğu üçün, məzmun yalnız
    // QALAN yarıya (clientWidth/2) sığmalıdır - sığırsa heç nə sürüşmür.
    overflowing = contentWidth > spelled.clientWidth / 2;
    spelled.style.paddingRight = overflowing ? `${Math.round(spelled.clientWidth / 2)}px` : '';
    if (!overflowing) spelled.scrollLeft = 0;
  };

  const centerActive = () => {
    if (!overflowing) return; // sığan mətndə heç vaxt sürüşmə yoxdur
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
  // yeni mətn yazılanda sığma qərarı dərhal yenidən hesablansın.
  new MutationObserver((mutations) => {
    if (mutations.some((m) => m.type === 'childList')) recomputeMode();
    centerActive();
  }).observe(spelled, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ['class'],
  });
  recomputeMode();
}

// OYNATMA qrupunun (#topbar-ctrl-t2s) sol kənarını stage-in 2-ci panelinin
// (skeleton görüntüsü) sol kənarı ilə piksel-dəqiq eyniləşdirir. CSS-də eyni
// grid-template-columns-u təkrarlamaq (stage 1fr/1fr + panel 1fr/340px iç-içə)
// mümkün olsa da kövrək olardı (fərqli gap-lar: 14px stage, 16px panel) - ona
// görə faktiki render olunmuş mövqeyə görə margin-left hesablanır, pəncərə
// ölçüsü dəyişəndə yenidən ölçülür.
function alignOynatmaToStage() {
  const group = document.querySelector('#topbar-ctrl-t2s');
  const stage = document.querySelector('#tab-t2s .stage');
  const panel = document.querySelector('#tab-t2s');
  if (!group || !stage || !panel) return;

  const sync = () => {
    if (!document.body.classList.contains('is-fit')) {
      group.style.marginLeft = '';
      return;
    }
    const secondViewport = stage.querySelectorAll('.viewport')[1];
    if (!secondViewport) return;
    group.style.marginLeft = '0px'; // təbii mövqeyi ölçmək üçün sıfırla
    const naturalLeft = group.getBoundingClientRect().left;
    const targetLeft = secondViewport.getBoundingClientRect().left;
    const delta = Math.round(targetLeft - naturalLeft);
    // Məhdudiyyət yoxdur: dar ekranlarda (məs. 1280x600) stage-in sütunları
    // topbar-ın sabit-enli uşaqlarından (tablar/status) daha sürətlə daralır,
    // bu da target-i natural mövqedən SOLA keçirə bilər (delta mənfi olur) -
    // köhnə `Math.max(16, delta)` bunu +16-ya məcburi döndərüb qrupu
    // target-dən 40px+ sağa itələyirdi.
    group.style.marginLeft = `${delta}px`;
  };

  new ResizeObserver(sync).observe(stage);
  window.addEventListener('resize', sync);
  // Panel `#tab-t2s` ilkin yüklənmədə `display:none`-dır (is-active deyil) -
  // o anda stage-in 2-ci viewport-u ölçüsüz olur (0,0,0,0), ona görə ilkin
  // sync() səhv/köhnə qiymət qoyur və sonra heç nə onu düzəltmir (stage-in
  // öz qutusu "is-active" olandan sonra dəyişmir, ResizeObserver yenidən
  // işə düşmür). Düzgün tətik nöqtəsi: panel HƏQİQƏTƏN is-active olanda
  // (mövcud `initTopbarControls`-un izlədiyi eyni .panel class atributu).
  new MutationObserver(sync).observe(panel, { attributes: true, attributeFilter: ['class'] });
  // Kök səbəb (uzun axtarışla tapıldı): tab t2s-ə keçəndə app.js `#status-text`-i
  // "Avatar yüklənir…" sonra "Avatar hazır" edir (VRM asinxron yüklənir). `.status`
  // elementi `.topbar`-da #topbar-ctrl-t2s-dən ƏVVƏL gəlir və qrup flex:1-dir - mətn
  // eni dəyişəndə qrupun başladığı nöqtə sürüşür, amma bunu heç bir ResizeObserver/
  // panel-class müşahidəçisi tutmur (stage-in öz qutusu dəyişmir). Ona görə status
  // mətnini də izləyirik.
  const statusText = document.querySelector('#status-text');
  if (statusText) new MutationObserver(sync).observe(statusText, { characterData: true, childList: true, subtree: true });
  sync();
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
  alignOynatmaToStage();
  initFit();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
