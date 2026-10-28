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

// OYNATMA token cərgəsinin (#spelled) tərpənişini idarə edir. app.js heç bir
// scroll/scrollIntoView etmir (yoxlandı) - #spelled-in öz UŞAQLARINA
// (app.js-in <i> elementlərinə) TOXUNULMUR, yalnız #spelled-in özü BİR DƏFƏ
// yeni `.spelled-clip` valideyninə sarılır (köhnə `.spelled{overflow-x:auto}`
// memarlığı böyük məzmunda flex-sətri konteynerdən daşırıb #current-letter-i
// ekrandan kənara itələyirdi - kök səbəb; indi klip xalis overflow:hidden
// qutudur, məzmun eni ona təsir etmir, #spelled transform:translateX ilə
// sürüşür).
//
// İki mərhələ:
//  A) yazma/boş (mode='idle') - L-R arasında P-nöqtəli böyümə (aşağıda).
//  B) oynatma (mode='playing') - aktiv token sabit ekran nöqtəsində qalır,
//     sətir onun ətrafında sürüşür; başlayanda SIÇRAYIŞ yoxdur (cari mövqe
//     olduğu kimi "oynatma nöqtəsi" kimi yadda saxlanılır).
//
// L/R/C definisiyaları (klip pəncərəsinin öz LOKAL koordinatında, 0 = klip
// pəncərəsinin öz sol kənarı):
//   L_local = 24   (sayğac-dan sonra 12px gap + 24px fade zonası artıq
//                   #topbar-ctrl-t2s-in gap:12px-i ilə klipdən ƏVVƏL verilir,
//                   klip daxilində isə yalnız 24px fade zonası qalır)
//   R_local = klip pəncərəsinin öz eni (WW) - sağ fade zonası simmetrik
//             tətbiq olunur, amma R-in özü WW-dir (tapşırıqdakı tərifə görə)
//   C_local = (L_local + R_local) / 2
function initSpelledTimeline() {
  const spelled = document.querySelector('#spelled');
  const group = document.querySelector('#topbar-ctrl-t2s');
  const panel = document.querySelector('#tab-t2s');
  const playProgress = document.querySelector('#play-progress');
  if (!spelled || !group || !panel) return;

  // bir dəfəlik sarma: #spelled-in öz valideynini #spelled-clip ilə əvəz
  // edir, #spelled-i onun daxilinə köçürür - #spelled-in uşaqları toxunulmur.
  let clip = spelled.parentElement;
  if (!clip || !clip.classList.contains('spelled-clip')) {
    clip = document.createElement('div');
    clip.className = 'spelled-clip';
    spelled.parentNode.insertBefore(clip, spelled);
    clip.appendChild(spelled);
  }

  const L_LOCAL = 24;
  let WW = 0;
  let R_LOCAL = 0;
  let C_LOCAL = 0;
  let currentOffset = 0;
  let pAnchor = null; // "P": yazma mərhələsində sətrin sağ kənarının sabit qaldığı nöqtə
  let playAnchor = null; // oynatma mərhələsində aktiv tokenin sabit qaldığı ekran nöqtəsi
  let mode = 'idle'; // 'idle' | 'playing'

  const recomputeBounds = () => {
    WW = clip.getBoundingClientRect().width;
    R_LOCAL = WW;
    C_LOCAL = (L_LOCAL + R_LOCAL) / 2;
  };

  const updateFade = (offset, contentWidth) => {
    const leftFade = offset < -0.5;
    const rightFade = offset + contentWidth > WW + 0.5;
    let mask = 'none';
    if (leftFade && rightFade) {
      mask = 'linear-gradient(90deg, transparent, #000 24px, #000 calc(100% - 24px), transparent)';
    } else if (leftFade) {
      mask = 'linear-gradient(90deg, transparent, #000 24px, #000 100%)';
    } else if (rightFade) {
      mask = 'linear-gradient(90deg, #000 0%, #000 calc(100% - 24px), transparent)';
    }
    clip.style.webkitMaskImage = mask;
    clip.style.maskImage = mask;
  };

  const moveSpelledTo = (offsetPx, animate) => {
    // Bu bütün yenidən-yerləşdirmə məntiqi yalnız fit rejimində mənalıdır
    // (#spelled-clip-ə position:absolute/transform yalnız body.is-fit
    // altında verilir) - fit olmayanda inline transform-u təmizləyib
    // bazanın öz flex-wrap sətir düzülüşünə qarışmırıq (əks halda inline
    // transform qeyri-mövqelənmiş elementi də sürüşdürüb cərgələri
    // üst-üstə salırdı).
    if (!document.body.classList.contains('is-fit')) {
      spelled.style.transform = '';
      clip.style.webkitMaskImage = '';
      clip.style.maskImage = '';
      return;
    }
    currentOffset = offsetPx;
    if (!animate) spelled.classList.add('no-anim');
    spelled.style.transform = `translate(${offsetPx}px, -50%)`;
    updateFade(offsetPx, spelled.scrollWidth);
    if (!animate) {
      void spelled.offsetWidth; // transition:none-u tətbiq etmək üçün reflow məcburiləşdirilir
      requestAnimationFrame(() => spelled.classList.remove('no-anim'));
    }
  };

  // A mərhələsi: tək token həmişə C-də mərkəzlənir (bu an "P" - sətrin sağ
  // kənarı - yenidən təyin olunur). Çox tokenli halda P SABİT qalır, sətir
  // sola doğru böyüyür; L-ə dəyəndə (offset < L_LOCAL) klamp olunur və
  // bundan sonra yazılan tokenlər sağa əlavə olunur (silinəndə tərsinə işləyir,
  // çünki Math.max(L_LOCAL, pAnchor - W) hər iki istiqamətdə simmetrikdir).
  const layoutPhaseA = (animate) => {
    const items = spelled.querySelectorAll('i');
    if (!items.length) {
      pAnchor = null;
      moveSpelledTo(0, false);
      return;
    }
    const W = spelled.scrollWidth;
    let offset;
    if (items.length === 1) {
      offset = C_LOCAL - W / 2;
      pAnchor = offset + W;
    } else {
      if (pAnchor == null) pAnchor = L_LOCAL + W; // ehtiyat: birbaşa çox-tokenli başlanğıc
      offset = Math.max(L_LOCAL, pAnchor - W);
    }
    moveSpelledTo(offset, animate);
  };

  // B mərhələsi başlayır: SIÇRAYIŞ YOXDUR - cari mövqe olduğu kimi qalır,
  // yalnız aktiv tokenin bu andakı ekran-lokal mərkəzi "oynatma nöqtəsi"
  // kimi yadda saxlanılır.
  const capturePlayAnchor = () => {
    const now = spelled.querySelector('i.now');
    if (!now) { playAnchor = null; return; }
    playAnchor = currentOffset + now.offsetLeft + now.offsetWidth / 2;
  };

  // B mərhələsi davam edir: yeni aktiv token playAnchor nöqtəsində qalacaq
  // şəkildə sətir sürüşür (hamar, CSS transition: transform 0.2s).
  const trackPlayAnchor = (animate) => {
    const now = spelled.querySelector('i.now');
    if (!now || playAnchor == null) return;
    const tokenCenterAtZero = now.offsetLeft + now.offsetWidth / 2;
    moveSpelledTo(playAnchor - tokenCenterAtZero, animate);
  };

  // Bir neçə müşahidəçi (childList, resize, panel-class) eyni tick-də
  // ardıcıl tətikə bilər (məs. ResizeObserver-in sub-piksel yenidən
  // hesablaması ilə mətn dəyişikliyi üst-üstə düşəndə) - əgər hər biri
  // BİRBAŞA layoutPhaseA/trackPlayAnchor çağırsaydı, sıra-asılı yarış
  // vəziyyəti yaranıb köhnə hesablama sonuncu tətbiq oluna bilərdi. Ona görə
  // bütün tətiklər TƏK bir rAF-planlaşdırılan yenidən-hesablamaya yönəlir -
  // faktiki tətbiq həmişə həmin an-ın ƏN SON DOM/mode vəziyyətinə görədir.
  let rafId = 0;
  const scheduleRecompute = (animate) => {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(() => {
      rafId = 0;
      if (mode === 'playing') trackPlayAnchor(animate);
      else layoutPhaseA(animate);
    });
  };

  // #play-progress mətni "N/M" olanda oynanır, əks halda ('hazır', '-') yox -
  // onLetter (.now/.done class) və onState (#play-progress mətni) app.js-də
  // eyni addımda, ardıcıl sinxron çağırılır, ona görə bu tək siqnal kifayətdir.
  const onPlayStateChange = () => {
    const text = playProgress?.textContent ?? '';
    const nowPlaying = /^\d+\/\d+$/.test(text);
    if (nowPlaying && mode !== 'playing') {
      mode = 'playing';
      capturePlayAnchor(); // B1: heç bir sıçrayış/hərəkət yoxdur - heç nə planlaşdırılmır
    } else if (!nowPlaying && mode === 'playing') {
      mode = 'idle';
      scheduleRecompute(true); // B5: dayananda/bitəndə A qaydası ilə hamar geri qayıdış
    } else if (nowPlaying) {
      scheduleRecompute(true); // B2: addım irəlilədikcə izləmə
    }
  };

  // mətn dəyişəndə (#spelled-in bütün tərkibi innerHTML ilə əvəz olunur - bu
  // childList mutasiyasıdır, class mutasiyası deyil) yalnız A mərhələsində
  // yenidən yerləşdirilir (oynatma zamanı yazma baş vermir).
  new MutationObserver(() => {
    if (mode === 'idle') scheduleRecompute(true);
  }).observe(spelled, { childList: true });

  if (playProgress) {
    new MutationObserver(onPlayStateChange).observe(playProgress, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  }

  // Pəncərə/panel ölçüsü dəyişəndə (və ya panel ilk dəfə is-active olanda,
  // klip hələ display:none ikən 0 ölçü ala bilər) hüdudlar yenidən hesablanır
  // və cari mərhələ animasiyasız yenidən tətbiq olunur (mövqe dəyişməsin,
  // yalnız yeni hüduda uyğunlaşsın).
  const resync = () => {
    recomputeBounds();
    scheduleRecompute(false);
  };

  new ResizeObserver(resync).observe(clip);
  new MutationObserver(resync).observe(panel, { attributes: true, attributeFilter: ['class'] });
  window.addEventListener('resize', resync);
  recomputeBounds();
  if (mode === 'playing') trackPlayAnchor(false);
  else layoutPhaseA(false);
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
  initSpelledTimeline();
  alignOynatmaToStage();
  initFit();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
