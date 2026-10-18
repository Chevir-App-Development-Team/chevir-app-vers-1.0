# Chevir — brend qaydaları

## Rəng

```
--chevir-blue       #0071EF   əsas vurğu - düymə, link, kanal A (həmişə sabit)
--bridge-blue       #0021BF   brend qradiyentin ikinci ucu, kanal B açıq fonda
--midnight          #091638   tünd bölmə fonu + açıq bölmənin mətn rəngi
--cloud              #F3F7FC   açıq bölmə fonu
--accent-b-on-dark  #6FB1FF   kanal B tünd fonda (Bridge Blue Midnight-da AA-nı
                              keçmir - kontrast 1.66; bu tonun kontrastı 7.95)
```

Brend qradiyenti: `linear-gradient(90deg, #0071EF, #0021BF)`.

**Kanal məntiqi.** Loqo iki əks istiqamətli, qradiyentli damladır; bu simvolik
cüt Chevir Blue / `--accent-b-on-dark` (və ya açıq fonda Bridge Blue) ilə
ifadə olunur - **yalnız qrafik elementlərdə**: 3D əlin iki tərəfi, Loop
diaqramının iki oxu/marker-i, Pipeline addım nömrələri, fon sxelet nöqtələri.
Heç vaxt override OLUNMUR - hər istehlakçı öz fonuna (tünd/açıq) uyğun sabiti
birbaşa seçir. **Kiçik mətndə işlədilmir**: Chevir Blue kiçik mətndə AA-nı
keçmir (ölçülüb: 3.89/4.24, tələb 4.5) - yalnız iri rəqəm/başlıq və qrafik
element (ox, marker, dairə) kimi işlət, adi abzas/etiket mətnində yox.

Tünd rejim `prefers-color-scheme` ilə DƏSTƏKLƏNMİR - `color-scheme: light`
qəsdən seçilib (bölmə ritmi OS temasından asılı olmamalıdır).

## Tipoqrafiya

Bir və ya iki ailə, artıq yox. Yumşaq premium üçün geometrik/qrotesk uyğundur,
serif yox.

Şrift: **Poppins**, öz saxlanılan woff2 fayllarımızla (`public/fonts/*.woff2`,
Google Fonts YOX). `ə Ə ğ Ğ ı İ ş Ş ç Ç ö Ö ü Ü` dəstəyi `pyftsubset`-lə açıq
`--unicodes` siyahısı ilə təmin olunub - yeni çəki əlavə etsən eyni subset
siyahısını (Latin + Latin Extended-A + U+0259/U+018F schwa + ümumi
punktuasiya) işlət, ttf mənbələrini çevirdikdən sonra repodan sil.

Ölçü: `clamp()` ilə axıcı şkala. Sətir uzunluğu 80 simvoldan az.

Tire işarəsi: yalnız qısa tire (-) işlədilir, uzun tire (—) işlədilmir.

## Motion spesifikasiyası

Beş adlandırılmış an. Bunlardan artığı yoxdur - hər bölməyə effekt əlavə etmə.

**1. Preloader.** Loqonun iki damlası (gradient-dolğulu, əks istiqamətli) və
iki nöqtəsi yumşaq opacity/scale ilə özünü göstərir (stagger), sonra
naviqasiyadakı yerinə uçur. Maksimum 1 saniyə - sayğac yoxdur.

**2. Hero.** Başlıq SplitText ilə söz-söz qalxır, hərf-hərf yox (uzun
Azərbaycan sözlərində hərf-hərf oxunuşu pozur). Loqonun yerində 3D əl nöqtə
buludu (Three.js, MediaPipe 21-nöqtə sxemi) yavaş fırlanır və jest pozaları
arasında keçir; kursor/toxunma ilə orbit edilə bilər. Ekrandan çıxanda render
dayanır. reduced-motion və ya WebGL yoxdursa, statik loqo qalır.

**3. Pipeline (sticky).** Sol tərəfdə görüntü pin olunur, sağda dörd addım
scroll olunur; hər addıma çatanda görüntü dəyişir və nömrənin rəngi kanal
rənginə keçir (01-02 Chevir Blue, 03-04 `--accent-b-on-dark`).

**4. Loop diaqramı.** Bölmə pin olunur, iki ox scroll ilə əks istiqamətlərdə
çəkilir (scrub: true).

**5. Shader.** Magenta-mavi gradient, yumşaq, yavaş. Yalnız "Bizə nə lazımdır"
bölməsinin fonunda. Hero-da yox - mobil cihazlarda ağırdır.

Ümumi qaydalar: yalnız `transform` və `opacity` animasiya olunur. Keçid
müddəti 0.4-0.8s, `power2.out`. Scroll-scrub olanlarda `scrub: 1` (0.5-dən az
sərt görünür).

## Naviqasiya

Navbar həmişə tünd rejimdədir, bölmə ritminə görə `.is-light` sinfi ilə
rənglər keçir (açıq bölmə üzərində loqo/mətn/hamburger rəngi Bridge Blue-ya
dönür, CTA isə midnight-pill-ə - hero üzərindəki istisnadan başqa, bax
`navbar.css` şərhləri). Açıq fonlu yeni bölmə əlavə olunarsa, naviqasiya
keçid məntiqi (`LIGHT_SECTION_SELECTOR`, `hero.js`) yenidən nəzərdən
keçirilməlidir.

## Əlçatanlıq - güzəşt yoxdur

Bu, eşitmə məhdudiyyətli insanlar üçün qurulan məhsuldur. Aşağıdakılar
estetik seçim deyil, tələbdir.

- `prefers-reduced-motion: reduce` seçilibsə **bütün** animasiya dayanır,
  hamar-scroll söndürülür, səhifə tam funksional qalır.
- Heç bir məlumat yalnız animasiya vasitəsilə ötürülmür.
- Video səssizdir, avtomatik oynayır, `playsinline`, hər birinin yanında
  mətn izahı var.
- Fon musiqisi və səsli effekt yoxdur.
- Klaviatura naviqasiyası və görünən fokus halqası işləyir.
- Kontrast WCAG AA (bax yuxarıda "Rəng").

## Məzmun qaydaları

- Terminologiya: "kar" sözü işlədilmir. Həmişə "eşitmə məhdudiyyətli" yazılır.
- AZ və EN bərabər hüquqludur - EN "tərcümə" kimi görünməməlidir.

## Qadağan siyahısı

Bunlar generik/şablon görünüşünün mənbəyidir:

- Hər bölmədə fade-up reveal
- Eyni radius və eyni boz kölgəli identik kartlar
- ALL-CAPS eyebrow etiketləri başlıqların üstündə
- Düymə mətnində `→`
- Krem fon (#F4F1EA və ya yaxını)
- Dekorativ gradient yuyuntusu
- Başlıqda tək sözü fərqli rəngə boyamaq
- Ardıcıllıq olmayan məzmunda 01/02/03 nömrələri
- Emoji ikonlar
