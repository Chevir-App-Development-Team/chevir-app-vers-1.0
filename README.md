# Chevir

Azərbaycan jest dili ilə danışıq Azərbaycan dili arasında tərcümə: layihənin saytı və
**tamamilə brauzerdə işləyən** tərcümə sistemi bir repo-da.

| Səhifə | Nə var |
|---|---|
| `/` | layihənin saytı — Vite, GSAP (ScrollTrigger), Lenis, three.js 3D əl |
| `/tercume/` | tərcümə sistemi: **Jest → Mətn** (kamera) və **Mətn → Jest** (VRM avatar + cyber skeleton) |

| Modul | Nə edir |
|---|---|
| **Jest → Mətn** | Veb-kamera → MediaPipe Hands → 20×63 → model → beam search + leksikon → Azərbaycan sözü |
| **Mətn → Jest** | Mətn → hərflər → real AzSLD pozaları → VRM avatar **və** cyber skeleton (yan-yana) |

Tərcümə üçün server yoxdur: video cihazdan çıxmır, gecikmə minimaldır, sayt statik fayl
kimi hər yerdə host oluna bilər (jest dili videosu istifadəçinin üzünü daşıyır — məxfilik
təsadüfi seçim deyil).

> **Hazırkı həcm:** sistem yalnız barmaq əlifbasını tanıyır və göstərir — 32 hərf və
> boşluq / enter / backspace. Söz və cümlə səviyyəsində tanıma, gloss və üz ifadələri hələ yoxdur.

## İşə salmaq

**Node.js 20+** lazımdır:

```bash
npm install
npm run dev
```

Sayt `http://localhost:5173/`, tərcümə sistemi `http://localhost:5173/tercume/` ünvanındadır.
Qısa yol: `./BAŞLAT.sh` (Linux / macOS) və ya `BASLAT.bat` (Windows) — paketləri ilk
dəfə özü qurur və brauzeri açır.

**Production:**

```bash
npm run build      # → dist/ (statik fayllar)
npm run preview    # build-i lokal yoxlamaq üçün
```

Vercel-də Vite layihəsi kimi avtomatik tanınır (build: `npm run build`, çıxış: `dist`).
Kamera yalnız HTTPS və ya `localhost` üzərində işləyir (brauzer qaydası).

### Avatar faylı

Avatar (`AvatarSample_Z.vrm`, pixiv VRoid Project) lisenziyası yenidən paylaşmağa icazə
vermir, ona görə repo-da **yoxdur**. Lokal işləmək üçün VRoid Hub-dan endirib
`public/tercume/assets/avatar.vrm` adı ilə qoyun. Fayl olmasa, "Mətn → Jest" cyber
skeleton ilə işləyir. Yayımlanan saytda avatarın görünməsi üçün yenidən paylaşmağa
icazə verən lisenziyalı VRM lazımdır (məsələn, VRoid Studio-da komandanın özünün yaratdığı model).

### Nə gözləmək

- **Jest → Mətn** — kamerasız da sınana bilər: klaviaturadan hərf yazın (`s a l a m`),
  sağda beam və leksikon canlı işləyir. `Enter` sözü tamamlayır.
- **Mətn → Jest** — mətn yazıb "Jest dilində göstər" düyməsinə basın, ya da əlifbadan hərfə toxunun.
- **Model** — memarlıq və real AzSLD üzərində ölçmələr.

## Qovluq quruluşu

```
index.html               sayt (/)
tercume/index.html       tərcümə sistemi (/tercume/)
src/
  main.js                sayt girişi
  sections/              sayt bölmələri; navbar.js hər iki səhifədə işlədilir
  lib/, styles/          sayt köməkçiləri və stilləri
  tercume/
    main.js              sistem girişi: saytın naviqasiyası + sistem
    model.js             Keras MLP-nin JS irəli keçidi (TensorFlow.js YOX — 4 matmul)
    decoder.js           beam search + hərf-bigram DM + leksikon yoxlaması
    hands.js             MediaPipe sarğısı + sürüşən kadr pəncərəsi (əl/hərəkət ölçüsü)
    s2t.js / t2s.js      jest → mətn idarəsi / mətn → jest oynatması
    skeleton.js          Three.js cyber skeleton (UnrealBloom)
    retarget.js          landmark → avatar: əl oriyentasiyası, qol IK, oynaq həddləri
    vrm.js               VRM 1.0 avatar: yaylı animasiya, nəfəs, göz qırpma, istirahət
    paths.js             statik faylların yolları
public/
  media/, *.svg, og-image.png   sayt faylları
  tercume/assets/        model.bin, lm.json, vocab.json, poses.json, hand_landmarker.task
  tercume/wasm/          MediaPipe wasm — @mediapipe/tasks-vision 0.10.14 ilə eyni olmalıdır
tools/
  export_model.py        model/fingerspelling_33.h5 → model.bin + model.json
  build_lexicon.py       model/lexicon.txt → bigram DM + lüğət + prefikslər
  build_poses.py         AzSLD şəkilləri → poses.json (+ model yoxlaması)
  verify_*.mjs/py        reqressiya yoxlamaları
model/
  fingerspelling_33.h5   barmaq əlifbası modeli (mənbə)
  lexicon.txt            dekoder üçün söz korpusu
docs/
  NSosyal_2026.docx      texniki hesabat
```

`@mediapipe/tasks-vision` versiyası dəqiq bağlanıb (`0.10.14`): paketi yeniləsəniz,
`node_modules/@mediapipe/tasks-vision/wasm/` fayllarını `public/tercume/wasm/`-a köçürün.

## Assetləri yenidən qurmaq

Saytı işə salmaq üçün lazım deyil — hazır fayllar `public/tercume/assets/`-dədir.
Yalnız model, leksikon və ya pozalar dəyişəndə (Python 3):

```bash
python3 -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/export_model.py     # model → model.bin + model.json
.venv/bin/python tools/build_lexicon.py    # leksikon → lm.json, vocab.json, prefixes.json
.venv/bin/python tools/build_poses.py      # AzSLD şəkilləri → poses.json
```

`build_poses.py` üçün AzSLD Fingerspelling datasetini (Zenodo 14222948, ~1.1 GB)
`data/AzSLD_Fingerspelling/` altına açın və ya `--fs-dir` ilə yolunu verin. `data/` git-ə düşmür.

## Modelin dəqiq portu

`fingerspelling_33.h5` sadə `Flatten → Dense(300) → Dense(256) → Dense(128) → Dense(35)`
olduğu üçün TensorFlow.js gətirmək lazım deyil: çəkilər xam `float32` kimi
çıxarılıb, irəli keçid JS-də 4 matmul ilə hesablanır (~492K parametr, 1.9 MB).

Port numpy referansı ilə yoxlanılır:

```bash
.venv/bin/python tools/verify_forward.py   # referans yaradır
node tools/verify_forward.mjs              # JS ilə tutuşdurur
# → maks fərq 4.8e-7 (float32 həddi)
```

## Dekoder

Top-3 proqnoz → beam → hərf-bigram dil modeli → leksikon yoxlaması, üç təkmilləşdirmə ilə:

1. **artımlı beam** — cartesian product yerinə addım-addım kəsmə
2. **loq fəzası** — ehtimal hasilləri sıfıra yuvarlanmır
3. **leksikon prefiks kəsməsi** — `strict` / `soft` / `off` rejimləri

Bigram hesablamasında söz-başı keçidi yalnız boşluq kontekstindən sayılır (ilkin
versiyada `j==0` halında sözün son hərfindən sayılırdı). Görünməmiş bigramlar üçün
Laplace hamarlaşdırma əlavə olunub.

```bash
node tools/verify_decoder.mjs
# "insan"-da 3-cü hərf p=0.25 verilsə belə leksikon onu düzəldir
```

## Kameradan hərf nə vaxt yazılır

Model 20 kadrlıq pəncərə ilə işləyir. Əvvəl pəncərə hər 30 kadrdan bir **şərtsiz**
hərf yazırdı — ona görə əlin kadra girdiyi və bir pozadan digərinə keçdiyi anlar da
hərfə çevrilirdi (ölçülüb: 5 boş kadr + 15 kadr "s" → model **"ö" 0.89** ilə deyir,
yəni sadəcə əminlik həddi bunu tutmur). İndi pəncərə sürüşür və hərf yalnız bu
şərtlər ödənəndə yazılır:

| Şərt | Dəyər | Niyə |
|---|---|---|
| Pəncərənin bütün kadrlarında əl görünür | 20/20 | əl kadra girəndə boş kadrlar dinamik hərf kimi oxunur |
| Hərəkət ölçüsü kiçikdir | < 0.05 | sabit poza 0.00–0.04, pozalar arası keçid 0.06–0.11 |
| Model əmindir | ≥ 0.45 | modelin özü bəzi hərflərdə alçaq qalır (sabit "m" → 0.56) |
| Eyni hərf ardıcıl təsdiqlənir | 3 proqnoz | anlıq titrəmə hərf yazmasın |
| Yazılan hərf kilidlənir | əl tərpənənə qədər | eyni hərf təkrar-təkrar yazılmasın |

Praktikada: hər hərfi ~1 saniyə sabit saxlamaq, sonra əli tərpədib növbətiyə keçmək.
Kameranın altındakı halqa vəziyyəti göstərir (əl yoxdur / hərəkət / saxla / yazıldı).

```bash
node tools/verify_s2t.mjs
# süni kamera ardıcıllığı: "salam" hərf-hərf yazılır, keçidlər və boş kadrlar heç nə yazmır
```

## Hərf pozaları

`poses.json` AzSLD barmaq əlifbası şəkillərindən (Zenodo 14222948, CC BY 4.0)
MediaPipe ilə çıxarılıb — **brauzerdə işlədilən eyni `hand_landmarker.task`**,
beləcə landmark konvensiyası uyğun qalır.

- 24 statik hərf → medoid poza (ən tipik nümunə)
- 8 dinamik hərf (ç d g k ö ü y z) → **tək video klipdən** 12–14 ardıcıl kadr.
  Qovluqda bir neçə klip var (kadr nömrəsində >20 boşluq klip sərhədidir); model
  onu yaxşı tanıyan, əlin tərəfi ardıcıl olan və ovucu titrəməyən klip seçilir

Hər kadr üçün normallaşdırılmış landmark-lardan başqa `world` (MediaPipe-in metrik
3D koordinatları), `size` (mənbə şəklin ölçüsü), `handedness` və `files` (mənbə şəkil) saxlanılır.

Çıxarma zamanı model real data üzərində yoxlanılır (bax `public/tercume/assets/eval.json`):

| Qrup | Hərf | Orta top-1 | Median | ≥90% |
|---|---|---|---|---|
| Statik | 24 | 77.7% | 91.9% | 15/24 |
| Dinamik | 8 | 1.9% | 0.0% | 0/8 |

**İki qeyd.** Dinamik hərflərin sıfıra yaxın nəticəsi modelin uğursuzluğu deyil:
ölçmədə bir kadr 20 dəfə təkrarlanır, hərəkət isə belə təmsil oluna bilmir
(real ardıcıllıqla `ç` və `ö` düzgün tanınır). Həmçinin bu şəkillər modelin
təlim datası ola bilər — rəqəmlər **optimistdir**, kənarlaşdırılmış test dəsti deyil.

## Avatar

VRM 1.0 (VRoid Studio), 30 barmaq sümüyü. Poza bütün qola ötürülür — çiyin, dirsək,
bilək, barmaqlar (`src/tercume/retarget.js` riyazi hissə, `src/tercume/vrm.js` animasiya):

1. **Əlin oriyentasiyası** şəkil koordinatlarından (en/hündürlük nisbəti düzəldilir),
   **barmaq forması** metrik 3D koordinatlardan (`world`).
2. **Qol** iki-sümüklü IK ilə qurulur. Dirsəyin fırlanması və əlin kiçik yerdəyişməsi
   elə axtarılır ki, bilək bükülməsi (−65°…90°), yana əyilmə və önqol burulması (±95°)
   anatomik həddlərdə qalsın, dirsək bədənə girməsin və "qanad" kimi qalxmasın.
   Barmaqları aşağı olan hərflərdə əl sinə səviyyəsində sallanır, yuxarı olanlarda çiyin önündədir.
3. **Barmaqlar**: MCP 2 sərbəstlik (bükülmə + yana açılma), PIP/DIP menteşə,
   baş barmaq istiqamət izləmə — hamısı oynaq həddləri ilə.
4. **Hərəkət** kritik sönümlü yaylarla: sürət kəsilmir, qol barmaqlardan yavaş çatır.
   Sözlər arasında əl hazır vəziyyətə keçir, işarə bitəndə qol aşağı enir.
5. **Canlılıq**: nəfəs, göz qırpma, baxış izləyiciyə, başın kiçik hərəkəti.

Datadan gələn üç tələ nəzərə alınıb:

- AzSLD-də bəzi hərflər sol əllə və ya güzgülü selfi ilə çəkilib. Əlin tərəfi etiketlə
  yox, barmaqların hansı tərəfə büküldüyü ilə təyin olunur və avatarın əlinə güzgülənir.
- MediaPipe bəzi kadrlarda dərinliyi güzgü həll edir (tərs əl kimi) — həmin kadrda z çevrilir.
- Dinamik hərfdə tək kadrlıq ovuc sıçrayışı (qonşulardan 60°+) oynadılmır.

İlkin (yalnız barmaq bükülməsi) və hazırkı avatar eyni metrika ilə, 32 hərf üzrə:

| | İlkin | Hazırkı |
|---|---|---|
| Əlin oriyentasiya xətası (orta) | 67.8° | 0.8° |
| Oriyentasiyası 45°-dən çox səhv olan hərf | 17/32 | 0/32 |
| Barmaq seqmenti xətası (orta) | 27.7° | 9.2° |
| Bilək həddini aşan hərf | — | 0 |

Barmaq xətasının qalanı əsasən oynaq həddlərindən gəlir: MediaPipe-in təxmin etdiyi
yana və ya arxaya bükülmə anatomik mümkün deyilsə, avatar onu təkrarlamır.

## Mənbələr

1. **World Health Organization (WHO)** - *Deafness and hearing loss* (2026). [Link](https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss)
2. **T.C. Sağlık Bakanlığı** - *Uluslararası İşitme Engelliler Haftası*. [Link](https://www.saglik.gov.tr/)
3. Traxler, C. B. (2000). *The Stanford Achievement Test, 9th Edition: National Norming and Performance Standards for Deaf and Hard-of-Hearing Students*. Journal of Deaf Studies and Deaf Education. [DOI: 10.1093/deafed/5.4.337](https://doi.org/10.1093/deafed/5.4.337)
4. Qi, S. & Mitchell, R. E. (2012). *Large-Scale Academic Achievement Testing of Deaf and Hard-of-Hearing Students*. [DOI: 10.1093/deafed/enr028](https://doi.org/10.1093/deafed/enr028)
5. Mayer, C., Trezek, B. J. & Hancock, G. R. (2021). *Reading Achievement of Deaf Students*. [DOI: 10.1093/deafed/enab013](https://doi.org/10.1093/deafed/enab013)
6. Öztürk, Ş. & Keleş, H. Y. (2024). *E-TSL: A Continuous Educational Turkish Sign Language Dataset with Baseline Methods*. [arXiv](https://arxiv.org/abs/2405.02984)
7. *Türk İşaret Dili Sisteminin Oluşturulması ve Uygulanmasına Yönelik Usul ve Esasların Belirlenmesine İlişkin Yönetmelik*, Resmî Gazete (2006). [Link](https://www.resmigazete.gov.tr/eskiler/2006/04/20060414-2.htm)
8. Signapse - *How does our AI technology work*. [Link](https://www.signapse.ai/post/how-does-our-ai-technology-work)
9. Google I/O 2025 - *SignGemma*. [Link](https://deepmind.google/blog/putting-sign-language-ai-into-users-hands/)
10. Zero Project (2025). *Signapse: AI Sign Language*. [Link](https://zeroproject.org/view/project/100ff85a-f64c-f011-8779-7c1e527683f1)
11. Sincan, O. M. & Keleş, H. Y. (2020). *AUTSL: A Large Scale Multi-Modal Turkish Sign Language Dataset and Baseline Methods*. IEEE Access. [DOI: 10.1109/ACCESS.2020.3028072](https://doi.org/10.1109/ACCESS.2020.3028072)
12. Camgöz, N. C., et al. (2016). *BosphorusSign: A Turkish Sign Language Recognition Corpus in Health and Finance Domains*. LREC'16. [PDF Link](https://aclanthology.org/L16-1220.pdf)
13. SyncWords & Signapse (2025). *SyncWords and Signapse Launch Live Automatic Sign Language for Streaming*. [Link](https://www.prnewswire.com/news-releases/syncwords-and-signapse-launch-live-automatic-sign-language-for-streaming-302605280.html)
14. Alishzade, N. & Hasanov, J. (2025). *AzSLD: Azerbaijani Sign Language Dataset for Fingerspelling, Word, and Sentence Translation with Baseline Software*. Data in Brief. [DOI: 10.1016/j.dib.2024.111230](https://doi.org/10.1016/j.dib.2024.111230)

**Üçüncü tərəf komponentlər:** three.js (MIT), @pixiv/three-vrm (MIT), GSAP, Lenis (MIT),
MediaPipe Tasks Vision və `hand_landmarker.task` (Apache-2.0), AzSLD Fingerspelling (CC BY 4.0).
Saytın ilkin versiyası [Gulshan-hu/chevir-site](https://github.com/Gulshan-hu/chevir-site) repo-sundan köçürülüb.
