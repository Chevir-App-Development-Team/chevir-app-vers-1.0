"""
Bir işarənin etiketləri → bir sinif, brauzer üçün (public/tercume/assets/sign_merge.json).

AzSLD söz formalarını ayrı etiketləyir (ACAM, ACMİŞAM), bəziləri isə eyni işarədir. Brauzer modelin
çıxışında belə etiketlərin ehtimallarını toplayır (recognizer.js signGroups): köhnə model də yeni
birləşmələrlə işləyir, yenidən öyrədilmiş modeldə isə birləşmə artıq etiketlərin özündədir.

Qaydalar ml/groups.py-dədir (notebook-lar da onu işlədir):
  - ml/gloss_spelling.csv: eyni sözün iki yazılışı (variant → glos), əvvəl tətbiq olunur;
  - ml/lemma_groups.csv: təsdiqlənmiş söz ailələri (confirmed = yes; ailənin əsas forması da);
  - ml/same_sign.csv: modelin və ya əllərin hərəkətinin ayıra bilmədiyi formalar (glos → qrup);
    təsdiqlənmiş ailəyə qoşulan qrup onun adını alır.

İstifadə: python -I tools/build_sign_merge.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'ml'))
from groups import sign_merge, spelling  # noqa: E402  (ml/groups.py: the same rules as the notebooks)


def main():
    sys.stdout.reconfigure(encoding='utf-8')      # -I ilə PYTHONIOENCODING oxunmur
    spelling_, merge = spelling(), sign_merge()
    out = ROOT / 'public' / 'tercume' / 'assets' / 'sign_merge.json'
    out.write_text(json.dumps({
        'source': 'ml/gloss_spelling.csv, ml/lemma_groups.csv (confirmed), ml/same_sign.csv',
        'spelling': dict(sorted(spelling_.items())),
        'merge': dict(sorted(merge.items())),
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    groups = len(set(merge.values()))
    print(f'{out.relative_to(ROOT)}: {len(spelling_)} yazılış variantı, {len(merge)} glos {groups} işarəyə birləşir')


if __name__ == '__main__':
    main()
