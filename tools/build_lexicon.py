"""
lexicon.txt -> hərf-bigram dil modeli + söz lüğəti (brauzer üçün JSON).

Müəllimin notebook-undaki (cell 3) məntiqi saxlanılır: söz-başı P(hərf|' '),
keçid P(hərf|əvvəlki) və söz-sonu P(' '|hərf) ehtimalları unikal sözlərdən
sayılır. İki dəyişiklik var, ikisi də şüurludur:

1. Orijinalda j==0 halında `else` bloku da işləyir və w[j-1] = w[-1]
   (sözün SON hərfi) olur — yəni söz-başına yanlış keçid sayılır. Burada
   söz-başı yalnız ' ' konteksti kimi sayılır.
2. Laplace (add-one) hamarlaşdırma əlavə olunub. Orijinalda görünməmiş
   bigram 0 verir və beam-də bütün namizədi sıfırlayır; hamarlaşdırma
   nadir, lakin düzgün hərf ardıcıllıqlarının kəsilməsinin qarşısını alır.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "model" / "lexicon.txt"
OUT_DIR = ROOT / "public" / "tercume" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHABET = ['a', 'b', 'c', 'ç', 'd', 'e', 'ə', 'f', 'g', 'ğ', 'h', 'x', 'ı', 'i',
            'j', 'k', 'q', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u',
            'ü', 'v', 'y', 'z']
STOP = {'da', 'də', 'nə'}
BOW = ' '          # söz sərhədi simvolu (notebook ilə eyni)


def load_words() -> list[str]:
    text = SRC.read_text(encoding='utf8').replace('\n', ' ').replace('\r', ' ')
    allowed = set(ALPHABET)
    out = []
    for w in text.split(' '):
        w = w.strip().lower()
        if len(w) < 2 or w in STOP:
            continue
        if not w.isalpha() or not set(w) <= allowed:
            continue
        out.append(w)
    return out


def main() -> None:
    words = load_words()
    freq = Counter(words)
    vocab = sorted(freq)
    print(f"korpus: {len(words):,} söz | unikal: {len(vocab):,}")

    syms = [BOW] + ALPHABET
    idx = {s: i for i, s in enumerate(syms)}
    n = len(syms)

    # counts[prev][next], BOW həm başlanğıc, həm son kontekstidir
    counts = [[0] * n for _ in range(n)]
    for w in vocab:
        prev = BOW
        for ch in w:
            counts[idx[prev]][idx[ch]] += 1
            prev = ch
        counts[idx[prev]][idx[BOW]] += 1

    # Laplace hamarlaşdırma + sətir üzrə normalizasiya
    logp = []
    import math
    for r in range(n):
        row_total = sum(counts[r]) + n
        logp.append([round(math.log((counts[r][c] + 1) / row_total), 5)
                     for c in range(n)])

    (OUT_DIR / "lm.json").write_text(json.dumps({
        "symbols": syms,
        "bow": BOW,
        "logp": logp,
        "note": "logp[prev][next] = ln P(next|prev), Laplace hamarlaşdırılmış",
    }, ensure_ascii=False))

    # Leksikon yoxlaması üçün: sözlər + prefiks dəsti (beam kəsmə üçün)
    prefixes = set()
    for w in vocab:
        for k in range(1, len(w) + 1):
            prefixes.add(w[:k])

    (OUT_DIR / "vocab.json").write_text(json.dumps({
        "words": vocab,
        "freq": {w: freq[w] for w in vocab},
    }, ensure_ascii=False))
    (OUT_DIR / "prefixes.json").write_text(json.dumps(sorted(prefixes),
                                                      ensure_ascii=False))

    for f in ("lm.json", "vocab.json", "prefixes.json"):
        print(f"  {f}: {(OUT_DIR / f).stat().st_size/1024:.0f} KB")
    print(f"prefiks sayı: {len(prefixes):,}")
    print("ən tez-tez: " + ", ".join(w for w, _ in freq.most_common(12)))


if __name__ == "__main__":
    main()
