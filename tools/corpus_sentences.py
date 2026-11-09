"""
HPLT 2.0 (Azərbaycan dili, CC0) parquet faylından qısa, təmiz cümlələr: işarələr → mətn modeli üçün
süni cütlərin (tools/build_pseudo_gloss.mjs) xammalı.

  .venv/Scripts/python -I tools/corpus_sentences.py data/corpus/hplt2_azj_00000.parquet data/corpus/hplt_sentences.txt

Sənəd cümlələrə bölünür; saxlanır: 3–12 söz, böyük hərflə başlayır və . ! ? ilə bitir, yalnız Azərbaycan
əlifbası, rəqəm və adi durğu işarələri (w, @, http yoxdur), ən azı bir ə/ı/ö/ü/ğ/ş/ç hərfi (ingiliscə,
türkcə mətn düşür), təkrarlar atılır. Mənbə: https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned (azj_Latn).
"""
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

SPLIT = re.compile(r'(?<=[.!?])\s+(?=["«(]?[A-ZƏÖÜĞIİŞÇ])|\n+')
OK = re.compile(r"^[A-Za-zƏəÖöÜüĞğIıİiŞşÇç0-9 .,!?;:\"«»()\-–—'’]+$")
AZ = re.compile(r'[əıöüğşçƏÖÜĞIİŞÇ]')


def sentences(doc):
    for s in SPLIT.split(doc):
        s = ' '.join(s.split())
        n = len(s.split())
        if (3 <= n <= 12 and s[0].isupper() and s[-1] in '.!?' and OK.match(s) and AZ.search(s)
                and 'w' not in s.lower() and s.upper() != s):
            yield s


def main(src, dst):
    sys.stdout.reconfigure(encoding='utf-8')
    f = pq.ParquetFile(src)
    seen, docs = set(), 0
    with open(dst, 'w', encoding='utf8') as out:
        for batch in f.iter_batches(columns=['text'], batch_size=2000):
            for doc in batch.column('text').to_pylist():
                docs += 1
                for s in sentences(doc or ''):
                    if s not in seen:
                        seen.add(s)
                        out.write(s + '\n')
    print(f'{docs} sənəd → {len(seen)} cümlə: {dst}')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
