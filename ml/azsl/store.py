"""Landmarks on disk: items packed into part-NNNNN.npz files plus one manifest.csv per subset.

A part holds a few hundred items concatenated along time (float16) with `ids` and
`offsets`, so thousands of clips load as a handful of files. Each manifest row says
which part and offset an item lives in, plus its metadata. Writes are resumable:
items already in the manifest are skipped, and a part is written before its rows are
appended, so an interrupted run never leaves manifest rows without data.
"""
import csv
import functools
import glob
import os

import numpy as np

from .landmarks import EMB_DIM, FIELDS

BASE_COLS = ['id', 'part', 'offset', 'length']


class PartWriter:
    def __init__(self, out_dir: str, items_per_part: int = 400, embeddings: bool = False):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.per_part, self.embeddings = out_dir, items_per_part, embeddings
        self.manifest = os.path.join(out_dir, 'manifest.csv')
        nums = [int(os.path.basename(p)[5:10]) for p in glob.glob(os.path.join(out_dir, 'part-*.npz'))]
        self.next_part = max(nums, default=-1) + 1
        self.cols = None
        if os.path.exists(self.manifest):
            with open(self.manifest, newline='', encoding='utf-8') as f:
                self.cols = next(csv.reader(f), None)
        self.buf = []

    def done_ids(self) -> set:
        if not os.path.exists(self.manifest):
            return set()
        with open(self.manifest, newline='', encoding='utf-8') as f:
            return {r['id'] for r in csv.DictReader(f)}

    def add(self, item_id: str, arrays: dict, meta: dict, face: dict | None = None):
        self.buf.append((item_id, arrays, meta, face))
        if len(self.buf) >= self.per_part:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        name = f'part-{self.next_part:05d}.npz'
        lengths = [a['hand'].shape[0] for _, a, _, _ in self.buf]
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        payload = {k: np.concatenate([a[k] for _, a, _, _ in self.buf]).astype(np.float16) for k in FIELDS}
        payload['ids'] = np.array([i for i, _, _, _ in self.buf])
        payload['offsets'] = offsets
        if self.embeddings:
            emb = np.full((len(self.buf), EMB_DIM), np.nan, np.float32)
            thumbs, toff = [], [0]
            for j, (_, _, _, face) in enumerate(self.buf):
                if face is not None:
                    emb[j] = face['emb']
                t = (face or {}).get('thumb') or b''
                thumbs.append(np.frombuffer(t, np.uint8))
                toff.append(toff[-1] + len(t))
            payload['emb'] = emb
            payload['thumb'] = np.concatenate(thumbs) if thumbs else np.zeros(0, np.uint8)
            payload['thumb_off'] = np.array(toff, np.int64)
        path = os.path.join(self.dir, name)
        np.savez_compressed(path + '.tmp.npz', **payload)
        os.replace(path + '.tmp.npz', path)

        rows = [{'id': i, 'part': name, 'offset': int(offsets[j]), 'length': lengths[j], **meta}
                for j, (i, _, meta, _) in enumerate(self.buf)]
        if self.cols is None:
            self.cols = BASE_COLS + [k for k in rows[0] if k not in BASE_COLS]
        extra = set(rows[0]) - set(self.cols)
        if extra:
            raise ValueError(f'new manifest columns {extra} — keep metadata keys stable within a subset')
        new = not os.path.exists(self.manifest)
        with open(self.manifest, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self.cols)
            if new:
                w.writeheader()
            w.writerows(rows)
        self.buf = []
        self.next_part += 1

    close = flush


class LandmarkStore:
    """Read side. `dirs` may span several outputs (e.g. one per Sentences shard)."""

    def __init__(self, dirs):
        import pandas as pd
        dirs = [dirs] if isinstance(dirs, str) else list(dirs)
        frames = []
        for d in dirs:
            m = os.path.join(d, 'manifest.csv')
            if os.path.exists(m):
                df = pd.read_csv(m, dtype={'id': str}, keep_default_na=False, na_values=[''])
                df['dir'] = d
                frames.append(df)
        if not frames:
            raise FileNotFoundError(f'no manifest.csv in {dirs}')
        self.manifest = pd.concat(frames, ignore_index=True).drop_duplicates('id', keep='last')
        self.index = self.manifest.set_index('id')

    def __len__(self):
        return len(self.manifest)

    @functools.lru_cache(maxsize=16)
    def _part(self, path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def get(self, item_id: str, dtype=np.float32) -> dict:
        r = self.index.loc[item_id]
        p = self._part(os.path.join(r['dir'], r['part']))
        s, n = int(r['offset']), int(r['length'])
        return {k: p[k][s:s + n].astype(dtype) for k in FIELDS}

    def embeddings(self):
        """→ (ids, emb [N, 128], thumbs list[bytes]) over every part that stored them."""
        ids, embs, thumbs = [], [], []
        for d, part in self.manifest[['dir', 'part']].drop_duplicates().itertuples(index=False):
            p = self._part(os.path.join(d, part))
            if 'emb' not in p:
                continue
            off = p['thumb_off']
            for j, i in enumerate(p['ids']):
                ids.append(str(i))
                embs.append(p['emb'][j])
                thumbs.append(p['thumb'][off[j]:off[j + 1]].tobytes())
        return ids, (np.stack(embs) if embs else np.zeros((0, EMB_DIM), np.float32)), thumbs
