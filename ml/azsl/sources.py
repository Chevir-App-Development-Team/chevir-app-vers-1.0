"""What is inside the AzSLD archives, as tables.

Layout (checked against Zenodo 14222948, Oct 2026):
  AzSLD_Sentences/<folder>/{Cam1,Cam2}/<YYYY-MM-DD HH-MM-SS>.mp4   + <folder>/ann/<same>.mp4.json
      Cam2 is the FRONTAL camera, and annotations belong to Cam2 (frame counts match);
      Cam1 is the side view and can be a few frames shorter. 1280×960, 30 fps.
      ann JSON (Supervisely): tags[{name, frameRange [start, end] 1-based inclusive, key}]
  AzSLD_Sentences/sentence_list.csv                                folder → sentence, gloss sequence
  AzSLD_Words_200/<GLOSS>/<key>.mp4    clips cut from Cam2 at a tag's frameRange, no margin;
      <key> is the tag key, which links a clip to its recording (and so to its signer)
  AzSLD_Fingerspelling/<LETTER>/*.jpg  40 images per static letter, ~800 frames per dynamic
      letter. "ID--<n>--__<uuid>.jpg" names carry a contributor id; others are numbered frames.
No signer ids anywhere — see split.py for how recordings are grouped by signer.
"""
import csv
import hashlib
import io
import json
import re
from datetime import datetime

from .zenodo import Member, read_many

DYNAMIC_LETTERS = set('çdgköüyz')          # same split as tools/build_poses.py
RUN_GAP = 20                               # frame-number gap that starts a new clip


def az_lower(s: str) -> str:
    """Azerbaijani lower-case: I → ı, İ → i (str.lower() gets these wrong)."""
    return s.replace('I', 'ı').replace('İ', 'i').lower()


def anon(raw: str) -> str:
    """Contributor ids in file names are Telegram user ids — keep only a short hash."""
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def parse_stamp(stem: str):
    m = re.match(r'(\d{4}-\d\d-\d\d \d\d-\d\d-\d\d)', stem)
    return datetime.strptime(m.group(1), '%Y-%m-%d %H-%M-%S') if m else None


def sentence_recordings(members: list[Member], camera: str = 'Cam2') -> list[Member]:
    """Video members of one camera, in a stable order."""
    rx = re.compile(rf'^AzSLD_Sentences/([^/]+)/{camera}/([^/]+)\.mp4$')
    return sorted((m for m in members if rx.match(m.name)), key=lambda m: m.name)


def rec_id(name: str) -> str:
    """'AzSLD_Sentences/12/Cam2/2022-06-01 15-17-31.mp4' → '12/2022-06-01 15-17-31'."""
    p = name.split('/')
    stem = p[-1]
    for suf in ('.mp4.json', '.mp4'):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
    return f'{p[1]}/{stem}'


def sentences_index(members: list[Member]):
    """→ (tags, recordings, sentences) DataFrames from the annotation JSONs + sentence_list.csv."""
    import pandas as pd
    anns = [m for m in members if re.match(r'^AzSLD_Sentences/[^/]+/ann/.+\.json$', m.name)]
    lst = [m for m in members if m.name.endswith('sentence_list.csv')]
    cams = {c: {rec_id(m.name): m.size for m in sentence_recordings(members, c)} for c in ('Cam1', 'Cam2')}

    tags, recs = [], []
    for m, data in read_many(anns):
        rid = rec_id(m.name)
        d = json.loads(data.decode('utf-8'))
        recs.append({
            'rec_id': rid, 'folder': rid.split('/')[0], 'stamp': parse_stamp(rid.split('/')[1]),
            'frames': d.get('framesCount'), 'width': d.get('size', {}).get('width'),
            'height': d.get('size', {}).get('height'), 'n_tags': len(d.get('tags', [])),
            'cam1': rid in cams['Cam1'], 'cam2': rid in cams['Cam2'],
        })
        for t in d.get('tags', []):
            fr = t.get('frameRange') or [None, None]
            tags.append({'key': t['key'], 'gloss': t['name'], 'rec_id': rid, 'start': fr[0], 'end': fr[1],
                         'labeler': t.get('labelerLogin'), 'created': t.get('createdAt')})
    tags, recs = pd.DataFrame(tags), pd.DataFrame(recs)
    tags['dup'] = _duplicates(tags)

    sentences = pd.DataFrame(columns=['folder', 'text', 'glosses'])
    if lst:
        rows = list(csv.reader(io.StringIO(next(read_many(lst))[1].decode('utf-8-sig'))))
        sentences = pd.DataFrame([{'folder': r[0].strip(), 'text': r[1].strip(), 'glosses': r[2].strip()}
                                  for r in rows[1:] if len(r) >= 3])
    return tags, recs, sentences


def _duplicates(tags):
    """Same gloss annotated twice over the same span of one recording (seen in the data) → later one is dup."""
    dup = [False] * len(tags)
    if tags.empty:
        return dup
    for _, g in tags.sort_values('created').groupby(['rec_id', 'gloss']):
        kept = []
        for idx, r in g.iterrows():
            s, e = r['start'], r['end']
            if any(min(e, ke) - max(s, ks) + 1 >= 0.5 * min(e - s + 1, ke - ks + 1) for ks, ke in kept):
                dup[tags.index.get_loc(idx)] = True
            else:
                kept.append((s, e))
    return dup


def words_listing(names: list[str], root: str = 'AzSLD_Words_200'):
    """Zip member names → key (tag key, links to a recording), gloss, member."""
    import pandas as pd
    rx = re.compile(rf'^{root}/([^/]+)/([0-9a-f]+)\.mp4$')
    rows = [{'key': mm.group(2), 'gloss': mm.group(1), 'member': n} for n in names if (mm := rx.match(n))]
    return pd.DataFrame(rows)


def fingerspelling_listing(names: list[str]):
    """One row per image: letter, contributor group (hashed) or clip run, frame number."""
    import pandas as pd
    rows = []
    for name in names:
        p = name.split('/')
        if len(p) != 3 or not p[2].lower().endswith('.jpg'):
            continue
        letter, file = az_lower(p[1]), p[2]
        cid = re.match(r'^ID--(\d+)--', file)
        num = re.search(r'(\d+)\.jpg$', file, re.I)
        rows.append({'file': file, 'member': name, 'letter': letter,
                     'kind': 'dynamic' if letter in DYNAMIC_LETTERS else 'static',
                     'contributor': anon(cid.group(1)) if cid else '',
                     'frame_no': int(num.group(1)) if (num and not cid) else -1})
    df = pd.DataFrame(rows)
    # numbered frames: a gap > RUN_GAP starts a new clip (as in tools/build_poses.py)
    df['run'] = ''
    for letter, g in df[df.contributor == ''].sort_values('frame_no').groupby('letter'):
        run, prev = 0, None
        for idx, n in g['frame_no'].items():
            if prev is not None and n - prev > RUN_GAP:
                run += 1
            df.at[idx, 'run'] = f'{letter}-run{run}'
            prev = n
    df['group'] = df.contributor.where(df.contributor != '', df.run)
    return df
