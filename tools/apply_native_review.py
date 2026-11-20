"""
Yerli rəy paketinin cavabları (tools/build_native_review.py, E36) → xəritələr və bayraqlar.

  D:/chevir/venv/Scripts/python -I tools/apply_native_review.py <cavablar.csv> [başqa.csv ...] [--dry-run] [--root <repo surəti>]

Cavab kodları yes / no / other / unsure; unsure və boş heç nəyi dəyişmir. Bir neçə rəy verən eyni bəndə fərqli cavab
veribsə, bənd keçilir və siyahıda göstərilir (qərar komandanındır). Bölmələr:

  slovo_map            yes → native = yes; no → sətir ml/slovo_azsld.csv-dən çıxır; other + tanıyıcının etiketi →
                       sətrin azsld-si o olur, başqa söz → sətir çıxır (namizəd jurnala yazılır)
  slovo_new_installed  yes → native = yes, basis-ə "native yes = rank 0" (SLOVO_NEW_MAX_RANK onu saxlayır);
  slovo_new_unchecked  no → sətir ml/slovo_new.csv-dən çıxır; other + AzSLD etiketi → sətir slovo_azsld.csv-yə
                       keçir, başqa söz → etiket o söz olur (native = yes)
  slovo_azsld55        yes → slovo_azsld.csv-yə yeni sətir (native = yes); other + tanıyıcının etiketi → o etiketlə
  d5                   cavablar ml/review/word_forms_review_w500.csv-yə yazılır, sonra tools/same_sign/apply_review.py
                       (ml/lemma_groups.csv) və tools/build_sign_merge.py (public/tercume/assets/sign_merge.json)
  literal              no → ml/label_flags.csv-də literal bayrağı (qeyddə doğma işarə); yes → bayraq çıxır
  n8                   ml/text_alt.csv: no → sətir çıxır, yes → qeyd "native yes"; təklif: yes → sətir əlavə olunur;
                       sonra avatar lüğətinin alt sahələri (public/tercume/words) text_alt.csv-yə uyğunlaşır
                       (JestDili.az-ın öz alt-ları qalır; tools/text_alt.py, build_avatar_words.py --alt-only də belə)
  n9                   yalnız jurnal: qaydalar kodun içindədir (src/tercume/words.js, gloss2text.js), əl ilə dəyişilir
  v2                   keçilir: node tools/review_summary.mjs <cavablar.csv>

ml/label_flags.csv-nin RSL bayraqları ml/slovo_new.csv-dən yenidən hesablanır: quraşdırılmış söz modelinin Slovo-dan
gələn hər sinfi rsl (təsdiqlənməyib), sətirlərinin hamısı native = yes-dirsə bayraqsız, heç sətri qalmayıbsa
rsl-rejected (model yenidən öyrədilənə qədər bu sinfi tanıyır). Hər tətbiq olunan cavab ml/review/native_decisions.csv
jurnalına yazılır (rəy verənin kodu; ad yox). Sonra: bütün node tools/verify_*.mjs və node tools/ci_gates.mjs;
xəritə dəyişibsə söz və cümlə modelləri yenidən öyrədilir (E36).
"""
import argparse
import csv
import datetime
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # -I ilə skriptin qovluğu yolda deyil
from text_alt import alt_map, sync_alt                        # noqa: E402

CODES = ('yes', 'no', 'other')
SHEET_ANSWER = {'yes': 'bəli', 'no': 'xeyr', 'unsure': 'bilmirəm'}
FLAG_COLS = ['label', 'flag', 'note']
LOG_COLS = ['date', 'reviewer', 'section', 'item', 'a', 'b', 'answer', 'correction', 'comment', 'action']


def az_lower(s):
    return s.replace('I', 'ı').replace('İ', 'i').lower()


def az_upper(s):
    return s.replace('i', 'İ').replace('ı', 'I').upper()


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        return list(r.fieldnames), list(r)


def write_csv(path, cols, rows, bom=False):
    with open(path, 'w', newline='', encoding='utf-8-sig' if bom else 'utf-8') as f:
        w = csv.DictWriter(f, cols, lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def add_note(old, new):
    return f'{old}; {new}' if old and new else old or new


def decide(rows):
    """Bir bəndin cavabları (bir neçə rəy verən) → (kod, düzəliş, şərh, kodlar) və ya ('conflict', ...)."""
    who = '+'.join(sorted({r['reviewer'] for r in rows}))
    comment = ' / '.join(r['comment'].strip() for r in rows if r.get('comment', '').strip())
    firm = [r for r in rows if r['answer'] in CODES]
    if not firm:
        return 'unsure', '', comment, who
    codes = {(r['answer'], az_lower(r['correction'].strip()) if r['answer'] == 'other' else '') for r in firm}
    if len(codes) > 1:
        return 'conflict', ' | '.join(f"{r['reviewer']}: {r['answer']} {r['correction']}".strip() for r in firm), comment, who
    corr = next((r['correction'].strip() for r in firm if r['correction'].strip()), '')
    return firm[0]['answer'], corr, comment, who


def main():
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('answers', nargs='+', type=Path)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--dry-run', action='store_true', help='heç nə yazma, yalnız nə dəyişəcəyini göstər')
    args = ap.parse_args()
    root, dry = args.root, args.dry_run
    today = datetime.date.today().isoformat()
    P = {k: root / v for k, v in {
        'map': 'ml/slovo_azsld.csv', 'new': 'ml/slovo_new.csv', 'flags': 'ml/label_flags.csv', 'alt': 'ml/text_alt.csv',
        'sheet': 'ml/review/word_forms_review_w500.csv', 'log': 'ml/review/native_decisions.csv',
        'words': 'public/tercume/words', 'manifest': 'public/tercume/assets/word_model/manifest.json',
        'label_text': 'ml/label_text.csv'}.items()}

    # cavablar: bənd üzrə
    by_item = defaultdict(list)
    v2 = 0
    for path in args.answers:
        _, rows = read_csv(path)
        for r in rows:
            if r['section'] == 'v2':
                v2 += 1
                continue
            if r.get('answer') or r.get('comment') or r.get('correction'):
                by_item[(r['section'], r['item'])].append(r)

    labels = json.loads(P['manifest'].read_text(encoding='utf-8'))['labels']
    text_label = {az_lower(r['text']): r['label'] for r in read_csv(P['label_text'])[1]}

    def to_label(word):
        """Rəy verənin yazdığı söz → tanıyıcının etiketi (yoxdursa None)."""
        w = word.strip().strip('«»"\'')
        for cand in (az_upper(w), text_label.get(az_lower(w))):
            if cand and cand in labels:
                return cand
        return None

    map_cols, map_rows = read_csv(P['map'])
    new_cols, new_rows = read_csv(P['new'])
    snap = lambda rows: json.dumps([{**r, 'native': r.get('native') or ''} for r in rows], ensure_ascii=False)  # noqa: E731
    map0, new0 = snap(map_rows), snap(new_rows)
    for cols in (map_cols, new_cols):
        if 'native' not in cols:
            cols.append('native')
    new_labels0 = {r['azsld'] for r in new_rows}
    flag_cols, flags = read_csv(P['flags']) if P['flags'].exists() else (FLAG_COLS, [])
    flags0 = json.dumps(flags, ensure_ascii=False)
    alt_cols, alt_rows = read_csv(P['alt'])
    alt_before = [dict(r) for r in alt_rows]
    sheet_cols, sheet = read_csv(P['sheet'])
    ans_col = next(c for c in sheet_cols if c.startswith('same_sign_answer'))
    log, conflicts, d5_written, todo_by_hand = [], [], 0, []

    def find(rows, **kv):
        return next((r for r in rows if all(r[k] == v for k, v in kv.items())), None)

    def set_flag(label, flag, note):
        row = find(flags, label=label, flag=flag)
        if row:
            row['note'] = note
        else:
            flags.append({'label': label, 'flag': flag, 'note': note})

    def drop_flag(label, flag):
        flags[:] = [f for f in flags if not (f['label'] == label and f['flag'] == flag)]

    for (section, item), rows in sorted(by_item.items()):
        code, corr, comment, who = decide(rows)
        a, b = rows[0]['a'], rows[0]['b']
        note = f'native review {today} ({who})' + (f': {corr}' if corr else '') + (f'; {comment}' if comment else '')
        action = 'none'
        if code == 'conflict':
            conflicts.append(f'{section} {item}: {corr}')
            action = 'skipped: reviewers disagree'
        elif code == 'unsure':
            action = 'none (unsure)'
        elif section == 'slovo_map':
            row = find(map_rows, slovo=a)
            if not row:
                action = 'row not found'
            elif code == 'yes':
                row['native'] = 'yes'
                action = 'confirmed'
            elif code == 'no' or not to_label(corr):
                map_rows.remove(row)
                action = 'removed from slovo_azsld.csv' + (f' (candidate: {corr})' if code == 'other' else '')
            else:
                row.update(azsld=to_label(corr), basis=add_note(row['basis'], f'native: {corr}'), native='yes')
                action = f'relabelled {b} → {row["azsld"]}'
        elif section in ('slovo_new_installed', 'slovo_new_unchecked'):
            row = find(new_rows, slovo=a)
            lab = to_label(corr) if code == 'other' else None
            if not row:
                action = 'row not found'
            elif code == 'yes':
                row['native'] = 'yes'
                if 'native yes' not in row['basis']:
                    row['basis'] = add_note(row['basis'], 'native yes = rank 0')
                action = 'confirmed'
            elif code == 'no':
                new_rows.remove(row)
                action = 'removed from slovo_new.csv'
            elif lab and lab not in new_labels0:
                new_rows.remove(row)
                if not find(map_rows, slovo=a):
                    map_rows.append({'slovo': a, 'azsld': lab, 'basis': f'native: {corr}', 'native': 'yes'})
                action = f'moved to slovo_azsld.csv as {lab}'
            else:
                row.update(azsld=lab or az_upper(corr), basis=add_note(row['basis'], f'native: {corr} = rank 0'), native='yes')
                action = f'relabelled {b} → {row["azsld"]}'
        elif section == 'slovo_azsld55':
            lab = b if code == 'yes' else to_label(corr) if code == 'other' else None
            if code == 'no':
                action = 'not added'
            elif not lab:
                action = f'not added (candidate: {corr})'
            elif find(map_rows, slovo=a):
                action = 'already in slovo_azsld.csv'
            else:
                map_rows.append({'slovo': a, 'azsld': lab, 'basis': 'native', 'native': 'yes'})
                action = f'added to slovo_azsld.csv as {lab}'
        elif section == 'd5':
            sid = item.split('/', 1)[1]
            row = find(sheet, id=sid)
            if not row or (row['gloss_a'], row['gloss_b']) != (a, b):
                action = 'row not found'
            else:
                row[ans_col] = SHEET_ANSWER[code if code != 'other' else 'unsure']
                row['reviewer'] = who
                row['comment'] = '; '.join(x for x in (corr, comment) if x)
                d5_written += 1
                action = f'sheet: {row[ans_col]}'
        elif section == 'literal':
            if code == 'no':
                set_flag(a, 'literal', note)
                action = 'flag literal'
            else:
                drop_flag(a, 'literal')
                action = 'native sign (no flag)'
        elif section == 'n8' and item.startswith('n8/poly/'):
            todo_by_hand.append(f'N8 {a}: {code} {corr} {comment}'.strip())
            action = 'logged (polysemy)'
        elif section == 'n8':
            row = find(alt_rows, text=a, alt=b)
            if row and code == 'yes':
                row['note'] = add_note(row['note'], f'native yes {today}')
                action = 'confirmed'
            elif row:
                alt_rows.remove(row)
                action = 'removed from text_alt.csv' + (f' (native: {corr})' if corr else '')
            elif code == 'yes':
                alt_rows.append({'text': a, 'alt': b, 'note': add_note(f'native yes {today}', comment)})
                action = 'added to text_alt.csv'
            else:
                action = 'proposal not added'
        elif section == 'n9':
            if code != 'yes':
                todo_by_hand.append(f'N9 {a}: {code} {corr} {comment}'.strip())
            action = 'logged (rule: change words.js / gloss2text.js by hand)' if code != 'yes' else 'rule confirmed'
        else:
            action = 'unknown section'
        log.append({'date': today, 'reviewer': who, 'section': section, 'item': item, 'a': a, 'b': b,
                    'answer': code if code != 'conflict' else 'conflict', 'correction': corr, 'comment': comment,
                    'action': action})

    # RSL bayraqları: quraşdırılmış modelin Slovo-dan gələn sinifləri, slovo_new.csv-nin indiki halından
    rsl_labels = ({f['label'] for f in flags if f['flag'].startswith('rsl')} | new_labels0) & set(labels)
    rows_of = defaultdict(list)
    for r in new_rows:
        rows_of[r['azsld']].append(r)
    for lab in sorted(rsl_labels):
        drop_flag(lab, 'rsl')
        drop_flag(lab, 'rsl-rejected')
        rs = rows_of.get(lab, [])
        if not rs:
            set_flag(lab, 'rsl-rejected', 'Slovo sign rejected in the native review; the installed word model keeps the class until retrained')
        elif not all(r.get('native') == 'yes' for r in rs):
            set_flag(lab, 'rsl', 'class trained on Slovo (Russian Sign Language) clips only; not confirmed by a native AzSL user')
    flags.sort(key=lambda f: (f['flag'], f['label']))

    # xülasə
    by_action = defaultdict(int)
    for x in log:
        by_action[(x['section'], x['action'].split(' (')[0].split(' → ')[0].split(' as ')[0])] += 1
    print(f'{sum(len(v) for v in by_item.values())} cavab sətri, {len(by_item)} bənd; V2 sətirləri keçildi: {v2}')
    for (sec, act), n in sorted(by_action.items()):
        print(f'  {sec:20s} {act:40s} {n}')
    print(f'slovo_azsld.csv sətirləri {len(read_csv(P["map"])[1])} → {len(map_rows)}, slovo_new.csv {len(read_csv(P["new"])[1])} → '
          f'{len(new_rows)}, text_alt.csv {len(alt_before)} → {len(alt_rows)}; bayraqlar: ' +
          ', '.join(f'{k} {sum(f["flag"] == k for f in flags)}' for k in sorted({f['flag'] for f in flags})))
    if conflicts:
        print('Rəy verənlər razılaşmır (keçildi):\n  ' + '\n  '.join(conflicts))
    if todo_by_hand:
        print('Əl ilə baxılacaq (N8 çoxmənalılıq, N9 qaydalar):\n  ' + '\n  '.join(todo_by_hand))
    if dry:
        print('--dry-run: heç nə yazılmadı')
        return

    # yalnız dəyişən fayllar yazılır (native sütunu ilk təsdiqdə əlavə olunur)
    if snap(map_rows) != map0:
        write_csv(P['map'], map_cols, map_rows)
    if snap(new_rows) != new0:
        write_csv(P['new'], new_cols, new_rows)
    if json.dumps(flags, ensure_ascii=False) != flags0 or not P['flags'].exists():
        write_csv(P['flags'], flag_cols, flags)
    if alt_rows != alt_before:
        write_csv(P['alt'], alt_cols, alt_rows)
    if d5_written:
        write_csv(P['sheet'], sheet_cols, sheet, bom=True)
    if log:
        old = not P['log'].exists()
        with open(P['log'], 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, LOG_COLS, lineterminator='\n')
            if old:
                w.writeheader()
            w.writerows(log)

    # D5: təsdiqlənmiş ailələr və brauzerin birləşməsi
    if d5_written:
        py = sys.executable
        subprocess.run([py, '-I', str(root / 'tools/same_sign/apply_review.py'), str(P['sheet']),
                        '--tag', f'native review {today}'], check=True)
        subprocess.run([py, '-I', str(root / 'tools/build_sign_merge.py')], check=True)

    # N8: avatar lüğətinin alt sahələri (yalnız text_alt.csv-dən gələnlər dəyişir; tools/text_alt.py)
    before, after = alt_map(alt_before), alt_map(alt_rows)
    if before != after:
        changed, missing = sync_alt(P['words'], after, before)
        print(f'avatar lüğəti: {changed} sözün alt sahəsi dəyişdi' + (f'; lüğətdə olmayan mətnlər: {missing}' if missing else ''))
    print(f'yazıldı; jurnal: {P["log"].relative_to(root)} (+{len(log)} sətir)')


if __name__ == '__main__':
    main()
