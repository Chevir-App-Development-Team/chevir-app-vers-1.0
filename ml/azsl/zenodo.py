"""Read AzSLD from Zenodo without downloading whole archives.

Zenodo answers HTTP Range requests, so the central directory of a zip and any
single member can be fetched on their own. AzSLD_Sentences.zip is 43.9 GB — more
than Kaggle keeps — so recordings are read one at a time and only landmarks stay.
"""
import io
import os
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass

RECORD = '14222948'
ARCHIVES = {
    'words200': 'AzSLD_Words_200.zip',
    'words100': 'AzSLD_Words_100.zip',
    'fingerspelling': 'AzSLD_Fingerspelling.zip',
    'sentences': 'AzSLD_Sentences.zip',
}
UA = {'User-Agent': 'azsl-landmarks/1.0 (+https://github.com/Chevir-App-Development-Team)'}


def archive_url(key: str) -> str:
    return f'https://zenodo.org/records/{RECORD}/files/{ARCHIVES[key]}?download=1'


def http_get(url: str, start: int | None = None, end: int | None = None,
             retries: int = 8, timeout: int = 120) -> bytes:
    """GET (optionally bytes [start, end)) with backoff — Zenodo rate-limits bursts (429)."""
    headers = dict(UA)
    if start is not None:
        headers['Range'] = f'bytes={start}-{end - 1}'
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                data = r.read()
            if start is not None and len(data) != end - start:
                raise IOError(f'short read: {len(data)} of {end - start} bytes')
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last, wait = e, min(300, 5 * 2 ** attempt)
            ra = e.headers.get('Retry-After') if e.headers else None
            if ra and ra.isdigit():
                wait = max(wait, int(ra))
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last, wait = e, min(300, 5 * 2 ** attempt)
        time.sleep(wait)
    raise IOError(f'GET {url} failed after {retries} tries: {last}')


def download(url: str, dest: str, chunk: int = 1 << 24) -> str:
    """Whole-file download in ranged chunks; resumes if `dest` is partly there."""
    total = int(_head_size(url))
    done = os.path.getsize(dest) if os.path.exists(dest) else 0
    with open(dest, 'ab') as f:
        while done < total:
            end = min(done + chunk, total)
            f.write(http_get(url, done, end))
            done = end
    return dest


def _head_size(url: str) -> int:
    req = urllib.request.Request(url, headers={**UA, 'Range': 'bytes=0-0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Range'].split('/')[-1])


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; just enough for zipfile to read the directory."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        self.size = _head_size(url)

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if not n:
            return b''
        if n > 4 * self.block:                     # big reads (the central directory) in one go
            data = http_get(self.url, self.pos, self.pos + n)
        else:
            out, p = bytearray(), self.pos
            while p < self.pos + n:
                b = p // self.block
                if b not in self.cache:
                    s = b * self.block
                    self.cache[b] = http_get(self.url, s, min(s + self.block, self.size))
                chunk, o = self.cache[b], p - b * self.block
                take = min(len(chunk) - o, self.pos + n - p)
                out += chunk[o:o + take]
                p += take
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


@dataclass(frozen=True)
class Member:
    """Picklable pointer to one zip member — workers fetch it with a single request."""
    url: str
    name: str
    offset: int          # local header offset
    csize: int
    size: int
    method: int
    crc: int


def list_members(url: str) -> list[Member]:
    z = zipfile.ZipFile(RangeFile(url))
    return [Member(url, i.filename, i.header_offset, i.compress_size, i.file_size, i.compress_type, i.CRC)
            for i in z.infolist() if not i.is_dir()]


def _slack(m: Member) -> int:
    return 30 + len(m.name.encode('utf-8')) + 1024


def _unpack(m: Member, raw: bytes) -> bytes:
    """`raw` starts at the member's local header; returns the verified, decompressed data."""
    if raw[:4] != b'PK\x03\x04':
        raise IOError(f'{m.name}: no local file header at {m.offset}')
    n, e = int.from_bytes(raw[26:28], 'little'), int.from_bytes(raw[28:30], 'little')
    start = 30 + n + e
    data = raw[start:start + m.csize]
    if len(data) < m.csize:
        data = http_get(m.url, m.offset + start, m.offset + start + m.csize)
    if m.method == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(data)
    elif m.method != zipfile.ZIP_STORED:
        raise IOError(f'{m.name}: unsupported compression {m.method}')
    if zlib.crc32(data) != m.crc:
        raise IOError(f'{m.name}: CRC mismatch')
    return data


def read_member(m: Member) -> bytes:
    """Local header + data in one Range request (a second one only if the local extra field is huge)."""
    return _unpack(m, http_get(m.url, m.offset, m.offset + _slack(m) + m.csize))


def read_many(members: list[Member], max_span: int = 8 << 20):
    """Small neighbouring members (annotation JSONs) fetched as shared spans: ~1 request per folder
    instead of one per file. Yields (member, data)."""
    ms = sorted(members, key=lambda m: m.offset)
    i = 0
    while i < len(ms):
        j, end = i, ms[i].offset + _slack(ms[i]) + ms[i].csize
        while j + 1 < len(ms):
            nxt = ms[j + 1]
            nend = nxt.offset + _slack(nxt) + nxt.csize
            if nxt.offset > end + (64 << 10) or nend - ms[i].offset > max_span:
                break
            j, end = j + 1, max(end, nend)
        span = http_get(ms[i].url, ms[i].offset, end)
        for m in ms[i:j + 1]:
            yield m, _unpack(m, span[m.offset - ms[i].offset:])
        i = j + 1
