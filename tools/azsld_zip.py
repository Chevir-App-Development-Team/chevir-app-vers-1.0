"""
AzSLD zip-lərindən tək faylı oxumaq — arxivi açmadan (Sentences 43.8 GB-dır).

Endirmə bitibsə, zip-in öz kataloqu oxunur (zipfile, zip64). Endirmə hələ gedirsə
(`<zip>.pieces` faylı var), kataloq Zenodo-dan bir dəfə götürülür (build_sentence_words.zip_index,
`<zip>.index.json` keşi) və fayl yalnız bütün baytları diskdə olanda oxunur.
"""
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PIECE = 32 << 20                  # tools/download_azsld.py-nin hissə ölçüsü
ZENODO = 'https://zenodo.org/records/14222948/files/'


class AzsldZip:
    def __init__(self, path):
        self.path = Path(path)
        self.pieces = Path(str(self.path) + '.pieces')
        if self.pieces.exists():
            import build_sentence_words as bsw
            bsw.URL = ZENODO + self.path.name
            self.ents = bsw.zip_index(self.path.with_suffix('.index.json'))
        else:
            with zipfile.ZipFile(self.path) as z:
                self.ents = {i.filename: [i.compress_type, i.compress_size, i.file_size, i.header_offset]
                             for i in z.infolist() if not i.is_dir()}

    def __contains__(self, name):
        return name in self.ents

    def available(self, name):
        if name not in self.ents:
            return False
        if not self.pieces.exists():
            return True
        done = set(json.loads(self.pieces.read_text()))
        _, csz, _, off = self.ents[name]
        return all(i in done for i in range(off // PIECE, (off + 30 + 1024 + csz) // PIECE + 1))

    def read(self, name):
        meth, csz, _, off = self.ents[name]
        with open(self.path, 'rb') as f:
            f.seek(off)
            head = f.read(30)
            if head[:4] != b'PK\x03\x04':
                raise IOError(f'{name}: lokal başlıq yoxdur')
            nl, el = struct.unpack('<HH', head[26:30])
            f.seek(off + 30 + nl + el)
            data = f.read(csz)
        return zlib.decompressobj(-15).decompress(data) if meth == 8 else data


def cam2(rec_id):
    """glosses_split.csv-dəki rec_id ("1/2022-05-31 15-58-30") → zip-dəki Cam2 videosu."""
    folder, stamp = rec_id.split('/', 1)
    return f'AzSLD_Sentences/{folder}/Cam2/{stamp}.mp4'
