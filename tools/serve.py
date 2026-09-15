"""Keşsiz lokal dev server (brauzer köhnə JS-i tutub saxlamasın)."""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCache(SimpleHTTPRequestHandler):
    # Windows-da Python MIME tiplərini reyestrdən oxuyur və .js-i tez-tez
    # text/plain verir — brauzer isə modulu yalnız JavaScript tipi ilə yükləyir.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.html': 'text/html; charset=utf-8',
        '.js': 'text/javascript; charset=utf-8',
        '.mjs': 'text/javascript; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
        '.json': 'application/json',
        '.wasm': 'application/wasm',
        '.vrm': 'model/gltf-binary',
        '.task': 'application/octet-stream',
    }

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

    def log_message(self, fmt, *args):
        msg = fmt % args
        if ' 200 ' not in msg:
            sys.stderr.write(msg + "\n")


if __name__ == '__main__':
    from pathlib import Path

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
    # Kök qovluq skriptin yerinə görə tapılır — hansı qovluqdan çağırılmasının
    # fərqi yoxdur (dostun iki dəfə klikləməsi də işləsin).
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / 'public'
    url = f"http://127.0.0.1:{port}"
    print(f"Chevir demo: {url}\n(dayandırmaq üçün Ctrl+C)", flush=True)
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    try:
        ThreadingHTTPServer(('127.0.0.1', port), partial(NoCache, directory=str(root))).serve_forever()
    except KeyboardInterrupt:
        print("\ndayandırıldı")
