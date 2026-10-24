#!/usr/bin/env python3
"""Serve the human-interface preview only. No Viam imports, credentials, API or robot proxy."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

STATIC = Path(__file__).resolve().parent / 'static'
ASSETS = {
    '/': ('language.html', 'text/html; charset=utf-8'),
    '/index.html': ('language.html', 'text/html; charset=utf-8'),
    '/language.html': ('language.html', 'text/html; charset=utf-8'),
    '/face-servo.html': ('face-servo.html', 'text/html; charset=utf-8'),
    '/language.css': ('language.css', 'text/css'),
    '/language-app.js': ('language-app.js', 'text/javascript'),
    '/language-parser.mjs': ('language-parser.mjs', 'text/javascript'),
}


class PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        asset = ASSETS.get(urlsplit(self.path).path)
        if asset is None:
            self.send_error(404, 'Preview asset not found; no robot endpoints are exposed')
            return
        name, mime = asset
        body = (STATIC / name).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        self.send_error(405, 'Offline preview: commands cannot be sent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8091)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), PreviewHandler)
    print(f'Offline human interface: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
