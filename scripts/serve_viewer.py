#!/usr/bin/env python3
"""Serve the viewer from the repository root, on the loopback interface only.

The page lives in `viewer/` and reads `data/derived/viewer/` and the original images
under `data/raw/`, all by relative path. On top of handing files out, the server runs the
demo engine (`caypollard.demo`): it analyses a picture through every representation the
experiments used, keeps the result in memory under a token, and answers searches over the
frozen pool for it. Heavy models load on first use and stay loaded.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import http.server
import json
import secrets
import sys
import threading
import traceback
import urllib.parse
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "data/derived/demo/queries"

_analyses: collections.OrderedDict[str, object] = collections.OrderedDict()
_pool = None
_engine_lock = threading.Lock()


def pool():
    global _pool
    if _pool is None:
        from caypollard.demo.pool import Pool

        _pool = Pool()
    return _pool


def remember(token: str, analysis) -> None:
    _analyses[token] = analysis
    while len(_analyses) > 24:
        _analyses.popitem(last=False)


class Handler(http.server.SimpleHTTPRequestHandler):
    # ---- plumbing ----
    def send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def query(self) -> dict[str, str]:
        parsed = urllib.parse.urlparse(self.path)
        return {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

    def route(self) -> str:
        return urllib.parse.urlparse(self.path).path

    def end_headers(self) -> None:
        if self.path.endswith((".json", ".bin", ".js", ".css", ".html", "/")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(format, *args)

    # ---- routes ----
    def do_GET(self) -> None:
        path = self.route()
        if path in ("", "/"):
            self.send_response(302)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return
        if path == "/api/demo/explain":
            from caypollard.demo.explain import DEFAULTS, STEPS
            from caypollard.demo.namer import available
            from caypollard.demo.pool import CHANNELS

            self.send_json(
                {
                    "steps": STEPS,
                    "defaults": DEFAULTS,
                    "channels": [
                        c.__dict__ | {"table": str(c.table.relative_to(ROOT))}
                        for c in CHANNELS.values()
                    ],
                    "namer_available": available(),
                }
            )
            return
        if path == "/api/demo/analyse":
            try:
                self.analyse(None)
            except Exception as error:
                traceback.print_exc()
                self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = self.route()
        try:
            if path == "/api/demo/analyse":
                length = int(self.headers.get("Content-Length", "0"))
                self.analyse(self.rfile.read(length) if length else None)
            elif path == "/api/demo/search":
                length = int(self.headers.get("Content-Length", "0"))
                self.search(json.loads(self.rfile.read(length) or b"{}"))
            else:
                self.send_json({"error": "route inconnue"}, 404)
        except Exception as error:
            traceback.print_exc()
            self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)

    def analyse(self, body: bytes | None) -> None:
        """Analyse an uploaded image (POST body) or a pool picture (`?item=<id>`)."""
        from caypollard.demo.pipeline import analyse

        q = self.query()
        params = json.loads(q.get("params", "{}") or "{}")
        namer = q.get("namer", "sonnet")
        item = q.get("item")
        QUERIES.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(8)
        if item:
            index = pool().index()
            if item not in index:
                self.send_json({"error": f"{item} n'est pas dans le pool"}, 404)
                return
            items = json.loads(
                (ROOT / "data/derived/viewer/items.json").read_text(encoding="utf-8")
            )["items"]
            image_path = ROOT / items[index[item]]["img"]
        else:
            if not body:
                self.send_json({"error": "aucune image reçue"}, 400)
                return
            image_path = QUERIES / f"{token}.img"
            image_path.write_bytes(body)
        with _engine_lock:
            analysis, result = analyse(image_path, params, namer=namer, pool=pool(), item_id=item)
        remember(token, analysis)
        result["token"] = token
        result["item"] = item
        self.send_json(result)

    def search(self, request: dict) -> None:
        analysis = _analyses.get(str(request.get("token", "")))
        if analysis is None:
            self.send_json({"error": "analyse inconnue ou expirée : relancer l'analyse"}, 404)
            return
        weights = {str(k): float(v) for k, v in (request.get("weights") or {}).items()}
        k = int(request.get("k", 50))
        exclude = request.get("exclude")
        result = pool().search(analysis.vectors, weights, k=k, exclude=exclude)
        self.send_json(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not (ROOT / "data/derived/viewer/items.json").exists():
        print(
            "data/derived/viewer/items.json is missing: run `make viewer-build` first",
            file=sys.stderr,
        )
        sys.exit(1)

    handler = functools.partial(Handler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    server.verbose = args.verbose  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{args.port}/viewer/"
    print(f"caypollard viewer: {url}  (Ctrl-C pour arrêter)", file=sys.stderr)
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
