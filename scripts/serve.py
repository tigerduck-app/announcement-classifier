"""Serve the announcement classifier over HTTP for the backend.

Run: .venv/bin/python scripts/serve.py <model dir> [--host 127.0.0.1] [--port 40002] [--threads 2] [--device cpu]
GET  /health       {"status": "ok", "labels": [...], "threshold": 0.35}
POST /v1/classify  {"posts": [{"subject": "<Moodle subject>", "message": "<Moodle HTML>"}, ...]}
                or {"texts": ["<title>\n\n<Markdown content>", ...]}, at most MAX_TEXTS either way
  -> {"results": [{"labels": [...], "scores": {"exam_info": 0.97, ...}}, ...]}
Posts are turned into text the way the training data was (moodle_text.post_text) and each result
also carries that "text", unmasked. The model reads it with URLs, emails and student IDs masked.
Requests are scored one at a time. Bodies are never logged: announcements carry names.
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from moodle_text import model_input, post_text
from predict import Classifier

MAX_TEXTS = 64
MAX_BODY_BYTES = 4 * 1024 * 1024


def classify(clf, texts: list[str]) -> list[dict]:
    return [{"labels": clf.labels_for(s), "scores": {l: round(p, 4) for l, p in zip(clf.labels, s)}}
            for s in clf.scores([model_input(t) for t in texts])]


def _is_post(p: object) -> bool:
    return isinstance(p, dict) and isinstance(p.get("subject"), str) and isinstance(p.get("message"), str)


def parse_body(body: bytes) -> tuple[list[str], bool] | None:
    """The texts to score and whether they came as posts, or None for a malformed body."""
    try:
        data = json.loads(body)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    posts, texts = data.get("posts"), data.get("texts")
    if isinstance(posts, list) and 0 < len(posts) <= MAX_TEXTS and all(map(_is_post, posts)):
        return [post_text(p["subject"], p["message"]) for p in posts], True
    if isinstance(texts, list) and 0 < len(texts) <= MAX_TEXTS and all(isinstance(t, str) for t in texts):
        return texts, False
    return None


def make_handler(clf):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                return self.send(404, {"error": "not found"})
            self.send(200, {"status": "ok", "labels": clf.labels, "threshold": clf.threshold})

        def do_POST(self) -> None:
            if self.path != "/v1/classify":
                return self.send(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                return self.send(413 if length else 400, {"error": f"body must be 1 to {MAX_BODY_BYTES} bytes"})
            parsed = parse_body(self.rfile.read(length))
            if parsed is None:
                return self.send(400, {"error": f'expected 1 to {MAX_TEXTS} "posts" ({{"subject", "message"}}) or "texts"'})
            texts, as_posts = parsed
            with lock:
                results = classify(clf, texts)
            if as_posts:
                results = [{**r, "text": t} for r, t in zip(results, texts)]
            self.send(200, {"results": results})

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=40002)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    clf = Classifier(args.model_dir).to(args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(clf))
    print(f"serving {args.model_dir} on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    assert parse_body(b'{"texts": ["a"]}') == (["a"], False)
    assert parse_body(b'{"posts": [{"subject": "HW1", "message": "<p>due</p>"}]}') == (["HW1\n\ndue"], True)
    for bad in (b'{"texts": []}', b'{"text": "a"}', b'{"posts": [{"subject": "a"}]}', b"[1]", b"x"):
        assert parse_body(bad) is None, bad
    sys.exit(main())
