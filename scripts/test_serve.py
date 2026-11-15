"""Check that serve.py answers with the same labels and scores as the model itself.

Run: .venv/bin/python scripts/test_serve.py <model dir>
Starts the server on a free local port, sends the synthetic posts in tests/fixtures as texts and as
raw Moodle posts, and a bad request.
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from moodle_text import model_input, post_text
from predict import Classifier
from serve import make_handler, read_release

ROOT = Path(__file__).resolve().parent.parent


def post_json(url: str, body: bytes) -> tuple[int, dict]:
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def main() -> int:
    clf = Classifier(sys.argv[1]).to("cpu")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(clf, read_release(sys.argv[1])))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    fixture = [json.loads(l) for l in open(ROOT / "tests/fixtures/synthetic_posts.jsonl", encoding="utf-8")]
    texts = [post_text(p["subject"], p["message"]) for p in fixture]

    with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
        health = json.load(r)
    assert health == {"status": "ok", "labels": clf.labels, "threshold": clf.threshold, "release": read_release(sys.argv[1])}

    status, answer = post_json(f"{base}/v1/classify", json.dumps({"texts": texts}).encode())
    assert status == 200 and len(answer["results"]) == len(texts)
    for got, want in zip(answer["results"], clf.scores([model_input(t) for t in texts])):
        assert got["labels"] == clf.labels_for(want) and "text" not in got
        assert all(abs(got["scores"][l] - p) < 1e-3 for l, p in zip(clf.labels, want))

    post = fixture[2]
    status, answer = post_json(f"{base}/v1/classify", json.dumps({"posts": [post]}).encode())
    want = clf.scores([model_input(post_text(post["subject"], post["message"]))])[0]
    assert status == 200 and answer["results"][0]["text"] == post_text(post["subject"], post["message"])
    assert answer["results"][0]["labels"] == clf.labels_for(want) and "exam_info" in answer["results"][0]["labels"]

    assert post_json(f"{base}/v1/classify", b'{"texts": []}')[0] == 400
    server.shutdown()
    print("serve.py ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
