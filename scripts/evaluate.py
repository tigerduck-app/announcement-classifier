"""Score the model on the gold set, pick its threshold and time CPU inference.

Run: .venv/bin/python scripts/evaluate.py <model dir>
The threshold is one global value. Its honest score comes from 2-fold cross-fitting on gold (pick on
one half, score the other); the deployed value is then picked on all 300 items and written to
<model dir>/threshold.json. Prints per-label scores and writes <model dir>/eval.json.
"""

import hashlib
import json
import sys
import time
from pathlib import Path


from compare import scores
from predict import Classifier

ROOT = Path(__file__).resolve().parent.parent
GRID = [round(0.05 * i, 2) for i in range(2, 19)]


def labeled(ids, probs, labels, t) -> dict[str, set[str]]:
    return {i: {l for l, p in zip(labels, ps) if p >= t} for i, ps in zip(ids, probs)}


def best_threshold(gold, ids, probs, labels) -> float:
    return max(GRID, key=lambda t: scores({i: gold[i] for i in ids}, labeled(ids, probs, labels, t))["micro_f1"])


def load_gold() -> tuple[list[dict], dict[str, set[str]]]:
    items = [json.loads(l) for l in open(ROOT / "gold/gold_items.jsonl", encoding="utf-8")]
    gold = {e["id"]: set(e["labels"]) for e in map(json.loads, open(ROOT / "gold/gold_labels.jsonl", encoding="utf-8"))}
    return items, gold


def cross_fitted(gold, probs: dict[str, list[float]], labels) -> dict[str, set[str]]:
    """Each gold item's labels under the threshold picked on the other half."""
    ids = list(probs)
    fold = {i: int(hashlib.sha1(i.encode()).hexdigest(), 16) % 2 for i in ids}
    crossed = {}
    for k in (0, 1):
        pick = [i for i in ids if fold[i] == k]
        rest = [i for i in ids if fold[i] != k]
        t = best_threshold(gold, pick, [probs[i] for i in pick], labels)
        crossed.update(labeled(rest, [probs[i] for i in rest], labels, t))
        print(f"fold {k}: threshold {t} picked on {len(pick)}, applied to {len(rest)}")
    return crossed


def main() -> int:
    model_dir = sys.argv[1]
    clf = Classifier(model_dir)
    items, gold = load_gold()
    ids = [e["id"] for e in items]
    probs = dict(zip(ids, clf.scores([e["text"] for e in items])))
    honest = scores(gold, cross_fitted(gold, probs, clf.labels))
    final_t = best_threshold(gold, ids, [probs[i] for i in ids], clf.labels)
    (Path(model_dir) / "threshold.json").write_text(json.dumps({"threshold": final_t}) + "\n")
    print(f"cross-fitted on gold: micro F1 {honest['micro_f1']:.3f}  macro F1 {honest['macro_f1']:.3f}  exact {honest['exact']:.3f}")
    for label, (support, p, r, f) in honest["rows"].items():
        print(f"  {label:18s} support {support:4d}  P {p:.2f}  R {r:.2f}  F1 {f:.2f}")
    print(f"deployed threshold {final_t}")

    cpu = clf.to("cpu")
    texts = [e["text"] for e in items[:64]]
    cpu.scores(texts[:4])
    start = time.time()
    for t in texts:
        cpu.scores([t])
    per_item = (time.time() - start) / len(texts) * 1000
    print(f"CPU latency, one announcement at a time: {per_item:.0f} ms")
    (Path(model_dir) / "eval.json").write_text(json.dumps({
        "gold_items": len(ids), "cross_fitted": {k: honest[k] for k in ("micro_f1", "macro_f1", "exact")},
        "per_label": {l: {"support": s, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
                      for l, (s, p, r, f) in honest["rows"].items()},
        "threshold": final_t, "cpu_ms_per_item": round(per_item),
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    assert labeled(["a"], [[0.7, 0.2]], ["x", "y"], 0.5) == {"a": {"x"}}
    sys.exit(main())
