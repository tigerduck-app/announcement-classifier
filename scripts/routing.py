"""How often a confidence band would send announcements to the LLM, and how good the rest are.

Run: .venv/bin/python scripts/routing.py <model dir> [--only label,label,...]
An announcement is "uncertain" when a watched label's score falls inside [low, high); every label is
watched unless --only names some. For each band the table shows the share routed to the LLM on the 300
gold items and on predictions/announcements_labeled.jsonl (all announcements), then for the gold items
the model keeps: the exact-match rate and the precision and recall of the --only labels, or without
--only of the exam labels (the ones the app surfaces).
"""

import json
import sys
from pathlib import Path

from predict import Classifier

ROOT = Path(__file__).resolve().parent.parent
EXAM = ["exam_info", "exam_change", "exam_seating", "exam_review", "exam_prep", "grade_release"]
BANDS = [(0.4, 0.4), (0.3, 0.6), (0.2, 0.7), (0.15, 0.8), (0.1, 0.9), (0.05, 0.95)]


def uncertain(scores: list[float], low: float, high: float) -> bool:
    return any(low <= p < high for p in scores)


def main() -> int:
    clf = Classifier(sys.argv[1])
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    watch = [clf.labels.index(l) for l in only] if only else list(range(len(clf.labels)))
    measured = set(only or EXAM)
    items = [json.loads(l) for l in open(ROOT / "gold/gold_items.jsonl", encoding="utf-8")]
    gold = {e["id"]: set(e["labels"]) for e in map(json.loads, open(ROOT / "gold/gold_labels.jsonl", encoding="utf-8"))}
    scores = dict(zip((e["id"] for e in items), clf.scores([e["text"] for e in items])))
    real_path = ROOT / "predictions/announcements_labeled.jsonl"
    real = [[r["scores"][clf.labels[k]] for k in watch] for r in map(json.loads, open(real_path, encoding="utf-8"))] \
        if real_path.exists() else []
    print(f"threshold {clf.threshold}; watched {only or 'all labels'}; P/R over {sorted(measured)}")
    print("band          routed  real-routed  kept-exact  kept-P  kept-R")
    for low, high in BANDS:
        kept = [i for i, s in scores.items() if not uncertain([s[k] for k in watch], low, high)]
        pred = {i: set(clf.labels_for(scores[i])) for i in kept}
        exact = sum(pred[i] == gold[i] for i in kept) / max(1, len(kept))
        tp = sum(len(pred[i] & gold[i] & measured) for i in kept)
        p = tp / max(1, sum(len(pred[i] & measured) for i in kept))
        r = tp / max(1, sum(len(gold[i] & measured) for i in kept))
        real_share = sum(uncertain(s, low, high) for s in real) / len(real) if real else float("nan")
        print(f"[{low:.2f},{high:.2f})  {1 - len(kept) / len(scores):6.1%}  {real_share:11.1%}  {exact:10.1%}  {p:6.1%}  {r:6.1%}")
    return 0


if __name__ == "__main__":
    assert uncertain([0.1, 0.5], 0.3, 0.6) and not uncertain([0.1, 0.9], 0.3, 0.6)
    sys.exit(main())
