"""Compare two or more label files on the same ids: agreement, per-label scores, disagreements.

Run: python3 compare.py <reference.jsonl> <other.jsonl> [--show <items.jsonl>]
Scores treat the first file as the reference. --show prints each disagreeing item with both label sets.
"""

import json
import re
import sys

from labels import LABELS


def load(path: str) -> dict[str, set[str]]:
    out = {}
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if "labels" in e:
            out[e["id"]] = set(e["labels"])
    return out


def scores(ref: dict, other: dict) -> dict:
    ids = ref.keys() & other.keys()
    rows = {}
    for label in LABELS:
        tp = sum(label in ref[i] and label in other[i] for i in ids)
        fp = sum(label not in ref[i] and label in other[i] for i in ids)
        fn = sum(label in ref[i] and label not in other[i] for i in ids)
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        rows[label] = (tp + fn, p, r, 2 * p * r / (p + r) if p + r else 0.0)
    tp = sum(len(ref[i] & other[i]) for i in ids)
    fp = sum(len(other[i] - ref[i]) for i in ids)
    fn = sum(len(ref[i] - other[i]) for i in ids)
    micro = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    macro = sum(v[3] for v in rows.values()) / len(rows)
    exact = sum(ref[i] == other[i] for i in ids) / len(ids)
    return {"n": len(ids), "rows": rows, "micro_f1": micro, "macro_f1": macro, "exact": exact}


def main() -> int:
    ref, other = load(sys.argv[1]), load(sys.argv[2])
    s = scores(ref, other)
    print(f"n {s['n']}  micro F1 {s['micro_f1']:.3f}  macro F1 {s['macro_f1']:.3f}  exact match {s['exact']:.3f}")
    for label, (support, p, r, f) in s["rows"].items():
        print(f"  {label:18s} support {support:4d}  P {p:.2f}  R {r:.2f}  F1 {f:.2f}")
    if "--show" in sys.argv:
        texts = {e["id"]: e["text"] for e in map(json.loads, open(sys.argv[sys.argv.index("--show") + 1], encoding="utf-8"))}
        for i in sorted(ref.keys() & other.keys()):
            if ref[i] != other[i]:
                print(f"\n### {i}  A={sorted(ref[i])}  B={sorted(other[i])}\n{re.sub(r'\n{2,}', chr(10), texts[i])[:700]}")
    return 0


if __name__ == "__main__":
    _s = scores({"a": {"exam_info"}, "b": set()}, {"a": {"exam_info", "materials"}, "b": set()})
    assert _s["rows"]["exam_info"][3] == 1.0 and _s["rows"]["materials"][1] == 0.0 and _s["exact"] == 0.5
    sys.exit(main())
