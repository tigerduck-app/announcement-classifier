"""Compare two models on gold, with a paired bootstrap for the difference.

Run: .venv/bin/python scripts/compare_models.py <model dir A> <model dir B>
Both models' labels are cross-fitted as in evaluate.py. The bootstrap resamples the 300 gold items
(the same draw for both models) and reports B - A with its 95% interval and how often B came out ahead.
"""

import sys

import numpy as np

from compare import scores
from evaluate import cross_fitted, load_gold
from labels import LABELS
from predict import Classifier

DRAWS = 5000
SEED = 7


def counts(gold, pred, ids) -> np.ndarray:
    """Per item and label: [true positive, false positive, false negative]."""
    out = np.zeros((len(ids), len(LABELS), 3))
    for n, i in enumerate(ids):
        for k, l in enumerate(LABELS):
            out[n, k] = (l in gold[i] and l in pred[i], l not in gold[i] and l in pred[i], l in gold[i] and l not in pred[i])
    return out


def metrics(c: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """micro F1, macro F1 and exact match for each draw; c is draws x labels x 3 summed counts."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    micro = 2 * tp.sum(-1) / np.maximum(2 * tp.sum(-1) + fp.sum(-1) + fn.sum(-1), 1)
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 1.0)
    macro = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0).mean(-1)
    return np.stack([micro, macro, exact], -1)


def main() -> int:
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    items, gold = load_gold()
    ids = [e["id"] for e in items]
    preds = []
    for d in (a_dir, b_dir):
        clf = Classifier(d)
        assert clf.labels == LABELS
        preds.append(cross_fitted(gold, dict(zip(ids, clf.scores([e["text"] for e in items]))), clf.labels))
    a, b = (scores(gold, p) for p in preds)
    print(f"{'':14s} {'A':>7s} {'B':>7s} {'B-A':>7s}   A = {a_dir}, B = {b_dir}")
    for key in ("micro_f1", "macro_f1", "exact"):
        print(f"{key:14s} {a[key]:7.3f} {b[key]:7.3f} {b[key] - a[key]:+7.3f}")

    ca, cb = (counts(gold, p, ids) for p in preds)
    ea, eb = (np.array([p[i] == gold[i] for i in ids], float) for p in preds)
    whole = metrics(np.stack([ca.sum(0), cb.sum(0)]), np.array([ea.mean(), eb.mean()]))
    assert np.allclose(whole, [[s[k] for k in ("micro_f1", "macro_f1", "exact")] for s in (a, b)])
    draw = np.random.default_rng(SEED).integers(0, len(ids), (DRAWS, len(ids)))
    diff = metrics(cb[draw].sum(1), eb[draw].mean(1)) - metrics(ca[draw].sum(1), ea[draw].mean(1))
    print(f"\npaired bootstrap, {DRAWS} draws of {len(ids)} items: B - A, 95% interval, share of draws B > A")
    for n, key in enumerate(("micro_f1", "macro_f1", "exact")):
        lo, hi = np.percentile(diff[:, n], [2.5, 97.5])
        print(f"  {key:12s} {diff[:, n].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  {(diff[:, n] > 0).mean():.0%}")

    wins = sum(preds[1][i] == gold[i] != preds[0][i] for i in ids)
    losses = sum(preds[0][i] == gold[i] != preds[1][i] for i in ids)
    print(f"\nitems only B gets exactly right {wins}, only A {losses}")
    print(f"\n{'label':18s} {'support':>7s} {'A F1':>6s} {'B F1':>6s} {'B-A':>6s}")
    for l in LABELS:
        s, fa, fb = a["rows"][l][0], a["rows"][l][3], b["rows"][l][3]
        print(f"{l:18s} {s:7d} {fa:6.2f} {fb:6.2f} {fb - fa:+6.2f}")
    return 0


if __name__ == "__main__":
    _c = np.array([[[1, 0, 0], [0, 1, 1]]], float)  # one draw, two labels
    assert np.allclose(metrics(_c, np.array([0.5])), [[0.5, 0.5, 0.5]])
    sys.exit(main())
