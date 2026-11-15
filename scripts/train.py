"""Fine-tune a multilingual encoder as a multi-label announcement classifier.

Run: .venv/bin/python scripts/train.py <announcements.jsonl> <out dir> [--limit N] [--epochs N] [--labels NAME]
         [--base NAME] [--lr X]
Targets are the mean of the Gemma and Qwen labels (labels/full_*.jsonl), or with --labels the hard
labels in labels/NAME.jsonl; gold items are held out.
Validation holds out whole courses (course code without its term), so a course's templated
announcements never sit on both sides. Writes the best model by validation loss to <out dir>.
"""

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from labels import INDEX, LABELS

ROOT = Path(__file__).resolve().parent.parent
BASE = "intfloat/multilingual-e5-base"
MAX_TOKENS = 384
BATCH = 16
LR = 3e-5
VAL_SHARE = 0.1
SEED = 20261114


def teacher_targets(source: str = "teachers") -> dict[str, list[float]]:
    """Mean of the Gemma and Qwen labels, or hard targets from labels/<source>.jsonl."""
    if source != "teachers":
        return {e["id"]: [1.0 if l in e["labels"] else 0.0 for l in LABELS]
                for e in map(json.loads, open(ROOT / f"labels/{source}.jsonl", encoding="utf-8"))}
    votes: dict[str, list[list[float]]] = {}
    for name in ("gemma", "qwen"):
        for e in map(json.loads, open(ROOT / f"labels/full_{name}.jsonl", encoding="utf-8")):
            if "labels" in e:
                v = [0.0] * len(LABELS)
                for label in e["labels"]:
                    v[INDEX[label]] = 1.0
                votes.setdefault(e["id"], []).append(v)
    return {i: [sum(col) / len(vs) for col in zip(*vs)] for i, vs in votes.items()}


def split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = sorted({e["course_codes"][0][4:] for e in items})
    random.Random(SEED).shuffle(groups)
    val_groups = set(groups[: max(1, int(len(groups) * VAL_SHARE))])
    return ([e for e in items if e["course_codes"][0][4:] not in val_groups],
            [e for e in items if e["course_codes"][0][4:] in val_groups])


def batches(items: list[dict], tokenizer, shuffle: bool):
    order = list(range(len(items)))
    if shuffle:
        random.shuffle(order)
    for s in range(0, len(order), BATCH):
        chunk = [items[i] for i in order[s:s + BATCH]]
        # Few distinct shapes: MPS caches memory per input shape, and with free lengths larger
        # models swapped within an hour.
        enc = tokenizer([e["text"] for e in chunk], truncation=True, max_length=MAX_TOKENS,
                        padding=True, pad_to_multiple_of=64, return_tensors="pt")
        yield enc, torch.tensor([e["target"] for e in chunk])


def evaluate(model, items, tokenizer, device) -> float:
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total = 0.0
    with torch.no_grad():
        for enc, target in batches(items, tokenizer, shuffle=False):
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            total += loss_fn(logits, target.to(device)).item()
    model.train()
    return total / (len(items) * len(LABELS))


def main() -> int:
    src, out = sys.argv[1], Path(sys.argv[2])
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    epochs = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 4
    source = sys.argv[sys.argv.index("--labels") + 1] if "--labels" in sys.argv else "teachers"
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else BASE
    lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else LR
    random.seed(SEED)
    torch.manual_seed(SEED)
    gold = {json.loads(l)["id"] for l in open(ROOT / "gold/gold_items.jsonl", encoding="utf-8")}
    targets = teacher_targets(source)
    items = [{**e, "target": targets[e["id"]]} for e in map(json.loads, open(src, encoding="utf-8"))
             if e["id"] in targets and e["id"] not in gold]
    if limit:
        items = items[:limit]
    train, val = split(items)
    print(f"train {len(train)}  val {len(val)}  held-out gold {len(gold)}", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(LABELS), problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id=INDEX,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train) / BATCH)
    warmup = int(steps * 0.06)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / max(1, warmup), max(0.0, (steps - s) / max(1, steps - warmup))))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best = math.inf
    start = time.time()
    step = 0
    for epoch in range(epochs):
        for enc, target in batches(train, tokenizer, shuffle=True):
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            loss = loss_fn(logits, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 200 == 0:
                if device.type == "mps":
                    torch.mps.empty_cache()
                print(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - start:.0f}s", flush=True)
        val_loss = evaluate(model, val, tokenizer, device)
        print(f"epoch {epoch + 1} val loss {val_loss:.4f}", flush=True)
        if val_loss < best:
            best = val_loss
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    _train, _val = split([{"course_codes": [c]} for c in ("T01ACS1", "T02ACS1", "T01AEE2", "T02AME3")])
    assert {e["course_codes"][0][4:] for e in _train}.isdisjoint(e["course_codes"][0][4:] for e in _val)
    sys.exit(main())
