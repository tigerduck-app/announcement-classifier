"""Classify announcements with the fine-tuned model.

Run: .venv/bin/python scripts/predict.py <model dir> <input.jsonl> <output.jsonl>
     .venv/bin/python scripts/predict.py <model dir> --text "<title>\n\n<content>"
Input lines carry "id" and "text" (title, blank line, content; masked or not). Output lines are
{"id", "labels", "scores"}. The threshold comes from <model dir>/threshold.json, else 0.5.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_TOKENS = 384
BATCH = 32


class Classifier:
    def __init__(self, model_dir: str) -> None:
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        path = Path(model_dir) / "threshold.json"
        self.threshold = json.loads(path.read_text())["threshold"] if path.exists() else 0.5

    def to(self, device: str) -> "Classifier":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def scores(self, texts: list[str]) -> list[list[float]]:
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), BATCH):
                enc = self.tokenizer(texts[s:s + BATCH], truncation=True, max_length=MAX_TOKENS,
                                     padding=True, return_tensors="pt").to(self.device)
                out += torch.sigmoid(self.model(**enc).logits).cpu().tolist()
        return out

    def labels_for(self, scores: list[float]) -> list[str]:
        return [l for l, p in zip(self.labels, scores) if p >= self.threshold]


def main() -> int:
    clf = Classifier(sys.argv[1])
    if sys.argv[2] == "--text":
        s = clf.scores([sys.argv[3].replace("\\n", "\n")])[0]
        print(json.dumps({"labels": clf.labels_for(s), "scores": dict(zip(clf.labels, map(lambda p: round(p, 3), s)))}, ensure_ascii=False))
        return 0
    items = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
    scores = clf.scores([e["text"] for e in items])
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        for e, s in zip(items, scores):
            f.write(json.dumps({"id": e["id"], "labels": clf.labels_for(s), "scores": [round(p, 4) for p in s]}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
