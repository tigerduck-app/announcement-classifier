# Announcement classifier

A multi-label classifier that tags NTUST Moodle course announcements with one or more of 20 labels. It is `intfloat/multilingual-e5-base` fine-tuned, about 35 ms per announcement on CPU.

[繁體中文](README.md)

## Labels

Definitions and decision rules are in [`labeling_guide.md`](labeling_guide.md) (Traditional Chinese).

- `exam_info` Exam info: date, time, place, scope, format or rules of a specific exam
- `exam_change` Exam change: a published exam is moved, relocated, rescoped or cancelled, or confirmed to go ahead
- `exam_seating` Exam seating: the seating chart or room assignment for an exam
- `exam_review` Exam review: when and where to see or collect graded exam papers
- `exam_prep` Exam material: past papers, mock exams, review sessions, solutions to exam and quiz questions
- `grade_release` Grade release: any grade published or corrected, or students asked to check theirs
- `assignment_new` New assignment: an assignment, project or report with its requirements and deadline
- `assignment_change` Assignment timing change: deadline changed, late or make-up submission opened, submission area reopened
- `assignment_note` Assignment note: follow-up clarification, correction or hint for a published assignment
- `class_cancel` Class cancelled: a session is not held
- `class_change` Class change: time, room or mode changed; make-up or swapped sessions
- `class_prep` Class prep: things to bring, read or install before the next class; textbook orders
- `attendance` Attendance: roll-call method, sign-in, absence records, leave rules
- `materials` Materials: slides, handouts, recordings, exercises and their solutions, readings
- `grading_policy` Grading policy: weights, criteria, bonus rules and changes to them
- `course_info` Course info: contacts, office hours, course groups, syllabus, schedule
- `grouping` Grouping: group sign-up forms, lists, deadlines, changes
- `admin` Administration: registration, add/drop, withdrawal, certificates and other procedures
- `survey` Survey: teaching evaluations, course surveys, feedback forms
- `event` Event: talks, seminars, competitions, exhibitions and other activities

## Metrics

On 300 human-labelled gold announcements, estimated with 2-fold cross-fitting (the threshold is picked on the other half):

| micro F1 | macro F1 | exact match |
|---|---|---|
| 0.870 | 0.843 | 0.703 |

One global threshold, 0.35 (`threshold.json`). Per label:

| Label | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| `exam_info` | 68 | 0.914 | 0.941 | 0.928 |
| `exam_change` | 13 | 0.909 | 0.769 | 0.833 |
| `exam_seating` | 7 | 1.000 | 1.000 | 1.000 |
| `exam_review` | 16 | 0.941 | 1.000 | 0.970 |
| `exam_prep` | 14 | 0.857 | 0.429 | 0.571 |
| `grade_release` | 35 | 0.889 | 0.914 | 0.901 |
| `assignment_new` | 41 | 0.872 | 0.829 | 0.850 |
| `assignment_change` | 7 | 0.714 | 0.714 | 0.714 |
| `assignment_note` | 14 | 1.000 | 0.571 | 0.727 |
| `class_cancel` | 38 | 0.925 | 0.974 | 0.949 |
| `class_change` | 36 | 0.875 | 0.778 | 0.824 |
| `class_prep` | 14 | 0.923 | 0.857 | 0.889 |
| `attendance` | 17 | 0.789 | 0.882 | 0.833 |
| `materials` | 63 | 0.862 | 0.889 | 0.875 |
| `grading_policy` | 13 | 0.750 | 0.692 | 0.720 |
| `course_info` | 19 | 1.000 | 0.895 | 0.944 |
| `grouping` | 14 | 0.778 | 1.000 | 0.875 |
| `admin` | 10 | 0.875 | 0.700 | 0.778 |
| `survey` | 6 | 0.750 | 1.000 | 0.857 |
| `event` | 13 | 0.786 | 0.846 | 0.815 |

## HTTP API

`scripts/serve.py` is a standard-library HTTP server. It scores one request at a time, never logs request bodies and has no authentication, so expose it only on a trusted network.

```
GET  /health       -> {"status": "ok", "labels": [...], "threshold": 0.35}
POST /v1/classify  {"posts": [{"subject": "...", "message": "<Moodle HTML>"}, ...]}
                or {"texts": ["<title>\n\n<Markdown content>", ...]}     (at most 64 items)
  -> {"results": [{"labels": ["exam_info"], "scores": {"exam_info": 0.97, ...}}, ...]}
```

`posts` are converted from Moodle HTML the way the training data was (`scripts/moodle_text.py`), and each result also carries that `text`. URLs, emails and student IDs are masked before the model reads them.

## Deploy

A release holds `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `threshold.json`, `eval.json` and `SHA256SUMS`.

```bash
deploy/install.sh model-2026-11-15   # download, verify SHA256SUMS, create the venv, run test_serve.py
.venv/bin/python scripts/serve.py model --host 127.0.0.1 --port 40002 --threads 2
```

`deploy/ai.tigerduck.classifier.plist` is a macOS launchd definition; adjust its paths to your setup. The service has no authentication, so keep the firewall closed to the LAN.

## Retrain

```bash
.venv/bin/python scripts/train.py <announcements.jsonl> <out dir> --labels reviewed
.venv/bin/python scripts/evaluate.py <out dir>
.venv/bin/python scripts/compare_models.py <model A> <model B>
```

`train.py` holds out 10% of courses (course code without its term) for validation, trains 4 epochs and keeps the epoch with the lowest validation loss. `evaluate.py` picks the threshold on gold and writes `eval.json`.

## Training data format

The training data is not published. The scripts expect:

- `<announcements.jsonl>`: one `{"id": "...", "text": "<title>\n\n<body>"}` per line, body as plain Markdown.
- `labels/<name>.jsonl`: one `{"id": "...", "labels": ["exam_info", ...]}` per line, using only the 20 labels above.
- `gold/gold_items.jsonl` (`id`, `text`) and `gold/gold_labels.jsonl` (`id`, `labels`): the human-labelled evaluation set, excluded from training.

## Limitations

- Built for Traditional Chinese (with some English) NTUST Moodle course announcements. Other schools or genres are not measured.
- The gold set has only 300 items and some labels have 6 to 13, so per-label numbers are noisy. `exam_prep`, `assignment_change`, `grading_policy` and `assignment_note` are weaker.
- Names are not fully masked in the training data.
- The test announcements in `tests/fixtures/synthetic_posts.jsonl` are invented.

## License

[AGPL-3.0](LICENSE). The base model [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) is MIT licensed; thanks to its authors.
