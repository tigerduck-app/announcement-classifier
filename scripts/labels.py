"""The 20 announcement labels, in a fixed order shared by labeling, evaluation and training."""

LABELS = [
    "exam_info", "exam_change", "exam_seating", "exam_review", "exam_prep",
    "grade_release",
    "assignment_new", "assignment_change", "assignment_note",
    "class_cancel", "class_change", "class_prep", "attendance",
    "materials", "grading_policy", "course_info", "grouping",
    "admin", "survey", "event",
]
INDEX = {label: i for i, label in enumerate(LABELS)}


def clean(labels: list) -> list[str]:
    """Known labels only, deduplicated, in LABELS order."""
    return [l for l in LABELS if l in set(labels)]


def guide_terms() -> dict[str, tuple[str, str]]:
    """label -> (Chinese name, definition), read from labeling_guide.md."""
    import re
    from pathlib import Path
    guide = (Path(__file__).resolve().parent.parent / "labeling_guide.md").read_text("utf-8")
    found = {m[0]: (m[1], m[2]) for m in re.findall(r"^- `(\w+)` (\S+?)：(.+)$", guide, re.M)}
    return {label: found[label] for label in LABELS}


if __name__ == "__main__":
    terms = guide_terms()
    assert terms["exam_info"][0] == "考試資訊" and len({n for n, _ in terms.values()}) == len(LABELS)
