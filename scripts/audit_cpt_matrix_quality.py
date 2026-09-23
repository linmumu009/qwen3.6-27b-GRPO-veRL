"""Bind frozen label and design reviews; keep all question-level details private."""
import argparse
import json
from pathlib import Path
from audit_cpt_operation_matrix import audit as audit_matrix
from audit_cpt_two_family import texts, grams
from cpt_two_family import read, sha, freeze, validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    matrix = read(root / "matrix.v2.private.json")
    spec_review = read(root / "matrix_review.v2.private.json")
    assert audit_matrix(matrix, spec_review)["authoring_released"]
    assert spec_review["reviewed_matrix_sha256"] == sha(root / "matrix.v2.private.json")
    assert spec_review["reviewed_source_sha256"] == sha(root / "source_only.private.json")
    train = read(root / "train_author.private.json")["candidates"]
    dev = read(root / "development_author.private.json")["items"]
    tasks = train + dev
    blind = read(root / "blind_review_input.private.json")
    for name, value in blind["input_sha256"].items():
        assert sha(root / name) == value, "draft changed after blind export"
    sources = {f["family_id"]: f["evidence_text"] for f in read(root / "source_only.private.json")["families"]}
    reviews = []
    for name, input_name in [("development_blind_review.private.json", "development_blind_input.private.json"),
                             ("train_blind_review.private.json", "blind_review_input.private.json")]:
        r = read(root / name)
        assert r["reviewed_input_sha256"] == sha(root / input_name)
        reviews.extend(r["items"])
    assert len(tasks) == len(reviews) == 28
    judgments = {r["id"]: r for r in reviews}
    assert len(judgments) == 28 and set(judgments) == {t["id"] for t in tasks}
    failures = []
    for task in tasks:
        validate(task, sources[task["family"]])
        r = judgments[task["id"]]
        flags = ["supported", "unambiguous", "self_contained", "scope_preserved", "not_answer_leaking", "design_satisfied", "pass"]
        if any(r.get(k) is not True for k in flags) or r["derived_indices"] != task["correct_indices"]:
            failures.append(task["id"])
    design = read(root / "design_review.private.json")
    for name in ("matrix.v2.private.json", "train_author.private.json", "development_author.private.json",
                 "development_blind_review.private.json", "train_blind_review.private.json"):
        assert design["input_sha256"][name] == sha(root / name)
    history = [Path("CPT_resources/llin-transfer-audit-20260915-01/frozen_cases.private.jsonl")]
    for directory in ("llin-logistika-independent-20260922-01", "llin-remaining-operations-20260923-01", "llin-two-family-20260923-01"):
        folder = Path("CPT_resources") / directory
        history.extend(sorted(set(folder.glob("author*.private.json")) | set(folder.glob("*author*.private.json")) | set(folder.glob("diagnostic.private.json"))))
    corpus = []
    for path in history:
        value = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.suffix == ".jsonl" else read(path)
        corpus.extend((str(path), grams(row["question"])) for row in texts(value))
    hits = []
    for task in tasks:
        current = grams(task["question"])
        for path, other in corpus:
            score = len(current & other) / len(current | other) if current | other else 0
            if score >= .6:
                hits.append({"id": task["id"], "history": path, "jaccard": score})
    passed = not failures and design.get("passed") is True and not design.get("remaining_blockers") and not design.get("cross_split_conflicts")
    report = dict(passed=passed, tasks=28, training_tasks=20, development_tasks=8,
                  label_or_content_failures=len(failures), design_passed=design.get("passed") is True,
                  cross_split_conflicts=len(design.get("cross_split_conflicts", [])),
                  historical_records=len(corpus), lexical_history_hits=len(hits),
                  historical_overlap_is_not_independence_proof=True, student_calls=0, training_allowed=False,
                  inputs={n: sha(root / n) for n in ("matrix.v2.private.json", "source_only.private.json", "train_author.private.json",
                      "development_author.private.json", "development_blind_review.private.json", "train_blind_review.private.json", "design_review.private.json")},
                  history_inputs={str(path): sha(path) for path in history})
    freeze(root / "quality_details.private.json", dict(failures=failures, history_hits=hits, design_blockers=design.get("remaining_blockers", [])))
    freeze(root / "quality.safe.json", report)
    freeze(a.output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
