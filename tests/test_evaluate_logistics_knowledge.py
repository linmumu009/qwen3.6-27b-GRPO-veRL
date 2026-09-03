import csv
import json
import zipfile
from pathlib import Path

import pytest

from scripts.evaluate_logistics_knowledge import (
    build_safe_result,
    compare_safe_results,
    load_logistika,
    load_sc_knowledge,
    mcnemar_exact_pvalue,
    parse_answers,
)


def _write_logistika(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question_id", "question_type", "question", "choices", "answer", "subject"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "1",
                "question_type": "multiple_choice",
                "question": "Which modes are used for freight?",
                "choices": json.dumps(["rail", "road", "telepathy"]),
                "answer": json.dumps([0, 1]),
                "subject": "transport",
            }
        )


def _write_sc_zip(path: Path) -> None:
    rows = {
        "multiple_choices_clean_final_clean.jsonl": {
            "question": "Select warehouse activities.",
            "options": [{"key": "A", "text": "Picking"}, {"key": "B", "text": "Putaway"}],
            "answer": ["A", "B"],
            "field": "Warehousing",
            "question_type": "multiple_choices",
        },
        "single_choices_clean_final_clean.jsonl": {
            "question": "Which is a transport mode?",
            "options": [{"key": "A", "text": "Rail"}, {"key": "B", "text": "Shelf"}],
            "answer": ["A"],
            "field": "Transport",
            "question_type": "single_choice",
        },
        "true_false_clean_final_clean.jsonl": {
            "question": "Cycle counting can support inventory accuracy.",
            "options": [],
            "answer": ["true"],
            "field": "Inventory",
            "question_type": "true_or_false",
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, row in rows.items():
            archive.writestr(f"SC-bench-main/data/{name}", json.dumps({"output": row}) + "\n")


def _private_row(item, *, correct: bool, parse_ok: bool = True) -> dict:
    return {
        "item_hash": item.item_hash,
        "dataset": item.dataset,
        "source_id": item.source_id,
        "category": item.category,
        "question_type": item.question_type,
        "question": item.question,
        "options": list(item.options),
        "expected": list(item.expected),
        "prediction": "",
        "parsed": list(item.expected) if correct else [1],
        "parse_ok": parse_ok,
        "correct": correct,
        "elapsed_sec": 0.2,
        "usage": None,
        "error": None,
    }


def test_loads_and_hashes_both_benchmarks(tmp_path: Path) -> None:
    logistika = tmp_path / "logistika.csv"
    sc_zip = tmp_path / "sc.zip"
    _write_logistika(logistika)
    _write_sc_zip(sc_zip)

    logistika_items = load_logistika(logistika)
    sc_items = load_sc_knowledge(sc_zip)
    assert logistika_items[0].expected == (0, 1)
    assert [item.expected for item in sc_items] == [(0, 1), (0,), (0,)]
    assert len({item.item_hash for item in logistika_items + sc_items}) == 4
    assert all(len(item.item_hash) == 64 for item in logistika_items + sc_items)


def test_parse_answers_requires_exact_json_contract() -> None:
    assert parse_answers('{"answers":[1,0]}', 3) == ((0, 1), True)
    assert parse_answers('prefix {"answers":"0, 2"} suffix', 3) == ((0, 2), True)
    assert parse_answers('{"answers":[3]}', 3) == ((), False)
    assert parse_answers('{"answer":[0]}', 3) == ((), False)


def test_safe_result_excludes_prompts_answers_and_predictions(tmp_path: Path) -> None:
    logistika = tmp_path / "logistika.csv"
    sc_zip = tmp_path / "sc.zip"
    _write_logistika(logistika)
    _write_sc_zip(sc_zip)
    item = load_logistika(logistika)[0]
    result = build_safe_result(
        [_private_row(item, correct=True)],
        model="model-a",
        endpoint_label="test",
        concurrency=64,
        input_hashes={"logistika": "a", "sc_zip": "b"},
        elapsed_sec=1.0,
    )
    serialized = json.dumps(result)
    assert item.question not in serialized
    assert "telepathy" not in serialized
    assert '"expected"' not in serialized
    assert result["accuracy"] == 1.0


def test_paired_comparison_counts_improvements_and_regressions(tmp_path: Path) -> None:
    logistika = tmp_path / "logistika.csv"
    sc_zip = tmp_path / "sc.zip"
    _write_logistika(logistika)
    _write_sc_zip(sc_zip)
    items = load_logistika(logistika) + load_sc_knowledge(sc_zip)
    base_rows = [_private_row(item, correct=value) for item, value in zip(items, [False, True, True, False])]
    candidate_rows = [_private_row(item, correct=value) for item, value in zip(items, [True, False, True, True])]
    kwargs = {
        "concurrency": 64,
        "input_hashes": {"logistika": "a", "sc_zip": "b"},
        "elapsed_sec": 1.0,
    }
    baseline = build_safe_result(base_rows, model="base", endpoint_label="base", **kwargs)
    candidate = build_safe_result(candidate_rows, model="candidate", endpoint_label="candidate", **kwargs)
    comparison = compare_safe_results(baseline, candidate)
    assert comparison["overall"]["improved_0_to_1"] == 2
    assert comparison["overall"]["regressed_1_to_0"] == 1
    assert comparison["overall"]["net_correct"] == 1
    assert comparison["overall"]["delta_accuracy_points"] == 25.0
    assert [row["items"] for row in comparison["by_dataset"]] == [1, 3]
    assert sum(row["net_correct"] for row in comparison["by_dataset"]) == 1
    assert sum(row["items"] for row in comparison["by_dataset_choice_count"]) == 4
    assert comparison["private_content_included"] is False


def test_comparison_rejects_different_inputs(tmp_path: Path) -> None:
    logistika = tmp_path / "logistika.csv"
    sc_zip = tmp_path / "sc.zip"
    _write_logistika(logistika)
    _write_sc_zip(sc_zip)
    item = load_logistika(logistika)[0]
    row = _private_row(item, correct=True)
    baseline = build_safe_result(
        [row], model="base", endpoint_label="base", concurrency=1, input_hashes={"x": "1"}, elapsed_sec=1
    )
    candidate = build_safe_result(
        [row], model="candidate", endpoint_label="candidate", concurrency=1, input_hashes={"x": "2"}, elapsed_sec=1
    )
    with pytest.raises(ValueError, match="input hashes differ"):
        compare_safe_results(baseline, candidate)


def test_mcnemar_exact_pvalue_is_symmetric() -> None:
    assert mcnemar_exact_pvalue(0, 0) == 1.0
    assert mcnemar_exact_pvalue(2, 1) == pytest.approx(mcnemar_exact_pvalue(1, 2))
    assert mcnemar_exact_pvalue(10, 0) == pytest.approx(0.001953125)
