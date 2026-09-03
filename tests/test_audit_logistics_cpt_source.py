import csv
import json
import zipfile
from pathlib import Path

from scripts.audit_logistics_cpt_source import benchmark_overlap, build_report, profile_source


def _write_sc_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in (
            "multiple_choices_clean_final_clean.jsonl",
            "single_choices_clean_final_clean.jsonl",
            "true_false_clean_final_clean.jsonl",
        ):
            row = {
                "output": {
                    "question": "Which warehouse policy minimizes unnecessary travel during order picking?",
                    "options": ["Use batch picking for compatible orders", "Ignore item locations"],
                    "field": "Warehousing",
                }
            }
            archive.writestr(f"SC-bench-main/data/{name}", json.dumps(row) + "\n")


def test_profile_detects_structure_noise_and_rights(tmp_path: Path) -> None:
    book = tmp_path / "book.txt"
    chapters = "\n".join(f"{index:02d}" for index in range(1, 45))
    repeated = "A sufficiently long repeated logistics paragraph for duplicate detection across the complete private source."
    book.write_text(
        "CONTENTS\n"
        + chapters
        + f"\nPART ONE\n\n{repeated}\n\n"
        + f"{repeated}\n\n●\n"
        + "No use by any artificial intelligence (AI) or machine learning system without prior written permission.\n"
        + "REFERENCES\nReference entry\nINDEX\nwarehouse 1\n",
        encoding="utf-8",
    )
    profile, _ = profile_source(book)
    assert profile["toc_unique_chapters_01_44"] == 44
    assert profile["noise_indicators"]["graphic_placeholder_lines"] == 1
    assert profile["noise_indicators"]["repeated_substantive_paragraph_instances"] == 1
    assert profile["rights_indicator"]["explicit_ai_ml_training_restriction_detected"] is True


def test_overlap_reports_only_hashes_and_aggregates() -> None:
    source = "Batch picking for compatible orders reduces warehouse travel distance and improves productivity."
    rows = [
        {
            "id": "secret-id",
            "question": "Why does batch picking for compatible orders reduces warehouse travel distance and improves productivity?",
            "options": ["Because orders share a route"],
            "category": "warehouse",
        }
    ]
    report = benchmark_overlap(source, "example", rows)
    serialized = json.dumps(report)
    assert "secret-id" not in serialized
    assert "batch picking" not in serialized.casefold()
    assert report["any_exact_contiguous_ngram_in_question_or_option"]["8"]["count"] == 1


def test_build_report_reads_both_benchmarks(tmp_path: Path) -> None:
    book = tmp_path / "book.txt"
    book.write_text("PART ONE\nGeneral logistics content.\nREFERENCES\nINDEX\n", encoding="utf-8")
    logistika = tmp_path / "logistika.csv"
    with logistika.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "question", "choices", "subject"])
        writer.writeheader()
        writer.writerow({"question_id": "1", "question": "A transport question", "choices": "[]", "subject": "transport"})
    sc_zip = tmp_path / "sc.zip"
    _write_sc_zip(sc_zip)
    report = build_report(book, logistika, sc_zip)
    assert [row["items"] for row in report["benchmark_overlap"]] == [1, 3]
    assert report["source_content_included"] is False
