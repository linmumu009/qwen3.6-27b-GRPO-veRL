"""Audit a private logistics CPT source without emitting copyrighted content.

The report contains only hashes, counts, structural indicators, and benchmark
overlap aggregates.  Source passages and benchmark prompts are never written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-’'][A-Za-z0-9]+)*")
SPACE_RE = re.compile(r"\s+")
NGRAM_SIZES = (8, 12, 16, 24)
SC_KNOWLEDGE_FILES = (
    "SC-bench-main/data/multiple_choices_clean_final_clean.jsonl",
    "SC-bench-main/data/single_choices_clean_final_clean.jsonl",
    "SC-bench-main/data/true_false_clean_final_clean.jsonl",
)
LOGISTIKA_LISTED_SOURCE_MARKERS = (
    "MITx MicroMasters SCM Key Concepts",
    "Global Supply Chain and Operations Management",
    "Ports and Waterways: Navigating the changing world",
    "Challenges and Opportunities in International Business",
    "Glossary for Transport Statistics",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def normalize_text(text: str) -> str:
    return " ".join(normalize_words(text))


def _find_line(lines: list[str], value: str, start: int = 0) -> int | None:
    target = value.casefold()
    for index in range(start, len(lines)):
        if lines[index].strip().casefold() == target:
            return index
    return None


def _paragraphs(text: str) -> list[str]:
    return [SPACE_RE.sub(" ", part).strip() for part in re.split(r"(?:\r?\n){2,}", text) if part.strip()]


def profile_source(path: Path, tokenizer_json: Path | None = None) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    lines = text.splitlines()
    paragraphs = _paragraphs(text)
    words = normalize_words(text)

    part_one = _find_line(lines, "PART ONE")
    references = _find_line(lines, "REFERENCES", (part_one or 0) + 1)
    index = _find_line(lines, "INDEX", (references or part_one or 0) + 1)
    main_end = references if references is not None else (index if index is not None else len(lines))
    main_start = part_one if part_one is not None else 0
    reference_end = index if index is not None else len(lines)

    toc_end = part_one if part_one is not None else min(len(lines), 2_000)
    toc_chapters = {
        int(line.strip())
        for line in lines[:toc_end]
        if re.fullmatch(r"(?:0[1-9]|[1-3][0-9]|4[0-4])", line.strip())
    }

    normalized_paragraphs = [normalize_text(paragraph) for paragraph in paragraphs]
    substantive_paragraphs = [paragraph for paragraph in normalized_paragraphs if len(paragraph) >= 80]
    paragraph_counts = Counter(substantive_paragraphs)
    repeated_paragraph_instances = sum(count - 1 for count in paragraph_counts.values() if count > 1)

    normalized_lines = [normalize_text(line) for line in lines]
    substantive_lines = [line for line in normalized_lines if len(line) >= 30]
    line_counts = Counter(substantive_lines)
    repeated_line_instances = sum(count - 1 for count in line_counts.values() if count > 1)

    main_text = "\n".join(lines[main_start:main_end])
    references_text = "\n".join(lines[references:reference_end]) if references is not None else ""
    index_text = "\n".join(lines[index:]) if index is not None else ""
    lower = SPACE_RE.sub(" ", text.casefold())

    exact_token_counts: dict[str, Any] | None = None
    if tokenizer_json is not None:
        from tokenizers import Tokenizer

        tokenizer_bytes = tokenizer_json.read_bytes()
        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        exact_token_counts = {
            "tokenizer_json_sha256": sha256_bytes(tokenizer_bytes),
            "raw": len(tokenizer.encode(text, add_special_tokens=False).ids),
            "main_body": len(tokenizer.encode(main_text, add_special_tokens=False).ids),
            "special_tokens_added": False,
        }

    profile = {
        "file_name": path.name,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "characters": len(text),
        "lines": len(lines),
        "blank_lines": sum(not line.strip() for line in lines),
        "paragraphs": len(paragraphs),
        "lexical_tokens": len(words),
        "estimated_model_tokens": {
            "low": round(len(text) / 4.2),
            "high": round(len(text) / 3.3),
            "method": "UTF-8 character heuristic retained for portability; use exact_qwen_token_counts when present",
        },
        "exact_qwen_token_counts": exact_token_counts,
        "utf8_valid": True,
        "replacement_characters": text.count("\ufffd"),
        "nul_characters": text.count("\x00"),
        "toc_unique_chapters_01_44": len(toc_chapters),
        "main_body": {
            "boundary_detected": part_one is not None and main_end > main_start,
            "lexical_tokens": len(normalize_words(main_text)),
        },
        "references": {
            "boundary_detected": references is not None,
            "lexical_tokens": len(normalize_words(references_text)),
        },
        "index": {
            "boundary_detected": index is not None,
            "lexical_tokens": len(normalize_words(index_text)),
        },
        "noise_indicators": {
            "graphic_placeholder_lines": sum(line.strip() in {"●", "•"} for line in lines),
            "repeated_substantive_line_instances": repeated_line_instances,
            "repeated_substantive_paragraph_instances": repeated_paragraph_instances,
        },
        "rights_indicator": {
            "explicit_ai_ml_training_restriction_detected": (
                "artificial intelligence (ai) or machine learning system" in lower
                and "prior written permission" in lower
            ),
            "training_permission_evidence_in_file": False,
            "status": "blocked_pending_rights_confirmation",
        },
    }
    return profile, text


def load_logistika(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                choices = json.loads(row.get("choices") or "[]")
            except json.JSONDecodeError:
                choices = []
            rows.append(
                {
                    "id": str(row.get("question_id") or len(rows) + 1),
                    "question": str(row.get("question") or ""),
                    "options": [str(value) for value in choices],
                    "category": str(row.get("subject") or "unknown"),
                }
            )
    return rows


def load_sc_knowledge(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for member in SC_KNOWLEDGE_FILES:
            with archive.open(member) as handle:
                for line_index, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue
                    outer = json.loads(raw_line.decode("utf-8"))
                    row = outer.get("output", outer)
                    raw_options = row.get("options") or []
                    options = [
                        str(option.get("text", "")) if isinstance(option, dict) else str(option)
                        for option in raw_options
                    ]
                    rows.append(
                        {
                            "id": f"{Path(member).name}:{line_index}",
                            "question": str(row.get("question") or ""),
                            "options": options,
                            "category": str(row.get("field") or "unknown"),
                        }
                    )
    return rows


def _ngrams(tokens: list[str], size: int) -> Iterable[tuple[str, ...]]:
    for start in range(0, len(tokens) - size + 1):
        yield tuple(tokens[start : start + size])


def _safe_id(dataset: str, item_id: str) -> str:
    return hashlib.sha256(f"{dataset}\0{item_id}".encode("utf-8")).hexdigest()[:16]


def benchmark_overlap(source_text: str, dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_tokens = normalize_words(source_text)
    source_normalized = " ".join(source_tokens)
    source_ngrams = {size: set(_ngrams(source_tokens, size)) for size in NGRAM_SIZES}
    full_question_matches: list[str] = []
    ngram_matches: dict[int, list[str]] = {size: [] for size in NGRAM_SIZES}

    for row in rows:
        safe_id = _safe_id(dataset, str(row["id"]))
        question_tokens = normalize_words(row["question"])
        if len(question_tokens) >= 12 and " ".join(question_tokens) in source_normalized:
            full_question_matches.append(safe_id)

        segments = [question_tokens, *(normalize_words(option) for option in row["options"])]
        for size in NGRAM_SIZES:
            if any(any(ngram in source_ngrams[size] for ngram in _ngrams(segment, size)) for segment in segments):
                ngram_matches[size].append(safe_id)

    count = len(rows)
    return {
        "dataset": dataset,
        "items": count,
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "exact_full_question": {
            "count": len(full_question_matches),
            "rate": round(len(full_question_matches) / count, 6) if count else 0.0,
            "matched_item_hashes": full_question_matches,
        },
        "any_exact_contiguous_ngram_in_question_or_option": {
            str(size): {
                "count": len(ngram_matches[size]),
                "rate": round(len(ngram_matches[size]) / count, 6) if count else 0.0,
                "matched_item_hashes": ngram_matches[size],
            }
            for size in NGRAM_SIZES
        },
        "raw_prompts_included": False,
    }


def build_report(
    book: Path,
    logistika: Path,
    sc_zip: Path,
    tokenizer_json: Path | None = None,
) -> dict[str, Any]:
    source_profile, source_text = profile_source(book, tokenizer_json)
    normalized_source = normalize_text(source_text)
    return {
        "schema_version": 1,
        "audit_type": "private_logistics_cpt_source_quality_and_eval_overlap",
        "source_profile": source_profile,
        "benchmark_overlap": [
            benchmark_overlap(source_text, "LogistikaBench", load_logistika(logistika)),
            benchmark_overlap(source_text, "SC-bench-knowledge", load_sc_knowledge(sc_zip)),
        ],
        "listed_logistika_source_title_matches": {
            marker: normalize_text(marker) in normalized_source for marker in LOGISTIKA_LISTED_SOURCE_MARKERS
        },
        "source_content_included": False,
        "benchmark_prompt_content_included": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--logistika", type=Path, required=True)
    parser.add_argument("--sc-zip", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.book, args.logistika, args.sc_zip, args.tokenizer_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
