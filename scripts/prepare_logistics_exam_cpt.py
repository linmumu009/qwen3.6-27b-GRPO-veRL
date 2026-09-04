#!/usr/bin/env python3
"""Build private, packed causal-LM corpora from frozen logistics exam cases.

The direct arm uses each original stem plus only its gold option text.  The
rewritten arm substitutes a separately verified paraphrase of the stem while
keeping the gold option text byte-identical after whitespace normalization.
Distractors and option labels are never rendered into the training document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from transformers import AutoTokenizer


FORMAT_VERSION = "logistics-exam-gold-text-cpt-v1"
TRUE_VALUES = {"true", "correct", "yes"}
FALSE_VALUES = {"false", "incorrect", "no"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def normalize_space(value: str) -> str:
    return " ".join(str(value).split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def write_jsonl_private(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    os.chmod(path, 0o600)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_source_row(row: dict[str, Any], index: int) -> None:
    required = {"dataset", "source_id", "question_type", "question", "options", "expected", "item_hash"}
    if not required <= set(row):
        raise ValueError(f"source row {index} missing fields: {sorted(required - set(row))}")
    if not normalize_space(row["question"]):
        raise ValueError(f"source row {index} has empty question")
    options = row["options"]
    expected = row["expected"]
    if not isinstance(options, list) or not options or any(not normalize_space(value) for value in options):
        raise ValueError(f"source row {index} has invalid options")
    if (
        not isinstance(expected, list)
        or not expected
        or any(isinstance(value, bool) or not isinstance(value, int) for value in expected)
        or any(value < 0 or value >= len(options) for value in expected)
    ):
        raise ValueError(f"source row {index} has invalid expected indices")


def gold_texts(row: dict[str, Any]) -> list[str]:
    return [normalize_space(row["options"][index]) for index in sorted(set(row["expected"]))]


def render_training_document(row: dict[str, Any], question: str) -> str:
    """Render one truthful document with no distractors or option labels."""

    stem = normalize_space(question)
    answers = gold_texts(row)
    if str(row["question_type"]) == "true_or_false":
        if len(answers) != 1:
            raise ValueError(f"true/false item {row['item_hash']} must have one gold answer")
        normalized = answers[0].casefold().rstrip(".")
        if normalized in TRUE_VALUES:
            verdict = "true"
        elif normalized in FALSE_VALUES:
            verdict = "false"
        else:
            raise ValueError(f"unrecognized true/false answer for {row['item_hash']}")
        return f'Statement: "{stem}"\nThis statement is {verdict}.'
    label = "Correct answer" if len(answers) == 1 else "Correct answers"
    return f"Question: {stem}\n{label}: {'; '.join(answers)}"


def load_rewrites(path: Path, source_rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    rows = read_jsonl(path)
    indexed: dict[str, str] = {}
    source_hashes = {str(row["item_hash"]): text_hash(normalize_space(row["question"])) for row in source_rows}
    for row in rows:
        item_hash = str(row.get("item_hash") or "")
        if not item_hash or item_hash in indexed or item_hash not in source_hashes:
            raise ValueError("rewrites contain an unknown or duplicate item hash")
        if row.get("semantic_validation_passed") is not True:
            raise ValueError(f"rewrite semantic validation did not pass: {item_hash}")
        if str(row.get("original_question_sha256") or "") != source_hashes[item_hash]:
            raise ValueError(f"rewrite source hash mismatch: {item_hash}")
        question = normalize_space(row.get("rewritten_question") or "")
        if not question:
            raise ValueError(f"empty rewritten question: {item_hash}")
        indexed[item_hash] = question
    if set(indexed) != set(source_hashes):
        raise ValueError("rewrites do not cover the frozen source exactly")
    return indexed


def token_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
    return [int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"]]


def _render_group(documents: Sequence[str], separator: str) -> str:
    return separator.join(documents)


def pack_indexed_documents(
    documents: Sequence[str],
    tokenizer: Any,
    *,
    block_count: int,
    max_content_tokens: int,
) -> list[list[int]]:
    """Use deterministic longest-first bin packing for balanced exact blocks."""

    if block_count < 1 or block_count > len(documents):
        raise ValueError("block_count must be within the document count")
    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_token:
        raise ValueError("tokenizer must define eos_token")
    separator = f"{eos_token}\n\n"
    document_lengths = [len(token_ids(tokenizer, document)) for document in documents]
    if any(length > max_content_tokens for length in document_lengths):
        raise ValueError("one rendered exam item exceeds the content-token limit")
    bins: list[list[int]] = [[] for _ in range(block_count)]
    bin_lengths = [0] * block_count
    for document_index in sorted(range(len(documents)), key=lambda index: (-document_lengths[index], index)):
        document = documents[document_index]
        placed = False
        for bin_index in sorted(range(block_count), key=lambda index: (bin_lengths[index], index)):
            candidate_indices = bins[bin_index] + [document_index]
            candidate_text = _render_group([documents[index] for index in candidate_indices], separator)
            candidate_length = len(token_ids(tokenizer, candidate_text))
            if candidate_length <= max_content_tokens:
                bins[bin_index].append(document_index)
                bin_lengths[bin_index] = candidate_length
                placed = True
                break
        if not placed:
            raise ValueError("documents cannot be packed within the requested block count and token limit")
    if any(not group for group in bins):
        raise ValueError("balanced packing produced an empty block")
    for group in bins:
        group.sort()
    return bins


def pack_documents(
    documents: Sequence[str],
    tokenizer: Any,
    *,
    block_count: int,
    max_content_tokens: int,
) -> list[list[str]]:
    return [
        [documents[index] for index in group]
        for group in pack_indexed_documents(
            documents,
            tokenizer,
            block_count=block_count,
            max_content_tokens=max_content_tokens,
        )
    ]


def build_corpus(
    source_rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    arm: str,
    rewrites: dict[str, str] | None,
    block_count: int,
    max_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if arm not in {"direct", "rewritten"}:
        raise ValueError("arm must be direct or rewritten")
    if (arm == "rewritten") != (rewrites is not None):
        raise ValueError("rewritten arm requires rewrites and direct arm forbids them")
    if len({str(row["item_hash"]) for row in source_rows}) != len(source_rows):
        raise ValueError("source item hashes are not unique")

    private_items: list[dict[str, Any]] = []
    documents: list[str] = []
    changed_questions = 0
    true_false_items = 0
    answer_text_count = 0
    for index, row in enumerate(source_rows):
        validate_source_row(row, index)
        original_question = normalize_space(row["question"])
        question = rewrites[str(row["item_hash"])] if rewrites is not None else original_question
        if question != original_question:
            changed_questions += 1
        document = render_training_document(row, question)
        answers = gold_texts(row)
        documents.append(document)
        answer_text_count += len(answers)
        true_false_items += str(row["question_type"]) == "true_or_false"
        private_items.append(
            {
                "format_version": FORMAT_VERSION,
                "arm": arm,
                "item_hash": str(row["item_hash"]),
                "dataset": str(row["dataset"]),
                "source_id": str(row["source_id"]),
                "question_type": str(row["question_type"]),
                "original_question_sha256": text_hash(original_question),
                "training_question_sha256": text_hash(question),
                "gold_text_sha256": [text_hash(value) for value in answers],
                "training_document": document,
            }
        )

    indexed_groups = pack_indexed_documents(
        documents,
        tokenizer,
        block_count=block_count,
        max_content_tokens=max_length - 1,
    )
    separator = f"{tokenizer.eos_token}\n\n"
    blocks: list[dict[str, Any]] = []
    for block_index, group_indices in enumerate(indexed_groups):
        group = [documents[index] for index in group_indices]
        text = _render_group(group, separator)
        count = len(token_ids(tokenizer, text))
        item_hashes = [str(source_rows[index]["item_hash"]) for index in group_indices]
        blocks.append(
            {
                "text": text,
                "record_id": f"{arm}-{block_index:04d}",
                "arm": arm,
                "item_count": len(group),
                "token_count": count,
                "item_hashes_sha256": text_hash("\n".join(item_hashes)),
            }
        )
    packed_indices = [index for group in indexed_groups for index in group]
    if sorted(packed_indices) != list(range(len(source_rows))):
        raise AssertionError("packed item accounting mismatch")
    lengths = [int(block["token_count"]) for block in blocks]
    document_lengths = [len(token_ids(tokenizer, document)) for document in documents]
    summary = {
        "schema_version": 1,
        "format_version": FORMAT_VERSION,
        "private_content_included": False,
        "arm": arm,
        "items": len(source_rows),
        "blocks": len(blocks),
        "answer_texts": answer_text_count,
        "true_false_items": true_false_items,
        "changed_question_items": changed_questions,
        "distractors_rendered": 0,
        "option_indices_or_letters_rendered_by_template": False,
        "chat_template_applied": False,
        "item_separator_is_tokenizer_eos": True,
        "content_tokens": sum(lengths),
        "sequence_tokens_with_terminal_eos": sum(lengths) + len(lengths),
        "minimum_block_tokens": min(lengths),
        "median_block_tokens": statistics.median(lengths),
        "maximum_block_tokens": max(lengths),
        "minimum_document_tokens": min(document_lengths),
        "median_document_tokens": statistics.median(document_lengths),
        "maximum_document_tokens": max(document_lengths),
        "max_length": max_length,
    }
    return private_items, blocks, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--arm", choices=("direct", "rewritten"), required=True)
    parser.add_argument("--rewrites", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--block-count", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--private-items-output", type=Path, required=True)
    parser.add_argument("--parquet-output", type=Path, required=True)
    parser.add_argument("--safe-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.arm == "rewritten" and not args.rewrites:
        raise ValueError("--rewrites is required for the rewritten arm")
    if args.arm == "direct" and args.rewrites:
        raise ValueError("--rewrites is forbidden for the direct arm")
    source_rows = read_jsonl(args.source)
    rewrites = load_rewrites(args.rewrites, source_rows) if args.rewrites else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    private_items, blocks, summary = build_corpus(
        source_rows,
        tokenizer,
        arm=args.arm,
        rewrites=rewrites,
        block_count=args.block_count,
        max_length=args.max_length,
    )

    write_jsonl_private(args.private_items_output, private_items)
    args.parquet_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = args.parquet_output.with_name(f".{args.parquet_output.name}.tmp-{os.getpid()}")
    pd.DataFrame(blocks).to_parquet(temporary_parquet, index=False)
    os.replace(temporary_parquet, args.parquet_output)
    os.chmod(args.parquet_output, 0o600)
    summary.update(
        {
            "source_sha256": sha256_file(args.source),
            "rewrites_sha256": sha256_file(args.rewrites) if args.rewrites else None,
            "private_items_sha256": sha256_file(args.private_items_output),
            "parquet_sha256": sha256_file(args.parquet_output),
            "tokenizer_json_sha256": sha256_file(args.model_path / "tokenizer.json"),
        }
    )
    write_json(args.safe_output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
