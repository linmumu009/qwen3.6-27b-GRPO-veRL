"""Prepare an auditable, private causal-LM dataset from the logistics handbook.

The private JSONL contains copyrighted source text and must not be committed.  The
safe manifest contains only hashes, counts, and transformation metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


PART_RE = re.compile(r"^PART (ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN)$")
CHAPTER_SUFFIX_RE = re.compile(r"(?:^|\s)(0[1-9]|[1-3][0-9]|4[0-4])$")
PAGE_NUMBER_RE = re.compile(r"[1-9][0-9]{0,3}")
BULLET_LINES = {"●", "•"}
TERMINAL_RE = re.compile(r"[.!?;:]\s*$")
WORD_RE = re.compile(r"[A-Za-z]+(?:[-’'][A-Za-z]+)*")


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> Any: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...


@dataclass(frozen=True)
class Block:
    text: str
    source_line_start: int
    source_line_end: int
    part: int | None
    chapter: int | None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _token_ids(tokenizer: TokenizerLike, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = getattr(encoded, "ids", encoded)
    return list(ids)


def _find_exact_line(lines: list[str], target: str, start: int = 0) -> int | None:
    folded = target.casefold()
    for index in range(start, len(lines)):
        if lines[index].strip().casefold() == folded:
            return index
    return None


def _next_nonempty(lines: list[str], index: int, limit: int = 12) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for cursor in range(index + 1, min(len(lines), index + limit + 1)):
        text = lines[cursor].strip()
        if text:
            found.append((cursor, text))
    return found


def find_chapter_markers(lines: list[str], start: int, end: int) -> dict[int, int]:
    """Find the 44 sequential chapter markers using their nearby Introduction heading."""

    markers: dict[int, int] = {}
    cursor = start
    for chapter in range(1, 45):
        expected = f"{chapter:02d}"
        match_index: int | None = None
        fallback_candidates: list[int] = []
        for index in range(cursor, end):
            match = CHAPTER_SUFFIX_RE.search(lines[index].strip())
            if match is None or match.group(1) != expected:
                continue
            nearby = [text.casefold() for _, text in _next_nonempty(lines, index, 12)]
            if "introduction" in nearby:
                match_index = index
                break
            if (
                lines[index].strip() == expected
                and index > 0
                and index + 1 < len(lines)
                and not lines[index - 1].strip()
                and not lines[index + 1].strip()
                and _next_nonempty(lines, index, 4)
            ):
                fallback_candidates.append(index)
        if match_index is None and len(fallback_candidates) == 1:
            match_index = fallback_candidates[0]
        if match_index is None:
            raise ValueError(f"Could not locate a high-confidence marker for chapter {chapter:02d}")
        markers[chapter] = match_index
        cursor = match_index + 1
    return markers


def find_part_markers(lines: list[str], start: int, end: int) -> dict[int, int]:
    names = ("ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN")
    markers: dict[int, int] = {}
    for number, name in enumerate(names, start=1):
        index = _find_exact_line(lines, f"PART {name}", start)
        if index is None or index >= end:
            raise ValueError(f"Could not locate PART {name} in the main body")
        markers[number] = index
    if list(markers.values()) != sorted(markers.values()):
        raise ValueError("Part markers are not in ascending order")
    return markers


def _marker_at_or_before(markers: dict[int, int], line_index: int) -> int | None:
    value: int | None = None
    for number, marker_index in markers.items():
        if marker_index > line_index:
            break
        value = number
    return value


def _normalized_header(line: str) -> str:
    return " ".join(word.casefold() for word in WORD_RE.findall(line))


def find_running_artifacts(
    lines: list[str], start: int, end: int, chapter_lines: set[int], part_lines: set[int]
) -> tuple[set[int], set[int]]:
    """Conservatively find page numbers and repeated headers around page breaks."""

    normalized = [_normalized_header(line.strip()) for line in lines]
    counts = Counter(value for value in normalized[start:end] if len(value) >= 8)
    repeated_headers = {value for value, count in counts.items() if count >= 3}
    candidates: list[tuple[int, int, int | None]] = []

    for index in range(start, end):
        value = lines[index].strip()
        if PAGE_NUMBER_RE.fullmatch(value) is None:
            continue
        if index == 0 or index + 1 >= len(lines):
            continue
        if lines[index - 1].strip() or lines[index + 1].strip():
            continue

        following = _next_nonempty(lines, index, 12)
        following_lines = {line_index for line_index, _ in following}
        first_nonnumeric = next(
            (
                (line_index, text)
                for line_index, text in following
                if PAGE_NUMBER_RE.fullmatch(text) is None
            ),
            None,
        )
        repeated = (
            first_nonnumeric[0]
            if first_nonnumeric is not None
            and _normalized_header(first_nonnumeric[1]) in repeated_headers
            else None
        )
        structural = bool({line_index for line_index, _ in following[:6]} & (chapter_lines | part_lines))
        if repeated is not None or structural:
            candidates.append((index, int(value), repeated))

    value_counts = Counter(value for _, value, _ in candidates)
    # If the same apparent page number occurs more than once, at least one
    # occurrence can be a table cell. Keep every ambiguous occurrence rather
    # than risk deleting domain facts.
    selected = [candidate for candidate in candidates if value_counts[candidate[1]] == 1]
    page_lines = {index for index, _, _ in selected}
    header_lines = {header for _, _, header in selected if header is not None}
    return page_lines, header_lines


def _continues_across_blank(previous: str, current: str) -> bool:
    if previous == "-":
        return True
    if not previous or not current:
        return False
    if current[0].islower():
        return True
    return not TERMINAL_RE.search(previous) and len(previous) >= 90


def _join_line(previous: str, current: str) -> tuple[str, bool]:
    if previous == "-":
        return f"- {current}", False
    if previous.endswith("-") and current and current[0].islower():
        return previous[:-1] + current, True
    return f"{previous} {current}", False


def build_blocks(text: str) -> tuple[list[Block], dict[str, Any]]:
    """Extract the main body and conservatively reconstruct wrapped paragraphs."""

    lines = text.splitlines()
    main_start = _find_exact_line(lines, "PART ONE")
    if main_start is None:
        raise ValueError("Main-body start marker PART ONE was not found")
    references = _find_exact_line(lines, "REFERENCES", main_start + 1)
    if references is None:
        raise ValueError("Main-body end marker REFERENCES was not found")

    chapter_markers = find_chapter_markers(lines, main_start, references)
    part_markers = find_part_markers(lines, main_start, references)
    chapter_by_line = {line_index: number for number, line_index in chapter_markers.items()}
    page_lines, header_lines = find_running_artifacts(
        lines,
        main_start,
        references,
        set(chapter_markers.values()),
        set(part_markers.values()),
    )

    blocks: list[Block] = []
    current = ""
    current_start = 0
    current_end = 0
    current_part: int | None = None
    current_chapter: int | None = None
    blank_seen = False
    hyphen_repairs = 0
    bullet_conversions = 0

    def flush() -> None:
        nonlocal current
        if current.strip():
            blocks.append(
                Block(
                    text=current.strip(),
                    source_line_start=current_start + 1,
                    source_line_end=current_end + 1,
                    part=current_part,
                    chapter=current_chapter,
                )
            )
        current = ""

    for index in range(main_start, references):
        raw = lines[index].strip()
        if index in page_lines or index in header_lines:
            blank_seen = True
            continue
        if not raw:
            blank_seen = True
            continue

        part = _marker_at_or_before(part_markers, index)
        chapter = _marker_at_or_before(chapter_markers, index)
        structural = PART_RE.fullmatch(raw) is not None or index in chapter_by_line
        if raw in BULLET_LINES:
            raw = "-"
            bullet_conversions += 1
            structural = True
        elif index in chapter_by_line:
            match = CHAPTER_SUFFIX_RE.search(raw)
            title = raw[: match.start()].strip() if match is not None else ""
            raw = f"{title}\nChapter {chapter_by_line[index]}" if title else f"Chapter {chapter_by_line[index]}"

        if current and (structural or part != current_part or chapter != current_chapter):
            flush()
        elif current and blank_seen and not _continues_across_blank(current, raw):
            flush()

        if not current:
            current = raw
            current_start = index
            current_part = part
            current_chapter = chapter
        else:
            current, repaired = _join_line(current, raw)
            hyphen_repairs += int(repaired)
        current_end = index
        blank_seen = False

    flush()
    if not blocks:
        raise ValueError("No main-body blocks were produced")

    stats = {
        "source_lines": len(lines),
        "main_body_source_line_start": main_start + 1,
        "main_body_source_line_end_exclusive": references + 1,
        "chapters_detected": len(chapter_markers),
        "parts_detected": len(part_markers),
        "blocks": len(blocks),
        "page_number_lines_removed": len(page_lines),
        "running_header_lines_removed": len(header_lines),
        "bullet_markers_converted": bullet_conversions,
        "line_break_hyphen_repairs": hyphen_repairs,
        "references_and_index_excluded": True,
        "front_matter_excluded": True,
        "aggressive_figure_or_table_deletion": False,
    }
    return blocks, stats


def _split_oversize_block(block: Block, tokenizer: TokenizerLike, target_tokens: int) -> list[Block]:
    ids = _token_ids(tokenizer, block.text)
    pieces: list[Block] = []
    for start in range(0, len(ids), target_tokens):
        text = tokenizer.decode(ids[start : start + target_tokens], skip_special_tokens=False).strip()
        if text:
            pieces.append(
                Block(
                    text=text,
                    source_line_start=block.source_line_start,
                    source_line_end=block.source_line_end,
                    part=block.part,
                    chapter=block.chapter,
                )
            )
    return pieces


def chunk_blocks(
    blocks: list[Block], tokenizer: TokenizerLike, target_tokens: int
) -> list[dict[str, Any]]:
    if target_tokens < 32:
        raise ValueError("target_tokens must be at least 32")

    expanded: list[Block] = []
    for block in blocks:
        if len(_token_ids(tokenizer, block.text)) > target_tokens:
            expanded.extend(_split_oversize_block(block, tokenizer, target_tokens))
        else:
            expanded.append(block)

    records: list[dict[str, Any]] = []
    pending: list[Block] = []

    def flush() -> None:
        if not pending:
            return
        text = "\n\n".join(block.text for block in pending)
        token_count = len(_token_ids(tokenizer, text))
        records.append(
            {
                "record_id": f"handbook8e-{len(records) + 1:05d}",
                "part": pending[0].part,
                "chapter": pending[0].chapter,
                "source_line_start": min(block.source_line_start for block in pending),
                "source_line_end": max(block.source_line_end for block in pending),
                "text_sha256": sha256_bytes(text.encode("utf-8")),
                "token_count": token_count,
                "text": text,
            }
        )
        pending.clear()

    for block in expanded:
        if pending and (block.part, block.chapter) != (pending[0].part, pending[0].chapter):
            flush()
        candidate = "\n\n".join([*(item.text for item in pending), block.text])
        if pending and len(_token_ids(tokenizer, candidate)) > target_tokens:
            flush()
        pending.append(block)
    flush()

    if any(record["token_count"] > target_tokens for record in records):
        raise AssertionError("A prepared record exceeds the requested token limit")
    return records


def validate_rights_attestation(path: Path, source_sha256: str) -> dict[str, Any]:
    attestation = json.loads(path.read_text(encoding="utf-8"))
    if attestation.get("source_sha256") != source_sha256:
        raise ValueError("Rights attestation source hash does not match the source file")
    if attestation.get("user_confirmed_written_permission_covering_ai_ml_training") is not True:
        raise ValueError("Rights attestation does not confirm written AI/ML training permission")
    return {
        "basis": "user_attested_written_permission",
        "attestation_sha256": sha256_bytes(path.read_bytes()),
        "permission_document_reviewed_by_agent": bool(
            attestation.get("permission_document_provided_to_or_reviewed_by_agent", False)
        ),
        "terms_independently_verified": bool(attestation.get("permission_terms_independently_verified", False)),
        "legal_determination": False,
    }


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    os.chmod(path, 0o600)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_dataset(
    source: Path,
    tokenizer_json: Path,
    rights_attestation: Path,
    private_output: Path,
    safe_manifest_output: Path,
    target_tokens: int = 4096,
) -> dict[str, Any]:
    from tokenizers import Tokenizer

    source_bytes = source.read_bytes()
    source_sha256 = sha256_bytes(source_bytes)
    source_text = source_bytes.decode("utf-8", errors="strict")
    rights = validate_rights_attestation(rights_attestation, source_sha256)
    blocks, cleaning = build_blocks(source_text)
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    records = chunk_blocks(blocks, tokenizer, target_tokens)
    for record in records:
        record["source_sha256"] = source_sha256

    _atomic_write_jsonl(private_output, records)
    private_sha256 = sha256_bytes(private_output.read_bytes())
    token_counts = [record["token_count"] for record in records]
    manifest = {
        "schema_version": 1,
        "dataset_type": "causal_lm_continued_pretraining",
        "created_date": "2026-09-03",
        "source": {
            "file_name": source.name,
            "sha256": source_sha256,
            "bytes": len(source_bytes),
        },
        "rights": rights,
        "tokenizer": {
            "tokenizer_json_sha256": sha256_bytes(tokenizer_json.read_bytes()),
            "add_special_tokens_during_counting": False,
        },
        "cleaning": cleaning,
        "records": {
            "count": len(records),
            "target_max_tokens": target_tokens,
            "content_tokens": sum(token_counts),
            "minimum_tokens": min(token_counts),
            "maximum_tokens": max(token_counts),
            "shorter_than_1024": sum(count < 1024 for count in token_counts),
            "append_one_eos_per_record_at_indexing": True,
            "planned_single_exposure": True,
        },
        "private_jsonl_sha256": private_sha256,
        "private_output_mode_requested": "0600",
        "source_content_included": False,
    }
    _atomic_write_json(safe_manifest_output, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--rights-attestation", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--safe-manifest-output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = prepare_dataset(
        source=args.source,
        tokenizer_json=args.tokenizer_json,
        rights_attestation=args.rights_attestation,
        private_output=args.private_output,
        safe_manifest_output=args.safe_manifest_output,
        target_tokens=args.target_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
