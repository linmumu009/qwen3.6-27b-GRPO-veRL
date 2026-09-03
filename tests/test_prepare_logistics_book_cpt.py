from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_logistics_book_cpt import (
    Block,
    build_blocks,
    chunk_blocks,
    validate_rights_attestation,
)


class WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        return " ".join(f"token{value}" for value in ids)


def _synthetic_book() -> str:
    lines = ["copyright and table of contents", "PART ONE", "Concepts"]
    for chapter in range(1, 45):
        lines.extend(
            [
                f"Chapter title {chapter}",
                f"{chapter:02d}",
                "Introduction",
                f"This is chapter {chapter} and it contains a substantive logistics sentence.",
            ]
        )
        if chapter == 5:
            lines.extend(
                [
                    "Repeated running header",
                    "",
                    "17",
                    "",
                    "Repeated running header",
                    "A sentence split at a line-",
                    "break for testing.",
                    "",
                    "●",
                    "",
                    "inventory planning",
                ]
            )
        if chapter == 15:
            lines.append("PART TWO")
        if chapter == 20:
            lines.append("PART THREE")
        if chapter == 25:
            lines.append("PART FOUR")
        if chapter == 30:
            lines.append("PART FIVE")
        if chapter == 35:
            lines.append("PART SIX")
        if chapter == 40:
            lines.append("PART SEVEN")
    lines.extend(["Repeated running header", "REFERENCES", "excluded reference", "INDEX", "excluded index"])
    return "\n".join(lines) + "\n"


def test_build_blocks_keeps_main_body_structure_and_excludes_tail() -> None:
    blocks, stats = build_blocks(_synthetic_book())
    joined = "\n".join(block.text for block in blocks)
    assert "copyright and table of contents" not in joined
    assert "excluded reference" not in joined
    assert "Chapter 1" in joined
    assert "Chapter 44" in joined
    assert "linebreak for testing" in joined
    assert "- inventory planning" in joined
    assert stats["chapters_detected"] == 44
    assert stats["parts_detected"] == 7
    assert stats["bullet_markers_converted"] == 1
    assert stats["line_break_hyphen_repairs"] == 1


def test_chunk_blocks_respects_chapter_and_token_limit() -> None:
    blocks = [
        Block(" ".join(f"a{index}" for index in range(25)), 1, 1, 1, 1),
        Block(" ".join(f"b{index}" for index in range(10)), 2, 2, 1, 1),
        Block("nine ten", 3, 3, 1, 2),
    ]
    rows = chunk_blocks(blocks, WhitespaceTokenizer(), target_tokens=32)
    assert [row["chapter"] for row in rows] == [1, 1, 2]
    assert max(row["token_count"] for row in rows) <= 32
    assert all(row["text_sha256"] != row["text"] for row in rows)


def test_validate_rights_attestation_requires_matching_authorized_source(tmp_path: Path) -> None:
    source_hash = hashlib.sha256(b"source").hexdigest()
    path = tmp_path / "rights.json"
    path.write_text(
        json.dumps(
            {
                "source_sha256": source_hash,
                "user_confirmed_written_permission_covering_ai_ml_training": True,
                "permission_document_provided_to_or_reviewed_by_agent": False,
                "permission_terms_independently_verified": False,
            }
        ),
        encoding="utf-8",
    )
    result = validate_rights_attestation(path, source_hash)
    assert result["basis"] == "user_attested_written_permission"
    assert result["permission_document_reviewed_by_agent"] is False

    with pytest.raises(ValueError, match="does not match"):
        validate_rights_attestation(path, "0" * 64)


def test_safe_rights_result_never_contains_source_text(tmp_path: Path) -> None:
    source_hash = hashlib.sha256(b"private copyrighted sentence").hexdigest()
    path = tmp_path / "rights.json"
    path.write_text(
        json.dumps(
            {
                "source_sha256": source_hash,
                "user_confirmed_written_permission_covering_ai_ml_training": True,
            }
        ),
        encoding="utf-8",
    )
    safe = json.dumps(validate_rights_attestation(path, source_hash))
    assert "private copyrighted sentence" not in safe
