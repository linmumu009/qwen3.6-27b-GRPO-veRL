#!/usr/bin/env python3
"""Seed a selective API rewrite from a verified prior DWH sandbox.

The output is deliberately an ``.incomplete`` sandbox.  Rows in the selected
difficulty bands are omitted so the API rewriter must regenerate them; all
other rows are reused only after plan, gold, role, provenance, and semantic
validation checks pass against the new deterministic base.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence


Validator = Callable[[dict[str, Any], str], list[str]]


def _load_validator(script: Path | None = None) -> Validator:
    if script is not None:
        spec = importlib.util.spec_from_file_location("llin_dwh_revision_validator", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load validator script: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.validate_rewrite
    try:
        from scripts.rewrite_plan_first_dwh_instructions_api import validate_rewrite
    except ModuleNotFoundError:  # Direct execution: python scripts/<name>.py
        from rewrite_plan_first_dwh_instructions_api import validate_rewrite
    return validate_rewrite


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def seed_revision(
    base_dir: Path,
    previous_api_dir: Path,
    incomplete_dir: Path,
    rewrite_bands: set[int],
    validator: Validator | None = None,
    rewrite_invalid: bool = False,
) -> dict[str, Any]:
    if not rewrite_bands <= set(range(1, 7)):
        raise ValueError("rewrite bands must be a subset of 1..6")
    if not rewrite_bands and not rewrite_invalid:
        raise ValueError("select at least one rewrite band or enable rewrite-invalid")
    if incomplete_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {incomplete_dir}")

    base_rows = _read_jsonl(base_dir / "dwh_tasks.jsonl")
    previous_rows = _read_jsonl(previous_api_dir / "dwh_tasks.jsonl")
    if len(base_rows) != 300 or len(previous_rows) != 300:
        raise ValueError("both source sandboxes must contain exactly 300 tasks")
    previous_by_id = {str(row["task_id"]): row for row in previous_rows}
    if len(previous_by_id) != 300:
        raise ValueError("previous API sandbox contains duplicate task IDs")

    validate = validator or _load_validator()
    reused: list[dict[str, Any]] = []
    selected_rows = 0
    for base in base_rows:
        band = int(base["difficulty_band"])
        if band in rewrite_bands:
            selected_rows += 1
            continue
        task_id = str(base["task_id"])
        previous = previous_by_id.get(task_id)
        if previous is None:
            raise ValueError(f"previous API sandbox is missing task: {task_id}")
        if previous.get("query_plan") != base.get("query_plan"):
            raise ValueError(f"query plan changed outside selected bands: {task_id}")
        if previous.get("gold_answer") != base.get("gold_answer"):
            raise ValueError(f"gold changed outside selected bands: {task_id}")
        if previous.get("instruction_role") != base.get("instruction_role"):
            raise ValueError(f"instruction role changed outside selected bands: {task_id}")
        generation = previous.get("instruction_generation") or {}
        if generation.get("semantic_validation_passed") is not True:
            raise ValueError(f"prior instruction was not validated: {task_id}")
        instruction = str(previous.get("natural_language_instruction") or "")
        reasons = validate(base, instruction)
        if reasons:
            if rewrite_invalid:
                selected_rows += 1
                continue
            raise ValueError(f"prior instruction no longer validates: {task_id}: {reasons}")

        merged = dict(base)
        for key in (
            "natural_language_instruction",
            "instruction_variants",
            "instruction_style",
            "instruction_generation",
            "generation_contract",
        ):
            merged[key] = previous[key]
        reused.append(merged)

    if selected_rows + len(reused) != 300:
        raise ValueError("selective revision partition does not cover all tasks")

    shutil.copytree(base_dir, incomplete_dir)
    _write_jsonl(incomplete_dir / "dwh_tasks.jsonl", reused)
    return {
        "contract": "llin-plan-first-dwh-selective-api-revision-seed-v1",
        "base_task_count": len(base_rows),
        "reused_validated_rows": len(reused),
        "pending_rewrite_rows": selected_rows,
        "rewrite_bands": sorted(rewrite_bands),
        "training_allowed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sandbox", type=Path, required=True)
    parser.add_argument("--previous-api-sandbox", type=Path, required=True)
    parser.add_argument("--output-incomplete", type=Path, required=True)
    parser.add_argument("--rewrite-band", type=int, action="append", default=[])
    parser.add_argument("--rewrite-invalid", action="store_true")
    parser.add_argument("--validator-script", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = seed_revision(
        args.base_sandbox,
        args.previous_api_sandbox,
        args.output_incomplete,
        set(args.rewrite_band),
        validator=_load_validator(args.validator_script),
        rewrite_invalid=args.rewrite_invalid,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
