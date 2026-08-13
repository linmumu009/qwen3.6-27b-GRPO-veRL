#!/usr/bin/env python3
"""CPU-only token audit for variable-size disjoint first-error pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from omegaconf import OmegaConf
from verl.utils.tokenizer import hf_tokenizer

from scripts.prepare_repair_sft_dataset import sha256_file
from scripts.qwen36_semantic_delta_margin_dataset import Qwen36SemanticDeltaMarginDataset


TRAINING_DATA_CONTRACT = "current-definition-disjoint-first-error-pairs-v1"
EVALUATION_DATA_CONTRACT = "current-definition-disjoint-first-error-evaluation-v1"
NATIVE_CANDIDATE_DATA_CONTRACT = "current-definition-native-first-error-training-candidates-v1"


def check(data_file: Path, contract_file: Path, model_path: str, max_length: int) -> dict:
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    contract_name = contract.get("contract")
    if contract_name not in {
        TRAINING_DATA_CONTRACT,
        EVALUATION_DATA_CONTRACT,
        NATIVE_CANDIDATE_DATA_CONTRACT,
    }:
        raise ValueError("unexpected disjoint first-error pair contract")
    pairs = int(contract.get("pairs") or 0)
    rows = int(contract.get("rows") or 0)
    evaluation_only = contract_name == EVALUATION_DATA_CONTRACT
    candidate_only = contract_name == NATIVE_CANDIDATE_DATA_CONTRACT
    if evaluation_only:
        expected_pairs = int(contract.get("expected_pairs") or 0)
        if (
            contract.get("pair_evaluation_gate_passed") is not True
            or pairs != expected_pairs
            or expected_pairs <= 0
        ):
            raise ValueError("disjoint first-error evaluation pair gate did not pass")
        for key in ("evaluation_only",):
            if contract.get(key) is not True:
                raise ValueError(f"disjoint evaluation contract failed: {key}")
        for key in ("may_be_used_as_training_data", "training_allowed", "promotion_allowed"):
            if contract.get(key) is not False:
                raise ValueError(f"disjoint evaluation contract is not fail closed: {key}")
        minimum_pairs = None
    elif candidate_only:
        expected_pairs = int(contract.get("expected_pairs") or 0)
        if (
            contract.get("candidate_pair_gate_passed") is not True
            or contract.get("candidate_only") is not True
            or contract.get("evaluation_only") is not False
            or pairs != expected_pairs
            or expected_pairs <= 0
        ):
            raise ValueError("native first-error candidate pair gate did not pass")
        for key in ("may_be_used_as_training_data", "training_allowed", "promotion_allowed"):
            if contract.get(key) is not False:
                raise ValueError(f"native candidate contract is not fail closed: {key}")
        minimum_pairs = None
    else:
        minimum_pairs = int(contract.get("minimum_pairs") or 0)
        if contract.get("pair_count_gate_passed") is not True or pairs < minimum_pairs:
            raise ValueError("disjoint first-error pair count gate did not pass")
    if rows != 2 * pairs:
        raise ValueError("pair contract does not contain exactly two rows per pair")
    if contract.get("output_sha256") != sha256_file(data_file):
        raise ValueError("disjoint first-error pair Parquet hash differs from its contract")
    for key in (
        "chosen_queries_mechanically_verified",
        "all_first_error_tool_results_observed",
        "pair_prefix_identical_through_observed_error_result",
    ):
        if contract.get(key) is not True:
            raise ValueError(f"disjoint pair contract failed: {key}")
    rejected_key = (
        "rejected_queries_are_actual_model_first_errors"
        if candidate_only
        else "rejected_queries_are_actual_step120_first_errors"
    )
    if contract.get(rejected_key) is not True:
        raise ValueError(f"disjoint pair contract failed: {rejected_key}")

    tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
    config = OmegaConf.create(
        {
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "enable_thinking_default": False,
            "pad_mode": "no_padding",
            "max_length": max_length,
            "truncation": "error",
            "ignore_input_ids_mismatch": False,
            "apply_chat_template_kwargs": {},
        }
    )
    dataset = Qwen36SemanticDeltaMarginDataset(str(data_file), tokenizer, config)
    if len(dataset) != rows:
        raise ValueError(f"tokenized row count differs from contract: {len(dataset)} != {rows}")
    evidence = {str(row["task_id"]): row for row in contract.get("evidence") or []}
    if len(evidence) != pairs:
        raise ValueError("pair evidence count differs from contract")

    labels: Counter[str] = Counter()
    samples = []
    expected_order = [
        item for pair_index in range(pairs) for item in ((pair_index, "chosen"), (pair_index, "rejected"))
    ]
    for index in range(len(dataset)):
        row = dataset.dataframe.iloc[index]
        current_task_id = str(row["source_task_id"])
        label = str(row["candidate_label"])
        pair_index = int(row["pair_index"])
        if current_task_id not in evidence:
            raise ValueError(f"tokenized pair is absent from evidence: {current_task_id}")
        if (pair_index, label) != expected_order[index]:
            raise ValueError("disjoint pair rows are not adjacent chosen/rejected pairs")
        item = dataset[index]
        sql_mask = item["sql_shell_mask"].tolist()
        delta_mask = item["semantic_delta_mask"].tolist()
        if not all(not delta or sql for delta, sql in zip(delta_mask, sql_mask, strict=True)):
            raise ValueError(f"{current_task_id}::{label}: delta mask extends outside SQL")
        delta_tokens = int(sum(delta_mask))
        if delta_tokens <= 0:
            raise ValueError(f"{current_task_id}::{label}: empty semantic-delta mask")
        expected_sign = 1 if label == "chosen" else -1
        if set(item["candidate_sign"].tolist()) != {expected_sign}:
            raise ValueError(f"{current_task_id}::{label}: candidate sign differs")
        if set(item["pair_index"].tolist()) != {pair_index}:
            raise ValueError(f"{current_task_id}::{label}: pair index differs")
        labels[label] += 1
        samples.append(
            {
                "task_id": current_task_id,
                "candidate_label": label,
                "sql_tokens": int(sum(sql_mask)),
                "semantic_delta_tokens": delta_tokens,
                "sequence_tokens": int(item["input_ids"].numel()),
            }
        )
    if labels != Counter({"chosen": pairs, "rejected": pairs}):
        raise ValueError(f"disjoint pair candidate imbalance: {dict(labels)}")
    return {
        "contract": (
            "current-definition-disjoint-pair-evaluation-token-gate-v1"
            if evaluation_only
            else (
                "current-definition-disjoint-pair-candidate-token-gate-v1"
                if candidate_only
                else "current-definition-disjoint-pair-token-gate-v1"
            )
        ),
        "rows": rows,
        "pairs": pairs,
        "minimum_pairs": minimum_pairs,
        "expected_pairs": pairs if (evaluation_only or candidate_only) else None,
        "candidate_rows": dict(sorted(labels.items())),
        "all_delta_masks_nonempty": True,
        "all_delta_masks_subset_of_sql": True,
        "all_pairs_adjacent_chosen_then_rejected": True,
        "all_candidate_signs_and_pair_indices_match": True,
        "truncation": "error",
        "max_length": max_length,
        "samples": samples,
        "npu_required": False,
        "evaluation_only": evaluation_only,
        "candidate_only": candidate_only,
        "may_be_used_as_training_data": False if (evaluation_only or candidate_only) else None,
        "training_allowed": False,
        "promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.data_file, args.contract, args.model_path, args.max_length)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
