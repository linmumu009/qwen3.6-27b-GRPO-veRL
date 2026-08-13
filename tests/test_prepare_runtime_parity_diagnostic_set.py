import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.prepare_runtime_parity_diagnostic_set import prepare


GUIDANCE = (
    "沙箱布局：\n- 数据库：/workspace/logistics.sqlite（用 sqlite3 查询，只读，勿修改）\n"
    "- 知识库文档：/workspace/documents/（业务规范/操作手册）\n"
    "- 数据字典：/workspace/schema_dictionary.md（列出所有表及用途，查库前先读以避免盲目探索）\n\n任务："
)


def row(index: int, answer_type: str) -> dict:
    value = index if answer_type == "numeric" else [{"category": f"c{index}", "value": index}]
    return {
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": GUIDANCE + f"question {index}"},
        ],
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "task_id": f"task_{index}",
                "environment_id": "sft/20260628_v15",
                "task_family": "dwh",
                "answer_type": answer_type,
                "expected_value_json": json.dumps(value),
            },
        },
        "extra_info": {"alignment_reviewed": True},
    }


def test_prepare_freezes_five_per_type_and_sensitive_permissions(tmp_path: Path):
    rows = [row(i, "numeric") for i in range(8)] + [row(i + 8, "table") for i in range(8)]
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    output = tmp_path / "out"
    manifest = prepare(source, output, 5, "seed")
    assert manifest["selected_tasks"] == 10
    assert manifest["answer_types"] == {"numeric": 5, "table": 5}
    assert manifest["training_allowed"] is False
    tasks = [json.loads(line) for line in (output / "pi_tasks.sensitive.jsonl").read_text().splitlines()]
    assert len(tasks) == 10
    assert all(not item["instruction_without_guidance"].startswith("沙箱布局") for item in tasks)


def test_runtime_parity_contract_does_not_claim_strict_identical_runtime_configuration():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "configs" / "runtime_parity_10x8_contract_20260813.json").read_text(
            encoding="utf-8"
        )
    )

    assert contract["strict_runtime_configuration_matched"] is False
    assert contract["effective_sampling_audit"]["ordinary_verl_validation_default_temperature"] == 0
    assert contract["effective_sampling_audit"]["ordinary_validation_default_used_by_this_experiment"] is False
    assert "per_assistant_request_limit" in contract["known_native_runtime_differences"]
