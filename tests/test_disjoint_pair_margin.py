import math

from scripts.analyze_disjoint_pair_margin import analyze


def fixtures(pairs: int, chosen_preferred: int):
    evidence = [{"task_id": f"task_{index:06d}"} for index in range(pairs)]
    rows = []
    for index in range(pairs):
        task = f"task_{index:06d}"
        preferred = index < chosen_preferred
        for label in ("chosen", "rejected"):
            chosen = label == "chosen"
            nll = 1.0 if chosen == preferred else 2.0
            rows.append(
                {
                    "task_id": f"{task}::{label}",
                    "components": {
                        "semantic_delta": {"mean_nll": nll},
                        "sql_shell": {"mean_nll": nll + 0.1},
                    },
                    "sql_token_rank": {
                        "first_nongreedy_offset": 0,
                        "first_nongreedy_target_id": 1,
                        "first_nongreedy_rank": 2,
                        "first_nongreedy_target_token": "SUM" if index % 2 else "SELECT",
                    },
                }
            )
    diagnostic = {
        "contract": "repair-sft-teacher-forced-component-diagnostic-v3",
        "forward_only": True,
        "optimizer_initialized": False,
        "data_sha256": "hash",
        "components": {"semantic_delta": {}},
        "per_task": rows,
    }
    contract = {
        "contract": "current-definition-disjoint-first-error-pairs-v1",
        "pairs": pairs,
        "rows": 2 * pairs,
        "minimum_pairs": 48,
        "pair_count_gate_passed": True,
        "output_sha256": "hash",
        "evidence": evidence,
    }
    return diagnostic, contract


def test_margin_gate_uses_dynamic_pair_count_and_75_percent_threshold():
    diagnostic, contract = fixtures(48, 35)
    result = analyze(diagnostic, contract)

    assert result["task_count"] == 48
    assert result["semantic_delta_margin"]["chosen_preferred"] == 35
    assert result["semantic_delta_margin"]["preference_threshold_75pct"] == 36
    assert result["decision"]["one_step_training_allowed"] is True
    assert result["semantic_delta_margin"]["by_critical_token_family"] == {
        "aggregation_function": {
            "tasks": 24,
            "chosen_preferred": 17,
            "mean_margin": (17 - 7) / 24,
            "median_margin": 1.0,
        },
        "query_start": {
            "tasks": 24,
            "chosen_preferred": 18,
            "mean_margin": (18 - 6) / 24,
            "median_margin": 1.0,
        },
    }

    diagnostic, contract = fixtures(50, 38)
    result = analyze(diagnostic, contract)
    assert result["semantic_delta_margin"]["preference_threshold_75pct"] == math.ceil(37.5)
    assert result["decision"]["one_step_training_allowed"] is False
