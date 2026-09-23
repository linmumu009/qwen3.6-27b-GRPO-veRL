import copy
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("matrix_audit", Path(__file__).parents[1] / "scripts/audit_cpt_operation_matrix.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    pairs = [{"id": f"{p}{i}", "family": p,
              "D": {"concrete_input_state": {"x": 1}, "atomic_fact_ids": ["a"]},
              "T": {"atomic_fact_ids": ["a"]}, "exposure_limitation": "one vs two"}
             for p in ("U", "T") for i in range(1, 6)]
    reserves = [{"id": f"M{p}{i}", "components": [f"{p}1"]} for p in ("U", "T") for i in range(1, 5)]
    matrix = {"training_pairs": pairs, "reserved_transfer": reserves,
              "policy": {"training_allowed": False, "development_measurement_not_independent_holdout": True}}
    review = {"teaching": [{"id": r["id"], "pass": True, "source_supported": True, "contrast_genuine": True, "no_overclaim": True} for r in pairs],
              "transfer": [{"id": r["id"], "pass": True, "source_supported": True, "allocation_distinct": True} for r in reserves],
              "overall_authoring_release": True, "remaining_blockers": []}
    return matrix, review


def test_specification_release_never_allows_training():
    m, r = fixture()
    result = module.audit(m, r)
    assert result["authoring_released"] and not result["training_allowed"]


def test_hold_or_missing_review_blocks_release():
    m, r = fixture()
    for mutate in (lambda x: x["remaining_blockers"].append("unbound state"),
                   lambda x: x["teaching"].pop(),
                   lambda x: x["transfer"][0].update({"pass": False})):
        changed = copy.deepcopy(r)
        mutate(changed)
        assert not module.audit(m, changed)["authoring_released"]


def test_different_facts_or_dangling_dependencies_block_release():
    m, r = fixture()
    m["training_pairs"][0]["T"]["atomic_fact_ids"] = ["b"]
    assert not module.audit(m, r)["authoring_released"]
    m, r = fixture()
    m["reserved_transfer"][0]["components"] = ["absent"]
    assert not module.audit(m, r)["authoring_released"]
