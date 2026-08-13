from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validator_recomputes_identity_margins_and_blocks_training():
    source = (
        ROOT / "scripts" / "validate_disjoint_real_state_eval22_artifacts.py"
    ).read_text(encoding="utf-8")
    assert "pairs != 22 or rows != 44" in source
    assert "eval22 Parquet pair order changed" in source
    assert "eval22 diagnostic candidate identities differ" in source
    assert "semantic mean margin" in source
    assert "full-SQL mean margin" in source
    assert '"use_eval22_as_training_data": False' in source
    assert '"training_now": False' in source
    assert '"training_allowed": False' in source
    assert '"promotion_allowed": False' in source
