from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_acquisition_supervisor_uses_three_full_hosts_and_strict_confirmation() -> None:
    text = (ROOT / "scripts" / "run_qwen38_fresh_acquisition_host.py").read_text(
        encoding="utf-8"
    )
    for fragment in (
        '"m05": {"ssh": None',
        '"m06": {"ssh": args.remote_host',
        '"m00": {"ssh": args.m00_host',
        '"npus": 16, "dp": 4',
        '"npus": 12, "dp": 3',
        '"--max-num-seqs", "16"',
        '"window": 80',
        '"window": 60',
        '"--confirm-candidates"',
        '"/models/Qwen3.8-27B"',
        '"queue_wait_counts_toward_timeout": False',
        '"trajectory_timeout_seconds": 1800',
        '"minimum_robust_candidates_for_canary": 24',
        'runtime_projection',
        'scripts/prepare_qwen38_fresh_v22_v26.py',
    ):
        assert fragment in text


def test_fresh_acquisition_freezes_v22_and_samples_only_v23_v26() -> None:
    text = (ROOT / "scripts" / "prepare_qwen38_fresh_v22_v26.py").read_text(
        encoding="utf-8"
    )
    assert '"v22_full500.sensitive.parquet"' in text
    assert '"versions": ["v23", "v24", "v25", "v26"]' in text
    assert '"v23_pilot100": pilot' in text
    assert '"v23_rest400"' in text
    assert '"queue_wait_counts_toward_timeout": False' in text
