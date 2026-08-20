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


def test_fresh_acquisition_cleanup_reaps_detached_vllm_workers() -> None:
    text = (ROOT / "scripts" / "run_qwen38_fresh_acquisition_host.py").read_text(
        encoding="utf-8"
    )
    assert 'VLLM_ORPHAN_PATTERN = "^VLLM::"' in text
    assert '["pkill", "-TERM", "-f", "--", VLLM_ORPHAN_PATTERN]' in text
    assert '["pkill", "-KILL", "-f", "--", VLLM_ORPHAN_PATTERN]' in text
    stop_body = text[text.index("def stop") : text.index("def execute")]
    assert stop_body.index('"ray stop --force"') < stop_body.rindex("vllm_shell")


def test_cleanup_guardian_waits_for_finished_marker_and_checks_all_hosts() -> None:
    text = (ROOT / "scripts" / "guard_qwen38_fresh_cleanup_host.py").read_text(
        encoding="utf-8"
    )
    assert 'finished = args.supervisor_dir / "finished_at"' in text
    assert "acquisition.stop(args)" in text
    assert "for host in acquisition.specs(args)" in text
    assert "wait_until_idle(args, host)" in text
    assert "time.monotonic() + timeout_seconds" in text
