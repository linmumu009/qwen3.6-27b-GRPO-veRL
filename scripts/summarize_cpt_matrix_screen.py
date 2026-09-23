"""Verify all 80 saved predictions and report the registered readiness decision."""
import argparse
from pathlib import Path
from cpt_two_family import read, sha, freeze, screen_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit-output", type=Path)
    a = p.parse_args()
    root = a.root
    reg = read(root / "execution.safe.json")
    remote = root / "remote_run"
    assert read(remote / "execution.safe.json") == reg
    assert sha(root / "packet.private.json") == reg["packet_sha256"]
    assert sha(root / "quality.safe.json") == reg["quality_sha256"]
    assert read(remote / "status.safe.json")["state"] == "complete_pending_local_verification"
    folder = remote / "p1"
    completed = read(folder / "completed.safe.json")
    assert completed["calls"] == 80 and sha(folder / "predictions.private.json") == completed["predictions_sha256"]
    rows = read(folder / "predictions.private.json")
    batch_rows = []
    for start in range(0, 80, 16):
        name = f"batch-{start:03d}"
        result, reservation, done = (folder / (name + suffix) for suffix in (".private.json", ".reserved.safe.json", ".completed.safe.json"))
        r, d = read(reservation), read(done)
        assert r["start"] == start and r["calls"] == d["calls"] == 16
        assert r["packet_sha256"] == reg["packet_sha256"] and sha(result) == d["sha256"]
        batch_rows.extend(read(result))
    assert batch_rows == rows
    ledger, summary = screen_score(read(root / "packet.private.json"), rows)
    summary.update(training_started=False, completed_batches=5,
                   packet_sha256=reg["packet_sha256"], predictions_sha256=completed["predictions_sha256"],
                   source_condition_is_not_pure_recall=True,
                   lower_order_item=next({k: g[k] for k in ("id", "closed", "source", "repeat_consistent")} for g in ledger if g["id"] == "MU4"),
                   decision="proceed_to_token_and_installed_input_gates" if summary["training_value_passed"] else "closed_at_registered_screen_gate")
    freeze(root / "screen_ledger.private.json", ledger)
    freeze(a.output, summary)
    if a.audit_output:
        audit_path = root / "independent_execution_audit.private.json"
        audit = read(audit_path)
        assert audit["integrity_passed"] is True and audit["integrity_issues"] == []
        for name, fingerprint in audit["input_sha256"].items():
            assert sha(root / name) == fingerprint
        assert audit["counts"]["audited"] == 80 and audit["counts"]["invalid"] == summary["invalid"]
        assert audit["counts"]["truncated"] == summary["truncated"]
        for family in summary["families"]:
            independent = audit["counts"]["families"][family["family"]]
            assert independent["closed_all_four_correct_items"] == family["closed_stable"]
            assert independent["source_all_four_correct_items"] == family["source_stable"]
        safe = {k: audit[k] for k in ("input_sha256", "audit_method", "integrity_passed", "integrity_issues", "counts", "limits")}
        safe["private_audit_sha256"] = sha(audit_path)
        freeze(a.audit_output, safe)
    print(summary)


if __name__ == "__main__":
    main()
