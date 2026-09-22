"""Offline exposure/order and measurement-gap audit; never calls a model.

Uses the frozen sampler digest and per-step token totals to check attribution.
Batch losses are mixed-example losses, not source-specific learning curves.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_logistics_cpt_run import parse_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    import pyarrow.parquet as pq
    import torch

    gate = read(root / "results/gate.safe.json")
    order = torch.randperm(333, generator=torch.Generator().manual_seed(1)).tolist()
    order_sha = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    assert order_sha == gate["sampler_sha256"], "sampler reconstruction mismatch"
    reg = read(root / "results/registration.safe.json")
    sources = read(root / "sources.private.json")
    probes = read(root / "probes.private.json")
    results = {"new_generation_calls": 0, "new_training_steps": 0,
               "sampler_sha256": order_sha, "arms": {}, "units": [],
               "limitations": ["Mixed batch loss is not source-only loss.",
                               "Clipping frequency is not proof of inadequate learning.",
                               "No same-training-prompt generation or teacher-forced source loss measured."]}
    position = {idx: i for i, idx in enumerate(order)}
    for arm in ("K", "R"):
        data = root / "prepared" / arm / "train.parquet"
        assert digest(data) == reg["files"][f"prepared/{arm}/train.parquet"]
        rows = pq.read_table(data).to_pylist()
        messages_path = root / "prepared" / arm / "train.messages.private.jsonl"
        assert digest(messages_path) == reg["files"][f"prepared/{arm}/train.messages.private.jsonl"]
        messages = [json.loads(x) for x in messages_path.read_text(encoding="utf8").splitlines()]
        logs = sorted((root / "results" / (arm + "_training") / "torchrun_logs").glob("*/attempt_0/*/stdout.log"))
        metrics = parse_metrics("\n".join(p.read_text(encoding="utf8", errors="replace") for p in logs))
        assert len(rows) == len(messages) == 333
        assert sorted(metrics) == list(range(1, 112))
        steps = []
        for step in range(1, 112):
            indices = order[(step-1)*3:step*3]
            assert sum(len(rows[i]["input_ids"]) for i in indices) == metrics[step]["train/global_tokens"]
            steps.append({"step": step, "new_records": sum(i >= 315 for i in indices),
                          "loss": metrics[step]["train/loss"], "grad_norm": metrics[step]["train/grad_norm"]})
        old = sum(r["loss_tokens"] for r in rows[:315])
        new = sum(r["loss_tokens"] for r in rows[315:])
        results["arms"][arm] = {
            "old_supervised_tokens": old, "new_supervised_tokens": new,
            "new_supervised_share": new/(old+new),
            "new_record_share": 18/333,
            "steps_containing_new_records": [r["step"] for r in steps if r["new_records"]],
            "all_111_step_token_totals_match": True,
            "gradient_norm_above_clip_1_steps": sum(r["grad_norm"] > 1 for r in steps),
            "gradient_norm_median": statistics.median(r["grad_norm"] for r in steps),
            "batch_metrics": steps,
        }
        if arm == "K":
            for source in sources:
                uid = source["unit_id"]
                indices = [i for i, r in enumerate(rows) if r["id"].startswith("K-" + uid + "-")]
                assert len(indices) == 3
                assert all(messages[i]["messages"][-1]["content"] == source["body"] for i in indices)
                assert hashlib.sha256(source["body"].encode()).hexdigest() == source["body_sha256"]
                teaching = messages[indices[0]]["messages"][:-1]
                q = [p for p in probes if p["unit_id"] == uid]
                assert len(q) == 2
                exposures = sorted(position[i]//3+1 for i in indices)
                results["units"].append({
                    "unit_id": uid, "body_sha256": source["body_sha256"],
                    "supervised_tokens_per_exposure": [rows[i]["loss_tokens"] for i in indices],
                    "exposure_steps": exposures, "steps_after_last_exposure": 111-max(exposures),
                    "probe_system_equals_training": [p["messages"][0] == teaching[0] for p in q],
                    "probe_user_equals_training": [p["messages"][1] == teaching[1] for p in q],
                    "exposure_batch_losses": [metrics[s]["train/loss"] for s in exposures],
                })
    assert results["arms"]["K"]["old_supervised_tokens"] == results["arms"]["R"]["old_supervised_tokens"]
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.write_text(json.dumps(audit(args.root), indent=2) + "\n", encoding="utf8")
