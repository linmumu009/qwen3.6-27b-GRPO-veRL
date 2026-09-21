"""Summarize actual run artifacts; never promote preflight to training success."""
import json
from pathlib import Path
import re


def collect(root):
    result = {}
    for kind in ("SFT", "trajory-SFT", "GRPO", "agentic-GRPO", "CPT"):
        runs = []
        for directory in sorted((root / "outputs" / kind).glob("*")):
            if not directory.is_dir():
                continue
            run = {"path": str(directory.relative_to(root))}
            status = directory / "status.json"
            if status.exists():
                run.update(json.loads(status.read_text()))
            else:
                run["state"] = "preflight_only" if (directory / "preflight.json").exists() else "incomplete"
            exit_file = directory / "exit_code"
            if exit_file.exists():
                run["wrapper_exit_code"] = int(exit_file.read_text())
            tool_audit = directory / "tool_calls.jsonl"
            if tool_audit.exists():
                calls = [json.loads(line) for line in tool_audit.read_text().splitlines() if line.strip()]
                run["tool_audit"] = {
                    "calls": len(calls),
                    "successful": sum(bool(call["success"]) for call in calls),
                    "unique_instances": len({call["instance_id"] for call in calls}),
                }
            export_manifest = directory / "hf_export/llin_export_manifest.json"
            if export_manifest.exists():
                verification = json.loads(export_manifest.read_text())["verification"]
                run["hf_export_verification"] = {
                    key: verification[key] for key in (
                        "valid", "base_tensor_count", "output_tensor_count", "referenced_shard_count",
                        "missing_tensor_count", "extra_tensor_count", "shape_mismatch_count")
                }
                if run["state"] == "exported" and not verification["valid"]:
                    run["state"] = "export_verification_failed"
            manifests = []
            for manifest in directory.glob("checkpoints/global_step_*/**/ckpt_contents.json"):
                value = json.loads(manifest.read_text())
                model = value.get("contents", {}).get("model", {})
                checkpoint = manifest.parent / model.get("path", "model/dist_ckpt")
                shards = list(checkpoint.glob("*.distcp"))
                manifests.append({"path": str(manifest.relative_to(root)),
                                  "format": model.get("format"), "shards": len(shards),
                                  "bytes": sum(s.stat().st_size for s in shards),
                                  "metadata_exists": (checkpoint / ".metadata").exists()})
            run["checkpoint_manifests"] = manifests
            if run["state"] == "completed":
                run["checkpoint_integrity_gate"] = bool(manifests) and all(
                    m["shards"] > 0 and m["bytes"] > 0 and m["metadata_exists"] for m in manifests)
                if not run["checkpoint_integrity_gate"]:
                    run["state"] = "checkpoint_verification_failed"
            log = directory / "driver.log"
            if log.exists():
                text = log.read_text(errors="replace")
                validation = re.findall(r"Final validation metrics: (\{[^\n]+\})", text)
                if validation:
                    run["final_validation_metrics"] = validation[-1]
                for metric in ("train/loss", "train/grad_norm", "actor/grad_norm", "critic/score/mean"):
                    values = re.findall(re.escape(metric) + r":\s*([-+\d.eE]+)", text)
                    if values:
                        run[metric] = float(values[-1])
            runs.append(run)
        result[kind] = runs
    model = root / "model/qwen3.6-27b"
    result["model_is_standalone"] = model.is_dir() and not model.is_symlink()
    result["model_checksums_present"] = (model / "delivery_checksums.json").is_file()
    result["offline_image_present"] = (root / "environment/ascend-verl-base.tar").is_file()
    (root / "docs/verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(collect(Path(__file__).resolve().parents[1]), indent=2))
