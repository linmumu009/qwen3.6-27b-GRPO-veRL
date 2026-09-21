"""Small common launcher; all per-recipe parameters live in Train/*/config.yaml."""
import argparse
import asyncio
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.prepare_data import prepare
from tools.checkpoint import checkpoint_ready


def validate_data(root, kind, model):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from tools.calculator import CalculatorTool, SCHEMA, calculate
    from tools.reward import compute_score
    from verl.tools.schemas import OpenAIFunctionToolSchema
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    assert calculate({"a": 123, "b": 41, "operation": "multiply"}) == 5043
    for invalid in ({"a": True, "b": 2, "operation": "add"},
                    {"a": 1, "b": 2, "operation": "eval"},
                    {"a": 1000001, "b": 2, "operation": "add"}):
        try:
            calculate(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Calculator accepted invalid input")
    assert compute_score("demo", "Answer: 5043", "5043") == 1
    assert compute_score("demo", "Answer: 5042", "5043") == 0
    assert compute_score("demo", "5043", "5043") == 0
    async def tool_test():
        tool = CalculatorTool({}, OpenAIFunctionToolSchema.model_validate(SCHEMA))
        instance, _ = await tool.create()
        answer, _, _ = await tool.execute(instance, {"a": 123, "b": 41, "operation": "multiply"})
        assert answer.text == "5043"
        await tool.release(instance)
    # Do not mix preflight calls with actual online rollout evidence.
    audit = os.environ.pop("DELIVERY_TOOL_AUDIT", None)
    try:
        asyncio.run(tool_test())
    finally:
        if audit is not None:
            os.environ["DELIVERY_TOOL_AUDIT"] = audit
    result = {"calculator_unit_check": True, "reward_positive_negative_checks": True}
    if kind in {"SFT", "trajory-SFT", "CPT"}:
        from tools.loaders.qwen36_assistant_mask_sft_dataset import Qwen36AssistantMaskSFTDataset
        from tools.loaders.qwen36_causal_lm_dataset import Qwen36CausalLMDataset
        config = OmegaConf.create({"max_length": 1024, "truncation": "error", "pad_mode": "no_padding",
                                   "messages_key": "messages", "tools_key": "tools", "text_key": "text",
                                   "enable_thinking_default": False})
        cls = Qwen36CausalLMDataset if kind == "CPT" else Qwen36AssistantMaskSFTDataset
        details = {}
        for split in ("train", "val"):
            ds = cls(str(root / "datasets" / kind / f"{split}.parquet"), tokenizer, config)
            supervised = []
            for index in range(len(ds)):
                sample = ds[index]
                ids, mask = sample["input_ids"], sample["loss_mask"]
                assert len(ids) == len(mask) and 0 < int(mask.sum()) <= len(ids)
                if kind == "CPT":
                    assert mask[0] == 0 and bool((mask[1:] == 1).all())
                    assert ids[-1] == tokenizer.eos_token_id
                else:
                    assert 0 < int(mask.sum()) < len(ids)
                supervised.append(int(mask.sum()))
            details[split] = {"rows": len(ds), "min_supervised_tokens": min(supervised),
                              "max_supervised_tokens": max(supervised)}
        result["loader_checks"] = details
    else:
        import pyarrow.parquet as pq
        details = {}
        for split in ("train", "val"):
            rows = pq.read_table(root / "datasets" / kind / f"{split}.parquet").to_pylist()
            lengths = []
            for row in rows:
                assert row["reward_model"]["ground_truth"]
                rendered = tokenizer.apply_chat_template(row["prompt"], tools=[SCHEMA] if kind == "agentic-GRPO" else None,
                                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
                length = len(tokenizer.encode(rendered, add_special_tokens=False))
                assert length <= 1024
                lengths.append(length)
            details[split] = {"rows": len(rows), "max_prompt_tokens": max(lengths)}
        result["prompt_checks"] = details
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["SFT", "trajory-SFT", "GRPO", "agentic-GRPO", "CPT"])
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate data, tools and resolved trainer config; no training")
    mode.add_argument("--compose", action="store_true", help="Resolve trainer config only")
    mode.add_argument("--export-from", type=Path, help="Export an existing model checkpoint to Hugging Face format")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = Path("/models/base")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.export_from:
        checkpoint = args.export_from.resolve()
        if not checkpoint.is_relative_to((root / "outputs").resolve()):
            raise ValueError("Export checkpoint must be inside this package's outputs directory")
        command = [sys.executable, str(root / "tools/export_megatron_dist_to_hf.py"),
                   "--actor-checkpoint", str(checkpoint), "--base-model", str(model),
                   "--output-dir", str(out / "hf_export")]
        result = subprocess.run(command, timeout=1800)
        (out / "status.json").write_text(json.dumps({"state": "exported" if result.returncode == 0 else "export_failed",
                                                       "exit_code": result.returncode}))
        raise SystemExit(result.returncode)
    config = json.loads((root / "Train" / args.kind / "config.yaml").read_text())
    (out / "recipe.json").write_text(json.dumps(config, indent=2))
    substitutions = {"ROOT": str(root), "MODEL": str(model), "OUT": str(out), "KIND": args.kind}
    overrides = [value.format(**substitutions) for value in config["overrides"]]
    # This saved vector is reviewable and can be rerun without a shell parser.
    base = [sys.executable, "-m", config["module"], *overrides]
    (out / "command.json").write_text(json.dumps(base, indent=2))
    with (out / "resolved_config.yaml").open("w") as stream:
        subprocess.run([*base, "--cfg", "job", "--resolve"], stdout=stream, check=True)
    if args.compose:
        print("Configuration composition passed; training not started.")
        return
    prepare(root)
    checks = validate_data(root, args.kind, model)
    import resource
    checks["memlock_limit_bytes"] = list(resource.getrlimit(resource.RLIMIT_MEMLOCK))
    effective_caps = next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                          if line.startswith("CapEff:"))
    checks["ipc_lock_capability"] = bool(int(effective_caps, 16) & (1 << 14))
    if args.kind in {"GRPO", "agentic-GRPO"} and checks["memlock_limit_bytes"][0] != resource.RLIM_INFINITY:
        raise RuntimeError("GRPO requires Docker --ulimit memlock=-1:-1 for pinned-host-memory offload")
    if args.kind in {"GRPO", "agentic-GRPO"} and not checks["ipc_lock_capability"]:
        raise RuntimeError("Ascend GRPO host-memory offload requires Docker --cap-add IPC_LOCK")
    (out / "preflight.json").write_text(json.dumps(checks, indent=2))
    if args.check:
        print(json.dumps(checks, indent=2))
        print("Data/tool/config checks passed; training not started.")
        return
    if config["launcher"] == "torchrun":
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   "--nproc_per_node=16", "-m", config["module"], *overrides]
    else:
        command = base
    (out / "status.json").write_text(json.dumps({"state": "training"}))
    try:
        completed = subprocess.run(command, timeout=int(config.get("timeout_seconds", 1800)))
    except subprocess.TimeoutExpired:
        (out / "status.json").write_text(json.dumps({"state": "timed_out", "exit_code": 124}))
        raise SystemExit(124)
    checkpoints = list((out / "checkpoints").glob("global_step_*"))
    status = {"state": "completed" if completed.returncode == 0 else "failed",
              "exit_code": completed.returncode, "checkpoints": [str(path.relative_to(out)) for path in checkpoints]}
    if completed.returncode == 0 and (not checkpoints or not all(checkpoint_ready(path) for path in checkpoints)):
        status["state"] = "checkpoint_missing"
    if args.kind == "agentic-GRPO":
        audit = out / "tool_calls.jsonl"
        calls = [json.loads(line) for line in audit.read_text().splitlines()] if audit.exists() else []
        status["successful_online_tool_calls"] = sum(bool(call["success"]) for call in calls)
        if not status["successful_online_tool_calls"] and completed.returncode == 0:
            status["state"] = "online_tool_call_missing"
    (out / "status.json").write_text(json.dumps(status, indent=2))
    raise SystemExit(0 if status["state"] == "completed" else (completed.returncode or 4))


if __name__ == "__main__":
    main()
