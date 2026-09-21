"""Delivery contracts that can be checked without an Ascend runtime."""
import importlib.util
import json
from pathlib import Path
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1] / "delivery/business_training_delivery"
KINDS = ["SFT", "trajory-SFT", "GRPO", "agentic-GRPO", "CPT"]


class DeliveryTests(unittest.TestCase):
    def test_weight_sync_socket_ranks_match_across_physical_device_slices(self):
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location("delivery_sync", ROOT / "tools/weight_sync.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = SimpleNamespace(tensor_parallel_size=4, pipeline_parallel_size=1, data_parallel_size=1)
        senders = {(physical // 4, physical % 4) for physical in range(16)}
        receivers = {(replica, module.colocated_local_rank(local, config))
                     for replica in range(4) for local in range(4)}
        self.assertEqual(senders, receivers)
        with self.assertRaises(ValueError):
            module.colocated_local_rank(4, config)
        config.data_parallel_size = 2
        with self.assertRaises(ValueError):
            module.colocated_local_rank(0, config)

    def test_collector_rejects_invalid_export_despite_success_exit(self):
        spec = importlib.util.spec_from_file_location("delivery_collect", ROOT / "tools/collect_verification.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs").mkdir()
            run = root / "outputs/SFT/demo"
            (run / "hf_export").mkdir(parents=True)
            (run / "status.json").write_text(json.dumps({"state": "exported", "exit_code": 0}))
            verification = dict(valid=False, base_tensor_count=2, output_tensor_count=1,
                                referenced_shard_count=1, missing_tensor_count=1,
                                extra_tensor_count=0, shape_mismatch_count=0)
            (run / "hf_export/llin_export_manifest.json").write_text(json.dumps({"verification": verification}))
            self.assertEqual(module.collect(root)["SFT"][0]["state"], "export_verification_failed")

    def test_pageable_offload_patch_requires_synchronous_copy_and_is_idempotent(self):
        spec = importlib.util.spec_from_file_location("delivery_offload_patch", ROOT / "environment/apply_pageable_offload.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = (module.OLD + "buffer.param_data.cpu_data.copy_(buffer.param_data.data, non_blocking=False)\n"
                  'tensor.data = tensor.data.to("cpu", non_blocking=True)\n')
        patched = module.patch_text(source)
        self.assertIn("pin_memory=False", patched)
        self.assertIn('.to("cpu", non_blocking=False)', patched)
        self.assertEqual(module.patch_text(patched), patched)
        with self.assertRaises(ValueError):
            module.patch_text(source.replace("non_blocking=False", "non_blocking=True"))

    def test_incomplete_checkpoint_is_not_reported_as_success(self):
        spec = importlib.util.spec_from_file_location("delivery_checkpoint", ROOT / "tools/checkpoint.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertFalse(module.checkpoint_ready(directory))
            manifest = directory / "ckpt_contents.json"
            value = {"contents": {"model": {"format": "megatron_dist_checkpoint", "path": "model/dist_ckpt"}}}
            manifest.write_text(json.dumps(value))
            weights = directory / "model/dist_ckpt"
            weights.mkdir(parents=True)
            (weights / ".metadata").write_bytes(b"metadata")
            (weights / "rank0.distcp").write_bytes(b"")
            self.assertFalse(module.checkpoint_ready(directory))
            (weights / "rank0.distcp").write_bytes(b"model weights")
            self.assertTrue(module.checkpoint_ready(directory))
            value["contents"]["model"]["path"] = "../external"
            manifest.write_text(json.dumps(value))
            self.assertFalse(module.checkpoint_ready(directory))

    def test_reward_rejects_reasoning_only_wrong_and_unmarked_answers(self):
        spec = importlib.util.spec_from_file_location("delivery_reward", ROOT / "tools/reward.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        score = lambda value: module.compute_score("demo", value, "42")
        self.assertEqual(score("Answer: 42"), 1)
        self.assertEqual(score("<think>Answer: 41</think>\n答案：42"), 1)
        for wrong in ("Answer: 41", "42", "<think>Answer: 42", "<think>Answer: 42</think>"):
            self.assertEqual(score(wrong), 0, wrong)

    def test_data_split_and_trajectory_tool_contract(self):
        for kind in KINDS:
            rows = {}
            for split, expected in (("train", 32), ("val", 8)):
                rows[split] = [json.loads(line) for line in (ROOT / "datasets" / kind / f"{split}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows[split]), expected)
            encoded = lambda rs: {json.dumps(r, sort_keys=True) for r in rs}
            self.assertFalse(encoded(rows["train"]) & encoded(rows["val"]))
            if kind == "trajory-SFT":
                for row in rows["train"] + rows["val"]:
                    call, tool, answer = row["messages"][1:]
                    self.assertEqual(call["tool_calls"][0]["id"], tool["tool_call_id"])
                    parameters = call["tool_calls"][0]["function"]["arguments"]
                    gold = parameters["a"] + parameters["b"] if parameters["operation"] == "add" else parameters["a"] * parameters["b"]
                    self.assertEqual(tool["content"], str(gold))
                    self.assertEqual(answer["content"], f"Answer: {gold}")

    def test_recipes_do_not_depend_on_historical_data_or_checkpoints(self):
        for kind in KINDS:
            config = json.loads((ROOT / "Train" / kind / "config.yaml").read_text())
            for value in config["overrides"]:
                self.assertNotIn("/workspace/", value)
                self.assertNotIn("192.168.", value)
                self.assertNotIn("resume_path", value)
                value.format(ROOT="/delivery", MODEL="/models/base", OUT="/delivery/outputs/test", KIND=kind)
            self.assertIn("trainer.total_training_steps=1", config["overrides"])
            self.assertIn("trainer.resume_mode=disable", config["overrides"])


if __name__ == "__main__":
    unittest.main()
