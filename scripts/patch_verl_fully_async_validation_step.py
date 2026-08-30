#!/usr/bin/env python3
"""Make fully-async validation artifacts use the trainer policy step."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


TRAINER_MARKER = "LLIN_FULLY_ASYNC_VALIDATION_STEP"
ROLLOUTER_MARKER = "LLIN_FULLY_ASYNC_VALIDATION_STEP"
BASE_TRAINER_MARKER = "LLIN_VALIDATION_IDENTITY_JOIN_V1"


def _replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"expected validation patch anchor not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def patch_trainer(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if TRAINER_MARKER in text:
        return "already-patched"

    old = "val_metrics = await self.rollouter.do_validate.remote()"
    count = text.count(old)
    if count != 2:
        raise RuntimeError(f"expected two validation RPC anchors in {path}, found {count}")
    new_call = "val_metrics = await self.rollouter.do_validate.remote(self.current_param_version)"
    text = text.replace(old, new_call)
    match = re.search(rf"(?m)^([ \t]+){re.escape(new_call)}$", text)
    if match is None:
        raise RuntimeError(f"expected indented validation RPC not found in {path}")
    indent = match.group(1)
    first_indented_call = f"{indent}{new_call}"
    marker_block = (
        f"{indent}# LLIN_FULLY_ASYNC_VALIDATION_STEP: pass the trainer policy version;\n"
        f"{indent}# the rollouter's own global_steps is a different data counter.\n"
        f"{first_indented_call}"
    )
    path.write_text(text.replace(first_indented_call, marker_block, 1), encoding="utf-8")
    return "patched"


def patch_rollouter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if ROLLOUTER_MARKER in text:
        return "already-patched"

    old = '''\
    def do_validate(self):
        """Run validation and return metrics"""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return timing_raw | val_metrics
'''
    new = '''\
    def do_validate(self, validation_step: int | None = None):
        """Run validation and name artifacts with the trainer policy version."""
        timing_raw = {}
        original_global_steps = self.global_steps
        try:
            # LLIN_FULLY_ASYNC_VALIDATION_STEP: inherited validation dumping uses
            # self.global_steps, but this actor normally stores a rollout-data
            # counter there. Temporarily expose the trainer's policy step.
            if validation_step is not None:
                self.global_steps = int(validation_step)
            with marked_timer("rollouter/validate_time", timing_raw, color="green"):
                val_metrics: dict = self._validate()
        finally:
            self.global_steps = original_global_steps
        return timing_raw | val_metrics
'''
    if old not in text:
        raise RuntimeError(f"expected rollouter validation anchor not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def patch_base_trainer(path: Path) -> str:
    """Join partial validation output to prompts by frozen trajectory identity."""

    text = path.read_text(encoding="utf-8")
    if BASE_TRAINER_MARKER in text:
        return "already-patched"

    import_old = "from verl.trainer.ppo.reward import extract_reward\n"
    import_new = """\
from verl.trainer.ppo.reward import extract_reward

# LLIN_VALIDATION_IDENTITY_JOIN_V1: Fastest-K and UNKNOWN handling can return
# fewer physical validation trajectories than were requested.  Align only real
# output by the frozen task/prefix/policy/slot identity; never positional-unpad,
# duplicate, truncate, or synthesize a trajectory to satisfy DataProto.union.
from llin_verl.validation_identity import (
    IDENTITY_KEY,
    PADDING_KEY,
    POLICY_KEY,
    SLOT_KEY,
    align_returned_validation,
    apply_judge_states,
    build_validation_identities,
    mark_padding_identities,
    write_identity_status,
)
"""
    text = _replace_once(text, import_old, import_new, path)

    collector_old = """\
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
"""
    collector_new = """\
        sample_turns = []
        sample_uids = []
        validation_identity_status_rows = []

        for test_data in self.val_dataloader:
"""
    text = _replace_once(text, collector_old, collector_new, path)

    repeat_old = """\
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
"""
    repeat_new = """\
            validation_samples_per_state = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
            test_batch = test_batch.repeat(
                repeat_times=validation_samples_per_state, interleave=True
            )
            extra_info_values = test_batch.non_tensor_batch.get("extra_info")
            if extra_info_values is None:
                raise ValueError("identity-safe validation requires extra_info")
            expected_validation_ids, validation_slots, validation_policies = build_validation_identities(
                extra_info_values.tolist(),
                samples_per_state=validation_samples_per_state,
                policy_version=int(self.global_steps),
            )
            test_batch.non_tensor_batch[IDENTITY_KEY] = np.asarray(expected_validation_ids, dtype=object)
            test_batch.non_tensor_batch[SLOT_KEY] = np.asarray(validation_slots, dtype=np.int64)
            test_batch.non_tensor_batch[POLICY_KEY] = np.asarray(validation_policies, dtype=np.int64)

            test_gen_batch = self._get_gen_batch(test_batch)
"""
    text = _replace_once(text, repeat_old, repeat_new, path)

    padding_old = """\
            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
"""
    padding_new = """\
            # Pad only for worker divisibility.  Padding rows receive unique
            # identities so a partial or reordered return can be joined safely.
            expected_validation_count = len(test_gen_batch)
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            padded_validation_ids, padded_validation_mask = mark_padding_identities(
                test_gen_batch_padded.non_tensor_batch[IDENTITY_KEY].tolist(),
                expected_count=expected_validation_count,
            )
            test_gen_batch_padded.non_tensor_batch[IDENTITY_KEY] = np.asarray(
                padded_validation_ids, dtype=object
            )
            test_gen_batch_padded.non_tensor_batch[PADDING_KEY] = np.asarray(
                padded_validation_mask, dtype=bool
            )
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
"""
    text = _replace_once(text, padding_old, padding_new, path)

    unpad_old = """\
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")
"""
    unpad_new = """\
            returned_identity_values = test_output_gen_batch_padded.non_tensor_batch.get(IDENTITY_KEY)
            returned_padding_values = test_output_gen_batch_padded.non_tensor_batch.get(PADDING_KEY)
            if returned_identity_values is None or returned_padding_values is None:
                raise ValueError("validation rollout did not preserve frozen identity/padding fields")
            returned_validation_ids = [str(value) for value in returned_identity_values.tolist()]
            returned_validation_padding = [bool(value) for value in returned_padding_values.tolist()]
            expected_indices, returned_indices, batch_identity_status = align_returned_validation(
                expected_validation_ids,
                returned_validation_ids,
                returned_padding=returned_validation_padding,
            )
            if not returned_indices:
                val_data_dir = self.config.trainer.get("validation_data_dir", None)
                if val_data_dir:
                    write_identity_status(
                        os.path.join(val_data_dir, f"{self.global_steps}.identity_status.jsonl"),
                        validation_identity_status_rows + batch_identity_status,
                    )
                raise RuntimeError("validation returned no real trajectory identities")
            test_output_gen_batch = test_output_gen_batch_padded.select_idxs(returned_indices)
            test_batch = test_batch.select_idxs(expected_indices)
            returned_real_validation_ids = [returned_validation_ids[index] for index in returned_indices]

            print(
                "validation generation end "
                f"expected={len(expected_validation_ids)} returned={len(returned_real_validation_ids)} "
                f"missing={len(expected_validation_ids) - len(returned_real_validation_ids)} pad_size={pad_size}"
            )
"""
    text = _replace_once(text, unpad_old, unpad_new, path)

    union_old = """\
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
"""
    union_new = """\
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            # Store original inputs
"""
    text = _replace_once(text, union_old, union_new, path)

    reward_old = """\
            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
"""
    reward_new = """\
            # evaluate using reward_function.  Judge states are joined back to
            # the requested-slot ledger by identity; UNKNOWN remains a masked
            # resample state and no score/correctness value is rewritten.
            reward_tensor, reward_extra_info = extract_reward(test_batch)
            batch_identity_status = apply_judge_states(
                batch_identity_status,
                returned_real_validation_ids,
                reward_extra_info.get("judge_state"),
            )
            validation_identity_status_rows.extend(batch_identity_status)

            scores = reward_tensor.sum(-1).cpu().tolist()
"""
    text = _replace_once(text, reward_old, reward_new, path)

    dump_old = """\
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
"""
    dump_new = """\
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            write_identity_status(
                os.path.join(val_data_dir, f"{self.global_steps}.identity_status.jsonl"),
                validation_identity_status_rows,
            )
            self._dump_generations(
"""
    text = _replace_once(text, dump_old, dump_new, path)

    path.write_text(text, encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer",
        default="/verl/verl/experimental/fully_async_policy/fully_async_trainer.py",
    )
    parser.add_argument(
        "--rollouter",
        default="/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py",
    )
    parser.add_argument(
        "--base-trainer",
        default="/verl/verl/trainer/ppo/ray_trainer.py",
    )
    args = parser.parse_args()
    print(f"{patch_trainer(Path(args.trainer))}: {args.trainer}")
    print(f"{patch_rollouter(Path(args.rollouter))}: {args.rollouter}")
    print(f"{patch_base_trainer(Path(args.base_trainer))}: {args.base_trainer}")


if __name__ == "__main__":
    main()
