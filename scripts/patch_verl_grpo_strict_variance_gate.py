#!/usr/bin/env python3
"""Patch veRL separation training with a strict-correctness GRPO gate."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_STRICT_CORRECTNESS_GROUP_GATE"
V2_MARKER = "LLIN_STRICT_STALENESS_ZERO_V2"


def _upgrade_v2(text: str, path: Path) -> str:
    if V2_MARKER in text:
        return text
    text = text.replace(
        "            batch, strict_group_metrics = apply_strict_correctness_group_gate(batch)\n",
        "            # LLIN_STRICT_STALENESS_ZERO_V2: require the rollout policy used for this batch.\n"
        "            batch.meta_info[\"strict_expected_policy_version\"] = self.global_steps - 1\n"
        "            batch, strict_group_metrics = apply_strict_correctness_group_gate(batch)\n",
        1,
    )
    text = text.replace(
        "        timing_raw = self.timing_raw\n        if batch.meta_info.get(\"strict_group_should_update_actor\") is False:\n",
        "        timing_raw = self.timing_raw\n"
        "        metrics[\"training/nominal_group_step\"] = float(self.global_steps)\n"
        "        if not hasattr(self, \"strict_optimizer_steps\"):\n"
        "            self.strict_optimizer_steps = 0\n"
        "        if batch.meta_info.get(\"strict_group_should_update_actor\") is False:\n",
        1,
    )
    text = text.replace(
        "            metrics[\"actor/grad_norm\"] = 0.0\n            return batch\n",
        "            metrics[\"actor/grad_norm\"] = 0.0\n"
        "            metrics[\"training/optimizer_step\"] = float(self.strict_optimizer_steps)\n"
        "            return batch\n",
        1,
    )
    text = text.replace(
        "            metrics.update(actor_output_metrics)\n        return batch\n",
        "            metrics.update(actor_output_metrics)\n"
        "            self.strict_optimizer_steps += 1\n"
        "            metrics[\"training/optimizer_step\"] = float(self.strict_optimizer_steps)\n"
        "        return batch\n",
        1,
    )
    text = text.replace(
        "    def _fit_update_weights(self):\n        timing_raw = self.timing_raw\n",
        "    def _fit_update_weights(self):\n"
        "        timing_raw = self.timing_raw\n"
        "        if self.metrics.get(\"actor/update_skipped_no_strict_mixed\") == 1.0:\n"
        "            self.metrics[\"training/weight_sync_skipped_no_optimizer_step\"] = 1.0\n"
        "            return\n"
        "        self.metrics[\"training/weight_sync_skipped_no_optimizer_step\"] = 0.0\n",
        1,
    )
    if V2_MARKER not in text:
        raise RuntimeError(f"failed to upgrade strict group gate to v2 in {path}")
    return text


def patch_trainer(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        upgraded = _upgrade_v2(text, path)
        if upgraded != text:
            path.write_text(upgraded, encoding="utf-8")
            return "upgraded-v2"
        return "already-patched"

    advantage_anchor = '''\
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
        return batch
'''
    advantage_replacement = '''\
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
            # LLIN_STRICT_CORRECTNESS_GROUP_GATE: the scalar reward and KL
            # penalty have already been assembled.  Mask uniform correctness
            # groups last so no proxy or regularizer can revive their update.
            from llin_verl.grpo_group_gate import apply_strict_correctness_group_gate

            # LLIN_STRICT_STALENESS_ZERO_V2: require the rollout policy used for this batch.
            batch.meta_info["strict_expected_policy_version"] = self.global_steps - 1
            batch, strict_group_metrics = apply_strict_correctness_group_gate(batch)
            metrics.update(strict_group_metrics)
        return batch
'''
    if text.count(advantage_anchor) != 1:
        raise RuntimeError(f"expected one advantage anchor in {path}")
    text = text.replace(advantage_anchor, advantage_replacement, 1)

    update_anchor = '''\
    def _fit_update_actor(self, batch: DataProto) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        # implement critic warmup
'''
    update_replacement = '''\
    def _fit_update_actor(self, batch: DataProto) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw
        metrics["training/nominal_group_step"] = float(self.global_steps)
        if not hasattr(self, "strict_optimizer_steps"):
            self.strict_optimizer_steps = 0
        if batch.meta_info.get("strict_group_should_update_actor") is False:
            # LLIN_STRICT_CORRECTNESS_GROUP_GATE: skip the optimizer call, not
            # merely its loss, so Adam momentum cannot move an all-uniform batch.
            metrics["actor/update_skipped_no_strict_mixed"] = 1.0
            metrics["actor/grad_norm"] = 0.0
            metrics["training/optimizer_step"] = float(self.strict_optimizer_steps)
            return batch
        metrics["actor/update_skipped_no_strict_mixed"] = 0.0
        # implement critic warmup
'''
    if text.count(update_anchor) != 1:
        raise RuntimeError(f"expected one actor-update anchor in {path}")
    text = text.replace(update_anchor, update_replacement, 1)
    actor_output_anchor = '''\
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
        return batch
'''
    actor_output_replacement = '''\
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
            self.strict_optimizer_steps += 1
            metrics["training/optimizer_step"] = float(self.strict_optimizer_steps)
        return batch
'''
    if text.count(actor_output_anchor) != 1:
        raise RuntimeError(f"expected one actor output anchor in {path}")
    text = text.replace(actor_output_anchor, actor_output_replacement, 1)
    weight_anchor = '''\
    def _fit_update_weights(self):
        timing_raw = self.timing_raw
        if self.config.trainer.critic_warmup <= self.global_steps:
'''
    weight_replacement = '''\
    def _fit_update_weights(self):
        timing_raw = self.timing_raw
        if self.metrics.get("actor/update_skipped_no_strict_mixed") == 1.0:
            self.metrics["training/weight_sync_skipped_no_optimizer_step"] = 1.0
            return
        self.metrics["training/weight_sync_skipped_no_optimizer_step"] = 0.0
        if self.config.trainer.critic_warmup <= self.global_steps:
'''
    if text.count(weight_anchor) != 1:
        raise RuntimeError(f"expected one weight update anchor in {path}")
    path.write_text(text.replace(weight_anchor, weight_replacement, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer",
        default="/verl/verl/experimental/separation/ray_trainer.py",
    )
    args = parser.parse_args()
    print(f"{patch_trainer(Path(args.trainer))}: {args.trainer}")


if __name__ == "__main__":
    main()
