#!/usr/bin/env python3
"""Patch veRL separation training with a strict-correctness GRPO gate."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_STRICT_CORRECTNESS_GROUP_GATE"


def patch_trainer(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
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
        if batch.meta_info.get("strict_group_should_update_actor") is False:
            # LLIN_STRICT_CORRECTNESS_GROUP_GATE: skip the optimizer call, not
            # merely its loss, so Adam momentum cannot move an all-uniform batch.
            metrics["actor/update_skipped_no_strict_mixed"] = 1.0
            metrics["actor/grad_norm"] = 0.0
            return batch
        metrics["actor/update_skipped_no_strict_mixed"] = 0.0
        # implement critic warmup
'''
    if text.count(update_anchor) != 1:
        raise RuntimeError(f"expected one actor-update anchor in {path}")
    path.write_text(text.replace(update_anchor, update_replacement, 1), encoding="utf-8")
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
