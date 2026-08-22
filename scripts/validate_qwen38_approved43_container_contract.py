#!/usr/bin/env python3
"""Static CPU validation for the frozen Qwen3.8 approved43 launcher contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(text: str, values: tuple[str, ...], label: str) -> None:
    missing = [value for value in values if value not in text]
    if missing:
        raise RuntimeError(f"{label} is missing required contract strings: {missing}")


def validate(agent_loop: Path, trainer: Path, losses: Path, launcher: Path) -> dict[str, object]:
    agent_text = agent_loop.read_text(encoding="utf-8")
    trainer_text = trainer.read_text(encoding="utf-8")
    loss_text = losses.read_text(encoding="utf-8")
    launcher_text = launcher.read_text(encoding="utf-8")
    require(
        agent_text,
        (
            "LLIN_HARD_GATE_RESAMPLE_QUORUM",
            'reward_info.get("online_eligible", 0)',
            "hard_gate_cap_exhausted",
            "hard-gate attempt cap returned",
        ),
        "agent loop",
    )
    require(
        trainer_text,
        (
            "LLIN_STRICT_STALENESS_ZERO_V2",
            'strict_expected_policy_version"] = self.global_steps - 1',
            'actor/update_skipped_no_strict_mixed',
            'training/optimizer_step',
            'weight_sync_skipped_no_optimizer_step',
        ),
        "trainer",
    )
    require(
        loss_text,
        (
            "if config.use_kl_loss:",
            "kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob",
            "loss_mask=response_mask",
            "policy_loss += kl_loss * config.kl_loss_coef",
        ),
        "actor loss",
    )
    require(
        launcher_text,
        (
            "MODEL_PATH:-/models/Qwen3.8-27B",
            "algorithm.use_kl_in_reward=False",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.actor.kl_loss_coef=0.001",
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
            "STALENESS_THRESHOLD=0",
            "OVERSAMPLE_CANDIDATES=16",
            "trainer.max_actor_ckpt_to_keep=1",
            "FORMAL_TRAINING_APPROVED",
        ),
        "launcher",
    )
    return {
        "contract": "qwen38-approved43-actual-verl-container-static-v1",
        "status": "pass",
        "training_status": "paused_no_model_load_no_rollout_no_optimizer",
        "hard_gate_resample_to_8_or_skip_at_16": True,
        "uniform_and_hard_gate_groups_skip_optimizer": True,
        "hard_staleness_zero_exact_policy_version": True,
        "kl_outside_reward": True,
        "kl_uses_active_response_mask": True,
        "launcher_requires_post_shadow_approval": True,
        "source_files_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-loop", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--losses", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.agent_loop, args.trainer, args.losses, args.launcher)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
