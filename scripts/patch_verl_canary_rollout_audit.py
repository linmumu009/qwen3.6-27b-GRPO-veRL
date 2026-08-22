#!/usr/bin/env python3
"""Persist non-sensitive per-trajectory advantage/token evidence in veRL dumps."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_CANARY_ROLLOUT_AUDIT_V1"


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"
    anchor = '''\
    def _fit_dump_data(self, batch: DataProto):
        timing_raw = self.timing_raw
        reward_extra_infos_dict = self.reward_extra_infos_dict
'''
    replacement = '''\
    def _fit_dump_data(self, batch: DataProto):
        timing_raw = self.timing_raw
        # LLIN_CANARY_ROLLOUT_AUDIT_V1: add only numeric trajectory evidence;
        # prompts, gold and raw tool output remain in the private rollout dump.
        reward_extra_infos_dict = dict(self.reward_extra_infos_dict)
        response_mask = batch.batch["response_mask"]
        active_denominator = response_mask.sum(dim=-1).clamp_min(1)
        reward_extra_infos_dict["trajectory_advantage_mean"] = (
            (batch.batch["advantages"] * response_mask).sum(dim=-1) / active_denominator
        ).detach().cpu().tolist()
        reward_extra_infos_dict["trajectory_return_mean"] = (
            (batch.batch["returns"] * response_mask).sum(dim=-1) / active_denominator
        ).detach().cpu().tolist()
        reward_extra_infos_dict["trajectory_active_response_tokens"] = (
            response_mask.sum(dim=-1).detach().cpu().tolist()
        )
        response_width = batch.batch["responses"].shape[-1]
        reward_extra_infos_dict["trajectory_total_response_tokens"] = (
            batch.batch["attention_mask"][:, -response_width:].sum(dim=-1).detach().cpu().tolist()
        )
'''
    if text.count(anchor) != 1:
        raise RuntimeError(f"expected one rollout-audit anchor in {path}")
    path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer",
        default="/verl/verl/experimental/separation/ray_trainer.py",
    )
    args = parser.parse_args()
    print(f"{patch(Path(args.trainer))}: {args.trainer}")


if __name__ == "__main__":
    main()
