#!/usr/bin/env python3
"""Make fastest-K select H=1 trajectories and fail closed at the attempt cap."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_HARD_GATE_RESAMPLE_QUORUM"


def patch(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"
    old = """\
                    for task in done:
                        if len(outputs) < fastest_k:
                            outputs.append(await task)
                            selected_indices.append(task_to_index[task])
                        else:
                            await task
                            completed_but_discarded += 1

                pending_indices = [task_to_index[task] for task in pending]
"""
    new = """\
                    # LLIN_HARD_GATE_RESAMPLE_QUORUM: reward is computed inside
                    # each completed agent loop. Only H=1 candidates can fill
                    # the GRPO group. H=0 candidates are retained only as
                    # fail-closed placeholders if the physical attempt cap is
                    # exhausted before eight eligible trajectories arrive.
                    if 'hard_gate_rejected' not in locals():
                        hard_gate_rejected = []
                    for task in done:
                        candidate = await task
                        reward_info = candidate.extra_fields.get("reward_extra_info", {})
                        eligible = bool(reward_info.get("online_eligible", 0))
                        if eligible and len(outputs) < fastest_k:
                            outputs.append(candidate)
                            selected_indices.append(task_to_index[task])
                        else:
                            hard_gate_rejected.append((task_to_index[task], candidate))
                            completed_but_discarded += 1

                pending_indices = [task_to_index[task] for task in pending]
"""
    if old not in text:
        raise RuntimeError("fastest-K selection anchor not found; apply fastest-K patch first")
    text = text.replace(old, new, 1)
    old = '''\
            print(
                "[LLIN_FASTEST_K] "
'''
    new = '''\
            eligible_selected = len(outputs)
            hard_gate_cap_exhausted = eligible_selected < fastest_k
            if hard_gate_cap_exhausted:
                for rejected_index, rejected_output in hard_gate_rejected:
                    if len(outputs) >= fastest_k:
                        break
                    outputs.append(rejected_output)
                    selected_indices.append(rejected_index)
            if len(outputs) != fastest_k:
                raise RuntimeError(
                    f"hard-gate attempt cap returned {len(outputs)} of {fastest_k} required placeholders"
                )
            print(
                "[LLIN_FASTEST_K] "
'''
    if old not in text:
        raise RuntimeError("fastest-K observability anchor not found")
    text = text.replace(old, new, 1)
    text = text.replace(
        'f"physical_aborts={physically_aborted} "',
        'f"physical_aborts={physically_aborted} "\n                f"hard_gate_eligible={eligible_selected} "\n                f"hard_gate_rejected={len(hard_gate_rejected)} "\n                f"hard_gate_cap_exhausted={hard_gate_cap_exhausted} "',
        1,
    )
    path.write_text(text, encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-loop", default="/verl/verl/experimental/agent_loop/agent_loop.py")
    args = parser.parse_args()
    print(f"{patch(Path(args.agent_loop))}: {args.agent_loop}")


if __name__ == "__main__":
    main()
