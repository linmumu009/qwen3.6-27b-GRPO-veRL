#!/usr/bin/env python3
"""Add exact fully-async queue, training-stage and prewarm telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path


TRAINER_MARKER = "LLIN_FULLY_ASYNC_STAGE_TIMING"
MAIN_MARKER = "LLIN_FULLY_ASYNC_PREWARM"


def _replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"expected patch anchor not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def patch_trainer(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if TRAINER_MARKER in text:
        return "already-patched"

    deserialize_old = """\
        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
"""
    deserialize_new = """\
        deserialize_start = time.time()
        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        deserialize_time = time.time() - deserialize_start

        # Assemble batch - now working directly with RolloutSample objects.
        assemble_start = time.time()
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)
        assemble_time = time.time() - assemble_start

        # LLIN_FULLY_ASYNC_STAGE_TIMING: these values are logged separately
        # from actor compute so queue starvation is never called training time.
        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        batch.meta_info["fully_async/deserialize_time"] = deserialize_time
        batch.meta_info["fully_async/assemble_time"] = assemble_time
        print(
            f"[LLIN_QUEUE_STAGE] step={self.global_steps} "
            f"wait_s={total_wait_time:.6f} "
            f"deserialize_s={deserialize_time:.6f} "
            f"assemble_s={assemble_time:.6f} "
            f"groups={len(queue_samples)} queue_after={queue_len}",
            flush=True,
        )
"""
    text = _replace_once(text, deserialize_old, deserialize_new, path)

    log_old = """\
        await self._fit_validate()
        self._fit_save_checkpoint()
"""
    log_new = """\
        queue_wait = float(batch.meta_info.get("fully_async/total_wait_time", 0.0))
        deserialize_time = float(batch.meta_info.get("fully_async/deserialize_time", 0.0))
        assemble_time = float(batch.meta_info.get("fully_async/assemble_time", 0.0))
        print(
            f"[LLIN_TRAIN_STAGE] step={self.global_steps} "
            f"queue_wait_s={queue_wait:.6f} "
            f"deserialize_s={deserialize_time:.6f} "
            f"assemble_s={assemble_time:.6f} "
            f"reward_s={float(self.timing_raw.get('reward', 0.0)):.6f} "
            f"old_log_prob_s={float(self.timing_raw.get('old_log_prob', 0.0)):.6f} "
            f"ref_log_prob_s={float(self.timing_raw.get(str(Role.RefPolicy), 0.0)):.6f} "
            f"adv_s={float(self.timing_raw.get('adv', 0.0)):.6f} "
            f"update_actor_s={float(self.timing_raw.get('update_actor', 0.0)):.6f} "
            f"step_s={float(self.timing_raw.get('step', 0.0)):.6f}",
            flush=True,
        )

        await self._fit_validate()
        self._fit_save_checkpoint()
"""
    path.write_text(_replace_once(text, log_old, log_new, path), encoding="utf-8")
    return "patched"


def patch_main(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MAIN_MARKER in text:
        return "already-patched"

    text = _replace_once(text, "import threading\n", "import threading\nimport time\n", path)
    run_old = """\
        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()
        trainer_future = self.components["trainer"].fit.remote()

        futures = [rollouter_future, trainer_future]
"""
    run_new = """\
        print("[ASYNC MAIN] Starting Rollouter and Trainer...")
        rollouter_future = self.components["rollouter"].fit.remote()

        # LLIN_FULLY_ASYNC_PREWARM: start the trainer only after a configured
        # number of complete GRPO groups are buffered. This removes the empty
        # queue cold start without weakening group atomicity.
        prewarm_groups = int(self.components["config"].async_training.get("prewarm_groups", 0))
        if prewarm_groups > 0:
            max_queue_size = ray.get(self.components["message_queue"].get_statistics.remote())["max_queue_size"]
            if prewarm_groups > max_queue_size:
                raise ValueError(
                    f"prewarm_groups ({prewarm_groups}) exceeds max_queue_size ({max_queue_size})"
                )
            prewarm_start = time.time()
            while True:
                stats = ray.get(self.components["message_queue"].get_statistics.remote())
                if int(stats["queue_size"]) >= prewarm_groups:
                    break
                done, _ = ray.wait([rollouter_future], timeout=0)
                if done:
                    ray.get(done[0])
                    raise RuntimeError("rollouter exited before prewarm completed")
                time.sleep(1)
            print(
                f"[LLIN_PREWARM] groups={stats['queue_size']} "
                f"queued_tokens={stats.get('queued_tokens', 0)} "
                f"wait_s={time.time() - prewarm_start:.6f}",
                flush=True,
            )

        trainer_future = self.components["trainer"].fit.remote()
        futures = [rollouter_future, trainer_future]
"""
    path.write_text(_replace_once(text, run_old, run_new, path), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer",
        default="/verl/verl/experimental/fully_async_policy/fully_async_trainer.py",
    )
    parser.add_argument(
        "--main",
        default="/verl/verl/experimental/fully_async_policy/fully_async_main.py",
    )
    args = parser.parse_args()
    print(f"{patch_trainer(Path(args.trainer))}: {args.trainer}")
    print(f"{patch_main(Path(args.main))}: {args.main}")


if __name__ == "__main__":
    main()
