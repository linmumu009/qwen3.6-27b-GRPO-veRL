#!/usr/bin/env python3
"""Add group-safe fastest-K oversampling with physical vLLM request aborts.

``rollout.n`` remains the GRPO comparison-group size. A separate
``async_training.oversample_candidates`` controls how many physical candidates
are launched for one prompt. The worker returns the first
``async_training.fastest_k`` successful candidates and aborts the rest without
resetting the prefix cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROLLOUTER_MARKER = "LLIN_FASTEST_K_OVERSAMPLE_BATCH"
AGENT_MARKER = "LLIN_FASTEST_K_QUORUM"
TOOL_MARKER = "LLIN_FASTEST_K_REQUEST_ID"
CLIENT_MARKER = "LLIN_FASTEST_K_PHYSICAL_ABORT"
GROUP_SCOPE_MARKER = "LLIN_FASTEST_K_PER_PROMPT_GROUP_V4"


def _replace_once(text: str, old: str, new: str, path: Path) -> str:
    if old not in text:
        raise RuntimeError(f"expected patch anchor not found in {path}: {old[:120]!r}")
    return text.replace(old, new, 1)


def _upgrade_optional_async_config(text: str) -> tuple[str, bool]:
    """Let the inactive Fastest-K patch coexist with standard PPO configs."""
    old = 'self.config.async_training.get('
    new = 'self.config.get("async_training", {}).get('
    upgraded = old in text
    return text.replace(old, new), upgraded


def _upgrade_fastest_k_group_scope(text: str, path: Path) -> tuple[str, bool]:
    """Do not treat an arbitrary agent-worker shard as one Fastest-K group."""

    if GROUP_SCOPE_MARKER in text:
        return text, False
    old = """\
        fastest_k = int(self.config.get("async_training", {}).get("fastest_k", 0))
        selected_indices = list(range(len(tasks)))
        if fastest_k > 0 and len(tasks) > fastest_k:
"""
    new = """\
        fastest_k = int(self.config.get("async_training", {}).get("fastest_k", 0))
        oversample_candidates = int(
            self.config.get("async_training", {}).get("oversample_candidates", 0)
        )
        selected_indices = list(range(len(tasks)))
        # LLIN_FASTEST_K_PER_PROMPT_GROUP_V4: fastest_k == oversample_candidates
        # means no physical oversampling was requested.  Agent-loop batches are
        # arbitrary worker shards, not prompt groups, so never trim such a shard.
        if oversample_candidates > fastest_k > 0 and len(tasks) > fastest_k:
"""
    if old not in text:
        raise RuntimeError(f"expected Fastest-K group-scope anchor not found in {path}")
    return text.replace(old, new, 1), True


def patch_rollouter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if ROLLOUTER_MARKER in text:
        text, upgraded = _upgrade_optional_async_config(text)
        if upgraded:
            path.write_text(text, encoding="utf-8")
            return "upgraded"
        return "already-patched"

    old = """\
            full_batch = prepare_single_generation_data(batch_dict, self.config)

            sample_id = f"sample_{epoch}_{self.global_steps}"
"""
    new = """\
            full_batch = prepare_single_generation_data(batch_dict, self.config)

            # LLIN_FASTEST_K_OVERSAMPLE_BATCH: rollout.n remains the atomic
            # GRPO group size consumed by the trainer. Only the physical
            # generation batch is expanded; the worker returns fastest_k.
            async_training = self.config.get("async_training", {})
            fastest_k = int(async_training.get("fastest_k", 0))
            oversample_candidates = int(async_training.get("oversample_candidates", 0))
            if oversample_candidates > 0:
                expected_group_size = int(self.config.actor_rollout_ref.rollout.n)
                if fastest_k != expected_group_size:
                    raise ValueError(
                        f"fastest_k ({fastest_k}) must equal rollout.n ({expected_group_size})"
                    )
                if oversample_candidates < fastest_k:
                    raise ValueError(
                        f"oversample_candidates ({oversample_candidates}) must be >= fastest_k ({fastest_k})"
                    )
                if oversample_candidates > len(full_batch):
                    full_batch = full_batch[:1].repeat(
                        repeat_times=oversample_candidates,
                        interleave=True,
                    )

            sample_id = f"sample_{epoch}_{self.global_steps}"
"""
    path.write_text(_replace_once(text, old, new, path), encoding="utf-8")
    return "patched"


def patch_tool_agent_loop(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if TOOL_MARKER in text:
        return "already-patched"

    old = "        request_id = uuid4().hex\n"
    new = """\
        # LLIN_FASTEST_K_REQUEST_ID: name each candidate so a discarded
        # trajectory can be mapped to its active physical vLLM request.
        request_id = str(kwargs.get("__llin_request_id") or uuid4().hex)
"""
    path.write_text(_replace_once(text, old, new, path), encoding="utf-8")
    return "patched"


def patch_llm_client(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if CLIENT_MARKER in text:
        return "already-patched"

    init_old = """\
        self.config = config
        self._load_balancer = load_balancer_handle
"""
    init_new = """\
        self.config = config
        self._load_balancer = load_balancer_handle
        # LLIN_FASTEST_K_PHYSICAL_ABORT: logical trajectory id -> active
        # physical requests. A multi-turn trajectory normally has one active.
        self._llin_active_requests: dict[str, dict[str, tuple[str, Any]]] = {}
        self._llin_active_requests_lock = asyncio.Lock()
"""
    text = _replace_once(text, init_old, init_new, path)

    generate_old = """\
            output: TokenOutput = await server.generate.remote(
                request_id=uuid4().hex,  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **priority_kwargs,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            return output
        finally:
            self._release_server(server_id)
"""
    generate_new = """\
            physical_request_id = uuid4().hex
            async with self._llin_active_requests_lock:
                self._llin_active_requests.setdefault(str(request_id), {})[physical_request_id] = (
                    server_id,
                    server,
                )
            try:
                output: TokenOutput = await server.generate.remote(
                    request_id=physical_request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=image_data,
                    video_data=video_data,
                    **multimodal_kwargs,
                    **priority_kwargs,
                    **kwargs,
                )
                global_steps = output.extra_fields.get("global_steps")
                output.extra_fields.setdefault("min_global_steps", global_steps)
                output.extra_fields.setdefault("max_global_steps", global_steps)
                return output
            finally:
                async with self._llin_active_requests_lock:
                    active = self._llin_active_requests.get(str(request_id))
                    if active is not None:
                        active.pop(physical_request_id, None)
                        if not active:
                            self._llin_active_requests.pop(str(request_id), None)
        finally:
            self._release_server(server_id)

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = False) -> dict[str, Any]:
        \"\"\"Abort every active physical request for one logical trajectory.\"\"\"
        async with self._llin_active_requests_lock:
            active = list(self._llin_active_requests.get(str(request_id), {}).items())
        if not active:
            return {"logical_request_id": str(request_id), "aborted_count": 0, "results": []}

        results = await asyncio.gather(
            *[
                server.abort_request.remote(physical_id, reset_prefix_cache)
                for physical_id, (_server_id, server) in active
            ],
            return_exceptions=True,
        )
        aborted_count = sum(
            isinstance(result, dict) and bool(result.get("aborted"))
            for result in results
        )
        return {
            "logical_request_id": str(request_id),
            "aborted_count": aborted_count,
            "physical_request_count": len(active),
            "reset_prefix_cache": bool(reset_prefix_cache),
            "results": results,
        }
"""
    path.write_text(_replace_once(text, generate_old, generate_new, path), encoding="utf-8")
    return "patched"


def patch_agent_loop(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if AGENT_MARKER in text:
        text, optional_upgraded = _upgrade_optional_async_config(text)
        text, scope_upgraded = _upgrade_fastest_k_group_scope(text, path)
        if optional_upgraded or scope_upgraded:
            path.write_text(text, encoding="utf-8")
            return "upgraded"
        return "already-patched"

    text = _replace_once(text, "import random\n", "import random\nimport time\n", path)

    block_old = """\
        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(
            outputs, input_non_tensor_batch=batch.non_tensor_batch, validate=batch.meta_info.get("validate", False)
        )
"""
    block_new = """\
        tasks = []
        llin_request_ids = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items() if k != "__do_sample__"}
            logical_request_id = f"llin-fastest-k-{uuid4().hex}"
            kwargs["__llin_request_id"] = logical_request_id
            llin_request_ids.append(logical_request_id)
            sample_sampling_params = dict(sampling_params)
            if not validate and per_sample_do_sample is not None and not bool(per_sample_do_sample[i]):
                apply_greedy_sampling_params(sample_sampling_params)
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sample_sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )

        # LLIN_FASTEST_K_QUORUM: return as soon as fastest_k successful
        # trajectories finish, then physically abort and cancel stragglers.
        fastest_k = int(self.config.get("async_training", {}).get("fastest_k", 0))
        oversample_candidates = int(
            self.config.get("async_training", {}).get("oversample_candidates", 0)
        )
        selected_indices = list(range(len(tasks)))
        # LLIN_FASTEST_K_PER_PROMPT_GROUP_V4: fastest_k == oversample_candidates
        # means no physical oversampling was requested.  Agent-loop batches are
        # arbitrary worker shards, not prompt groups, so never trim such a shard.
        if oversample_candidates > fastest_k > 0 and len(tasks) > fastest_k:
            started_at = time.monotonic()
            task_to_index = {task: index for index, task in enumerate(tasks)}
            pending = set(tasks)
            outputs = []
            selected_indices = []
            completed_but_discarded = 0
            try:
                while pending and len(outputs) < fastest_k:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if len(outputs) < fastest_k:
                            outputs.append(await task)
                            selected_indices.append(task_to_index[task])
                        else:
                            await task
                            completed_but_discarded += 1

                pending_indices = [task_to_index[task] for task in pending]
                abort_results = await asyncio.gather(
                    *[
                        self.llm_client.abort_request(
                            llin_request_ids[index],
                            reset_prefix_cache=False,
                        )
                        for index in pending_indices
                    ],
                    return_exceptions=True,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            except BaseException:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise

            physically_aborted = sum(
                result.get("aborted_count", 0)
                for result in abort_results
                if isinstance(result, dict)
            )
            print(
                "[LLIN_FASTEST_K] "
                f"candidates={len(tasks)} selected={len(outputs)} "
                f"discarded={len(tasks) - len(outputs)} "
                f"completed_discarded={completed_but_discarded} "
                f"physical_aborts={physically_aborted} "
                f"quorum_s={time.monotonic() - started_at:.6f} "
                "reset_prefix_cache=False"
            )
        else:
            outputs = await asyncio.gather(*tasks)

        selected_non_tensor_batch = {
            key: value[selected_indices]
            for key, value in batch.non_tensor_batch.items()
        }
        output = self._postprocess(
            outputs,
            input_non_tensor_batch=selected_non_tensor_batch,
            validate=batch.meta_info.get("validate", False),
        )
"""
    path.write_text(_replace_once(text, block_old, block_new, path), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouter",
        default="/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py",
    )
    parser.add_argument(
        "--agent-loop",
        default="/verl/verl/experimental/agent_loop/agent_loop.py",
    )
    parser.add_argument(
        "--tool-agent-loop",
        default="/verl/verl/experimental/agent_loop/tool_agent_loop.py",
    )
    parser.add_argument(
        "--llm-server",
        default="/verl/verl/workers/rollout/llm_server.py",
    )
    args = parser.parse_args()
    print(f"{patch_rollouter(Path(args.rollouter))}: {args.rollouter}")
    print(f"{patch_agent_loop(Path(args.agent_loop))}: {args.agent_loop}")
    print(f"{patch_tool_agent_loop(Path(args.tool_agent_loop))}: {args.tool_agent_loop}")
    print(f"{patch_llm_client(Path(args.llm_server))}: {args.llm_server}")


if __name__ == "__main__":
    main()
