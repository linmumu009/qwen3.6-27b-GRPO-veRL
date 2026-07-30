#!/usr/bin/env python3
"""Make veRL fully-async queue group-atomic and token-budget bounded."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_GROUP_TOKEN_QUEUE"

PUT_START = "    async def put_sample(self, sample: Any) -> bool:\n"
PUT_END = "    async def get_sample(self) -> Any | None:\n"
PUT_NEW = """\
    async def put_sample(self, sample: Any, token_count: int = 0) -> bool:
        \"\"\"Put one complete GRPO group into the bounded queue.

        LLIN_GROUP_TOKEN_QUEUE: producers wait for both the group-count and
        token budgets instead of evicting an older group. A single oversized
        group is admitted only when the queue is empty, which guarantees
        progress without ever splitting a GRPO comparison group.
        \"\"\"
        token_count = max(0, int(token_count))
        async with self._lock:
            while self.running and (
                len(self.queue) >= self.max_queue_size
                or (
                    self.max_queue_tokens > 0
                    and len(self.queue) > 0
                    and self.queued_tokens + token_count > self.max_queue_tokens
                )
            ):
                await self._producer_condition.wait()
            if not self.running:
                return False

            self.queue.append((sample, token_count))
            self.queued_tokens += token_count
            self.total_produced += 1
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(
                    f"MessageQueue stats: produced={self.total_produced}, "
                    f"queue_size={len(self.queue)}, queued_tokens={self.queued_tokens}"
                )
            return True

"""


def replace_function(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))
    if start_index < 0 or end_index < 0:
        raise RuntimeError(f"expected function block not found: {start.strip()}")
    return text[:start_index] + replacement + text[end_index:]


def patch_message_queue(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"

    text = text.replace(
        "        self.queue = deque(maxlen=self.max_queue_size)\n",
        (
            "        self.queue = deque()\n"
            '        self.max_queue_tokens = int(config.async_training.get("max_queue_tokens", 0))\n'
            "        self.queued_tokens = 0\n"
        ),
        1,
    )
    text = text.replace(
        "        self._consumer_condition = asyncio.Condition(self._lock)\n",
        (
            "        self._consumer_condition = asyncio.Condition(self._lock)\n"
            "        self._producer_condition = asyncio.Condition(self._lock)\n"
        ),
        1,
    )
    text = replace_function(text, PUT_START, PUT_END, PUT_NEW)
    text = text.replace(
        "            data = self.queue.popleft()\n"
        "            self.total_consumed += 1\n"
        "            return data, len(self.queue)\n",
        (
            "            data, token_count = self.queue.popleft()\n"
            "            self.queued_tokens -= token_count\n"
            "            self.total_consumed += 1\n"
            "            self._producer_condition.notify_all()\n"
            "            return data, len(self.queue)\n"
        ),
        1,
    )
    text = text.replace(
        '                "max_queue_size": self.max_queue_size,\n',
        (
            '                "max_queue_size": self.max_queue_size,\n'
            '                "queued_tokens": self.queued_tokens,\n'
            '                "max_queue_tokens": self.max_queue_tokens,\n'
        ),
        1,
    )
    text = text.replace(
        "            self.queue.clear()\n"
        '            logger.info(f"Cleared {cleared_count} samples from queue")\n',
        (
            "            self.queue.clear()\n"
            "            self.queued_tokens = 0\n"
            "            self._producer_condition.notify_all()\n"
            '            logger.info(f"Cleared {cleared_count} samples from queue")\n'
        ),
        1,
    )
    text = text.replace(
        "            self._consumer_condition.notify_all()\n"
        '        logger.info("MessageQueue shutdown")\n',
        (
            "            self._consumer_condition.notify_all()\n"
            "            self._producer_condition.notify_all()\n"
            '        logger.info("MessageQueue shutdown")\n'
        ),
        1,
    )
    text = text.replace(
        "                sample = list(self.queue)[0]\n",
        "                sample = list(self.queue)[0][0]\n",
        1,
    )
    text = text.replace(
        "    async def put_sample(self, sample: Any) -> bool:\n"
        '        """Put batch into queue (async)"""\n'
        "        future = self.queue_actor.put_sample.remote(sample)\n",
        (
            "    async def put_sample(self, sample: Any, token_count: int = 0) -> bool:\n"
            '        """Put one complete group into the queue (async)."""\n'
            "        future = self.queue_actor.put_sample.remote(sample, token_count)\n"
        ),
        1,
    )
    if MARKER not in text or "max_queue_tokens" not in text:
        raise RuntimeError(f"failed to patch bounded queue in {path}")
    path.write_text(text, encoding="utf-8")
    return "patched"


def patch_rollouter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    marker = "LLIN_GROUP_TOKEN_COUNT"
    if marker in text:
        return "already-patched"
    old = """\
        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )
"""
    new = """\
        # LLIN_GROUP_TOKEN_COUNT: full_batch contains all rollout.n responses
        # for one prompt, so this is an indivisible GRPO group and its exact
        # post-generation token count can safely drive queue backpressure.
        token_count = int(rollout_sample.full_batch.batch["attention_mask"].sum().item())
        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
            token_count=token_count,
        )
"""
    if old not in text:
        raise RuntimeError(f"expected rollouter queue call not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--message-queue",
        default="/verl/verl/experimental/fully_async_policy/message_queue.py",
    )
    parser.add_argument(
        "--rollouter",
        default="/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py",
    )
    args = parser.parse_args()
    print(f"{patch_message_queue(Path(args.message_queue))}: {args.message_queue}")
    print(f"{patch_rollouter(Path(args.rollouter))}: {args.rollouter}")


if __name__ == "__main__":
    main()
