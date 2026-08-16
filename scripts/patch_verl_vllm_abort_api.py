#!/usr/bin/env python3
"""Use vLLM's public external-request abort API on vLLM 0.18."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "LLIN_VLLM_PUBLIC_ABORT_V5"
PREVIOUS_MARKER = "LLIN_VLLM_PUBLIC_ABORT_V4"


def patch_server(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already-patched"

    if PREVIOUS_MARKER in text:
        old_registration = '''\
        try:
            # vLLM 0.18 request_states is keyed by internal IDs. The public
            # AsyncLLM.abort API resolves the external ID through
            # output_processor.external_req_ids before touching EngineCore.
            external_req_ids = getattr(self.engine.output_processor, "external_req_ids", {})
            registered = bool(external_req_ids.get(request_id))
            if not registered:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            await self.engine.abort(request_id)
'''
        new_registration = '''\
        partial_response_tokens = 0
        try:
            # vLLM 0.18 maps external IDs to one or more internal request
            # states. Snapshot detokenized output lengths before the public
            # abort removes those states, without copying token content.
            output_processor = self.engine.output_processor
            external_req_ids = getattr(output_processor, "external_req_ids", {})
            internal_id_value = external_req_ids.get(request_id, [])
            internal_ids = (
                [internal_id_value]
                if isinstance(internal_id_value, str)
                else list(internal_id_value or [])
            )
            if not internal_ids:
                return {
                    "aborted": False,
                    "partial_response_tokens": 0,
                    "error": f"Request {request_id} not found",
                }
            request_states = getattr(output_processor, "request_states", {})
            for internal_id in internal_ids:
                state = request_states.get(internal_id)
                detokenizer = getattr(state, "detokenizer", None)
                if detokenizer is None:
                    continue
                counter = getattr(detokenizer, "num_output_tokens", None)
                if callable(counter):
                    partial_response_tokens += max(0, int(counter()))
                elif counter is not None:
                    partial_response_tokens += max(0, int(counter))
                else:
                    partial_response_tokens += len(
                        getattr(detokenizer, "output_token_ids", []) or []
                    )

            await self.engine.abort(request_id)
'''
        if old_registration not in text:
            raise RuntimeError(f"expected V4 registration anchor not found in {path}")
        text = text.replace(old_registration, new_registration, 1)
        text = text.replace(
            'return {"aborted": True, "request_id": request_id}',
            'return {\n'
            '                "aborted": True,\n'
            '                "request_id": request_id,\n'
            '                "partial_response_tokens": partial_response_tokens,\n'
            '            }',
            1,
        )
        text = text.replace(
            'return {"aborted": False, "request_id": request_id, "error": str(e)}',
            'return {\n'
            '                "aborted": False,\n'
            '                "request_id": request_id,\n'
            '                "partial_response_tokens": partial_response_tokens,\n'
            '                "error": str(e),\n'
            '            }',
            1,
        )
        text = text.replace(PREVIOUS_MARKER, MARKER, 1)
        path.write_text(text, encoding="utf-8")
        return "patched"

    old = '''\
    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}
'''
    new = '''\
    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """LLIN_VLLM_PUBLIC_ABORT_V5: abort and report partial token count."""
        partial_response_tokens = 0
        try:
            # vLLM 0.18 maps external IDs to one or more internal request
            # states. Snapshot detokenized output lengths before the public
            # abort removes those states, without copying token content.
            output_processor = self.engine.output_processor
            external_req_ids = getattr(output_processor, "external_req_ids", {})
            internal_id_value = external_req_ids.get(request_id, [])
            internal_ids = (
                [internal_id_value]
                if isinstance(internal_id_value, str)
                else list(internal_id_value or [])
            )
            if not internal_ids:
                return {
                    "aborted": False,
                    "partial_response_tokens": 0,
                    "error": f"Request {request_id} not found",
                }
            request_states = getattr(output_processor, "request_states", {})
            for internal_id in internal_ids:
                state = request_states.get(internal_id)
                detokenizer = getattr(state, "detokenizer", None)
                if detokenizer is None:
                    continue
                counter = getattr(detokenizer, "num_output_tokens", None)
                if callable(counter):
                    partial_response_tokens += max(0, int(counter()))
                elif counter is not None:
                    partial_response_tokens += max(0, int(counter))
                else:
                    partial_response_tokens += len(
                        getattr(detokenizer, "output_token_ids", []) or []
                    )

            await self.engine.abort(request_id)

            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request through public AsyncLLM API: {request_id}")
            return {
                "aborted": True,
                "request_id": request_id,
                "partial_response_tokens": partial_response_tokens,
            }

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {
                "aborted": False,
                "request_id": request_id,
                "partial_response_tokens": partial_response_tokens,
                "error": str(e),
            }
'''
    if old not in text:
        raise RuntimeError(f"expected abort_request anchor not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py",
    )
    args = parser.parse_args()
    print(f"{patch_server(Path(args.target))}: {args.target}")


if __name__ == "__main__":
    main()
