"""Bounded, deterministic arithmetic tool; no eval, shell, network or database."""
import json
import os
from pathlib import Path
from uuid import uuid4

SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Compute integer addition or multiplication. Use this tool for arithmetic.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"}, "b": {"type": "integer"},
                "operation": {"type": "string", "enum": ["add", "multiply"]},
            },
            "required": ["a", "b", "operation"], "additionalProperties": False,
        },
    },
}


def calculate(parameters):
    if set(parameters) != {"a", "b", "operation"}:
        raise ValueError("Expected a, b and operation")
    a, b = parameters["a"], parameters["b"]
    if any(type(v) is not int or abs(v) > 1_000_000 for v in (a, b)):
        raise ValueError("Operands must be integers between -1000000 and 1000000")
    if parameters["operation"] == "add":
        return a + b
    if parameters["operation"] == "multiply":
        return a * b
    raise ValueError("Unsupported operation")


# Lazy module dependency keeps the pure calculator independently testable.
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse


class CalculatorTool(BaseTool):
    async def create(self, instance_id=None, **kwargs):
        return instance_id or uuid4().hex, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        try:
            value = calculate(parameters)
            response = str(value)
            success = True
        except (TypeError, ValueError) as exc:
            response, success = str(exc), False
        audit = os.environ.get("DELIVERY_TOOL_AUDIT")
        if audit:
            Path(audit).parent.mkdir(parents=True, exist_ok=True)
            record = json.dumps({"instance_id": instance_id, "parameters": parameters,
                                 "response": response, "success": success}) + "\n"
            fd = os.open(audit, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, record.encode())
            finally:
                os.close(fd)
        return ToolResponse(text=response), 0.0, {"calculator_success": float(success)}
