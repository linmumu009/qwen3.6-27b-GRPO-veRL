"""Strict outcome reward for the Step-120-open-source recovery run.

The reward deliberately has no process or formatting component.  A wrong
answer always receives zero, even when it is well formatted, so a uniformly
wrong GRPO group cannot create a formatting-only learning signal.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_CUE_RE = re.compile(
    r"(?:final\s+answer|answer|答案)\s*(?:is|为|[:：=])?\s*(.+)",
    re.IGNORECASE,
)
_CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", re.IGNORECASE)
_NUMBER_RE = re.compile(
    r"[-+−]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?"
)


def _extract_braced(text: str, marker: str) -> list[str]:
    values: list[str] = []
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            break
        cursor = index + len(marker)
        depth = 1
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            values.append(text[index + len(marker) : cursor - 1])
        start = max(cursor, index + len(marker))
    return values


def extract_explicit_answer(text: str) -> tuple[str, bool]:
    """Return the last explicit final answer and whether one was present."""

    if not text:
        return "", False
    boxed = _extract_braced(text, r"\boxed{") + _extract_braced(text, r"\fbox{")
    if boxed:
        return boxed[-1].strip(), True
    tagged = _ANSWER_TAG_RE.findall(text)
    if tagged:
        return tagged[-1].strip(), True
    cues = _FINAL_CUE_RE.findall(text[-1024:])
    if cues:
        return cues[-1].strip().splitlines()[0].strip(), True
    return "", False


def _strip_math_wrappers(value: str) -> str:
    value = value.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if value.startswith(left) and value.endswith(right) and len(value) >= len(left) + len(right):
            value = value[len(left) : -len(right)].strip()
    return value


def normalize_math_text(value: str) -> str:
    value = _strip_math_wrappers(value)
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\cup", "∪").replace(r"\infty", "∞")
    value = value.replace(r"\,", "").replace(r"\!", "")
    value = value.replace(" ", "").replace("\n", "")
    return value.strip(".;。")


def _parse_number(value: str) -> float | None:
    value = normalize_math_text(value).replace(",", "")
    percent = value.endswith(r"\%") or value.endswith("%")
    if value.endswith(r"\%"):
        value = value[:-2]
    elif value.endswith("%"):
        value = value[:-1]
    frac = re.fullmatch(r"[-+]?\s*\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", value)
    try:
        if frac:
            result = float(Fraction(int(frac.group(1)), int(frac.group(2))))
        elif re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", value):
            result = float(Fraction(value))
        else:
            result = float(value)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return result / 100.0 if percent else result


def _read_brace_group(value: str, start: int) -> tuple[str, int]:
    if start >= len(value) or value[start] != "{":
        raise ValueError("expected braced LaTeX argument")
    depth = 1
    cursor = start + 1
    while cursor < len(value) and depth:
        if value[cursor] == "{":
            depth += 1
        elif value[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        raise ValueError("unbalanced LaTeX braces")
    return value[start + 1 : cursor - 1], cursor


def _read_latex_argument(value: str, start: int) -> tuple[str, int]:
    while start < len(value) and value[start].isspace():
        start += 1
    if start >= len(value):
        raise ValueError("missing LaTeX argument")
    if value[start] == "{":
        return _read_brace_group(value, start)
    if value[start] in "+-" and start + 1 < len(value):
        return value[start : start + 2], start + 2
    return value[start], start + 1


def _latex_to_plain(value: str) -> str:
    value = _strip_math_wrappers(value)
    output: list[str] = []
    cursor = 0
    while cursor < len(value):
        if value.startswith(r"\frac", cursor):
            numerator, after_numerator = _read_latex_argument(value, cursor + 5)
            denominator, after_denominator = _read_latex_argument(value, after_numerator)
            output.append(f"(({_latex_to_plain(numerator)})/({_latex_to_plain(denominator)}))")
            cursor = after_denominator
            continue
        if value.startswith(r"\sqrt", cursor):
            radicand, after_radicand = _read_latex_argument(value, cursor + 5)
            output.append(f"sqrt({_latex_to_plain(radicand)})")
            cursor = after_radicand
            continue
        output.append(value[cursor])
        cursor += 1
    plain = "".join(output)
    plain = plain.replace(r"\cdot", "*").replace(r"\times", "*")
    replacements = {
        r"\pi": "pi", r"\theta": "theta", r"\omega": "omega",
        r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma", r"\delta": "delta",
        r"\lambda": "lambda_", r"\phi": "phi", r"\varphi": "varphi",
        r"\rho": "rho", r"\mu": "mu", r"\sigma": "sigma",
        r"\varepsilon": "varepsilon", r"\epsilon": "epsilon",
        r"\sin": "sin", r"\cos": "cos", r"\tan": "tan", r"\csc": "csc",
        r"\arctan": "atan", r"\log": "log", r"\ln": "log", r"\exp": "exp",
        r"\max": "max_",
    }
    for source, target in replacements.items():
        plain = plain.replace(source, target)
    plain = plain.replace(r"\left", "").replace(r"\right", "")
    plain = re.sub(r"_\{([A-Za-z0-9_]+)\}", r"_\1", plain)
    plain = plain.replace("{", "(").replace("}", ")")
    plain = plain.replace(r"\,", "").replace(r"\!", "")
    return plain.strip()


def _sympy_value(value: str) -> Any:
    if not value or len(value) > 512:
        raise ValueError("symbolic answer is empty or too long")
    # Reject prose and control sequences outside a deliberately small math set.
    commands = re.findall(r"\\([A-Za-z]+)", value)
    allowed = {
        "frac", "sqrt", "pi", "cdot", "times", "sin", "cos", "tan",
        "log", "ln", "exp", "infty", "pm", "theta", "omega", "alpha",
        "beta", "gamma", "delta", "lambda", "phi", "varphi", "rho", "mu", "sigma",
        "varepsilon", "epsilon", "max", "left", "right", "arctan", "csc", "cup",
    }
    if any(command not in allowed for command in commands):
        raise ValueError("unsupported symbolic command")
    plain = _latex_to_plain(value)
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,^=\s]+", plain):
        raise ValueError("unsupported character in symbolic answer")
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", plain))
    if len(names) > 64 or names & {"Symbol", "Integer", "Float", "Rational", "__import__"}:
        raise ValueError("unsafe or overly complex symbolic answer")

    from sympy import E, Float, Integer, Rational, Symbol, atan, cos, csc, exp, log, pi, sin, sqrt, tan
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    known: dict[str, Any] = {
        "sqrt": sqrt, "sin": sin, "cos": cos, "tan": tan, "csc": csc,
        "atan": atan, "log": log,
        "exp": exp, "pi": pi, "E": E,
    }
    local_dict = {name: known.get(name, Symbol(name)) for name in names}
    global_dict = {
        "__builtins__": {}, "Symbol": Symbol, "Integer": Integer,
        "Float": Float, "Rational": Rational,
    }
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)

    def parse_one(expression: str) -> Any:
        return parse_expr(
            expression,
            local_dict=local_dict,
            global_dict=global_dict,
            transformations=transformations,
            evaluate=True,
        )

    if plain.count("=") == 1:
        left, right = plain.split("=", 1)
        return parse_one(left) - parse_one(right)
    if "=" in plain:
        raise ValueError("multiple equations are unsupported")
    return parse_one(plain)


def _answer_variants(value: str) -> list[str]:
    variants = [value]
    if value.count("=") == 1:
        right = value.split("=", 1)[1].strip()
        if right:
            variants.append(right)
    return variants


def math_equal(prediction: str, golden: str) -> bool:
    prediction_variants = _answer_variants(prediction)
    golden_variants = _answer_variants(golden)
    for pred_value in prediction_variants:
        for gold_value in golden_variants:
            pred_norm = normalize_math_text(pred_value)
            gold_norm = normalize_math_text(gold_value)
            if not pred_norm or not gold_norm:
                continue
            if pred_norm.casefold() == gold_norm.casefold():
                return True
            pred_number = _parse_number(pred_norm)
            gold_number = _parse_number(gold_norm)
            if pred_number is not None and gold_number is not None:
                if math.isclose(pred_number, gold_number, rel_tol=1e-6, abs_tol=1e-6):
                    return True
                continue
            try:
                from sympy import simplify

                pred_expr = _sympy_value(pred_value)
                gold_expr = _sympy_value(gold_value)
                if bool(simplify(pred_expr - gold_expr) == 0):
                    return True
            except Exception:
                continue
    return False


def _choice_answer(value: str) -> str:
    explicit, present = extract_explicit_answer(value)
    target = explicit if present else value[-128:]
    matches = _CHOICE_RE.findall(target)
    return matches[-1].upper() if matches else ""


def _numeric_answer(value: str) -> str:
    explicit, present = extract_explicit_answer(value)
    if present:
        return explicit
    matches = _NUMBER_RE.findall(value[-512:])
    return matches[-1] if matches else ""


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
) -> dict[str, float]:
    """Score one response with a strict binary final-answer reward."""

    del data_source, extra_info
    answer_type = str(ground_truth.get("answer_type") or "math").casefold()
    golden = str(ground_truth.get("answer") or "")
    explicit_answer, explicit_present = extract_explicit_answer(solution_str)

    if answer_type == "choice":
        prediction = _choice_answer(solution_str)
        correct = bool(prediction and prediction == _choice_answer(golden))
    elif answer_type == "numeric":
        prediction = _numeric_answer(solution_str)
        correct = math_equal(prediction, golden)
    else:
        prediction = explicit_answer if explicit_present else ""
        correct = math_equal(prediction, golden)

    score = float(correct)
    return {
        "score": score,
        "correct": score,
        "explicit_final": float(explicit_present),
        "verifier_valid": float(bool(golden)),
    }
