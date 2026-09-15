"""Numeric/exact-text outcome reward for engineering smoke tests, not a MATH evaluator."""
import math
import re
from fractions import Fraction


def explicit_answer(text):
    values = []
    for match in re.finditer(r'\\(?:boxed|fbox)\{', text):
        start = pos = match.end()
        depth = 1
        while pos < len(text) and depth:
            depth += (text[pos] == '{') - (text[pos] == '}')
            pos += 1
        if not depth:
            values.append((match.start(), text[start:pos-1]))
    if values:
        return max(values)[1].strip()
    tags = re.findall(r'<answer>(.*?)</answer>', text, re.S | re.I)
    if tags:
        return tags[-1].strip()
    cues = re.findall(r'(?:final\s+answer|answer|答案)\s*(?:is|为|[:：=])\s*([^\n]+)', text, re.I)
    return cues[-1].strip() if cues else None


def normalize(value):
    return str(value).strip().strip('$').replace('−', '-').replace(' ', '').rstrip('.;。')


def numeric(value):
    value = normalize(value).replace(',', '').replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    scale = 100 if value.endswith(('%', r'\%')) else 1
    value = value.removesuffix('%').removesuffix('\\') if scale == 100 else value
    match = re.fullmatch(r'([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}', value)
    try:
        if match:
            result = float(Fraction(int(match[2]), int(match[3]))) * (-1 if match[1] == '-' else 1)
        else:
            result = float(Fraction(value))
        return result / scale if math.isfinite(result) else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gold = ground_truth.get('golden_answer') if isinstance(ground_truth, dict) else ground_truth
    answer = explicit_answer(solution_str)
    if answer is None or gold is None or str(gold).strip() == '':
        return 0.0
    predicted, expected = numeric(answer), numeric(gold)
    if predicted is not None and expected is not None:
        return float(abs(predicted - expected) < 1e-3)
    return float(normalize(answer) == normalize(gold))
