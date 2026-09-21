"""Exact final-line reward shared by arithmetic GRPO and agentic GRPO."""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, extra_info, kwargs
    answer = ground_truth.get("answer") if isinstance(ground_truth, dict) else ground_truth
    # Ignore model reasoning; only an explicit final answer is rewarded.
    if "<think>" in solution_str and "</think>" not in solution_str:
        return 0.0
    final = solution_str.rsplit("</think>", 1)[-1].strip()
    match = re.search(r"(?:答案|Answer)\s*[:：]\s*(-?\d+)\s*$", final)
    return float(match is not None and int(match.group(1)) == int(answer))
