from scripts.generate_book_knowledge_probes import (
    extract_json_object,
    has_shared_ngram,
    rotate_options,
    validate_probe,
)
from scripts.run_vllm_logistics_mcq import repeat_path


def test_probe_parser_and_overlap_gate() -> None:
    source = "Batch picking reduces warehouse travel when several compatible orders are collected together."
    copied = {
        "question": "Why are several compatible orders are collected together?",
        "options": ["It reduces warehouse travel for operators", "b", "c", "d"],
        "answer": 0,
    }
    assert has_shared_ngram(source, copied["question"], 4)
    assert validate_probe(copied, source)[1] == "question_4gram_overlap"

    paraphrased = {
        "question": "What operational benefit can arise from grouping suitable customer requests?",
        "options": [
            "Workers spend less time moving through storage aisles",
            "Every shipment automatically receives a lower freight rate",
            "Cycle counting becomes unnecessary for stored inventory",
            "All replenishment decisions can be postponed indefinitely",
        ],
        "answer": 0,
    }
    assert validate_probe(paraphrased, source)[0] == paraphrased


def test_extract_and_rotate_preserve_correct_option() -> None:
    value = extract_json_object('prefix {"question":"q","options":["a","b","c","d"],"answer":2} suffix')
    assert value is not None
    rotated = rotate_options(value, 9)
    assert rotated["options"][rotated["answer"]] == "c"


def test_repeat_paths_keep_first_result_at_requested_location() -> None:
    from pathlib import Path

    path = Path("result.safe.json")
    assert repeat_path(path, 0) == path
    assert repeat_path(path, 1) == Path("result.safe.repeat2.json")
