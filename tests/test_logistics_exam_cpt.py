import json
from pathlib import Path

import pytest

from scripts.finalize_logistics_exam_rewrite_tail import IDENTITY_MODEL, identity_row
from scripts.prepare_logistics_exam_cpt import (
    build_corpus,
    load_rewrites,
    normalize_space,
    pack_documents,
    render_training_document,
    text_hash,
)
from scripts.rewrite_logistics_exam_stems import (
    deterministic_validation,
    item_validation,
    normalize_rewritten_form,
    verify_payload,
)
from scripts.summarize_logistics_exam_cpt_subsets import paired_summary, subset_summary


class FakeTokenizer:
    eos_token = "<eos>"

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


def source_row(**updates):
    row = {
        "dataset": "benchmark",
        "source_id": "1",
        "question_type": "single_choice",
        "question": "Which mode is fastest?",
        "options": ["Air freight", "Sea freight", "Rail freight"],
        "expected": [0],
        "item_hash": "item-1",
    }
    row.update(updates)
    return row


def test_direct_render_uses_gold_text_without_label_or_distractors():
    text = render_training_document(source_row(), "Which mode is fastest?")

    assert text == "Question: Which mode is fastest?\nCorrect answer: Air freight"
    assert "Sea freight" not in text
    assert "Rail freight" not in text
    assert "A." not in text


def test_multiple_choice_preserves_all_gold_text_in_source_order():
    row = source_row(expected=[2, 0], question_type="multiple_choices")

    text = render_training_document(row, row["question"])

    assert text.endswith("Correct answers: Air freight; Rail freight")


@pytest.mark.parametrize(
    ("gold", "verdict"),
    [("true", "This statement is true."), ("false", "This statement is false.")],
)
def test_historical_true_false_preserves_statement_and_appends_verdict(gold: str, verdict: str):
    row = source_row(
        question_type="true_or_false",
        question="Inventory is free.",
        options=["true", "false"],
        expected=[0 if gold == "true" else 1],
    )

    text = render_training_document(row, row["question"])

    assert text == f'Statement: "Inventory is free."\n{verdict}'


def test_pack_documents_reaches_exact_block_count_and_inserts_eos():
    tokenizer = FakeTokenizer()
    documents = [f"document {index} has tokens" for index in range(12)]

    groups = pack_documents(documents, tokenizer, block_count=4, max_content_tokens=20)

    assert len(groups) == 4
    assert sorted(item for group in groups for item in group) == sorted(documents)
    assert all(len(tokenizer.encode("<eos>\n\n".join(group))) <= 20 for group in groups)


def test_rewritten_corpus_keeps_gold_text_and_changes_only_question():
    tokenizer = FakeTokenizer()
    rows = [source_row()]
    rewrites = {"item-1": "What is the quickest transport mode?"}

    items, blocks, summary = build_corpus(
        rows,
        tokenizer,
        arm="rewritten",
        rewrites=rewrites,
        block_count=1,
        max_length=100,
    )

    assert "What is the quickest transport mode?" in blocks[0]["text"]
    assert "Air freight" in blocks[0]["text"]
    assert "Sea freight" not in blocks[0]["text"]
    assert items[0]["gold_text_sha256"] == [text_hash("Air freight")]
    assert summary["changed_question_items"] == 1
    assert summary["distractors_rendered"] == 0


def test_load_rewrites_requires_exact_hash_and_semantic_gate(tmp_path: Path):
    rows = [source_row()]
    path = tmp_path / "rewrites.jsonl"
    path.write_text(
        json.dumps(
            {
                "item_hash": "item-1",
                "original_question_sha256": text_hash(normalize_space(rows[0]["question"])),
                "rewritten_question": "What is the quickest transport mode?",
                "semantic_validation_passed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_rewrites(path, rows) == {"item-1": "What is the quickest transport mode?"}


def test_deterministic_rewrite_gate_rejects_number_and_polarity_changes():
    assert "unchanged" in deterministic_validation("Which is not valid in 2025?", "Which is not valid in 2025?")
    assert "numbers_changed" in deterministic_validation("What changed in 2025?", "What changed in 2026?")
    assert "logic_family_changed:negative" in deterministic_validation(
        "Which option is not valid?", "Which option is valid?"
    )


def test_semantic_verifier_payload_is_fail_closed():
    assert verify_payload(
        {"equivalent": True, "same_correct_answer": True, "issues": []}
    ) == (True, [])
    assert verify_payload(
        {"equivalent": True, "same_correct_answer": False, "issues": []}
    )[0] is False


def test_item_gate_rejects_appended_choice_text_and_non_question():
    row = source_row()

    assert "choice_text_added" in item_validation(row, "Which mode is quickest, such as Sea freight?")
    assert "not_a_question" in item_validation(row, "Identify the quickest transport mode")
    assert normalize_rewritten_form(row, "Identify the quickest transport mode.").endswith("?")


def test_rewrite_tail_identity_fallback_is_explicit_and_hash_bound():
    row = source_row()

    fallback = identity_row(row)

    assert fallback["generation_model"] == IDENTITY_MODEL
    assert fallback["identity_fallback"] is True
    assert fallback["semantic_validation_method"] == "identity_by_construction"
    assert fallback["rewritten_question"] == row["question"]
    assert fallback["original_question_sha256"] == text_hash(row["question"])
    assert fallback["rewritten_question_sha256"] == text_hash(row["question"])


def test_subset_summary_keeps_pairing_and_reports_exact_net_change():
    baseline = {
        "a": {"correct": False},
        "b": {"correct": True},
        "c": {"correct": False},
    }
    candidate = {
        "a": {"correct": True},
        "b": {"correct": False},
        "c": {"correct": True},
    }

    paired = paired_summary([baseline[key] for key in ("a", "b", "c")], [candidate[key] for key in ("a", "b", "c")])
    summary = subset_summary(["base", "candidate"], [baseline, candidate], ["a", "b", "c"])

    assert paired["improved_0_to_1"] == 2
    assert paired["regressed_1_to_0"] == 1
    assert paired["net_correct"] == 1
    assert summary["models"]["base"]["correct"] == 1
    assert summary["models"]["candidate"]["correct"] == 2
    assert summary["pairwise"][0]["net_correct"] == 1


def test_exam_cpt_training_arms_are_independent_equal_step_contamination_probes():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_logistics_exam_cpt_arm.sh").read_text(encoding="utf-8")

    assert "checkpoint_initialization=model_only_dist_ckpt" in script
    assert "purpose=controlled_benchmark_contamination_${ARM}_gold_text_cpt" in script
    assert "promotion_allowed=false" in script
    assert "delivery_metric_valid=false" in script
    assert "distractors_rendered=0" in script
    assert "records_per_exposure=${RECORDS_PER_EXPOSURE}" in script
    assert 'TOTAL_EXPOSURES="${TOTAL_EXPOSURES:-4}"' in script
    assert 'TOTAL_STEPS="${TOTAL_STEPS:-64}"' in script
    assert 'trainer.save_freq=${TOTAL_STEPS}' in script
    assert "'checkpoint.load_contents=[]'" in script
    assert "'checkpoint.save_contents=[model,extra]'" in script
