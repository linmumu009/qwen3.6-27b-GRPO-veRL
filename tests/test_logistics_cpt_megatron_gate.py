from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cpt_dataset_is_raw_text_all_token_with_eos_and_wraparound_mask():
    source = (ROOT / "scripts" / "qwen36_causal_lm_dataset.py").read_text(encoding="utf-8")

    assert 'tokenizer(text, add_special_tokens=False)' in source
    assert "input_id_list.append(int(eos_token_id))" in source
    assert "loss_mask[0] = 0" in source
    assert "apply_chat_template" not in source
    assert "class Qwen36CausalLMDataset" in source
    assert "del processor" in source


def test_cpt_one_step_gate_reuses_step120_megatron_and_freezes_mtp():
    script = (ROOT / "scripts" / "run_logistics_cpt_megatron_one_step_gate.sh").read_text(encoding="utf-8")

    assert "llin-step120-opensource-20260825-02/checkpoints/global_step_120" in script
    assert "checkpoint_initialization=model_only_dist_ckpt" in script
    assert "Qwen36CausalLMDataset" in script
    assert "trainer.total_training_steps=1" in script
    assert "'checkpoint.save_contents=[extra]'" in script
    assert "promotion_allowed=false" in script
    assert "model.mtp.enable=false" in script
    assert "model.mtp.enable_train=false" in script
    assert "chat_template_applied=false" in script
    assert "record_selection=single_longest_token_block" in script
    assert "+data.sort_by_token_count_desc=true" in script
    assert "data.train_max_samples=1" in script
    assert "optimizer_placement=device_side_matching_step120" in script
    assert "+optim.override_optimizer_config.optimizer_cpu_offload=false" in script
    assert "engine.optimizer_offload=false" in script
    assert "TRANSFORMERS_VERBOSITY=error" in script
    assert '--log-dir="${OUTPUT_DIR}/torchrun_logs" --redirects=3 --tee=0' in script


def test_cpt_private_converter_and_gate_do_not_emit_source_text_to_safe_summary():
    converter = (ROOT / "scripts" / "convert_logistics_cpt_jsonl_to_parquet.py").read_text(encoding="utf-8")
    gate = (ROOT / "scripts" / "check_logistics_cpt_dataset.py").read_text(encoding="utf-8")

    assert "os.chmod(output_path, 0o600)" in converter
    assert '"source_content_included": False' in gate
    assert '"chat_template_applied": False' in gate


def test_cpt_formal_pilot_is_one_exposure_model_only_from_step120_and_not_promotable():
    script = (ROOT / "scripts" / "run_logistics_cpt_book_one_epoch.sh").read_text(encoding="utf-8")

    assert "purpose=single_book_cpt_pilot" in script
    assert "promotion_allowed=false" in script
    assert "planned_exposures=1" in script
    assert "expected_sequence_tokens_with_eos=336702" in script
    assert 'TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"' in script
    assert 'TOTAL_STEPS="${TOTAL_STEPS:-29}"' in script
    assert 'LEARNING_RATE="${LEARNING_RATE:-5e-7}"' in script
    assert "checkpoint_initialization=model_only_dist_ckpt" in script
    assert "model.mtp.enable=false" in script
    assert "model.mtp.enable_train=false" in script
    assert "optimizer_checkpoint_saved=false" in script
    assert "'checkpoint.save_contents=[model,extra]'" in script
    assert "trainer.total_epochs=1" in script
