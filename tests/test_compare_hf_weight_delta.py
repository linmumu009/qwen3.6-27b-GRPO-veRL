from scripts.compare_hf_weight_delta import (
    add_stats,
    new_accumulator,
    summarize_accumulator,
    tensor_group,
)


def test_tensor_group_recognizes_language_and_frozen_components() -> None:
    assert tensor_group("model.language_model.layers.7.mlp.up_proj.weight") == ("language_layer_07", "mlp")
    assert tensor_group("model.visual.blocks.2.attn.qkv.weight")[0] == "visual_encoder"
    assert tensor_group("mtp.pre_fc_norm_embedding.weight")[0] == "mtp"


def test_accumulator_summary_is_element_weighted() -> None:
    accumulator = new_accumulator()
    add_stats(
        accumulator,
        {
            "elements": 4,
            "changed_elements": 2,
            "base_sq": 16.0,
            "candidate_sq": 16.0,
            "delta_sq": 1.0,
            "dot": 15.5,
            "max_abs_delta": 0.5,
        },
    )
    result = summarize_accumulator(accumulator)
    assert result["changed_fraction"] == 0.5
    assert result["relative_l2"] == 0.25
