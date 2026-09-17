from tempfile import TemporaryDirectory

import torch

from skyrl.models.grug import (
    GrugMoeConfig,
    grug_long_layer_flags,
    is_grug_router_bias,
    jax_top_k,
)


def snowball_config(**overrides) -> GrugMoeConfig:
    values = {
        "architectures": ["GrugMoeForCausalLM"],
        "vocab_size": 128256,
        "hidden_size": 2560,
        "num_hidden_layers": 26,
        "num_attention_heads": 20,
        "num_key_value_heads": 5,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "sliding_window": 2048,
        "rms_norm_eps": 1e-5,
        "initializer_range": 0.009882117688026186,
        "rope_theta": 10000.0,
        "num_experts": 256,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 1280,
        "shared_expert_intermediate_size": 2560,
        "qk_mult": 1.5703274004183787,
        "grugmoe_artifact_schema_version": 1,
    }
    values.update(overrides)
    return GrugMoeConfig(**values)


def test_snowball_config_aliases_and_attention_pattern() -> None:
    config = snowball_config()
    expected_attention_pattern = [
        "full_attention" if is_long else "sliding_attention" for is_long in grug_long_layer_flags(26)
    ]

    assert config.model_type == "grug_moe"
    assert config.intermediate_size == config.moe_intermediate_size == 1280
    assert config.num_local_experts == config.num_experts == 256
    assert config.num_experts_per_token == config.num_experts_per_tok == 4
    assert config.attention_head_dim == config.head_dim == 128
    assert config.grug_attention_layer_types == expected_attention_pattern
    assert config.layer_types == ["attention"] * 26
    assert [index for index, is_long in enumerate(grug_long_layer_flags(26)) if is_long] == [3, 7, 11, 15, 19, 23, 25]


def test_snowball_normalizes_legacy_layer_types_for_vllm() -> None:
    legacy_attention_pattern = [
        "full_attention" if is_long else "sliding_attention" for is_long in grug_long_layer_flags(26)
    ]

    config = snowball_config(layer_types=legacy_attention_pattern)

    assert config.grug_attention_layer_types == legacy_attention_pattern
    assert config.layer_types == ["attention"] * 26


def test_snowball_config_round_trip_preserves_vllm_attention_marker() -> None:
    config = snowball_config()

    with TemporaryDirectory() as directory:
        config.save_pretrained(directory)
        reloaded = GrugMoeConfig.from_pretrained(directory)

    assert reloaded.layer_types == ["attention"] * 26
    assert reloaded.grug_attention_layer_types == config.grug_attention_layer_types


def test_stateful_grug_keeps_hybrid_layer_markers() -> None:
    config = snowball_config(
        grugmoe_artifact_schema_version=2,
        sconv=True,
    )

    assert config.layer_types == config.grug_attention_layer_types
    assert any(layer_type != "attention" for layer_type in config.layer_types)


def test_jax_top_k_prefers_lower_expert_index_on_ties() -> None:
    values = torch.tensor([[2.0, 3.0, 3.0, 1.0, 3.0]])

    top_values, top_indices = jax_top_k(values, 3)

    torch.testing.assert_close(top_values, torch.tensor([[3.0, 3.0, 3.0]]))
    assert top_indices.tolist() == [[1, 2, 4]]


def test_only_grug_router_bias_requires_fp32_weight_sync() -> None:
    assert is_grug_router_bias("grug_moe", "model.layers.25.mlp.router.bias")
    assert not is_grug_router_bias("grug_moe", "model.layers.25.mlp.router.weight")
    assert not is_grug_router_bias("qwen3_moe", "model.layers.25.mlp.router.bias")
