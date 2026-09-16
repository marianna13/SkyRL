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

    assert config.model_type == "grug_moe"
    assert config.intermediate_size == config.moe_intermediate_size == 1280
    assert config.num_local_experts == config.num_experts == 256
    assert config.num_experts_per_token == config.num_experts_per_tok == 4
    assert config.attention_head_dim == config.head_dim == 128
    assert config.layer_types == [
        "full_attention" if is_long else "sliding_attention" for is_long in grug_long_layer_flags(26)
    ]
    assert [index for index, is_long in enumerate(grug_long_layer_flags(26)) if is_long] == [3, 7, 11, 15, 19, 23, 25]


def test_jax_top_k_prefers_lower_expert_index_on_ties() -> None:
    values = torch.tensor([[2.0, 3.0, 3.0, 1.0, 3.0]])

    top_values, top_indices = jax_top_k(values, 3)

    torch.testing.assert_close(top_values, torch.tensor([[3.0, 3.0, 3.0]]))
    assert top_indices.tolist() == [[1, 2, 4]]


def test_only_grug_router_bias_requires_fp32_weight_sync() -> None:
    assert is_grug_router_bias("grug_moe", "model.layers.25.mlp.router.bias")
    assert not is_grug_router_bias("grug_moe", "model.layers.25.mlp.router.weight")
    assert not is_grug_router_bias("qwen3_moe", "model.layers.25.mlp.router.bias")
