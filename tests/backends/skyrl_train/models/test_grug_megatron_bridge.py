from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.bridge")

from megatron.bridge import AutoBridge

import skyrl.backends.skyrl_train.workers.megatron.grug_bridge as grug_bridge_module
from skyrl.backends.skyrl_train.workers.megatron.grug_bridge import (
    GrugModelProvider,
    GrugMoeBridge,
    _accumulate_grug_stacked_gated_export,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronWeightExtractor,
)
from skyrl.models.grug import GrugMoeConfig


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
        "dtype": "bfloat16",
    }
    values.update(overrides)
    return GrugMoeConfig(**values)


def test_snowball_dispatches_to_grug_provider() -> None:
    # Keep the module import live: importing it performs bridge registration.
    assert grug_bridge_module is not None
    bridge = AutoBridge.from_hf_config(snowball_config())

    assert isinstance(bridge._model_bridge, GrugMoeBridge)
    provider = bridge.to_megatron_provider(load_weights=False)

    assert isinstance(provider, GrugModelProvider)
    assert provider.num_layers == 26
    assert provider.hidden_size == 2560
    assert provider.num_attention_heads == 20
    assert provider.num_query_groups == 5
    assert provider.kv_channels == 128
    assert provider.num_moe_experts == 256
    assert provider.moe_router_topk == 4
    assert provider.moe_ffn_hidden_size == 1280
    assert provider.moe_shared_expert_intermediate_size == 2560
    assert provider.window_size == (2047, 0)
    assert provider.rotary_percent == 0.5
    assert provider.moe_permute_fusion is True


def test_bridge_covers_every_snowball_weight_family() -> None:
    registry = GrugMoeBridge().mapping_registry()
    weight_names = [
        "model.embed_tokens.weight",
        "model.embed_norm.weight",
        "model.embed_gated_norm.down_proj.weight",
        "model.embed_gated_norm.up_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.attn_gated_norm.down_proj.weight",
        "model.layers.0.attn_gated_norm.up_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.attn_gate.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp_gated_norm.down_proj.weight",
        "model.layers.0.mlp_gated_norm.up_proj.weight",
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.mlp.experts.up_proj.weight",
        "model.layers.0.mlp.experts.down_proj.weight",
        "model.layers.0.shared_expert.gate_proj.weight",
        "model.layers.0.shared_expert.up_proj.weight",
        "model.layers.0.shared_expert.down_proj.weight",
        "model.norm.weight",
        "model.final_gated_norm.down_proj.weight",
        "model.final_gated_norm.up_proj.weight",
        "lm_head.weight",
    ]

    missing = [name for name in weight_names if registry.hf_to_megatron_lookup(name) is None]

    assert missing == []


def test_megatron_rejects_newer_grug_artifact_schema() -> None:
    bridge = AutoBridge.from_hf_config(
        snowball_config(
            grugmoe_artifact_schema_version=2,
            latent_dim=128,
        )
    )

    with pytest.raises(NotImplementedError, match="schema 1 only"):
        bridge.to_megatron_provider(load_weights=False)


def test_grug_weight_sync_preserves_router_bias_fp32() -> None:
    class FakeBridge:
        hf_pretrained = type(
            "Pretrained",
            (),
            {"config": type("Config", (), {"model_type": "grug_moe"})()},
        )()

        def export_hf_weights(self, *args, **kwargs):
            del args, kwargs
            yield "model.layers.0.mlp.router.weight", torch.empty(2, 2)
            yield "model.layers.0.mlp.router.bias", torch.empty(2, dtype=torch.float32)

    extractor = MegatronWeightExtractor(FakeBridge(), actor_module=None)

    metadata = extractor.get_weight_metadata(torch.bfloat16)

    assert metadata["dtype_names"] == ["bfloat16", "float32"]


def test_grug_grouped_export_keeps_gate_and_up_separate() -> None:
    gate_name = "model.layers.0.mlp.experts.gate_proj.weight"
    up_name = "model.layers.0.mlp.experts.up_proj.weight"
    mapping = SimpleNamespace(
        hf_param={"gate": gate_name, "up": up_name},
        ep_size=2,
        group_key="model.layers.0.mlp.experts.gate_up",
    )
    model_config = SimpleNamespace(num_moe_experts=4)
    buffers: dict[str, dict[int, torch.Tensor]] = {}

    first = _accumulate_grug_stacked_gated_export(
        SimpleNamespace(mapping=mapping, param_name="decoder.layers.0.mlp.experts.linear_fc1.weight0"),
        {
            gate_name: torch.tensor([[[10.0]], [[30.0]]]),
            up_name: torch.tensor([[[110.0]], [[130.0]]]),
        },
        model_config,
        buffers,
    )
    second = _accumulate_grug_stacked_gated_export(
        SimpleNamespace(mapping=mapping, param_name="decoder.layers.0.mlp.experts.linear_fc1.weight1"),
        {
            gate_name: torch.tensor([[[20.0]], [[40.0]]]),
            up_name: torch.tensor([[[120.0]], [[140.0]]]),
        },
        model_config,
        buffers,
    )

    assert first is None
    assert set(second) == {gate_name, up_name}
    assert second[gate_name].flatten().tolist() == [10.0, 20.0, 30.0, 40.0]
    assert second[up_name].flatten().tolist() == [110.0, 120.0, 130.0, 140.0]
    assert buffers == {}
