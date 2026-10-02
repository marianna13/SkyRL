"""save_hf_configs keeps a GrugMoE checkpoint's config.json byte-identical to the source."""

import json

from transformers import AutoConfig, LlamaConfig

from skyrl.backends.skyrl_train.distributed.strategy import DistributedStrategy
from skyrl.models.grug import GRUG_MOE_MODEL_TYPE, GrugMoeConfig

AutoConfig.register(GRUG_MOE_MODEL_TYPE, GrugMoeConfig, exist_ok=True)

SNOWBALL_SCHEMA1 = {
    "architectures": ["GrugMoeForCausalLM"],
    "model_type": "grug_moe",
    "vocab_size": 128256,
    "hidden_size": 2560,
    "num_hidden_layers": 26,
    "num_attention_heads": 20,
    "num_key_value_heads": 5,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "sliding_window": 2048,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "num_experts": 256,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 1280,
    "shared_expert_intermediate_size": 2560,
    "qk_mult": 1.75,
    "tie_word_embeddings": False,
    "grugmoe_attention_mode": "production",
    "grugmoe_artifact_schema_version": 1,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
}


def test_grug_export_copies_source_config_verbatim(tmp_path):
    src = tmp_path / "base"
    src.mkdir()
    (src / "config.json").write_text(json.dumps(SNOWBALL_SCHEMA1, indent=2))
    config = AutoConfig.from_pretrained(str(src))
    assert config.model_type == GRUG_MOE_MODEL_TYPE

    out = tmp_path / "export"
    DistributedStrategy.save_hf_configs(None, config, str(out))

    assert (out / "config.json").read_bytes() == (src / "config.json").read_bytes()
    exported = json.loads((out / "config.json").read_text())
    assert "layer_types" not in exported  # never the attention-only markers vLLM 0.26 plugins reject
    reloaded = AutoConfig.from_pretrained(str(out))
    assert reloaded.grug_attention_layer_types == config.grug_attention_layer_types


def test_non_grug_export_still_serializes(tmp_path):
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4)
    out = tmp_path / "export"
    DistributedStrategy.save_hf_configs(None, config, str(out))
    assert json.loads((out / "config.json").read_text())["model_type"] == "llama"
