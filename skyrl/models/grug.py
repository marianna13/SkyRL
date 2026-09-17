# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from transformers.configuration_utils import PretrainedConfig

# Snowball used a fixed interval. Hero exports write the interval into config.
_FULL_ATTENTION_INTERVAL = 4
_HERO_ARTIFACT_SCHEMA_VERSION = 2
_SUPPORTED_SCONV_SITES = frozenset({"k", "attn", "mlp"})

GRUG_MOE_MODEL_TYPE = "grug_moe"
GRUG_ROUTER_BIAS_SUFFIX = ".mlp.router.bias"
GRUG_GATED_NORM_RANK = 128
GRUG_ROUTING_RENORM_SUM = 2.5
GRUG_ATTN_GATE_SCALE = 2.0
GRUG_QK_RMS_NORM_EPS = 1e-6
GRUG_XSA_EPS = 1e-6
GRUG_ROUTER_RENORM_EPS = 1e-9


def is_grug_router_bias(model_type: str | None, name: str) -> bool:
    """Return whether a state entry is Grug's persistent fp32 router bias."""

    return model_type == GRUG_MOE_MODEL_TYPE and name.endswith(GRUG_ROUTER_BIAS_SUFFIX)


def grug_long_layer_flags(num_layers: int) -> tuple[bool, ...]:
    """Return the decoder layers that use full attention without RoPE."""

    return tuple(idx % 4 == 3 or idx == num_layers - 1 for idx in range(num_layers))


def grug_rms_norm_no_weight(hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply Grug's weightless fp32-accumulating RMS normalization."""

    input_dtype = hidden_states.dtype
    fp32 = hidden_states.float()
    variance = fp32.square().mean(dim=-1, keepdim=True)
    return (fp32 * torch.rsqrt(variance + GRUG_QK_RMS_NORM_EPS)).to(input_dtype)


def jax_top_k(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k using JAX's stable lower-index tie rule."""

    sorted_values, sorted_indices = torch.sort(values, dim=-1, descending=True, stable=True)
    return sorted_values[..., :k], sorted_indices[..., :k]


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def grug_moe_layer_types(
    num_layers: int,
    global_every: int = _FULL_ATTENTION_INTERVAL,
) -> list[str]:
    return [
        (
            "full_attention"
            if (layer_index + 1) % global_every == 0 or layer_index == num_layers - 1
            else "sliding_attention"
        )
        for layer_index in range(num_layers)
    ]


def grug_moe_rope_theta(config: Any) -> float:
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        return float(rope_theta)

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        for key in ("rope_theta", "theta"):
            if key in rope_parameters:
                return float(rope_parameters[key])

    rope = getattr(config, "rope", None)
    if isinstance(rope, dict) and "theta" in rope:
        return float(rope["theta"])
    if hasattr(rope, "theta"):
        return float(rope.theta)
    return 10000.0


class GrugMoeConfig(PretrainedConfig):
    model_type = GRUG_MOE_MODEL_TYPE
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int | None = None,
        hidden_size: int | None = None,
        intermediate_dim: int | None = None,
        intermediate_size: int | None = None,
        moe_intermediate_size: int | None = None,
        shared_expert_intermediate_dim: int | None = None,
        shared_expert_intermediate_size: int | None = None,
        num_shared_experts: int = 1,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_experts_per_tok: int | None = None,
        num_layers: int | None = None,
        num_hidden_layers: int | None = None,
        num_heads: int | None = None,
        num_attention_heads: int | None = None,
        num_kv_heads: int | None = None,
        num_key_value_heads: int | None = None,
        local_kv_heads: int | None = None,
        global_kv_heads: int | None = None,
        head_dim: int | None = None,
        attention_head_dim: int | None = None,
        max_seq_len: int | None = None,
        max_position_embeddings: int | None = None,
        sliding_window: int | None = None,
        global_every: int = _FULL_ATTENTION_INTERVAL,
        layer_types: list[str] | None = None,
        grug_attention_layer_types: list[str] | None = None,
        layer_norm_eps: float | None = None,
        rms_norm_eps: float | None = None,
        initializer_std: float | None = None,
        initializer_range: float | None = None,
        qk_mult: float = 1.0,
        qk_mult_long_scale: float = 1.0,
        latent_dim: int | None = None,
        disable_pko: bool = True,
        disable_long_rope: bool = True,
        rope_fused: bool = False,
        sconv: bool = False,
        sconv_kernel: int = 4,
        sconv_sites: list[str] | tuple[str, ...] = ("k", "attn", "mlp"),
        grugmoe_artifact_schema_version: int = 1,
        rope: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_scaling: dict[str, Any] | None = None,
        rope_theta: float | None = None,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        hidden_size = int(_coalesce(hidden_size, hidden_dim, 2048))
        intermediate_size = int(_coalesce(intermediate_size, moe_intermediate_size, intermediate_dim, 5632))
        shared_expert_intermediate_size = int(
            _coalesce(
                shared_expert_intermediate_size,
                shared_expert_intermediate_dim,
                5632,
            )
        )
        num_hidden_layers = int(_coalesce(num_hidden_layers, num_layers, 24))
        num_attention_heads = int(_coalesce(num_attention_heads, num_heads, 16))
        num_key_value_heads = int(_coalesce(num_key_value_heads, num_kv_heads, num_attention_heads))
        head_dim = _coalesce(attention_head_dim, head_dim)
        max_position_embeddings = int(_coalesce(max_position_embeddings, max_seq_len, 4096))
        rope_theta = grug_moe_rope_theta(
            SimpleNamespace(
                rope=rope,
                rope_parameters=rope_parameters,
                rope_theta=rope_theta,
            )
        )
        if rope is None:
            rope = {"theta": rope_theta}
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        global_every = int(global_every)
        if global_every <= 0:
            raise ValueError("global_every must be positive")
        resolved_layer_types = grug_moe_layer_types(num_hidden_layers, global_every)
        attention_only_layer_types = ["attention"] * num_hidden_layers
        transformers_attention_only_layer_types = ["full_attention"] * num_hidden_layers
        if layer_types is not None and layer_types not in (
            resolved_layer_types,
            attention_only_layer_types,
            transformers_attention_only_layer_types,
        ):
            raise ValueError(
                "layer_types must contain either the legacy GrugMoE attention " "pattern or vLLM attention-only markers"
            )
        if grug_attention_layer_types is not None and grug_attention_layer_types != resolved_layer_types:
            raise ValueError("grug_attention_layer_types must match the GrugMoE attention architecture")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_dim = intermediate_size
        self.moe_intermediate_size = intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.shared_expert_intermediate_dim = shared_expert_intermediate_size
        self.num_shared_experts = int(num_shared_experts)
        self.num_experts = int(_coalesce(num_experts, num_local_experts, 8))
        self.num_local_experts = self.num_experts
        self.num_experts_per_tok = int(_coalesce(num_experts_per_tok, num_experts_per_token, 2))
        self.num_experts_per_token = self.num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.num_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_kv_heads = num_key_value_heads
        self.local_kv_heads = None if local_kv_heads is None else int(local_kv_heads)
        self.global_kv_heads = None if global_kv_heads is None else int(global_kv_heads)
        self.head_dim = head_dim
        self.attention_head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_position_embeddings
        self.sliding_window = int(_coalesce(sliding_window, max_position_embeddings))
        self.global_every = global_every
        # vLLM reserves ``layer_types`` for deciding whether a model has
        # recurrent/Mamba state. Grug's full-vs-sliding distinction describes
        # two kinds of attention and must live in separate metadata. Without
        # this normalization schema-1 Snowball is incorrectly treated as a
        # hybrid Mamba model and generation fails while allocating Mamba state.
        self.grug_attention_layer_types = resolved_layer_types
        self.layer_types = resolved_layer_types if bool(sconv) else attention_only_layer_types
        self.rms_norm_eps = float(_coalesce(rms_norm_eps, layer_norm_eps, 1e-5))
        self.layer_norm_eps = self.rms_norm_eps
        self.initializer_range = float(_coalesce(initializer_range, initializer_std, 0.02))
        self.initializer_std = self.initializer_range
        self.qk_mult = qk_mult
        self.qk_mult_long_scale = qk_mult_long_scale
        self.latent_dim = None if latent_dim is None else int(latent_dim)
        self.disable_pko = disable_pko
        self.disable_long_rope = disable_long_rope
        self.rope_fused = bool(rope_fused)
        self.sconv = bool(sconv)
        self.sconv_kernel = int(sconv_kernel)
        self.sconv_sites = list(sconv_sites)
        self.grugmoe_artifact_schema_version = int(grugmoe_artifact_schema_version)
        self.rope = rope
        self.rope_parameters = rope_parameters
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.use_cache = use_cache

        if self.grugmoe_artifact_schema_version not in (
            1,
            _HERO_ARTIFACT_SCHEMA_VERSION,
        ):
            raise ValueError("unsupported grugmoe_artifact_schema_version=" f"{self.grugmoe_artifact_schema_version}")
        if self.grugmoe_artifact_schema_version == 1:
            hero_fields = []
            if self.num_shared_experts != 1:
                hero_fields.append("num_shared_experts")
            if self.latent_dim is not None:
                hero_fields.append("latent_dim")
            if self.local_kv_heads is not None or self.global_kv_heads is not None:
                hero_fields.extend(("local_kv_heads", "global_kv_heads"))
            if self.global_every != _FULL_ATTENTION_INTERVAL:
                hero_fields.append("global_every")
            if self.rope_fused:
                hero_fields.append("rope_fused")
            if self.sconv:
                hero_fields.append("sconv")
            if self.sconv_kernel != 4:
                hero_fields.append("sconv_kernel")
            if self.sconv_sites != ["k", "attn", "mlp"]:
                hero_fields.append("sconv_sites")
            if hero_fields:
                raise ValueError(", ".join(hero_fields) + " require grugmoe_artifact_schema_version=2")
        if self.num_shared_experts <= 0:
            raise ValueError("num_shared_experts must be positive")
        if self.latent_dim is not None and not 0 < self.latent_dim <= hidden_size:
            raise ValueError(f"latent_dim must be in (0, hidden_size={hidden_size}]")
        if (self.local_kv_heads is None) != (self.global_kv_heads is None):
            raise ValueError("local_kv_heads and global_kv_heads must be set together")
        if self.local_kv_heads is not None and self.global_kv_heads is not None:
            if self.local_kv_heads <= 0 or self.global_kv_heads <= 0:
                raise ValueError("local_kv_heads and global_kv_heads must be positive")
            if num_attention_heads % self.local_kv_heads != 0 or num_attention_heads % self.global_kv_heads != 0:
                raise ValueError("num_attention_heads must be divisible by both local and " "global KV-head counts")
            if num_key_value_heads != max(self.local_kv_heads, self.global_kv_heads):
                raise ValueError("num_key_value_heads must equal the stored maximum of " "local/global KV heads")
        if self.sconv and not 2 <= self.sconv_kernel <= 5:
            raise ValueError("sconv_kernel must be between 2 and 5")
        unknown_sconv_sites = set(self.sconv_sites) - _SUPPORTED_SCONV_SITES
        if unknown_sconv_sites:
            raise ValueError("unsupported sconv_sites: " + ", ".join(sorted(unknown_sconv_sites)))

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        if not self.sconv:
            # transformers>=5 remaps the legacy ``attention`` marker to
            # ``full_attention`` while validating PretrainedConfig. Restore it
            # after validation because vLLM 0.27 specifically uses
            # ``attention`` to identify an attention-only specialization of a
            # model class that is also capable of recurrent state.
            self.layer_types = attention_only_layer_types


__all__ = [
    "GRUG_ATTN_GATE_SCALE",
    "GRUG_GATED_NORM_RANK",
    "GRUG_MOE_MODEL_TYPE",
    "GRUG_ROUTER_BIAS_SUFFIX",
    "GRUG_ROUTER_RENORM_EPS",
    "GRUG_ROUTING_RENORM_SUM",
    "GRUG_XSA_EPS",
    "GrugMoeConfig",
    "grug_long_layer_flags",
    "grug_moe_layer_types",
    "grug_moe_rope_theta",
    "grug_rms_norm_no_weight",
    "is_grug_router_bias",
    "jax_top_k",
]
