"""Grug router instrumentation, ported from MarinSkyRL (cefd98ac) tests/cpu/models/test_router_instrumentation.py.

Only the Grug test is kept; the Qwen / grouped-MoE router tests cover MarinSkyRL layers not ported here.
"""

import torch

from skyrl.models.grug_moe import GrugMoeConfig, GrugMoeRouter
from skyrl.models.router_instrumentation import (
    instrument_moe_routers,
    observe_router_forwards,
)

def _grug_config() -> GrugMoeConfig:
    return GrugMoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        shared_expert_intermediate_size=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        sliding_window=4,
    )


def test_grug_router_observation_uses_biased_selection_and_native_weights() -> None:
    router = GrugMoeRouter(_grug_config())
    with instrument_moe_routers(router) as instrumentation:
        assert instrumentation.router_count == 1
    with torch.no_grad():
        router.weight.copy_(torch.eye(4, 8))
        router.bias.copy_(torch.tensor([0.0, 0.5, -0.25, 0.75]))
    hidden = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    observations = []

    with observe_router_forwards(observations.append):
        raw_logits, selected_experts, combine_weights = router(hidden)

    selection_logits = raw_logits + router.bias
    selected_raw_logits = raw_logits.gather(-1, selected_experts)
    expected_weights = selected_raw_logits.sigmoid()
    expected_weights *= 2.5 / expected_weights.sum(dim=-1, keepdim=True)

    assert len(observations) == 1
    observation = observations[0]
    torch.testing.assert_close(observation.router_inputs, hidden)
    torch.testing.assert_close(observation.selection_logits, selection_logits)
    torch.testing.assert_close(observation.selection_log_probs, selection_logits.log_softmax(dim=-1))
    torch.testing.assert_close(observation.natural_selected_experts, selected_experts)
    torch.testing.assert_close(observation.selected_experts, selected_experts)
    torch.testing.assert_close(observation.combine_weights, combine_weights)
    torch.testing.assert_close(combine_weights, expected_weights)
