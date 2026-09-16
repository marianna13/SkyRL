"""Register the out-of-tree GrugMoE implementation with vLLM.

The registration stays intentionally lightweight: the model class is passed as
an import string so loading this plugin in vLLM's controller and worker
processes does not initialize CUDA.
"""

from transformers import AutoConfig

from skyrl.models.grug import GRUG_MOE_MODEL_TYPE, GrugMoeConfig

GRUG_MOE_ARCHITECTURE = "GrugMoeForCausalLM"
GRUG_MOE_MODEL_CLASS = "skyrl.backends.skyrl_train.inference_servers.grug_vllm:GrugMoeForCausalLM"


def register() -> None:
    """Register Grug's config and lazy model class in the active vLLM process."""

    from vllm import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY[GRUG_MOE_MODEL_TYPE] = GrugMoeConfig
    AutoConfig.register(GRUG_MOE_MODEL_TYPE, GrugMoeConfig, exist_ok=True)
    ModelRegistry.register_model(GRUG_MOE_ARCHITECTURE, GRUG_MOE_MODEL_CLASS)


__all__ = ["register"]
