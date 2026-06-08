"""SkyRL-Train backend for TinkerEngine.

Uses SkyRL-Train infrastructure for supervised training with cross-entropy loss.
Currently supports a single model only.
"""

import asyncio
import os
import tarfile
import tempfile
import shutil
from typing import Any

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from tx.tinker import types
from tx.tinker.backends.backend import AbstractBackend
from tx.utils.log import logger
import traceback

try:  # Optional dependency: keep other backends importable without ray/skyrl-train.
    import ray
    from ray.util.placement_group import placement_group
    from skyrl_train.training_batch import TrainingInputBatch
    from skyrl_train.trainer import RayPPOTrainer
    from skyrl_train.utils.tracking import Tracking
    from skyrl_train.utils.utils import initialize_ray
    from skyrl_train.utils import get_ray_pg_ready_with_timeout
    from skyrl_train.config.utils import get_default_config
    from skyrl_train.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S
    from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
    from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

    SKYRL_TRAIN_AVAILABLE = True
except Exception as e:  # pragma: no cover - exercised only in non-ray installs
    ray = None
    placement_group = None
    TrainingInputBatch = Any
    RayPPOTrainer = Any
    Tracking = Any
    initialize_ray = None
    get_ray_pg_ready_with_timeout = None
    get_default_config = None
    SKYRL_RAY_PG_TIMEOUT_IN_S = None
    create_ray_wrapped_inference_engines_from_config = None
    InferenceEngineClient = Any
    SKYRL_TRAIN_AVAILABLE = False
    print("SkyRL-Train backend not available. Ray and skyrl-train dependencies are required.", flush=True)
    traceback.print_exc()


#  data.train_data="['$DATA_DIR/train.parquet']" \
#     data.val_data="['$DATA_DIR/validation.parquet']" \
#     trainer.fully_async.max_staleness_steps=${MAX_STALENESS_STEPS} \
#     trainer.fully_async.num_parallel_generation_workers=${NUM_PARALLEL_GENERATION_WORKERS} \
#     trainer.algorithm.advantage_estimator="grpo" \
#     trainer.algorithm.use_tis=$USE_TIS \
#     trainer.algorithm.tis_imp_ratio_cap=$TIS_IMP_RATIO_CAP \
#     trainer.algorithm.use_kl_loss=true \
#     trainer.placement.colocate_all=false \
#     trainer.strategy=fsdp2 \
#     trainer.policy.model.path=${MODEL_PATH} \
#     trainer.placement.policy_num_nodes=$POLICY_NUM_NODES \
#     trainer.placement.ref_num_nodes=$REF_NUM_NODES \
#     trainer.placement.policy_num_gpus_per_node=4 \
#     trainer.placement.ref_num_gpus_per_node=4 \
#     generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
#     generator.inference_engine_tensor_parallel_size=$TENSOR_PARALLEL_SIZE \
#     +generator.engine_init_kwargs.custom_chat_template_chat_completion_path=$CHAT_TEMPLATE_PATH \
#     trainer.epochs=$EPOCHS \
#     trainer.eval_batch_size=$EVAL_BATCH_SIZE \
#     trainer.eval_before_train=false \
#     trainer.eval_interval=$EVAL_INTERVAL \
#     trainer.update_epochs_per_batch=1 \
#     trainer.train_batch_size=$TRAIN_BATCH_SIZE \
#     trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
#     trainer.micro_forward_batch_size_per_gpu=4 \
#     trainer.micro_train_batch_size_per_gpu=4 \
#     trainer.ckpt_interval=5 \
#     trainer.max_prompt_length=2048 \
#     generator.sampling_params.max_generate_length=1024 \
#     trainer.policy.optimizer_config.lr=1.0e-6 \
#     trainer.policy.optimizer_config.weight_decay=0.01 \
#     generator.n_samples_per_prompt=8 \
#     generator.gpu_memory_utilization=0.8 \
#     generator.eval_n_samples_per_prompt=1 \
#     trainer.logger=wandb \
#     trainer.run_name="$RUN_NAME" \
#     trainer.resume_mode=latest \
#     generator.backend=vllm \
#     generator.run_engines_locally=true \
#     generator.weight_sync_backend=nccl \
#     generator.async_engine=true \
#     generator.batched=false \
#     generator.enable_http_endpoint=true \


class FSDPConfig(BaseModel, extra="allow"):
    cpu_offload: bool = False

class LoraConfig(BaseModel, extra="allow"):
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.1
    lora_sync_path: str = "/tmp/lora_sync"  # Shared path for synchronizing LoRA weights between trainer and inference engines

class ModelConfig(BaseModel, extra="allow"):
    lora: LoraConfig = LoraConfig()

class MegatronConfig(BaseModel, extra="allow"):
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int | None = 1
    transformer_config_kwargs: dict = {}
    optimizer_config_kwargs: dict = {}


class OptimizerConfig(BaseModel, extra="allow"):
    lr: float = 1e-6
    weight_decay: float = 0.01
    scheduler: str = "constant"
    adam_betas: tuple[float, float] = (0.9, 0.999)
    max_warmup_steps: int = 0
    lr_decay_steps: int | None = None
    max_grad_norm: float = 1.0

class PolicyConfig(BaseModel, extra="allow"):
    fsdp_config: FSDPConfig = FSDPConfig()
    model: ModelConfig = ModelConfig()
    megatron_config: MegatronConfig = MegatronConfig()
    optimizer_config: OptimizerConfig = OptimizerConfig()

class RefConfig(BaseModel, extra="allow"):
    model: ModelConfig = ModelConfig()
    fsdp_config: FSDPConfig = FSDPConfig()
    megatron_config: MegatronConfig = MegatronConfig()

class GeneratorConfig(BaseModel, extra="allow"):
    """Subset of SkyRL-Train config relevant for generator setup."""

    num_inference_engines: int = 0
    inference_model_path: str | None = None
    weight_sync_backend: str = "nccl"
    enable_http_endpoint: bool = False
    http_endpoint_host: str = "localhost"
    http_endpoint_port: int = 8800
    gpu_memory_utilization: float = 0.8
    inference_engine_tensor_parallel_size: int = 1
    model_dtype: str = "bfloat16"
    run_engines_locally: bool = True
    engine_init_kwargs: dict = {}

class AlgorithmConfig(BaseModel, extra="allow"):
    """Subset of SkyRL-Train config relevant for algorithm setup."""

    advantage_estimator: str = "grpo"
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.001

class PlacementConfig(BaseModel, extra="allow"):
    """Subset of SkyRL-Train config relevant for placement setup."""

    colocate_all: bool = False
    colocate_policy_ref: bool = True  # Whether to colocate policy and ref workers (if not colocate_all)
    policy_num_nodes: int = 1
    ref_num_nodes: int = 1
    policy_num_gpus_per_node: int = 4
    ref_num_gpus_per_node: int = 4


class TrainerConfig(BaseModel, extra="allow"):
    """Subset of SkyRL-Train config relevant for trainer setup."""

    strategy: str = "fsdp2"
    train_batch_size: int = 256
    eval_batch_size: int = 1024
    ckpt_path: str | None = None
    micro_forward_batch_size_per_gpu: int = 4
    micro_train_batch_size_per_gpu: int = 4
    flash_attn: bool = False
    use_sample_packing: bool = False
    epochs: int = 1
    eval_before_train: bool = False
    policy_mini_batch_size: int = 8
    algorithm: AlgorithmConfig = AlgorithmConfig()
    placement: PlacementConfig = PlacementConfig()
    policy: PolicyConfig = PolicyConfig()
    ref: RefConfig = RefConfig()


class SkyRLTrainBackendConfig(BaseModel, extra="allow"):
    """Configuration for the SkyRL-Train backend.

    Note: Currently uses SkyRL's default config for all parameters.
    TODO: Implement proper config management to allow Tinker users to override
    training and inference parameters via backend_config.
    """

    trainer: TrainerConfig = TrainerConfig()
    generator: GeneratorConfig = GeneratorConfig()
    dp_size: int = 1  # Data parallel size for training




def _build_config(
    base_model: str,
    config: SkyRLTrainBackendConfig,
    lora_config: types.LoraConfig | None = None,
    **kwargs,
):
    """Build config for SkyRL-Train workers using default config.

    Args:
        base_model: HuggingFace model path
        config: Backend configuration
        lora_config: LoRA configuration if using LoRA
    """
    cfg = get_default_config()
    cfg.trainer.policy.model.path = base_model

    # Disable scheduler - Tinker manages learning rate externally via set_lr()
    cfg.trainer.policy.optimizer_config.scheduler = config.trainer.policy.optimizer_config.scheduler
    cfg.trainer.policy.optimizer_config.num_warmup_steps = config.trainer.policy.optimizer_config.max_warmup_steps
    cfg.trainer.policy.optimizer_config.lr = config.trainer.policy.optimizer_config.lr
    cfg.trainer.policy.optimizer_config.weight_decay = config.trainer.policy.optimizer_config.weight_decay
    cfg.trainer.policy.optimizer_config.adam_betas = config.trainer.policy.optimizer_config.adam_betas
    cfg.trainer.policy.optimizer_config.max_grad_norm = config.trainer.policy.optimizer_config.max_grad_norm
    # print(f"Setting optimizer config: {config.trainer.policy.optimizer_config}")
    # setattr(cfg.trainer.policy.optimizer_config, "lr_decay_steps", config.trainer.policy.optimizer_config.lr_decay_steps)



    cfg.trainer.algorithm.advantage_estimator = config.trainer.algorithm.advantage_estimator
    cfg.trainer.algorithm.use_kl_loss = config.trainer.algorithm.use_kl_loss
    cfg.trainer.algorithm.kl_loss_coef = config.trainer.algorithm.kl_loss_coef
    cfg.trainer.placement.colocate_all = config.trainer.placement.colocate_all
    cfg.trainer.placement.colocate_policy_ref = config.trainer.placement.colocate_policy_ref
    cfg.trainer.logger = "console"
    cfg.trainer.strategy = config.trainer.strategy
    cfg.trainer.placement.policy_num_nodes = config.trainer.placement.policy_num_nodes
    cfg.trainer.placement.ref_num_nodes = config.trainer.placement.ref_num_nodes
    cfg.trainer.placement.policy_num_gpus_per_node = config.trainer.placement.policy_num_gpus_per_node
    cfg.trainer.placement.ref_num_gpus_per_node = config.trainer.placement.ref_num_gpus_per_node
    cfg.trainer.train_batch_size = config.trainer.train_batch_size
    cfg.trainer.eval_batch_size = config.trainer.eval_batch_size
    cfg.trainer.micro_forward_batch_size_per_gpu = config.trainer.micro_forward_batch_size_per_gpu
    cfg.trainer.micro_train_batch_size_per_gpu = config.trainer.micro_train_batch_size_per_gpu
    cfg.trainer.flash_attn = config.trainer.flash_attn
    cfg.trainer.use_sample_packing = config.trainer.use_sample_packing
    cfg.trainer.policy.fsdp_config = config.trainer.policy.fsdp_config.dict()
    cfg.trainer.ref.fsdp_config = config.trainer.ref.fsdp_config.dict()
    cfg.trainer.ckpt_path = config.trainer.ckpt_path
    cfg.trainer.eval_before_train = config.trainer.eval_before_train

    cfg.trainer.policy.model.lora.rank = lora_config.rank if lora_config else config.trainer.policy.model.lora.rank
    cfg.trainer.policy.model.lora.alpha = config.trainer.policy.model.lora.alpha
    cfg.trainer.policy.model.lora.dropout = config.trainer.policy.model.lora.dropout
    cfg.trainer.policy.model.lora.lora_sync_path = config.trainer.policy.model.lora.lora_sync_path

    # Policy Megatron Config
    for k, v in config.trainer.policy.megatron_config.dict().items():
        if hasattr(cfg.trainer.policy.megatron_config, k):
            setattr(cfg.trainer.policy.megatron_config, k, v)

    # Ref Megatron Config
    for k, v in config.trainer.ref.megatron_config.dict().items():
        if hasattr(cfg.trainer.ref.megatron_config, k):
            setattr(cfg.trainer.ref.megatron_config, k, v)
    cfg.trainer.ref.model.path = base_model

    cfg.generator.num_inference_engines = config.generator.num_inference_engines
    cfg.generator.inference_engine_tensor_parallel_size = config.generator.inference_engine_tensor_parallel_size
    cfg.generator.weight_sync_backend = config.generator.weight_sync_backend
    cfg.generator.enable_http_endpoint = config.generator.enable_http_endpoint
    cfg.generator.gpu_memory_utilization = config.generator.gpu_memory_utilization
    cfg.generator.http_endpoint_host = config.generator.http_endpoint_host
    cfg.generator.http_endpoint_port = config.generator.http_endpoint_port
    cfg.generator.model_dtype = config.generator.model_dtype
    cfg.generator.run_engines_locally = config.generator.run_engines_locally
    # cfg.generator.enable_lora = config.generator.enable_lora
    cfg.generator.engine_init_kwargs = config.generator.engine_init_kwargs
    if config.generator.inference_model_path:
        cfg.generator.inference_model_path = config.generator.inference_model_path




    # cfg.dp_size = config.dp_size

    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logger.warning(f"Unknown config key '{key}' - ignoring")

    return cfg


class SkyRLTrainBackend(AbstractBackend):
    """SkyRL-Train backend for supervised training."""

    def __init__(self, base_model: str, config: SkyRLTrainBackendConfig):
        logger.warning("=" * 80)
        logger.warning("SkyRLTrainBackend is currently EXPERIMENTAL!")
        logger.warning("=" * 80)

        if not SKYRL_TRAIN_AVAILABLE or ray is None:
            raise ImportError(
                "SkyRLTrainBackend requires `ray`. Install the appropriate extras (e.g. `--extra skyrl_train`)."
            )

        self.base_model = base_model
        self.config = config
        self._model_id: str | None = None
        self._model_metadata: types.ModelMetadata | None = None
        self._trainer: RayPPOTrainer | None = None
        self._cfg = None
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self._inference_engine_client = None  # InferenceEngineClient for sampling

    @property
    def dp_size(self) -> int:
        """Return the data parallel size from the trainer's dispatch."""
        if self._trainer is not None and self._trainer.dispatch is not None:
            return self._trainer.dispatch.get_lcm_dp_size()
        return 1

    def has_model(self, model_id: str) -> bool:
        return self._model_id == model_id

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        if self._model_id is not None:
            raise ValueError(f"Model '{self._model_id}' already exists. Only one model supported.")

        # Build config
        self._cfg = _build_config(self.base_model, self.config, lora_config)

        logger.info(f"Creating model {model_id} with config: {self._cfg}")

        if not ray.is_initialized():
            logger.info("Initializing Ray with runtime environment")
            initialize_ray(self._cfg)

        # Generator placement group.
        # ONLY build/pass a shared PG when inference is COLOCATED with training
        # (colocate_all=True). That path deliberately runs vLLM workers at 0.2
        # GPU so they time-share devices with the trainer
        # (create_ray_wrapped_inference_engines: use_hybrid_engine = shared_pg
        # is not None -> num_gpus=0.2/worker).
        #
        # For colocate_all=False the generator must own FULL GPUs (1.0/worker).
        # The old `elif num_inference_engines > 0` branch passed a shared PG
        # here too, wrongly flipping the engine helper into hybrid 0.2-GPU mode
        # -> Ray packs ~5 inference ranks onto each physical GPU -> first OOM
        # (multiple 16GB shards on one GPU), then NCCL "Duplicate GPU detected".
        # Leaving it None makes the helper build a dedicated 1-GPU/worker PG.
        colocate_pg = None
        if self._cfg.trainer.placement.colocate_all:
            colocate_pg = self._create_colocate_pg()

        # Create inference engine client
        logger.info(f"Creating {self._cfg.generator.num_inference_engines} inference engines")
        self._inference_engine_client = InferenceEngineClient(
            create_ray_wrapped_inference_engines_from_config(self._cfg, colocate_pg, self._tokenizer),
            self._tokenizer,
            self._cfg,
        )

        # Create trainer
        tracker = Tracking(
            project_name="tinker",
            experiment_name=model_id,
            backends=[],  # No logging backends
            config=self._cfg,
        )

        self._trainer = RayPPOTrainer(
            cfg=self._cfg,
            tracker=tracker,
            tokenizer=self._tokenizer,
            train_dataset=None,  # Not needed for tinker API
            eval_dataset=None,
            inference_engine_client=self._inference_engine_client,
            generator=None,  # TODO(tyler): Update for sampling + RL
            colocate_pg=colocate_pg if self._cfg.trainer.placement.colocate_all else None,
        )

        # Get worker types based on strategy
        if self._cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, CriticWorker, RefWorker
        elif self._cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, CriticWorker, RefWorker
        else:
            raise ValueError(f"Unknown strategy type: {self._cfg.trainer.strategy}")

        logger.info("Building models.")
        self._trainer.build_models(PolicyWorker, CriticWorker, RefWorker)

        logger.info("Initializing weight sync state.")
        self._trainer.init_weight_sync_state()

        self._model_id = model_id
        self._model_metadata = types.ModelMetadata(adapter_index=0, lora_config=lora_config)
        logger.info(f"Created model {model_id} using RayPPOTrainer")

    def _create_colocate_pg(self):
        """Create placement group for colocated training + inference (following main_base.py pattern)."""
        total_gpu_slots = (
            self._cfg.generator.num_inference_engines
            * self._cfg.generator.inference_engine_tensor_parallel_size
            * self._cfg.generator.inference_engine_pipeline_parallel_size
            * self._cfg.generator.inference_engine_data_parallel_size
        )
        logger.info(f"Creating placement group with {total_gpu_slots} GPU slots for colocated training+inference")
        pg = placement_group([{"GPU": 1, "CPU": 1}] * total_gpu_slots, strategy="PACK")

        logger.info("Waiting for placement group to be ready...")
        get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        logger.info("Placement group ready!")

        return pg

    def delete_model(self, model_id: str) -> None:
        if self._model_id != model_id:
            raise ValueError(f"Model {model_id} not found")
        raise NotImplementedError("Deleting models not yet implemented")

    def _to_training_batch(self, prepared_batch: types.PreparedModelPassBatch) -> TrainingInputBatch:
        """Convert PreparedModelPassBatch to TrainingInputBatch."""
        if not prepared_batch.all_input_ids:
            return TrainingInputBatch({})

        # SkyRL-Train shifts internally, so provide the full sequence length by
        # appending the last target token to each already-shifted input.
        full_sequences = [
            list(input_ids) + ([targets[-1]] if targets else [])
            for input_ids, targets in zip(prepared_batch.all_input_ids, prepared_batch.all_targets)
        ]

        max_seq_len = max(len(seq) for seq in full_sequences)
        max_response_len = max(len(weights) for weights in prepared_batch.all_token_weights)

        sequences, attention_masks, loss_masks, response_masks = [], [], [], []
        # RL fields: populate with the per-token sampling_logprobs / advantages
        # from the tinker request so that importance_sampling_loss has non-None
        # old_log_probs and advantages. Left-pad to max_response_len with zeros
        # (those positions have loss_mask=0 so they contribute no gradient).
        sampling_logprobs, advantages = [], []
        have_sampling_lp = (
            getattr(prepared_batch, "all_sampling_logprobs", None) is not None
            and len(prepared_batch.all_sampling_logprobs) == len(full_sequences)
        )
        have_adv = (
            getattr(prepared_batch, "all_advantages", None) is not None
            and len(prepared_batch.all_advantages) == len(full_sequences)
        )

        for i, (seq, weights) in enumerate(zip(full_sequences, prepared_batch.all_token_weights)):
            pad_len = max_seq_len - len(seq)
            sequences.append([self._tokenizer.pad_token_id] * pad_len + list(seq))
            attention_masks.append([0] * pad_len + [1] * len(seq))
            action_pad = max_response_len - len(weights)
            loss_masks.append([0.0] * action_pad + [float(w) for w in weights])
            response_masks.append([0] * action_pad + [1] * len(weights))

            if have_sampling_lp:
                lp = [float(x) for x in prepared_batch.all_sampling_logprobs[i]]
                # sampling_logprobs may have length != len(weights) if the tinker client
                # padded differently; left-pad/truncate to max_response_len.
                lp_pad = max_response_len - len(lp)
                if lp_pad >= 0:
                    sampling_logprobs.append([0.0] * lp_pad + lp)
                else:
                    sampling_logprobs.append(lp[-max_response_len:])

            if have_adv:
                av = [float(x) for x in prepared_batch.all_advantages[i]]
                av_pad = max_response_len - len(av)
                if av_pad >= 0:
                    advantages.append([0.0] * av_pad + av)
                else:
                    advantages.append(av[-max_response_len:])

        # Pad the batch dimension to be a multiple of dp_size. The dispatch layer
        # asserts len(batch) % dp_size == 0 and will not pad on its own. We append
        # dummy rows at the end (copy of last row with loss_mask=0) so the extras
        # contribute zero gradient. The real request_batch_slices indices 0..N-1
        # are unaffected, so response building in forward_backward stays correct.
        dp_size = self.dp_size
        n_real = len(sequences)
        remainder = n_real % dp_size
        if remainder != 0 and n_real > 0:
            pad_rows = dp_size - remainder
            logger.info(
                f"Padding batch from {n_real} to {n_real + pad_rows} rows "
                f"to be divisible by dp_size={dp_size}"
            )
            for _ in range(pad_rows):
                sequences.append(list(sequences[-1]))
                attention_masks.append(list(attention_masks[-1]))
                loss_masks.append([0.0] * len(loss_masks[-1]))
                response_masks.append(list(response_masks[-1]))
                if have_sampling_lp and sampling_logprobs:
                    sampling_logprobs.append([0.0] * len(sampling_logprobs[-1]))
                if have_adv and advantages:
                    advantages.append([0.0] * len(advantages[-1]))

        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_masks, dtype=torch.float32)
        response_mask_tensor = torch.tensor(response_masks, dtype=torch.long)

        batch_dict = {
            "sequences": sequences_tensor,
            "attention_mask": attention_mask_tensor,
            "loss_mask": loss_mask_tensor,
            "response_mask": response_mask_tensor,
        }
        if have_sampling_lp and sampling_logprobs:
            # action_log_probs consumed as old_action_log_probs on the training side.
            batch_dict["action_log_probs"] = torch.tensor(sampling_logprobs, dtype=torch.float32)
        if have_adv and advantages:
            batch_dict["advantages"] = torch.tensor(advantages, dtype=torch.float32)

        batch = TrainingInputBatch(batch_dict)
        batch.metadata = {"response_length": max_response_len}
        return batch

    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
        loss_fn: str = "cross_entropy",
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_input_ids:
            return {}

        batch = self._to_training_batch(prepared_batch)
        logger.info(f"batch length: {len(batch)}")

        # Populate base_action_log_probs via ref forward when KL loss is enabled.
        # Mirrors skyrl-train's trainer.py:994-1012 — the tx backend bypasses
        # trainer.fit() so we must dispatch the ref forward ourselves, otherwise
        # policy loss reads data["base_action_log_probs"]=None and crashes in
        # compute_approx_kl.
        if (
            getattr(self._cfg.trainer.algorithm, "use_kl_loss", False)
            and getattr(self._trainer, "ref_model", None) is not None
        ):
            data_fwd_pass = batch.select(
                keys=["sequences", "attention_mask"],
                metadata_keys=["response_length"],
            )
            ref_output = self._trainer.dispatch.forward("ref", data_fwd_pass)
            base_log_probs = ref_output["output"]
            try:
                self._trainer.dispatch.empty_cache("ref")
            except Exception:
                pass
            sequences_all: torch.Tensor = batch["sequences"]
            if base_log_probs is not None:
                base_log_probs = base_log_probs[: len(sequences_all)]
            batch["base_action_log_probs"] = base_log_probs

        data = self._trainer.dispatch.forward_backward("policy", batch, loss_fn=loss_fn)

        results = {}
        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                raw_output = data["loss_fn_outputs"][i]
                logprobs = list(raw_output.get("logprobs", []))
                elementwise_loss = list(raw_output.get("elementwise_loss", []))
                loss_fn_outputs.append(
                    {
                        "elementwise_loss": {
                            "data": elementwise_loss,
                            "dtype": "float32",
                            "shape": [len(elementwise_loss)],
                        },
                        "logprobs": {
                            "data": logprobs,
                            "dtype": "float32",
                            "shape": [len(logprobs)],
                        },
                    }
                )
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )
        return results

    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        raise NotImplementedError("Forward-only pass not supported")

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")

        # Apply learning rate from AdamParams before optimizer step
        # Note: beta1, beta2, eps are fixed at optimizer creation and cannot be changed dynamically
        adam_params = request_data.adam_params
        self._trainer.dispatch.set_lr("policy", adam_params.learning_rate)

        grad_norm = self._trainer.dispatch.optim_step("policy")
        logger.info(f"optim_step: lr={adam_params.learning_rate}, grad_norm={grad_norm}")
        return types.OptimStepOutput()

    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Generate samples using InferenceEngineClient.

        NOTE: Weight sync is NOT triggered automatically. The caller must call
        save_weights_for_sampler() explicitly before calling sample() if weights
        have been updated.
        """
        # 1. Validate inference is enabled
        if self._inference_engine_client is None:
            error = types.ErrorResponse(
                error="Sampling not enabled. Inference engines were not initialized (num_inference_engines=0 in SkyRL config).",
                status="error",
            )
            return {req_id: error for req_id, _, _, _, _ in prepared_batch.request_batch_slices}

        # 2. Validate single model
        unique_models = set(prepared_batch.all_model_ids)
        if unique_models != {self._model_id}:
            error = types.ErrorResponse(
                error=f"Model mismatch. Expected {self._model_id}, got {unique_models}", status="error"
            )
            return {req_id: error for req_id, _, _, _, _ in prepared_batch.request_batch_slices}

        # 3. Sample all prompts in parallel
        async def sample_all():
            tasks = []
            for i in range(len(prepared_batch.all_prompts)):
                prompt = prepared_batch.all_prompts[i]
                sampling_params = prepared_batch.all_sampling_params[i]

                # Pass through common fields; only stop needs name translation
                # (Tinker uses stop_strings/stop_tokens, vLLM uses stop/stop_token_ids)
                params_dict = {
                    "temperature": sampling_params.temperature,
                    "max_tokens": sampling_params.max_tokens,
                    "seed": sampling_params.seed,
                    "top_k": sampling_params.top_k,
                    "top_p": sampling_params.top_p,
                    # Ask vLLM to return the sampled token's logprob. Without this,
                    # response_logprobs comes back None and the importance-sampling
                    # loss collapses to prob_ratio = exp(new_logprob) since
                    # old_log_probs are filled with zeros downstream.
                    "logprobs": 1,
                }
                if sampling_params.stop_strings:
                    params_dict["stop"] = sampling_params.stop_strings
                if sampling_params.stop_tokens:
                    params_dict["stop_token_ids"] = sampling_params.stop_tokens

                tasks.append(
                    self._inference_engine_client.sample(
                        prompt_token_ids=prompt,
                        num_samples=1,  # Tinker batches multiple samples separately
                        sampling_params=params_dict,
                    )
                )

            return await asyncio.gather(*tasks, return_exceptions=True)

        # Backend runs in engine subprocess with no event loop
        sample_outputs = asyncio.run(sample_all())

        # Note: sample_outputs may contain Exception objects (from return_exceptions=True)
        # We preserve these to include error messages in responses

        # 4. Aggregate results by request
        return self._aggregate_sample_results(prepared_batch, sample_outputs)

    def _aggregate_sample_results(
        self,
        prepared_batch: types.PreparedSampleBatch,
        sample_outputs: list,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Convert InferenceEngineClient outputs to Tinker format."""
        results = {}

        for request_id, model_id, start_idx, end_idx, needs_prompt_logprobs in prepared_batch.request_batch_slices:
            sequences = []
            has_error = False
            error_msg = None

            for i in range(start_idx, end_idx):
                output = sample_outputs[i]

                # Check if sampling failed (Exception or None)
                if isinstance(output, Exception):
                    has_error = True
                    error_msg = f"Sampling failed for sample {i}: {type(output).__name__}: {str(output)}"
                    logger.error(error_msg)
                    break
                elif output is None:
                    has_error = True
                    error_msg = f"Sampling failed for sample {i}: Unknown error (output is None)"
                    logger.error(error_msg)
                    break

                # Extract tokens and logprobs
                response_tokens = output["response_ids"][0]
                response_logprobs = (output.get("response_logprobs") or [[]])[0]
                stop_reason_raw = output["stop_reasons"][0]

                # Map vLLM stop reason to Tinker format
                stop_reason = "stop" if stop_reason_raw in ["stop", "stop_token"] else "length"

                # Ensure logprobs exist (critical for RL)
                if response_logprobs is None or len(response_logprobs) == 0:
                    logger.warning("No logprobs returned - filling with zeros")
                    response_logprobs = [0.0] * len(response_tokens)

                sequences.append(
                    types.GeneratedSequence(
                        tokens=response_tokens,
                        logprobs=response_logprobs,
                        stop_reason=stop_reason,
                    )
                )

            if has_error:
                results[request_id] = types.ErrorResponse(
                    error=error_msg or "Unknown sampling error",
                    status="error",
                )
            else:
                # Note: prompt_logprobs not supported initially
                if needs_prompt_logprobs:
                    logger.warning("Prompt logprobs requested but not yet supported")

                results[request_id] = types.SampleOutput(
                    sequences=sequences,
                    prompt_logprobs=None,
                )

        return results

    def _validate_model_state(self, model_id: str) -> None:
        """Validate that model exists and is initialized."""
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")
        if self._trainer is None:
            raise RuntimeError("Model not initialized")

    def _create_tar_from_directory(self, source_dir: str, output_path: str) -> None:
        """Create an uncompressed tar archive from a directory."""
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use uncompressed tar - gzip adds 5-10min CPU time on 6-7GB FSDP checkpoints
        with tarfile.open(output_path, "w") as tar:
            tar.add(source_dir, arcname=".")

    def save_checkpoint(self, output_path, model_id: str) -> None:
        """Save full training checkpoint (model + optimizer + scheduler) as tar."""
        self._validate_model_state(model_id)

        ckpt_dir = os.path.join(self.config.trainer.ckpt_path, "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save checkpoint directory (includes optimizer state automatically)
        self._trainer.dispatch.save_checkpoint(model="policy", ckpt_dir=ckpt_dir, tokenizer=self._tokenizer)

        # Create tar archive
        self._create_tar_from_directory(ckpt_dir, output_path)

        logger.info(f"Saved checkpoint for {model_id} to {output_path}")

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        """Load full training checkpoint (model + optimizer + scheduler) from tar or directory."""
        self._validate_model_state(model_id)

        if os.path.isdir(checkpoint_path):
            # Already a directory, load directly
            ckpt_dir = checkpoint_path
            temp_dir = None
        else:
            # Extract tar to temp directory (filter='data' prevents path traversal attacks)
            temp_dir = os.path.join(self.config.trainer.ckpt_path, "temp_checkpoint")
            os.makedirs(temp_dir, exist_ok=True)
            with tarfile.open(checkpoint_path, "r") as tar:
                tar.extractall(temp_dir, filter="data")
            ckpt_dir = temp_dir

        # Load checkpoint (includes optimizer and scheduler states)
        self._trainer.dispatch.load_checkpoint(
            model="policy", ckpt_dir=ckpt_dir, load_optimizer_states=True, load_lr_scheduler_states=True
        )

        logger.info(f"Loaded checkpoint for {model_id} from {checkpoint_path}")

        if temp_dir:
            shutil.rmtree(temp_dir)

    def save_sampler_checkpoint(self, output_path, model_id: str, persist: bool = True) -> None:
        """Sync weights to colocated inference engines and optionally save to disk.

        The NCCL broadcast always runs so inference engines have the latest
        policy weights.  When ``persist`` is False (the common hot-path in RL
        loops) the expensive HuggingFace model export is skipped entirely.
        """
        self._validate_model_state(model_id)

        # Always sync weights to inference engines (in-memory NCCL broadcast)
        if self._inference_engine_client is not None:
            asyncio.run(self._trainer.dispatch.save_weights_for_sampler())
            logger.info(f"Synced weights for {model_id} to inference engines via NCCL")

        if persist:
            
            hf_dir = os.path.join(self.config.trainer.ckpt_path, "hf_model")
            os.makedirs(hf_dir, exist_ok=True)
            hf_dir = os.path.join(hf_dir, f"tinker_sampler_{model_id}")
            os.makedirs(hf_dir, exist_ok=True)
            self._trainer.dispatch.save_hf_model(model="policy", export_dir=hf_dir, tokenizer=self._tokenizer)
            self._create_tar_from_directory(hf_dir, output_path)
            logger.info(f"Saved sampler checkpoint for {model_id} to {output_path}")
        else:
            # Hot path: write a lightweight marker so the engine's checkpoint
            # bookkeeping stays consistent.  Actual weights live in GPU memory.
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with tarfile.open(output_path, "w"):
                pass  # empty tar — marker only
            logger.info(f"Synced weights for {model_id} (disk save skipped)")
