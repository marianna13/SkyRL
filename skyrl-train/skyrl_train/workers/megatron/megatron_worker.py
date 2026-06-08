import torch
import torch.nn as nn
import torch.distributed
import ray
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

import os
import traceback
from datetime import timedelta
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
from omegaconf import OmegaConf
from loguru import logger

from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.canonical_lora import CanonicalLoRA
import megatron.core.parallel_state as mpu
from megatron.core.optimizer import DistributedOptimizer, ChainedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from skyrl_train.config.config import MegatronDDPConfig, get_config_as_dict
from skyrl_train.distributed.megatron.optimizer import (
    init_megatron_optim_config,
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
)
from skyrl_train.distributed.dispatch import MeshRank
from megatron.core.transformer.enums import AttnBackend
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.distributed.megatron.megatron_utils import print_model_size, broadcast_object_across_pp_ranks
from skyrl_train.utils.utils import update_model_config, str_to_torch_dtype
from skyrl_train.env_vars import SKYRL_WORKER_NCCL_TIMEOUT_IN_S
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.workers.worker_utils import BatchIterator, reduce_metrics, all_reduce_metrics
from skyrl_train.workers.worker import (
    PolicyWorkerBase,
    RefWorkerBase,
    CriticWorkerBase,
)
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
from skyrl_train.utils.profiler import Profiler
from skyrl_train.weight_sync import WeightExtractor, WeightChunk


class MegatronWeightExtractor(WeightExtractor):
    """Extracts weights from Megatron model-parallel models.

    Uses Megatron's bridge to export weights in HuggingFace format.

    Args:
        bridge: Megatron AutoBridge instance for weight conversion
        actor_module: The actor module to extract weights from
        enable_bucketing: If True, group parameters into size-based buckets for packing
        bucket_size_threshold_GB: Size threshold in GB for bucketing (only used if enable_bucketing=True)
        training_dtype: Training dtype for size calculation (only used if enable_bucketing=True)
    """

    def __init__(
        self,
        bridge,
        actor_module,
        enable_bucketing: bool = False,
        bucket_size_threshold_GB: float = 1.0,
        training_dtype: torch.dtype = torch.bfloat16,
    ):
        self.bridge = bridge
        self.actor_module = actor_module
        self.enable_bucketing = enable_bucketing
        self.bucket_size_threshold_GB = bucket_size_threshold_GB
        self.training_dtype = training_dtype

        # Initialize bucketing if enabled
        if enable_bucketing:
            self._init_param_buckets()
        else:
            self.param_buckets = None

    def _init_param_buckets(self):
        """Initialize parameter buckets for packing."""
        # Get conversion tasks from bridge
        weight_conversion_tasks = self.bridge.get_conversion_tasks(self.actor_module)

        # Calculate size for each parameter
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.training_dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = param.element_size() * param.numel() * tp_size * ep_size * scale

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_object_across_pp_ranks(size_in_bytes)

        for task in weight_conversion_tasks:
            param_info.append(
                (
                    task,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )

        # Group parameters into buckets based on size threshold
        self.param_buckets = [[]]
        curr_size = 0
        for task, size in param_info:
            if curr_size + size > self.bucket_size_threshold_GB * 1024**3:
                self.param_buckets.append([])
                curr_size = 0
            self.param_buckets[-1].append(task)
            curr_size += size

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from Megatron model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or one per bucket if bucketing enabled)
        """
        device = torch.cuda.current_device()

        if not self.enable_bucketing:
            # No bucketing: yield one chunk per parameter
            hf_params_generator = self.bridge.export_hf_weights(
                self.actor_module,
                show_progress=False,
                conversion_tasks=None,
            )

            for name, tensor in hf_params_generator:
                # Move to device and convert dtype
                tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)

                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            # Bucketing mode: iterate over buckets, yield one chunk per bucket
            for bucket in self.param_buckets:
                hf_params_generator = self.bridge.export_hf_weights(
                    self.actor_module,
                    show_progress=False,
                    conversion_tasks=bucket,
                )

                # Collect all parameters in this bucket into one chunk
                names = []
                dtypes_list = []
                shapes = []
                tensors = []

                for name, tensor in hf_params_generator:
                    # Move to device and convert dtype
                    tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)

                    names.append(name)
                    dtypes_list.append(str(dtype))
                    shapes.append(list(tensor.shape))
                    tensors.append(tensor)

                # Yield one chunk containing all parameters in this bucket
                if tensors:
                    yield WeightChunk(
                        names=names,
                        dtypes=dtypes_list,
                        shapes=shapes,
                        tensors=tensors,
                    )


class MegatronWorker:
    def init_configs(
        self,
        model_path,
        megatron_config,
        model_config_kwargs,
        transformer_config_kwargs,
        bf16=True,
        flash_attn=False,
        lora_config=None,
    ):
        """
        Initialize the Megatron-Bridge bridge and provider objects + hf_config and tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        override_config_kwargs = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        override_config_kwargs.update(model_config_kwargs.get("model_config", {}))
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)

        # if flash_attn is enabled, we use flash attention backend, otherwise fall back to fused attention backend
        transformer_config_kwargs = (
            transformer_config_kwargs
            if isinstance(transformer_config_kwargs, dict)
            else OmegaConf.to_container(transformer_config_kwargs, resolve=True)
        )
        
        # Default attention backend
        default_attn_backend = "flash" if flash_attn else "fused"
        attn_backend_str = transformer_config_kwargs.get("attention_backend", default_attn_backend)

        if not self.cfg.trainer.gradient_checkpointing:
            for key in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
                transformer_config_kwargs[key] = None

        # If MEGATRON_CKPT_PATH points to a converted Megatron checkpoint, skip the
        # 240GB-CPU-RAM HF safetensor load and arrange to populate weights from the
        # Megatron shards instead (~15GB per rank, no host-OOM).
        # The Megatron checkpoint must have been produced with the matching tp/ep/pp
        # config; if the env var is set we treat that as the source of truth for
        # weights and use the HF path only for config + tokenizer.
        megatron_ckpt_path = os.environ.get("MEGATRON_CKPT_PATH")
        self._megatron_ckpt_path = (
            megatron_ckpt_path
            if (megatron_ckpt_path
                and os.path.isdir(megatron_ckpt_path)
                and os.path.exists(os.path.join(megatron_ckpt_path, "latest_checkpointed_iteration.txt")))
            else None
        )
        if self._megatron_ckpt_path is not None:
            logger.info(
                f"[megatron_worker] MEGATRON_CKPT_PATH detected: {self._megatron_ckpt_path}; "
                f"using AutoBridge.from_hf_pretrained (LAZY: config + safetensors index "
                f"only, no tensor load) + Megatron checkpoint weight load"
            )
            # from_hf_pretrained builds a LAZY PreTrainedCausalLM: it reads only
            # config.json + model.safetensors.index.json (key-name metadata via
            # SafeTensorsStateSource); weight tensors are NOT materialized at
            # construction (the 120b host-OOM came from to_megatron_provider(
            # load_weights=True), not from this). We need a real
            # PreTrainedCausalLM (not a bare config from from_hf_config) because
            # the Megatron->HF weight-sync (build_conversion_tasks) requires
            # hf_pretrained.state.source.get_all_keys() for weight ordering,
            # and provider_bridge/_megatron_to_hf need .config/.generation_config.
            # load_weights=False still skips the expensive HF->Megatron tensor
            # load, so MEGATRON_CKPT_PATH's purpose (weights from the Megatron
            # dist-ckpt) is preserved — only cheap index.json metadata is read.
            bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
            provider = bridge.to_megatron_provider(load_weights=False)
        else:
            bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
            provider = bridge.to_megatron_provider()
        provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
        provider.pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
        provider.context_parallel_size = megatron_config.context_parallel_size
        provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
        provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
        provider.sequence_parallel = megatron_config.tensor_model_parallel_size > 1
        
        # Map string to AttnBackend enum
        attn_map = {
            "flash": AttnBackend.flash,
            "fused": AttnBackend.fused,
            "unfused": AttnBackend.unfused,
            "local": AttnBackend.local,
            "auto": AttnBackend.auto,
        }
        provider.attention_backend = attn_map.get(attn_backend_str.lower(), AttnBackend.fused)
        
        if provider.attention_backend == AttnBackend.local and self.cfg.trainer.use_sample_packing:
            logger.warning("Megatron's local attention (DotProductAttention) does not support packed sequences. Forcing use_sample_packing=False")
            self.cfg.trainer.use_sample_packing = False

        # Must be True whenever the forward pass can produce variable-length
        # tensors across micro-batches.  remove_left_padding (used when
        # use_sample_packing=False) compresses the seq dimension to the
        # batch-max, which varies per micro-batch.  With PP>1 the pipeline
        # P2P buffers must match the actual tensor sizes, so we need
        # variable_seq_lengths=True in that case as well.
        provider.variable_seq_lengths = True
        provider.masked_softmax_fusion = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "none"

        for k, v in transformer_config_kwargs.items():
            if k == "attention_backend":
                continue
            if isinstance(v, str):
                try:
                    v = str_to_torch_dtype(v)
                except Exception:
                    pass
            setattr(provider, k, v)

        if provider.expert_tensor_parallel_size is None:
            provider.expert_tensor_parallel_size = provider.tensor_model_parallel_size

        provider.finalize()

        self.provider = provider
        self.bridge = bridge

        self.strategy.hf_config = hf_config
        self.tokenizer = tokenizer

    def configure_lora(self, lora_config, lora_type: Optional[str] = "lora"):
        if lora_type == "lora":
            self.lora_cls = LoRA(
                target_modules=(
                    ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
                    if lora_config.target_modules == "all-linear"
                    else lora_config.target_modules
                ),
                dim=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                lora_A_init_method=lora_config.init_method,
                lora_B_init_method="zero",
                exclude_modules=[] if lora_config.exclude_modules is None else lora_config.exclude_modules,
                lora_dtype=torch.bfloat16 if self.cfg.trainer.bf16 else torch.float32,
            )
        elif lora_type == "canonical_lora":
            self.lora_cls = CanonicalLoRA(
                target_modules=(
                    [
                        "linear_q",
                        "linear_k",
                        "linear_v",
                        "linear_proj",
                        "linear_fc1_up",
                        "linear_fc1_gate",
                        "linear_fc2",
                    ]
                    if lora_config.target_modules == "all-linear"
                    else lora_config.target_modules
                ),
                dim=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                lora_A_init_method=lora_config.init_method,
                lora_B_init_method="zero",
                exclude_modules=[] if lora_config.exclude_modules is None else lora_config.exclude_modules,
            )

    def make_megatron_module(
        self,
        wrap_with_ddp: bool = True,
        ddp_config: Optional[Union[MegatronDDPConfig, Dict[str, Any]]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        lora_type: Optional[str] = "lora",
        bf16: bool = True,
    ) -> List[nn.Module]:
        """
        Creates a megatron GPTModel (optionally DDP wrapped) using the bridge.
        """
        from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

        if lora_config is not None:
            self.configure_lora(lora_config, lora_type)

            def lora_pre_wrap_hook(model):
                lora_model = self.lora_cls(model, training=True)
                self.lora_cls.set_params_to_save(lora_model)

                return lora_model

            self.provider.register_pre_wrap_hook(lora_pre_wrap_hook)

        default_ddp_config = DistributedDataParallelConfig()
        if wrap_with_ddp:
            default_ddp_config.use_distributed_optimizer = True
        if ddp_config is not None:
            for k, v in get_config_as_dict(ddp_config).items():
                setattr(default_ddp_config, k, v)
        model = self.provider.provide_distributed_model(
            ddp_config=default_ddp_config, wrap_with_ddp=wrap_with_ddp, bf16=bf16
        )
        # If using a converted Megatron checkpoint, populate weights into the
        # just-built (empty) model from the on-disk Megatron shards. This avoids
        # the 240GB-per-rank HF safetensor load peak that triggers Ray OOM during
        # MegatronRefWorker.init_model. Each rank reads only its TP/EP shard
        # (~15GB) directly from the distcp files.
        if getattr(self, "_megatron_ckpt_path", None) is not None:
            from pathlib import Path as _Path
            ckpt_dir = _Path(self._megatron_ckpt_path)
            iter_dirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")]
            if iter_dirs:
                latest_iter = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "") or -1))
                shard_path = str(latest_iter)
            else:
                shard_path = str(ckpt_dir)
            logger.info(f"[megatron_worker] loading Megatron weights from {shard_path}")
            from megatron.core.dist_checkpointing.serialization import load as _dist_ckpt_load
            from megatron.core.dist_checkpointing.validation import StrictHandling
            # Unwrap any DDP/wrappers to get to the underlying Megatron module
            def _unwrap(m):
                while hasattr(m, "module"):
                    m = m.module
                return m
            # LoRA adapter keys ('.adapter.linear_in.weight' / '.adapter.linear_out.weight')
            # are added by lora_pre_wrap_hook AFTER provide_distributed_model creates the
            # model, but they're not in the on-disk Megatron ckpt (which was saved pre-LoRA).
            # The adapters are zero-initialized anyway, so we filter them from the request.
            def _strip_lora_keys(sd):
                if not isinstance(sd, dict):
                    return sd
                return {
                    k: _strip_lora_keys(v) for k, v in sd.items()
                    if ".adapter." not in str(k)
                }
            unwrapped = [_unwrap(m) for m in model]
            if len(unwrapped) == 1:
                sharded_sd = _strip_lora_keys(unwrapped[0].sharded_state_dict())
                loaded_sd = _dist_ckpt_load(
                    sharded_sd, shard_path, strict=StrictHandling.LOG_UNEXPECTED
                )
                unwrapped[0].load_state_dict(loaded_sd, strict=False)
            else:
                sharded_sd = {
                    f"model{i}": _strip_lora_keys(m.sharded_state_dict())
                    for i, m in enumerate(unwrapped)
                }
                loaded_sd = _dist_ckpt_load(
                    sharded_sd, shard_path, strict=StrictHandling.LOG_UNEXPECTED
                )
                for i, m in enumerate(unwrapped):
                    m.load_state_dict(loaded_sd[f"model{i}"], strict=False)
            logger.info(f"[megatron_worker] Megatron weights loaded successfully")
        return model

    def forward(self, data: TrainingInputBatch):
        """
        Override `Worker.forward` to support passing the full mini batch to the MegatronModelWrapper.forward method.
        """
        # Run in micro batches grouped into a single mini-batch
        micro_bsz = self.cfg.trainer.micro_forward_batch_size_per_gpu
        micro_batches = data.chunk(micro_bsz)

        # Build micro-batch dicts expected by policy.forward_mini_batch
        micro_dicts = []
        device = torch.cuda.current_device()
        for micro in micro_batches:
            micro.to(device)
            sequences = micro["sequences"]
            attention_mask = micro["attention_mask"]
            num_actions = micro.metadata["response_length"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            micro_dicts.append(
                {
                    "sequences": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": num_actions,
                }
            )

        self.model.eval()
        seq_len = micro_dicts[0]["sequences"].shape[1]
        mbs = micro_dicts[0]["sequences"].shape[0]
        with torch.no_grad():
            log_probs = self.model.forward(
                micro_batches=micro_dicts,
                seq_len=seq_len,
                micro_batch_size=mbs,
                temperature=self.cfg.generator.sampling_params.temperature,
            )

        log_probs = log_probs.to("cpu")
        output = TrainingOutputBatch({"output": log_probs})
        output.metadata = data.metadata
        return output

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.bridge,
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )


class MegatronPolicyWorkerBase(MegatronWorker, PolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None
        self.scheduler: OptimizerParamScheduler = None
        self.optimizer: DistributedOptimizer = None
        self.profiler: Profiler = None
        self._is_lora = self.cfg.trainer.policy.model.lora.rank > 0

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.actor_module, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(
            self.actor_module, self.optimizer, non_blocking, backload_optimizer, backload_model
        )

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        if not torch.distributed.is_initialized():
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        # Explicitly wrap torch.distributed.broadcast in torch.no_grad() to avoid a warning in Megatron training where the
        # autograd engine tries to track gradients through the default Torch kernel. This fixes a deprecated behaviour in
        # PyTorch, preventing potential silent errors in future versions.

        if not getattr(torch.distributed, "_skyrl_broadcast_no_grad_patched", False):
            _orig_broadcast = torch.distributed.broadcast

            def _broadcast_no_grad(*args, **kwargs):
                with torch.no_grad():
                    return _orig_broadcast(*args, **kwargs)

            torch.distributed.broadcast = _broadcast_no_grad
            torch.distributed._skyrl_broadcast_no_grad_patched = True

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.policy.megatron_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            seed=self.cfg.trainer.seed,
            is_lora=self._is_lora,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model, optimizer, and scheduler for the policy worker.
        """
        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            self.cfg.trainer.policy.megatron_config,
            self.cfg.trainer.policy.megatron_config.model_config_kwargs,
            self.cfg.trainer.policy.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.trainer.bf16,
            flash_attn=self.cfg.trainer.flash_attn,
        )

        # wrap with DDP for training
        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=True,
            ddp_config=self.cfg.trainer.policy.megatron_config.ddp_config,
            lora_config=self.cfg.trainer.policy.model.lora if self._is_lora else None,
            lora_type=self.cfg.trainer.policy.megatron_config.lora_config.lora_type,
            bf16=self.cfg.trainer.bf16,
        )

        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            snapshot_download(model_path)  # will be no-op if already downloaded
        torch.distributed.barrier()

        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create profiler
        if self.cfg.trainer.policy.megatron_config.torch_profiler_config.enable:
            self.profiler = Profiler(self.cfg.trainer.policy.megatron_config.torch_profiler_config)

        # create optimizer
        optim_config = init_megatron_optim_config(
            self.cfg.trainer.policy.optimizer_config, self.cfg.trainer.policy.megatron_config.optimizer_config_kwargs
        )
        self.optimizer = get_megatron_optimizer(self.actor_module, optim_config)

        # PyTorch AdamW does not allocate optimizer state ('exp_avg',
        # 'exp_avg_sq', 'step') until the first .step() runs. If save_checkpoint
        # fires before any train step (initial save_state, async/sync alike),
        # `optimizer.state[param]` is empty and the dist-checkpoint writer hits
        # KeyError 'exp_avg' inside get_parameter_state_dp_zero. This is GENERAL,
        # not specific to MEGATRON_CKPT_PATH: it also hits the from_hf_pretrained
        # path (e.g. dense Qwen3-8B with MEGATRON_CKPT_PATH unset). The previous
        # `_megatron_ckpt_path is not None` guard made the HF path skip seeding
        # and KeyError anyway. Seeding is idempotent (`not in _state` checks), so
        # run it unconditionally for every trainable param.
        if True:
            # NOTE: never touch ChainedOptimizer.optimizer directly — it's a
            # @property that raises AssertionError (not AttributeError, so
            # getattr's default does NOT save us) when it wraps >1 sub-optimizer.
            # Walk .chained_optimizers and pull each sub-optimizer's underlying
            # torch optimizer via its plain `.optimizer` attribute, guarded.
            def _torch_opt_of(o):
                # Returns the torch optimizer (has .param_groups/.state) or None.
                try:
                    inner = o.optimizer  # plain attr on MegatronOptimizer subclasses
                except (AttributeError, AssertionError):
                    inner = o
                if hasattr(inner, "param_groups") and hasattr(inner, "state"):
                    return inner
                if hasattr(o, "param_groups") and hasattr(o, "state"):
                    return o
                return None
            _chain = getattr(self.optimizer, "chained_optimizers", None)
            _subs = _chain if _chain is not None else [self.optimizer]
            _torch_opts = []
            for _sub in _subs:
                _u = _torch_opt_of(_sub)
                if _u is not None:
                    _torch_opts.append(_u)
            # Do NOT filter on requires_grad. Megatron's DistributedOptimizer
            # holds fp32 *main shard* params (in shard_fp32_from_float16_groups)
            # whose .requires_grad is False even though they ARE the params it
            # steps and that get_parameter_state_dp_zero serializes. Filtering on
            # requires_grad skipped every one of them -> _seeded=0 -> KeyError
            # 'exp_avg' still fired (seen on Qwen3-32B TP=4 from-HF). The inner
            # optimizer only contains params it will update, so seeding state for
            # all of them is correct and harmless.
            _seeded = 0
            _total = 0
            for _opt in _torch_opts:
                for _grp in getattr(_opt, "param_groups", []):
                    for _p in _grp.get("params", []):
                        if _p is None or not hasattr(_p, "data"):
                            continue
                        _total += 1
                        _state = _opt.state[_p]
                        if "exp_avg" not in _state:
                            _state["exp_avg"] = torch.zeros_like(_p.data, memory_format=torch.preserve_format)
                            _seeded += 1
                        if "exp_avg_sq" not in _state:
                            _state["exp_avg_sq"] = torch.zeros_like(_p.data, memory_format=torch.preserve_format)
                        # Do NOT seed a per-param "step". The optimizer is TE
                        # FusedAdam, whose state_dict() does
                        #   for name in state[param]: get_unscaled_state(param, name)
                        # and get_unscaled_state looks up name_to_dtype_map[name],
                        # which only has exp_avg / exp_avg_sq / master_param. TE
                        # keeps `step` at param-group level, NOT per-param, so
                        # injecting state[param]["step"] makes state_dict() raise
                        # KeyError: 'step' in name_to_dtype_map. exp_avg/exp_avg_sq
                        # are the only per-param states get_parameter_state_dp_zero
                        # needs pre-first-step.
            # Unconditional log (not rank-gated) so the seed count is visible on
            # whichever rank performs save_checkpoint.
            logger.info(
                f"[megatron_worker] rank={getattr(self, '_rank', '?')} seeded optimizer "
                f"state for {_seeded}/{_total} params (AdamW exp_avg/exp_avg_sq); "
                f"n_torch_opts={len(_torch_opts)}"
            )

        # create scheduler
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=self.cfg.trainer.policy.optimizer_config,
            num_training_steps=num_training_steps,
        )

        # create worker model
        self.model = MegatronModelWrapper(
            config=self.cfg,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
        )

        # Initialize weight extractor
        # TODO(haochen): Now bucketing is only enabled for the CUDA IPC
        # transfer strategy, we can enable it for other strategies as well.
        from skyrl_train.weight_sync import CudaIpcTransferStrategy

        self.weight_extractor = MegatronWeightExtractor(
            bridge=self.bridge,
            actor_module=self.actor_module,
            enable_bucketing=self._transfer_strategy_cls is CudaIpcTransferStrategy,
            bucket_size_threshold_GB=self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB,
            training_dtype=torch.bfloat16 if self.cfg.trainer.bf16 else torch.float32,
        )

        self.empty_cuda_cache = self.cfg.trainer.policy.megatron_config.empty_cuda_cache

    def forward_backward(
        self,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Perform forward and backward passes for a batch, handling micro-batching internally.

        The batch is split into micro batches based on micro_train_batch_size_per_gpu.
        Megatron Core's forward_backward_func handles gradient accumulation internally.

        Args:
            data: TrainingInputBatch (already DP-sharded by WorkerDispatch/MeshDispatch)
            loss_fn: Optional loss function name (e.g., "cross_entropy", "ppo").
                     If provided, overrides the config's policy_loss_type.
            loss_fn_config: Optional config overrides for the loss function.

        Returns:
            Aggregated metrics dict across all micro batches
        """
        self.model.train()
        for chunk in self.actor_module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

        micro_batch_size = self.cfg.trainer.micro_train_batch_size_per_gpu
        all_metrics = defaultdict(list)

        # Move data to GPU
        data.to(torch.cuda.current_device())

        # Build micro-batch dicts expected by forward_backward_mini_batch
        micro_buffer = []
        for experience in BatchIterator(data, micro_batch_size, drop_last=False):
            sequences = experience.sequences
            attention_mask = experience.attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            micro_buffer.append(
                {
                    "sequences": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": experience.num_actions,
                    "old_action_log_probs": experience.action_log_probs,
                    "base_action_log_probs": experience.base_action_log_probs,
                    "advantages": experience.advantages,
                    "loss_mask": experience.loss_mask,
                    "rollout_action_logprobs": experience.rollout_logprobs,
                    "action_mask": experience.action_mask,
                }
            )

        if not micro_buffer:
            return {}

        seq_len = micro_buffer[0]["sequences"].shape[1]
        micro_bsz = micro_buffer[0]["sequences"].shape[0]

        try:
            metrics_list = self.model.forward_backward_mini_batch(
                micro_batches=micro_buffer,
                seq_len=seq_len,
                micro_batch_size=micro_bsz,
                temperature=self.cfg.generator.sampling_params.temperature,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
        except ValueError as e:
            if "no dot product attention backend is available" in str(e).lower():
                print("\n" + "="*80)
                print("CRITICAL ERROR: TransformerEngine failed to find a compatible attention backend.")
                print("This model (GptOss/OpenThoughts) has specific requirements (sliding window, attention bias, learnable softmax).")
                print("\nPOSSIBLE SOLUTIONS:")
                print("1. Set 'attention_backend: local' in your transformer_config_kwargs to use MCore's implementation.")
                print("2. Set 'export NVTE_FUSED_ATTN=0' in your launch script to allow fallback.")
                print("="*80 + "\n", flush=True)
            raise e
        except Exception as e:
            raise e

        if self.empty_cuda_cache:
            torch.cuda.empty_cache()

        # Track number of micro-batches for metrics
        self._micro_batches_accumulated += len(micro_buffer)

        # Aggregate metrics across micro-batches
        all_loss_fn_outputs = []  # Handle separately from scalar metrics
        for metrics in metrics_list:
            # Extract loss_fn_outputs before reduce_metrics (it's not a scalar metric)
            if "loss_fn_outputs" in metrics:
                all_loss_fn_outputs.extend(metrics.pop("loss_fn_outputs"))
            for k, v in metrics.items():
                all_metrics[k].append(v)

        # Reduce and all-reduce metrics
        status = reduce_metrics(dict(all_metrics))
        status["policy_lr"] = self.optimizer.param_groups[0]["lr"]
        status = all_reduce_metrics(status, self.strategy)

        # Add loss_fn_outputs back (not reduced, kept as list)
        if all_loss_fn_outputs:
            status["loss_fn_outputs"] = all_loss_fn_outputs

        return status

    def optim_step(self) -> Optional[float]:
        """
        Perform optimizer step.

        Note: Unlike FSDP workers, Megatron doesn't need manual gradient scaling here
        because Megatron Core's forward_backward_func handles loss scaling internally.

        Returns:
            The gradient norm (before scaling, after clipping), or None if unavailable.
        """
        import os as _os
        if _os.environ.get("GRAD_DEBUG") == "1":
            try:
                _mod = self.actor_module[0] if isinstance(self.actor_module, (list, tuple)) else self.actor_module
                _seen = set()
                while hasattr(_mod, "module") and id(_mod) not in _seen:
                    _seen.add(id(_mod))
                    _mod = _mod.module

                # Walk model params — Megatron accumulates into .main_grad (fp32), not .grad.
                n_trainable = 0
                n_with_grad = 0
                n_with_main_grad = 0
                grad_sum = 0.0
                main_grad_sum = 0.0
                lora_n_trainable = 0
                lora_n_with_grad = 0
                lora_n_with_main_grad = 0
                lora_grad_sum = 0.0
                lora_main_grad_sum = 0.0
                sample_lora = []
                sample_nonlora = []
                for _n, _p in _mod.named_parameters():
                    _is_lora = (".adapter." in _n) or (".lora_" in _n.lower())
                    if _p.requires_grad:
                        n_trainable += 1
                        if _is_lora:
                            lora_n_trainable += 1
                    # .grad (standard pytorch)
                    if _p.requires_grad and _p.grad is not None:
                        n_with_grad += 1
                        _s = float(_p.grad.detach().abs().sum().item())
                        grad_sum += _s
                        if _is_lora:
                            lora_n_with_grad += 1
                            lora_grad_sum += _s
                    # .main_grad (Megatron fp32 accumulation buffer)
                    _mg = getattr(_p, "main_grad", None)
                    if _p.requires_grad and _mg is not None:
                        n_with_main_grad += 1
                        _s = float(_mg.detach().abs().sum().item())
                        main_grad_sum += _s
                        if _is_lora:
                            lora_n_with_main_grad += 1
                            lora_main_grad_sum += _s
                            if len(sample_lora) < 3:
                                sample_lora.append((_n, tuple(_p.shape), _s))
                        else:
                            if len(sample_nonlora) < 3:
                                sample_nonlora.append((_n, tuple(_p.shape), _s))

                # Match optimizer's internal params to LoRA by NAME (not id, because
                # DistributedOptimizer holds fp32 copies with different id()s).
                lora_names = {n for n, _ in _mod.named_parameters()
                              if (".adapter." in n) or (".lora_" in n.lower())}
                opt_n_groups = 0
                opt_n_params = 0
                opt_lora_param_nameset = 0  # by name lookup via mapping
                try:
                    _opt = self.optimizer
                    _opts = getattr(_opt, "chained_optimizers", None) or [_opt]
                    for _o in _opts:
                        for g in getattr(_o, "param_groups", []):
                            opt_n_groups += 1
                            params_list = g.get("params", [])
                            opt_n_params += len(params_list)
                    # Try to use Megatron's param→name mapping if exposed
                    _pname_map = None
                    for _o in _opts:
                        if hasattr(_o, "param_to_name"):
                            _pname_map = _o.param_to_name
                            break
                        if hasattr(_o, "_param_to_name"):
                            _pname_map = _o._param_to_name
                            break
                    if _pname_map is not None:
                        opt_lora_param_nameset = sum(
                            1 for _p, _nm in _pname_map.items()
                            if (".adapter." in _nm) or (".lora_" in _nm.lower())
                        )
                except Exception as _e:
                    print(f"[GRAD_DEBUG] opt inspect err: {_e}", flush=True)

                print(
                    f"[GRAD_DEBUG] n_trainable={n_trainable} lora_n_trainable={lora_n_trainable} "
                    f"n_with_grad={n_with_grad} lora_n_with_grad={lora_n_with_grad} "
                    f"n_with_main_grad={n_with_main_grad} lora_n_with_main_grad={lora_n_with_main_grad} "
                    f"grad_abs_sum={grad_sum:.6e} main_grad_abs_sum={main_grad_sum:.6e} "
                    f"lora_grad_abs_sum={lora_grad_sum:.6e} lora_main_grad_abs_sum={lora_main_grad_sum:.6e} "
                    f"opt_n_groups={opt_n_groups} opt_n_params={opt_n_params} "
                    f"opt_lora_by_name={opt_lora_param_nameset} "
                    f"sample_lora_main_grad={sample_lora} sample_nonlora_main_grad={sample_nonlora}",
                    flush=True,
                )
            except Exception as _e:
                import traceback as _tb
                print(f"[GRAD_DEBUG] outer err: {_e}\n{_tb.format_exc()}", flush=True)

        grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="actor")

        # Reset counter for next accumulation cycle
        self._micro_batches_accumulated = 0

        if grad_norm is not None:
            grad_norm = grad_norm.detach().cpu().item() if hasattr(grad_norm, "item") else grad_norm
        return grad_norm

    def get_lr(self) -> float:
        """
        Get current learning rate from optimizer.

        Handles both regular optimizers and ChainedOptimizer.
        """
        if isinstance(self.optimizer, ChainedOptimizer):
            return self.optimizer.chained_optimizers[0].param_groups[0]["lr"]
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, learning_rate: float) -> None:
        """
        Set learning rate for the optimizer.

        Handles both regular optimizers and ChainedOptimizer (used with
        distributed optimizer). Updates all param_groups across all
        underlying optimizers.

        Note: This bypasses the scheduler. The next scheduler.step() call
        will override this value unless the scheduler is configured for
        constant LR.
        """
        if isinstance(self.optimizer, ChainedOptimizer):
            # ChainedOptimizer wraps multiple optimizers (e.g., for different param groups)
            for opt in self.optimizer.chained_optimizers:
                for param_group in opt.param_groups:
                    param_group["lr"] = learning_rate
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

    async def _save_lora_adapters_and_sync(self, lora_sync_path, inference_engine_client):
        """Extract LoRA adapter weights from Megatron model and sync to inference engines via disk."""
        import os
        import json
        from safetensors.torch import save_file
        from skyrl_train.weight_sync import LoraLoadRequest

        if torch.distributed.get_rank() == 0:
            os.makedirs(lora_sync_path, exist_ok=True)

            # Use Bridge's export_hf_weights for proper Megatron→HF name mapping,
            # but only keep adapter/lora parameters
            lora_params = {}
            hf_params_generator = self.bridge.export_hf_weights(
                self.actor_module,
                show_progress=False,
                conversion_tasks=None,
            )
            for name, tensor in hf_params_generator:
                if ".adapter." in name or ".lora_" in name:
                    logger.info(f"[LoRA sync] export_hf_weights adapter param: {name} shape={tensor.shape}")
                    # Convert adapter names to PEFT format
                    hf_name = name
                    if ".adapter.linear_in.weight" in name:
                        hf_name = name.replace(".adapter.linear_in.weight", ".lora_A.weight")
                    elif ".adapter.linear_out.weight" in name:
                        hf_name = name.replace(".adapter.linear_out.weight", ".lora_B.weight")
                    elif ".adapter.lora_a.weight" in name:
                        hf_name = name.replace(".adapter.lora_a.weight", ".lora_A.weight")
                    elif ".adapter.lora_b.weight" in name:
                        hf_name = name.replace(".adapter.lora_b.weight", ".lora_B.weight")
                    hf_name = "base_model.model." + hf_name
                    logger.info(f"[LoRA sync] → mapped to: {hf_name}")
                    lora_params[hf_name] = tensor.cpu().to(torch.bfloat16)

            if lora_params:
                save_file(lora_params, os.path.join(lora_sync_path, "adapter_model.safetensors"))

                # Write minimal adapter config
                lora_config = self.cfg.trainer.policy.model.lora
                peft_config = {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": lora_config.rank,
                    "lora_alpha": lora_config.alpha if hasattr(lora_config, 'alpha') else lora_config.rank,
                    "lora_dropout": getattr(lora_config, 'dropout', 0.0),
                    "target_modules": list(lora_config.target_modules) if hasattr(lora_config, 'target_modules') and lora_config.target_modules else ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"],
                    "bias": "none",
                }
                with open(os.path.join(lora_sync_path, "adapter_config.json"), "w") as f:
                    json.dump(peft_config, f, indent=4)

                lora_request = LoraLoadRequest(lora_path=lora_sync_path)
                await inference_engine_client.update_named_weights(lora_request)
            else:
                logger.info("[LoRA sync] No LoRA params found, skipping sync (initial broadcast)")

        torch.distributed.barrier()

    async def broadcast_to_inference_engines(self, inference_engine_client):
        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        # Use LoRA disk sync only when inference model differs from training model (e.g. mxfp4 inference)
        _use_lora_sync = (self._is_lora
                          and self.cfg.trainer.policy.model.lora.lora_sync_path
                          and getattr(self.cfg.generator, 'inference_model_path', None))
        if _use_lora_sync:
            await self._save_lora_adapters_and_sync(
                self.cfg.trainer.policy.model.lora.lora_sync_path,
                inference_engine_client,
            )
        else:
            # Extract and send weights using the sender created at init time
            await self._weight_transfer_sender.send_chunks(self.weight_extractor.extract_weights(generator_dtype))

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronRefWorkerBase(MegatronWorker, RefWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.actor_module, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.actor_module, None, non_blocking)

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        if not torch.distributed.is_initialized():
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.ref.megatron_config,
            optimizer_config=None,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model for the ref worker.
        """
        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            self.cfg.trainer.ref.megatron_config,
            self.cfg.trainer.ref.megatron_config.model_config_kwargs,
            self.cfg.trainer.ref.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.trainer.bf16,
            flash_attn=self.cfg.trainer.flash_attn,
        )

        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=False,
            ddp_config=None,
            bf16=self.cfg.trainer.bf16,
        )

        # download model weights from huggingface (need to be done for ref worker as well, else errors when colocate_all=False)
        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            snapshot_download(model_path)  # will be no-op if already downloaded
        torch.distributed.barrier()

        # load weights
        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create worker model
        self.model = MegatronModelWrapper(config=self.cfg, actor_module=self.actor_module)

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronCriticWorkerBase(MegatronWorker, CriticWorkerBase):
    def __init__(self, **kwargs):
        raise NotImplementedError()


PolicyWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)
RefWorker = ray.remote(num_gpus=1)(MegatronRefWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(MegatronCriticWorkerBase)
