import os
import random
from datetime import timedelta
from typing import List, Union, Optional
from jaxtyping import Float
import gc, os, time, traceback

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import distributed as dist

# Allowlist common optimizer and scheduler classes for safe unpickling in PyTorch 2.6+
if hasattr(torch.serialization, 'add_safe_globals'):
    safe_globals = [
        torch.optim.AdamW,
        torch.optim.Adam,
        torch.optim.SGD,
        torch.optim.lr_scheduler.LRScheduler,
        torch.optim.lr_scheduler.LambdaLR,
        torch.optim.lr_scheduler.StepLR,
        torch.optim.lr_scheduler.MultiStepLR,
        torch.optim.lr_scheduler.ExponentialLR,
        torch.optim.lr_scheduler.CosineAnnealingLR,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        torch.optim.lr_scheduler.OneCycleLR,
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    ]
    try:
        from transformer_engine.pytorch.optimizers import FusedAdam
        safe_globals.append(FusedAdam)
    except ImportError:
        pass
    
    torch.serialization.add_safe_globals(safe_globals)

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl_train.utils.io import io
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
import megatron.core.parallel_state as mpu
from skyrl_train.distributed.megatron.megatron_utils import (
    offload_megatron_model_to_cpu,
    load_megatron_model_to_gpu,
    offload_megatron_optimizer,
    load_megatron_optimizer,
    offload_megatron_grads_to_cpu,
    load_megatron_grads_to_gpu,
)

from megatron.core.dist_checkpointing.strategies import base as ckpt_base
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy,
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from transformers import PreTrainedTokenizer
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def _mem(msg=""):
    # lightweight logging for both CPU & GPU
    rss_gb = None
    try:
        import psutil
        rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    except Exception:
        pass

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        maxa  = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[{time.strftime('%H:%M:%S')}] {msg} | "
              f"GPU alloc={alloc:.2f}G reserv={reserv:.2f}G max={maxa:.2f}G"
              + (f" | RSS={rss_gb:.2f}G" if rss_gb is not None else ""),
              flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class MegatronStrategy(DistributedStrategy):
    """
    The strategy for training with Megatron.
    """

    def __init__(
        self,
        megatron_config,
        optimizer_config=None,
        seed: int = 42,
        is_lora: bool = False,
    ) -> None:
        super().__init__()
        self.megatron_config = megatron_config
        self.optimizer_config = optimizer_config
        self.seed = seed
        self.hf_config = None  # Set by the megatron worker once configs are initialized.
        self.is_lora = is_lora

        # NOTE: Set Megatron dist checkpoint async backend to persistent to avoid `os.fork()`-ing
        # short-lived background workers, which does not work well with Ray.
        ckpt_base.async_calls = AsyncCallsQueue(persistent=True)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.device_count() > 0:
            from megatron.core import tensor_parallel

            tensor_parallel.model_parallel_cuda_manual_seed(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            expert_model_parallel_size=self.megatron_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.megatron_config.expert_tensor_parallel_size,
            use_sharp=False,
            context_parallel_size=self.megatron_config.context_parallel_size,
            nccl_communicator_config_path=None,
        )
        self.set_seed(self.seed)
        self.world_size = dist.get_world_size()

    def offload_to_cpu(
        self, model, optimizer, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True
    ):
        """
        Offload model weights and optimizer to CPU memory.
        """
        try:
            _mem("start offload_to_cpu")
            # Catch early errors from previous kernels
            torch.cuda.synchronize()

            if offload_model:
                _mem("before offload_megatron_model_to_cpu")
                offload_megatron_model_to_cpu(model, pin_memory=pin_memory, non_blocking=non_blocking)
                _mem("after offload_megatron_model_to_cpu")

            if optimizer and offload_optimizer:
                _mem("before offload_megatron_grads_to_cpu")
                offload_megatron_grads_to_cpu(model, non_blocking=non_blocking)
                _mem("after offload_megatron_grads_to_cpu")

                _mem("before offload_megatron_optimizer")
                offload_megatron_optimizer(optimizer)
                _mem("after offload_megatron_optimizer")

            # If there was a CUDA async error earlier, this is where it will surface.
            _mem("before final synchronize")
            torch.cuda.synchronize()
            _mem("after final synchronize")

        except Exception as e:
            print("OFFLOAD FAILED:", repr(e), flush=True)
            traceback.print_exc()
            # This helps catch “silent” CUDA errors:
            if torch.cuda.is_available():
                print("CUDA error state:", torch.cuda.get_device_properties(0), flush=True)
            
            if "unspecified launch failure" in str(e).lower():
                print("\nCRITICAL: CUDA 'unspecified launch failure' detected. "
                      "This is often caused by 'NVTE_FUSED_ATTN=1' on GH200/Hopper systems. "
                      "Try setting 'export NVTE_FUSED_ATTN=0' in your launch script.\n", flush=True)
            
            if "no dot product attention backend is available" in str(e).lower():
                print("\nCRITICAL: TransformerEngine 'No dot product attention backend is available' detected. "
                      "This usually happens when 'NVTE_FUSED_ATTN=1' is set but no compatible fused kernel was found. "
                      "Try setting 'export NVTE_FUSED_ATTN=0' in your launch script to allow fallback to non-fused backends.\n", flush=True)
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _mem("end offload_to_cpu")

    def backload_to_gpu(self, model, optimizer, non_blocking=True, backload_optimizer=True, backload_model=True):
        """Reload model weights back to GPU."""
        if backload_model:
            load_megatron_model_to_gpu(model, non_blocking=non_blocking)
        if optimizer and backload_optimizer:
            load_megatron_grads_to_gpu(model, non_blocking=non_blocking)
            load_megatron_optimizer(optimizer)
        torch.cuda.synchronize()

    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        raise NotImplementedError()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        _, grad_norm, _ = optimizer.step()
        scheduler.step(1)
        optimizer.zero_grad()
        return grad_norm

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        raise NotImplementedError()

    def save_checkpoint(
        self,
        model: MegatronModelWrapper,
        ckpt_dir: str,
        node_local_rank: int,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        # Extract base model.
        model: List[nn.Module] = model.actor_module
        assert len(model) == 1, "Megatron virtual pipeline parallel is not yet supported"
        unwrapped_model = model[0]
        while hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module

        # Create checkpoint directory if it doesn't exist.
        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        # All ranks wait for the checkpoint directory to be created before saving.
        dist.barrier()

        # Collect the sharded state dicts for model and optimizer, and full state dict for the scheduler.
        import time as _time
        _ckpt_t = {}
        _t0 = _time.perf_counter()
        sharded_state_dict = {}
        model_sharded_state_dict = unwrapped_model.sharded_state_dict()
        _ckpt_t["model_sharded_state_dict"] = _time.perf_counter() - _t0
        if not self.is_lora:
            sharded_state_dict["model"] = model_sharded_state_dict
        _t0 = _time.perf_counter()
        if optimizer:
            # Use 'fully_reshardable' format to correctly preserve EDP replica_id
            # for MoE expert params. The deprecated 'fully_sharded_model_space' format
            # discards EDP info via replica_id[:2], causing checkpoint validation
            # failures with EP>1. (verl#4303 / Megatron-Bridge#1564)
            optim_ckpt_metadata = {'distrib_optim_sharding_type': 'fully_reshardable'}
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                model_sharded_state_dict, metadata=optim_ckpt_metadata
            )
        _ckpt_t["optimizer_sharded_state_dict"] = _time.perf_counter() - _t0
        if scheduler:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

        # Save RNG state.
        sharded_state_dict["rng"] = self.get_rng_state()
        _t0 = _time.perf_counter()

        # Save the checkpoint across ranks in parallel.
        save_strategy = get_default_save_sharded_strategy("torch_dist")
        save_strategy = FullyParallelSaveStrategyWrapper(
            save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )

        with io.local_work_dir(ckpt_dir) as work_dir:
            # TODO(tgriggs): Support configurable async saves.
            async_save_request = dist_checkpointing.save(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=work_dir,
                sharded_strategy=save_strategy,
                async_sharded_save=False,
                validate_access_integrity=True,
            )
            assert async_save_request is None, "Async save is not yet supported for Megatron"

            # Only global rank 0 saves the Huggingface config and tokenizer.
            if self.is_rank_0():
                hf_dir = os.path.join(work_dir, "huggingface")
                self.save_hf_configs(self.hf_config, hf_dir, tokenizer)

        _ckpt_t["dist_checkpointing_save"] = _time.perf_counter() - _t0
        _t0 = _time.perf_counter()
        if self.is_lora:
            self._save_lora_adapters(unwrapped_model, ckpt_dir)
        _ckpt_t["save_lora_adapters"] = _time.perf_counter() - _t0

        dist.barrier()
        ckpt_base.async_calls.close()
        ckpt_base.async_calls = AsyncCallsQueue(persistent=True)
        self.print(
            "[ckpt-timing] " + " ".join(f"{k}={v:.1f}s" for k, v in _ckpt_t.items())
            + f" | is_lora={self.is_lora}"
        )
        self.print(f"Checkpoint successfully saved to {ckpt_dir}")

    def _get_rank_path(self, ckpt_dir, dp_rank=None):
        tp_rank = mpu.get_tensor_model_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        cp_rank = mpu.get_context_parallel_rank()
        if dp_rank is None:
            dp_rank = mpu.get_data_parallel_rank()
        ep_rank = mpu.get_expert_model_parallel_rank()
        etp_rank = mpu.get_expert_tensor_parallel_rank()

        return os.path.join(
            ckpt_dir, f"adapter_tp{tp_rank}_pp{pp_rank}_cp{cp_rank}_dp{dp_rank}_ep{ep_rank}_etp{etp_rank}.pt"
        )

    def _save_lora_adapters(self, model, ckpt_dir):
        """Save LoRA adapters to checkpoint."""
        if not self.is_lora:
            return

        assert isinstance(model, nn.Module), "Model must be a nn.Module"

        model_state_dict = {}
        for name, param in model.named_parameters():
            if ".adapter" in name.lower():
                model_state_dict[name] = param.data

        with io.local_work_dir(ckpt_dir) as work_dir:
            adapter_path = self._get_rank_path(work_dir)
            torch.save({"model_state_dict": model_state_dict}, adapter_path)
            self.print(f"Saved {len(model_state_dict)} LoRA adapter parameters to {adapter_path}")

    def load_checkpoint(
        self,
        model: MegatronModelWrapper,
        ckpt_dir: str,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        load_module_strict: bool = True,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ):
        if not ckpt_dir or not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        try:
            _mem(f"start load_checkpoint from {ckpt_dir}")
            torch.cuda.synchronize()
            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()

            # Extract base model.
            model_list: List[nn.Module] = model.actor_module
            assert len(model_list) == 1, "Megatron virtual pipeline parallel is not yet supported"
            unwrapped_model = model_list[0]
            while hasattr(unwrapped_model, "module"):
                unwrapped_model = unwrapped_model.module

            # Extract sharded state dicts.
            _mem("before extracting sharded state dict")
            sharded_state_dict = {}
            model_sharded_state_dict = unwrapped_model.sharded_state_dict()
            if not self.is_lora:
                sharded_state_dict["model"] = model_sharded_state_dict
            
            if optimizer and load_optimizer_states:
                _mem("before optimizer.sharded_state_dict")
                # Must match save format: 'fully_reshardable' preserves EDP replica_id
                optim_ckpt_metadata = {'distrib_optim_sharding_type': 'fully_reshardable'}
                sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                    model_sharded_state_dict, is_loading=True, metadata=optim_ckpt_metadata
                )
            
            if scheduler and load_lr_scheduler_states:
                sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

            _mem(f"after extracting sharded state dict (keys: {list(sharded_state_dict.keys())}), before dist_checkpointing.load")
            
            with io.local_read_dir(ckpt_dir) as read_dir:
                # Load the checkpoint in parallel.
                load_strategy = get_default_load_sharded_strategy(read_dir)
                load_strategy = FullyParallelLoadStrategyWrapper(
                    load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
                )
                state_dict = dist_checkpointing.load(
                    sharded_state_dict=sharded_state_dict, checkpoint_dir=read_dir, sharded_strategy=load_strategy
                )
            
            _mem("after dist_checkpointing.load")
            torch.cuda.synchronize()
            dist.barrier()

            if not self.is_lora:
                # Load the model, optimizer, and scheduler state dicts.
                assert (
                    "model" in state_dict
                ), f"Model state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
                model_list[0].load_state_dict(state_dict["model"], strict=load_module_strict)
                self.print("Loaded model state dict.")
            else:
                _mem("before _load_lora_adapters")
                self._load_lora_adapters(unwrapped_model, ckpt_dir)
                _mem("after _load_lora_adapters")

            if optimizer and load_optimizer_states:
                assert (
                    "optimizer" in state_dict
                ), f"Optimizer state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
                optimizer.load_state_dict(state_dict["optimizer"])
                self.print("Loaded optimizer state dict.")

            if scheduler and load_lr_scheduler_states:
                assert (
                    "lr_scheduler" in state_dict
                ), f"LR scheduler state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
                scheduler.load_state_dict(state_dict["lr_scheduler"])
                self.print("Loaded LR scheduler state dict.")

            # Load RNG state, if present.
            if "rng" in state_dict:
                self.load_rng_state(state_dict["rng"])

            _mem("end load_checkpoint")
            torch.cuda.synchronize()
            dist.barrier()
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.print(f"ERROR in load_checkpoint: {repr(e)}")
            traceback.print_exc()
            
            if "no dot product attention backend is available" in str(e).lower():
                self.print("\nCRITICAL: TransformerEngine 'No dot product attention backend is available' detected during model loading. "
                      "This usually happens when 'NVTE_FUSED_ATTN=1' is set but the model configuration or hardware is incompatible. "
                      "Try setting 'export NVTE_FUSED_ATTN=0' in your launch script to allow fallback to non-fused backends.\n")
            raise e

        return ckpt_dir, {}

    def _load_lora_adapters(self, model, ckpt_dir):
        """Load LoRA adapters from checkpoint."""
        # TODO (erictang000): Update this logic once LoRA checkpointing is upstreamed to Megatron-Bridge
        if not self.is_lora:
            return

        assert isinstance(model, nn.Module), "Model must be a nn.Module"

        with io.local_read_dir(ckpt_dir) as read_dir:
            adapter_path = self._get_rank_path(read_dir)

            # Fallback to dp0 if current dp_rank file does not exist
            if not os.path.exists(adapter_path):
                dp_rank = mpu.get_data_parallel_rank()
                if dp_rank > 0:
                    adapter_path_dp0 = self._get_rank_path(read_dir, dp_rank=0)
                    if os.path.exists(adapter_path_dp0):
                        self.print(
                            f"LoRA adapter for dp_rank {dp_rank} not found at {adapter_path}. "
                            f"Falling back to dp_rank 0 at {adapter_path_dp0}"
                        )
                        adapter_path = adapter_path_dp0

            state_dict = torch.load(adapter_path, map_location="cpu", weights_only=False)
            _, unexpected = model.load_state_dict(state_dict["model_state_dict"], strict=False)
            if len(unexpected) > 0:
                raise ValueError(f"Unexpected keys in LoRA adapter state dict: {unexpected}")
            self.print(f"Loaded {len(state_dict['model_state_dict'])} LoRA adapters from {adapter_path}.")

    def save_hf_model(self, bridge, model: MegatronModelWrapper, output_dir: str, tokenizer=None, **kwargs) -> None:
        # Create checkpoint directory if it doesn't exist.
        if self.is_rank_0():
            io.makedirs(output_dir, exist_ok=True)
        dist.barrier()

        # All ranks call into bridge.
        with io.local_work_dir(output_dir) as work_dir:
            bridge.save_hf_weights(model.actor_module, work_dir, strict=False)
            self.print(f"Successfully saved HF safetensors model to {output_dir}")

            # Only rank 0 saves the Huggingface config and tokenizer.
            if self.is_rank_0():
                self.save_hf_configs(self.hf_config, work_dir, tokenizer)
                self.print(f"Successfully saved HF config and tokenizer to {output_dir}")

        dist.barrier()
