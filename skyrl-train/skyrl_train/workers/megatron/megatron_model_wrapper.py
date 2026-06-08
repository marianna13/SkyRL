from typing import Optional, Callable, List, Dict, Any
from functools import partial
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from megatron.core.pipeline_parallel import get_forward_backward_func
import megatron.core.parallel_state as mpu
from megatron.core.distributed import finalize_model_grads

from skyrl_train.distributed.megatron.model_utils import from_parallel_logits_to_logprobs, vocab_parallel_entropy
from skyrl_train.distributed.megatron.megatron_utils import get_model_config
from skyrl_train.utils.ppo_utils import compute_approx_kl, PolicyLossRegistry
from skyrl_train.utils.torch_utils import masked_mean

from skyrl_train.distributed.megatron.megatron_utils import (
    make_batch_generator,
    preprocess_packed_seqs,
    postprocess_packed_seqs,
    remove_left_padding,
    recover_left_padding,
)


class MegatronModelWrapper:
    def __init__(
        self,
        config,
        actor_module: List[nn.Module],
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_loss_fn: Optional[Callable] = None,
    ):
        self.cfg = config
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.policy_loss_fn = policy_loss_fn
        self.use_sample_packing = self.cfg.trainer.use_sample_packing

        config = get_model_config(self.actor_module[0])
        # This is set to None by default: https://github.com/NVIDIA/Megatron-LM/blob/07b22a05136a3cb08ece05f7de38cf6aeeb165fb/megatron/core/model_parallel_config.py#L95
        # use the build in finalize_model_grads function to all reduce gradients across parallelism dimensions
        config.finalize_model_grads_func = finalize_model_grads

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward-only inference to compute log-probs over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: List of micro-batch dicts with keys: "sequences", "attention_mask", "position_ids",
                           and "num_actions".
            seq_len: Padded sequence length per sample.
            micro_batch_size: Per-micro-batch size.
            temperature: Optional temperature scaling for logits.

        Returns:
            torch.Tensor of concatenated log-probs across micro-batches (valid on pipeline last stage only).
        """
        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, data):
            sequences = data["sequences"]
            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            if temperature != 1.0:
                logits.div_(temperature)

            token_logprobs = from_parallel_logits_to_logprobs(
                logits,
                sequences,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                tp_group=tp_grp,
                inference_only=True,
                cp_group=None,  # we handle cp gathering in `postprocess_packed_seqs`
                chunk_size=None,
            )
            return torch.tensor(0.0, device=token_logprobs.device), {"log_probs": token_logprobs}

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            if self.use_sample_packing:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            outputs = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
                packed_seq_params=packed_seq_params,
            )

            if self.use_sample_packing:
                outputs = postprocess_packed_seqs(
                    outputs,
                    packed_seq_params,
                    attention_mask,
                    micro_batch_size,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )
            else:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

            return outputs, partial(collection_func, data=batch)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=True,
        )

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            log_probs = [o["log_probs"] for o in output]
            log_probs = torch.cat(log_probs, dim=0)
            # take last num_actions tokens per micro; concatenate later
            # Assume all micros have same num_actions
            num_actions = micro_batches[0]["num_actions"]
            log_probs = log_probs[:, -num_actions:]
        else:
            # return dummy tensor for non-last pp stages
            device = micro_batches[0]["sequences"].device
            log_probs = torch.zeros(size=(1, 1), dtype=torch.bfloat16, device=device)
        return log_probs

    def forward_backward_mini_batch(
        self,
        micro_batches: List[dict],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """
        Run forward-backward over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: A list of micro-batch dicts. Each dict must contain keys:
                "sequences", "attention_mask", "position_ids", "num_actions",
                "old_action_log_probs", "base_action_log_probs", "advantages",
                "loss_mask", "rollout_action_logprobs".
            seq_len: Sequence length (tokens) per sample (assumed same across micros after padding).
            micro_batch_size: Micro-batch size per forward pass.
            temperature: Optional temperature for logits scaling.
            loss_fn: Optional loss function name (e.g., "cross_entropy", "ppo").
                     If provided, overrides the config's policy_loss_type.
            loss_fn_config: Optional config overrides for the loss function.

        Returns:
            List[dict]: one metrics dict per micro-batch in order.
        """
        forward_backward_func = get_forward_backward_func()

        # Resolve loss function
        resolved_loss_name = loss_fn if loss_fn is not None else self.cfg.trainer.algorithm.policy_loss_type
        if loss_fn is not None:
            current_loss_fn = PolicyLossRegistry.get(loss_fn)
        else:
            current_loss_fn = self.policy_loss_fn

        # Build config for loss function, applying any overrides
        loss_config = self.cfg.trainer.algorithm
        if loss_fn_config is not None:
            loss_config = OmegaConf.merge(loss_config, OmegaConf.create(loss_fn_config))

        def loss_func(logits, data):
            sequences = data["sequences"]
            num_actions = data["num_actions"]
            old_action_log_probs = data["old_action_log_probs"]
            base_action_log_probs = data["base_action_log_probs"]
            advantages = data["advantages"]
            loss_mask = data["loss_mask"]
            rollout_action_logprobs = data["rollout_action_logprobs"]
            action_mask = data.get("action_mask")

            tp_grp = mpu.get_tensor_model_parallel_group()
            tp_rank = mpu.get_tensor_model_parallel_rank()

            # temperature normalization
            if temperature != 1.0:
                logits.div_(temperature)

            token_logprobs = from_parallel_logits_to_logprobs(
                logits,
                sequences,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                tp_group=tp_grp,
                inference_only=False,
                cp_group=None,  # we handle cp gathering in `postprocess_packed_seqs`
                chunk_size=None,
            )

            action_log_probs = token_logprobs[:, -num_actions:]

            import os as _osgd
            if _osgd.environ.get("GRAD_DEBUG") == "1":
                try:
                    # Check the model instance that was just called: does it actually
                    # contain LoRA adapter modules with requires_grad=True params?
                    _mods = self.actor_module if isinstance(self.actor_module, (list, tuple)) else [self.actor_module]
                    _lora_count = 0
                    _lora_rg_count = 0
                    _sample_lora_names = []
                    for _m in _mods:
                        _u = _m
                        _seen_u = set()
                        while hasattr(_u, "module") and id(_u) not in _seen_u:
                            _seen_u.add(id(_u))
                            _u = _u.module
                        for _pn, _pp in _u.named_parameters():
                            if (".adapter." in _pn) or (".lora_" in _pn.lower()):
                                _lora_count += 1
                                if _pp.requires_grad:
                                    _lora_rg_count += 1
                                if len(_sample_lora_names) < 2:
                                    _sample_lora_names.append(
                                        (_pn, tuple(_pp.shape), bool(_pp.requires_grad))
                                    )
                    print(
                        f"[GRAD_DEBUG_FWD] logits.requires_grad={logits.requires_grad} "
                        f"logits.grad_fn={type(logits.grad_fn).__name__ if logits.grad_fn is not None else None} "
                        f"token_logprobs.requires_grad={token_logprobs.requires_grad} "
                        f"action_log_probs.requires_grad={action_log_probs.requires_grad} "
                        f"torch.is_grad_enabled={torch.is_grad_enabled()} "
                        f"torch.is_inference_mode_enabled={torch.is_inference_mode_enabled() if hasattr(torch, 'is_inference_mode_enabled') else 'N/A'} "
                        f"actor_module_lora_params={_lora_count} lora_requires_grad_true={_lora_rg_count} "
                        f"sample_lora_in_actor={_sample_lora_names} "
                        f"actor_training={getattr(_mods[0], 'training', None)}",
                        flush=True,
                    )
                except Exception as _e:
                    import traceback as _tb
                    print(f"[GRAD_DEBUG_FWD] err: {_e}\n{_tb.format_exc()}", flush=True)

            # policy loss should be calculated based on the selected token logprobs
            policy_loss, loss_metrics = current_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=loss_config,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
            )

            # SFT path: cross_entropy loss (negative log likelihood)
            if resolved_loss_name == "cross_entropy":
                loss = policy_loss

                # Compute elementwise loss for Tinker API (per-token NLL)
                with torch.no_grad():
                    elementwise_loss = -action_log_probs
                    if loss_mask is not None:
                        elementwise_loss = elementwise_loss * loss_mask

                # Build per-sequence loss_fn_outputs
                batch_size = action_log_probs.shape[0]
                loss_fn_outputs = []
                for i in range(batch_size):
                    if action_mask is not None:
                        valid_len = int(action_mask[i].sum().item())
                    elif loss_mask is not None:
                        valid_len = int(loss_mask[i].sum().item())
                    else:
                        valid_len = action_log_probs.shape[1]

                    start = max(action_log_probs.shape[1] - valid_len, 0)
                    loss_fn_outputs.append(
                        {
                            "logprobs": action_log_probs[i, start:].detach().cpu().tolist(),
                            "elementwise_loss": elementwise_loss[i, start:].detach().cpu().tolist(),
                        }
                    )

                metrics = {
                    "loss": loss.detach().item(),
                    "response_length": num_actions,
                    "loss_fn_outputs": loss_fn_outputs,
                }
                return loss, metrics

            # RL path: add optional KL/entropy terms
            # entropy loss
            with torch.set_grad_enabled(loss_config.use_entropy_loss):
                action_logits = logits[:, -num_actions - 1 : -1, :]
                entropy_BS = vocab_parallel_entropy(action_logits)
                entropy = masked_mean(entropy_BS, loss_mask)

            if loss_config.use_entropy_loss:
                entropy_loss_term = entropy * loss_config.entropy_loss_coef
            else:
                entropy_loss_term = torch.tensor(0.0)

            if loss_config.use_kl_loss:
                kl_loss = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    loss_mask=loss_mask,
                    kl_estimator_type=loss_config.kl_estimator_type,
                )
                kl_loss = masked_mean(kl_loss, loss_mask, dim=-1).mean()
            else:
                kl_loss = torch.tensor(0.0)
            kl_loss_term = kl_loss * loss_config.kl_loss_coef

            loss = policy_loss + kl_loss_term - entropy_loss_term

            import os as _os
            if _os.environ.get("GRAD_DEBUG") == "1":
                try:
                    _pl = float(policy_loss.detach().item())
                    _kl = float(kl_loss.detach().item()) if hasattr(kl_loss, "item") else float(kl_loss)
                    _ent = float(entropy.detach().item()) if hasattr(entropy, "item") else float(entropy)
                    _loss_val = float(loss.detach().item())
                    _gfn = type(loss.grad_fn).__name__ if loss.grad_fn is not None else None
                    _requires_grad = bool(loss.requires_grad)
                    _adv_mean = float(advantages.float().mean().item()) if advantages is not None else None
                    _adv_max = float(advantages.float().abs().max().item()) if advantages is not None else None
                    _alp_mean = float(action_log_probs.detach().mean().item())
                    _olp_mean = float(old_action_log_probs.mean().item()) if old_action_log_probs is not None else None
                    _lm_sum = int(loss_mask.sum().item()) if loss_mask is not None else -1
                    _lm_numel = int(loss_mask.numel()) if loss_mask is not None else -1
                    print(
                        f"[GRAD_DEBUG_LOSS] policy_loss={_pl:.6e} kl={_kl:.6e} ent={_ent:.6e} "
                        f"total_loss={_loss_val:.6e} loss.requires_grad={_requires_grad} "
                        f"loss.grad_fn={_gfn} adv_mean={_adv_mean} adv_abs_max={_adv_max} "
                        f"action_logprob_mean={_alp_mean} old_logprob_mean={_olp_mean} "
                        f"loss_mask_sum={_lm_sum}/{_lm_numel}",
                        flush=True,
                    )
                except Exception as _e:
                    import traceback as _tb
                    print(f"[GRAD_DEBUG_LOSS] err: {_e}\n{_tb.format_exc()}", flush=True)

            # Build per-sequence loss_fn_outputs (same schema the SFT path uses above)
            # so the tinker backend (skyrl_train.py) can populate its per-request
            # response without a KeyError. Values are detached + no_grad — pure
            # telemetry, doesn't affect training.
            with torch.no_grad():
                _elementwise_loss_rl = -action_log_probs
                if loss_mask is not None:
                    _elementwise_loss_rl = _elementwise_loss_rl * loss_mask
            _bsz = action_log_probs.shape[0]
            _loss_fn_outputs = []
            for _i in range(_bsz):
                if action_mask is not None:
                    _valid_len = int(action_mask[_i].sum().item())
                elif loss_mask is not None:
                    _valid_len = int(loss_mask[_i].sum().item())
                else:
                    _valid_len = action_log_probs.shape[1]
                _start = max(action_log_probs.shape[1] - _valid_len, 0)
                _loss_fn_outputs.append(
                    {
                        "logprobs": action_log_probs[_i, _start:].detach().cpu().tolist(),
                        "elementwise_loss": _elementwise_loss_rl[_i, _start:].detach().cpu().tolist(),
                    }
                )

            metrics = {
                "final_loss": loss.detach().item(),
                "policy_loss": policy_loss.detach().item(),
                "policy_entropy": entropy.detach().item(),
                "policy_kl": kl_loss.detach().item(),
                "loss": loss.detach().item(),
                "response_length": num_actions,
                "loss_fn_outputs": _loss_fn_outputs,
            }
            for k, v in loss_metrics.items():
                metrics["loss_metrics/" + k] = v
            return loss, metrics

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            sequences = batch["sequences"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            if self.use_sample_packing:
                new_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                new_attention_mask = None
                new_position_ids = None
            else:
                new_sequences, new_attention_mask, new_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            print(f"[DEBUG] foward_step: use_sample_packing={self.use_sample_packing}, packed_seq_params={packed_seq_params}")
            outputs = model(
                new_sequences,
                new_position_ids,
                new_attention_mask,
                packed_seq_params=packed_seq_params,
            )
            import os as _osgdmo
            if _osgdmo.environ.get("GRAD_DEBUG") == "1":
                try:
                    _g = getattr(outputs, "grad_fn", None)
                    _gname = type(_g).__name__ if _g is not None else None
                    _rg = getattr(outputs, "requires_grad", None)
                    _shape = tuple(outputs.shape) if hasattr(outputs, "shape") else None
                    _is_pp_last = mpu.is_pipeline_last_stage(ignore_virtual=True)
                    print(
                        f"[GRAD_DEBUG_MODEL_OUT] outputs.grad_fn={_gname} "
                        f"requires_grad={_rg} shape={_shape} pp_last_stage={_is_pp_last}",
                        flush=True,
                    )
                except Exception as _e:
                    import traceback as _tb
                    print(f"[GRAD_DEBUG_MODEL_OUT] err: {_e}\n{_tb.format_exc()}", flush=True)

            if self.use_sample_packing:
                outputs = postprocess_packed_seqs(
                    outputs,
                    packed_seq_params,
                    attention_mask,
                    micro_batch_size,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )
            else:
                outputs = recover_left_padding(
                    outputs,
                    new_attention_mask,
                    attention_mask,
                    seq_len,
                    post_process=mpu.is_pipeline_last_stage(ignore_virtual=True),
                )

            return outputs, partial(loss_func, data=batch)

        # batch should be a list of micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        metrics_list = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )

        # broadcast metrics to all pp ranks
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            metrics_list = [None] * len(micro_batches)
        with torch.no_grad():
            torch.distributed.broadcast_object_list(
                metrics_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        return metrics_list
