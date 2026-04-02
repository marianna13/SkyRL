import os
from typing import List, Any, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass
from loguru import logger
from http import HTTPStatus
import ray
import torch
import asyncio
import vllm
from types import SimpleNamespace
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    CompletionRequest,
    CompletionResponse,
)
from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.lora.request import LoRARequest
from torch.distributed import destroy_process_group
from skyrl_train.distributed.utils import init_custom_process_group
from uuid import uuid4
import warnings
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.weight_sync import WeightLoader
from skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs
from loguru import logger
from skyrl_train.utils import str_to_torch_dtype, get_tcp_url
import time
from packaging import version


@dataclass
class Logprob:
    logprob: float
    rank: int
    token_id: str


def setup_envvars_for_vllm(kwargs, bundle_indices):
    noset_visible_devices = kwargs.pop("noset_visible_devices")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # TODO(Charlie): may not be needed.
    if kwargs.get("distributed_executor_backend") == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    num_gpus = kwargs.pop("num_gpus")
    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        logger.info(f"creating LLM with bundle_indices={bundle_indices}")


class WorkerWrap:
    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker"""
        return args, kwargs

    def init_weight_update_communicator(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
        override_existing: bool = False,
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        if getattr(self, "_model_update_group", None):
            if override_existing:
                logger.info("Destroying existing model update group")
                destroy_process_group(self._model_update_group)
                self._model_update_group = None
            else:
                warnings.warn(
                    "Detected an existing weights update group. For overriding, use `generator.override_existing_update_group=True`"
                )

        rank = torch.distributed.get_rank() + rank_offset
        logger.info(
            f"torch.distributed.get_rank(): {torch.distributed.get_rank()}, rank_offset: {rank_offset}, rank: {rank}, world_size: {world_size}, group_name: {group_name}"
        )

        self._model_update_group = init_custom_process_group(
            backend=backend,
            init_method=get_tcp_url(master_address, master_port),
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logger.info(
            f"init_weight_update_communicator: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

        # Create receiver now that we have all the state
        self._weight_receiver = VLLMWeightTransferReceiver(
            model_update_group=self._model_update_group,
            model_config=self.model_config,
            device=self.device,
        )

    @staticmethod
    def _apply_fp8_weight_loader_patches():
        """Patch Fp8LinearMethod.process_weights_after_loading to preserve weight_loader.

        Following verl's approach: after FP8 processing creates new Parameter objects,
        copy custom attributes (weight_loader, output_dim, input_dim, subclass_type)
        from the original specialized parameter so weight sync can reload weights.
        """
        import os
        if os.environ.get("SKYRL_FUSE_WEIGHTS", "0") != "1":
            return

        try:
            from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
        except ImportError:
            return

        original_process = Fp8LinearMethod.process_weights_after_loading

        def patched_process(self_method, layer, *args, **kwargs):
            # Save original param attributes before processing
            saved_attrs = {}
            for pname, param in layer.named_parameters():
                attrs = {}
                for attr in ('weight_loader', 'output_dim', 'input_dim',
                             '_output_dim', '_input_dim', 'packed_dim',
                             'packed_factor', 'tp_rank', 'tp_size',
                             'logical_widths', 'output_sizes'):
                    if hasattr(param, attr):
                        attrs[attr] = getattr(param, attr)
                attrs['subclass_type'] = type(param)
                saved_attrs[pname] = attrs

            # Call original process_weights_after_loading
            result = original_process(layer, *args, **kwargs)

            # Restore attributes on new parameters
            for pname, param in layer.named_parameters():
                if pname in saved_attrs:
                    for attr, value in saved_attrs[pname].items():
                        try:
                            setattr(param, attr, value)
                        except (AttributeError, TypeError):
                            pass

            return result

        Fp8LinearMethod.process_weights_after_loading = patched_process

    def begin_weight_update(self) -> None:
        """Start accumulating weights for batched load_weights call.

        When SKYRL_FUSE_WEIGHTS=1, weights are accumulated instead of loaded
        immediately. Call end_weight_update() to flush and apply them all at once
        via model.load_weights(), which handles packed module mapping (qkv_proj, gate_up_proj).
        Weights are stored on CPU to avoid GPU OOM during accumulation.
        """
        self._accumulated_weights = []

    def _is_fp8_model(self):
        """Check if the model uses FP8 quantization."""
        quant_config = getattr(self.model_runner.model, 'quant_config', None)
        if quant_config is None:
            return False
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config
        return isinstance(quant_config, Fp8Config)

    def _quantize_weights_for_fp8(self, weights):
        """Quantize BF16 weights to FP8 before loading into FP8 model.

        Follows verl's approach: quantize each weight tensor to FP8 with
        per-tensor scale, then yield (name, fp8_tensor) and (name_scale, scale).
        Non-linear weights (layernorm, embedding) are passed through as-is.
        """
        import torch
        from vllm._custom_ops import scaled_fp8_quant

        model = self.model_runner.model
        # Build set of parameter names that are FP8 quantized
        # These are the linear layer weights (not biases, not layernorms, not embeddings)
        fp8_param_names = set()
        for name, module in model.named_modules():
            from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
            if hasattr(module, 'quant_method') and isinstance(module.quant_method, Fp8LinearMethod):
                for pname, _ in module.named_parameters():
                    if 'weight' in pname and 'scale' not in pname:
                        full_name = f"{name}.{pname}" if name else pname
                        fp8_param_names.add(full_name)

        for name, tensor in weights:
            # Check if this weight maps to an FP8-quantized parameter
            # The name might be "layers.0.self_attn.q_proj.weight" but the
            # FP8 param is "layers.0.self_attn.qkv_proj.weight"
            # We need to check the ORIGINAL unfused name against the fused params
            is_fp8 = False
            packed_mapping = getattr(model, 'packed_modules_mapping', {})
            # Reverse mapping: q_proj -> qkv_proj, gate_proj -> gate_up_proj
            reverse_map = {}
            for fused, originals in packed_mapping.items():
                for orig in originals:
                    reverse_map[orig] = fused

            # Try to find the FP8 param name
            check_name = name
            parts = name.rsplit('.', 2)
            if len(parts) >= 2:
                module_part = parts[-2]  # e.g. "q_proj"
                if module_part in reverse_map:
                    check_name = name.replace(module_part, reverse_map[module_part])

            if check_name in fp8_param_names or name in fp8_param_names:
                is_fp8 = True

            if is_fp8 and tensor.dtype != torch.float8_e4m3fn:
                # Move to GPU, quantize, move back to CPU
                gpu_tensor = tensor.to(device='cuda', dtype=torch.bfloat16)
                fp8_tensor, scale = scaled_fp8_quant(gpu_tensor)
                yield (name, fp8_tensor.cpu())
                # Yield the scale with the FUSED param name
                scale_name = check_name.replace('.weight', '.weight_scale')
                yield (scale_name, scale.cpu())
                del gpu_tensor, fp8_tensor
            else:
                yield (name, tensor)

    def _restore_param_subclasses(self, model):
        """Temporarily restore param __class__ to subclass_type for weight loading.

        After process_weights_after_loading, params are plain Parameter but have
        subclass_type saved. Restoring __class__ makes weight_loader dispatch work.
        Returns list of (param, original_class) for cleanup.
        """
        patched = []
        for name, param in model.named_parameters():
            subclass_type = getattr(param, 'subclass_type', None)
            if subclass_type is not None and type(param) != subclass_type:
                original_class = type(param)
                param.__class__ = subclass_type
                patched.append((param, original_class))
        return patched

    def _undo_param_subclasses(self, patched):
        """Undo the temporary __class__ patching."""
        for param, original_class in patched:
            param.__class__ = original_class

    def end_weight_update(self) -> None:
        """Flush accumulated weights via model.load_weights().

        For FP8 models: quantizes BF16 weights to FP8 before loading,
        following verl's approach. Also temporarily restores param subclass
        types so weight_loader dispatch works correctly with FP8 params.
        """
        import gc
        if hasattr(self, "_accumulated_weights") and self._accumulated_weights:
            model = self.model_runner.model
            if self._is_fp8_model():
                import torch
                import gc
                from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
                from vllm._custom_ops import scaled_fp8_quant

                # Receiver-side FP8 quantization: BF16 weights arrive via NCCL,
                # fuse stacked params, quantize to FP8, write directly to model.
                weight_index = {name: tensor for name, tensor in self._accumulated_weights}
                stacked = [
                    ("qkv_proj", "q_proj", "q"),
                    ("qkv_proj", "k_proj", "k"),
                    ("qkv_proj", "v_proj", "v"),
                    ("gate_up_proj", "gate_proj", 0),
                    ("gate_up_proj", "up_proj", 1),
                ]

                for mname, module in model.named_modules():
                    if not (hasattr(module, 'quant_method') and isinstance(module.quant_method, Fp8LinearMethod)):
                        continue
                    param = module.weight
                    device = param.device
                    is_stacked = any(mname.endswith(pn) for pn, _, _ in stacked)

                    if is_stacked:
                        shard_list = []
                        for param_name, weight_name, shard_id in stacked:
                            if not mname.endswith(param_name):
                                continue
                            src_name = mname.replace(param_name, weight_name) + ".weight"
                            if src_name in weight_index:
                                shard_list.append(weight_index[src_name])
                        if shard_list:
                            full_bf16 = torch.cat(shard_list, dim=0).to(
                                device=device, dtype=torch.bfloat16, non_blocking=True)
                            torch.cuda.current_stream().synchronize()
                            fp8_full, scale = scaled_fp8_quant(full_bf16)
                            param.data.copy_(fp8_full)
                            if hasattr(module, 'weight_scale'):
                                module.weight_scale.data.copy_(scale.squeeze())
                            del full_bf16, fp8_full, scale, shard_list
                    else:
                        src_name = mname + ".weight"
                        if src_name in weight_index:
                            bf16_w = weight_index[src_name].to(
                                device=device, dtype=torch.bfloat16, non_blocking=True)
                            torch.cuda.current_stream().synchronize()
                            fp8_w, scale = scaled_fp8_quant(bf16_w)
                            param.data.copy_(fp8_w)
                            if hasattr(module, 'weight_scale'):
                                module.weight_scale.data.copy_(scale.squeeze())
                            del bf16_w, fp8_w, scale

                # Load non-FP8 params (layernorms, embeddings)
                params_dict = dict(model.named_parameters())
                for name, tensor in self._accumulated_weights:
                    if name in params_dict:
                        param = params_dict[name]
                        if param.dtype != torch.float8_e4m3fn:
                            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

                del weight_index

                gc.collect()
                torch.cuda.empty_cache()
            else:
                model.load_weights(weights=iter(self._accumulated_weights))
            self._accumulated_weights.clear()
            del self._accumulated_weights
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights using the receiver.

        This method is called via collective_rpc from VLLMWeightLoader.

        When SKYRL_FUSE_WEIGHTS=1 and begin_weight_update() was called,
        weights are accumulated on CPU instead of loaded immediately.

        Args:
            request: Weight update request with names, dtypes, shapes, etc.
        """
        weight_list = []
        for name, tensor in self._weight_receiver.receive_weights(request):
            weight_list.append((name, tensor))

        if hasattr(self, "_accumulated_weights"):
            # Batched mode: move to CPU and accumulate for later flush
            for name, tensor in weight_list:
                self._accumulated_weights.append((name, tensor.cpu()))
            del weight_list
        else:
            # Immediate mode (default): load right away
            self.model_runner.model.load_weights(weights=weight_list)
            for weight in weight_list:
                del weight

    # TODO (sumanthrh): Add destroy process group RPC as a atexit handler to Trainer code.
    def destroy_weights_update_group(self):
        if not self._model_update_group:
            warnings.warn("No model update group to destroy")
            return
        destroy_process_group(self._model_update_group)


class BaseVLLMInferenceEngine(InferenceEngineInterface):
    """Base class containing shared logic between sync and async VLLM engines."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        setup_envvars_for_vllm(kwargs, bundle_indices)
        vllm_v1_disable_multiproc = kwargs.pop("vllm_v1_disable_multiproc", False)
        if vllm_v1_disable_multiproc or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Store common attributes
        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self._pp_size = kwargs.get("pipeline_parallel_size", 1)
        self._dp_size = kwargs.get("data_parallel_size", 1)
        self._is_lora = kwargs.get("enable_lora", False)

        if "rope_scaling" in kwargs:
            kwargs.pop("rope_scaling")
        # Let subclass create the appropriate engine
        self.llm = self._create_engine(*args, **kwargs)

        # Weight loader is created by subclass after engine initialization
        self._weight_loader = None

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        return self._pp_size

    def dp_size(self):
        return self._dp_size

    def _create_engine(self, *args, **kwargs):
        """Abstract method for subclasses to implement engine creation."""
        raise NotImplementedError("Subclasses must implement _create_engine")

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Common prompt preprocessing logic."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert (
            prompts is None and prompt_token_ids is not None
        ), "VLLMInferenceEngine only accepts `prompt_token_ids`, not `prompts`."

        sampling_params = (
            SamplingParams(**request_sampling_params) if request_sampling_params is not None else SamplingParams()
        )

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Common output processing logic."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = []

        for output in outputs:
            # TODO(tgriggs): Support n>1 sampling.
            assert (
                len(output.outputs) == 1
            ), "Each prompt should have only one responses. n>1 sampling is supported by copying prompts."
            resp = output.outputs[0]
            responses.append(resp.text)
            stop_reasons.append(resp.finish_reason)
            response_ids.append(resp.token_ids)
            _logprobs = None
            if resp.logprobs:
                _logprobs = []
                for i, token_logprobs in enumerate(resp.logprobs):
                    token_logprobs: Dict[str, Logprob]
                    token_id = resp.token_ids[i]
                    logprob = token_logprobs[token_id].logprob
                    _logprobs.append(logprob)
                    del token_logprobs
            response_logprobs.append(_logprobs)

        if len(response_logprobs) and response_logprobs[0] is None:
            response_logprobs = None  # hack: assume uniform sampling params

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
        )

    def _get_engine(self):
        """Get the underlying engine for RPC calls."""
        return self.llm.engine if hasattr(self.llm, "engine") else self.llm

    def _is_lora_disk_loading_request(self, request: NamedWeightsUpdateRequest) -> bool:
        """Check if this is a LoRA disk loading request."""
        is_lora = request["names"][0] == "lora_disk_load"
        if is_lora:
            assert request.get("extras") and len(request["extras"]) > 0 and "lora_disk_path" in request["extras"][0], (
                "vLLM LoRA weight update requests must contain the disk load " "path under key `lora_disk_path`"
            )
        return is_lora

    def reset_prefix_cache(self):
        """Reset the prefix cache. Subclasses override for async version."""
        return self.llm.llm_engine.reset_prefix_cache()

    async def abort_generation(self) -> None:
        raise NotImplementedError("Abort generation is only supported for AsyncVLLMInferenceEngine.")


class VLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Synchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=False)

    def _create_engine(self, *args, **kwargs):
        # Pipeline parallelism requires AsyncLLMEngine
        if kwargs.get("pipeline_parallel_size", 1) > 1:
            raise ValueError(
                "Pipeline parallelism is only supported with AsyncVLLMInferenceEngine. "
                "Please set `generator.async_engine=true` in your config."
            )
        return vllm.LLM(*args, **kwargs)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        # Check if LoRA is enabled and create LoRA requests
        lora_requests = None
        if self._is_lora:
            lora_int_ids = list(self.llm.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                batch_size = len(prompt_token_ids)
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path")
                ] * batch_size

        outputs = await asyncio.to_thread(
            self.llm.generate,
            prompts=[TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids],
            sampling_params=sampling_params,
            lora_request=lora_requests,
        )

        return self._postprocess_outputs(outputs)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError()

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError()

    async def wake_up(self, *args: Any, **kwargs: Any):
        await asyncio.to_thread(self.llm.wake_up, tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine().llm_engine
        output_processor = engine.output_processor
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = list(output_processor.request_states.keys())
            await asyncio.to_thread(engine.abort_request, unfinished_request_ids)

        level = 1 if self._is_lora else kwargs.get("level", 2)
        await asyncio.to_thread(self.llm.sleep, level=level)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        engine = self._get_engine()
        return await asyncio.to_thread(
            engine.collective_rpc,
            "init_weight_update_communicator",
            args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
        )

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = self.llm.llm_engine.add_lora(lora_request)
        return result

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        if not len(request["names"]):
            raise ValueError("Update weight request should have at least one entry in 'names'")

        # Handle LoRA disk loading request
        if self._is_lora_disk_loading_request(request):
            lora_path = request["extras"][0]["lora_disk_path"]
            return await self._load_lora_from_disk(lora_path)

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def reset_prefix_cache(self):
        return await asyncio.to_thread(self.llm.llm_engine.reset_prefix_cache)

    async def _destroy_weights_update_group(self):
        engine = self._get_engine()
        return await asyncio.to_thread(engine.collective_rpc, "destroy_weights_update_group")

class V1LoggingStatLoggerFixed(LoggingStatLogger):
    """
    A fixed version of LoggingStatLogger that actually logs during the record method.
    The log method is otherwise not called in the VLLM codebase.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_interval = 5

    def record(self, *args: Any, **kwargs: Any) -> None:
        super().record(*args, **kwargs)
        now = time.monotonic()
        if now - self.last_log_time > self.log_interval:
            self.log()
            self.last_log_time = now

class AsyncVLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Asynchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=True)

    def _create_engine(self, *args, **kwargs):
        openai_kwargs = pop_openai_kwargs(kwargs)
        # TODO (erictang000): potentially enable log requests for a debugging mode
        custom_chat_template_path = kwargs.pop("custom_chat_template_chat_completion_path", None)
        stat_loggers = [V1LoggingStatLoggerFixed]
        engine_args = vllm.AsyncEngineArgs(**kwargs)

        if version.parse(vllm.__version__) >= version.parse("0.10.0"):
            engine_args = vllm.AsyncEngineArgs(enable_log_requests=False, **kwargs)
        else:
            engine_args = vllm.AsyncEngineArgs(disable_log_requests=True, **kwargs)
        engine = vllm.AsyncLLMEngine.from_engine_args(engine_args, stat_loggers=stat_loggers)


        # Adapted from https://github.com/volcengine/verl/blob/e90f18c40aa639cd25092b78a5ff7e2d2508c088/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L327
        model_config = engine.model_config
        model_path = kwargs.get("model")
        # Allow overriding the served model name (similar to vLLM's --served-model-name flag).
        # Useful for Harbor/LiteLLM compatibility where model names must have exactly one '/'.
        # See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        served_model_name = kwargs.get("served_model_name")
        model_name = served_model_name if served_model_name else model_path

        base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(engine, base_model_paths)

        # TODO(Charlie): adding custom chat template for chat completion. Hacky!
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                custom_chat_template_content = f.read()
            logger.info(f"Initializing OpenAIServingChat with custom_chat_template read from: {custom_chat_template_path}")
        else:
            custom_chat_template_content = None

        # TODO(Charlie): revisit kwargs `enable_auto_tools` and `tool_parser` when we need to
        # support OAI-style tool calling; and `request_logger` for better debugging.
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=engine,
            models=models,
            response_role="assistant",
            request_logger=None,
            chat_template=custom_chat_template_content,
            chat_template_content_format="auto",
            **openai_kwargs,
        )

        # TODO(Charlie): revisit kwargs `return_tokens_as_token_ids`,
        # `enable_prompt_tokens_details`, `enable_force_include_usage`.
        self.openai_serving_completion = OpenAIServingCompletion(
            engine_client=engine,
            models=models,
            request_logger=None,
        )
        return engine

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = await self.llm.add_lora(lora_request)
        return result

    async def _collect_outputs(self, prompt_token_ids, request_id: str, sampling_params: SamplingParams):
        """Collect outputs for a single prompt."""
        # Check if LoRA is enabled and create LoRA request
        final_output = None
        lora_request = None

        if self._is_lora:
            lora_int_ids = list(await self.llm.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_request = LoRARequest(
                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path"
                )

        async for request_output in self.llm.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        ):
            final_output = request_output

        return final_output

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using vLLM's async engine."""
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        tasks = []
        for prompt in prompt_token_ids:
            # Schedule the collection of outputs for each prompt.
            # Avoid duplicate request_ids
            request_id = str(uuid4().hex)
            task = asyncio.create_task(self._collect_outputs(prompt, request_id, sampling_params))
            tasks.append(task)
        outputs = await asyncio.gather(*tasks)

        return self._postprocess_outputs(outputs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        await self.llm.wake_up(tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine()
        output_processor = engine.output_processor
        # make sure that the engine is alive
        engine.engine_core.ensure_alive()
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = list(output_processor.request_states.keys())
            await engine.abort(unfinished_request_ids)

        # TODO(team): remove once vllm fixes this
        # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
        await self.reset_prefix_cache()
        level = 1 if self._is_lora else kwargs.get("level", 2)
        await self.llm.sleep(level=level)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        engine = self._get_engine()
        return await engine.collective_rpc(
            "init_weight_update_communicator",
            args=(master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing),
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        if not len(request["names"]):
            raise ValueError("Update weight request should have atleast one entry in 'names'")

        # Check for LoRA disk loading request
        if self._is_lora_disk_loading_request(request):
            lora_path = request["extras"][0]["lora_disk_path"]
            return await self._load_lora_from_disk(lora_path)

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def begin_weight_update(self):
        """Signal engines to start accumulating weights for batched loading."""
        engine = self._get_engine()
        return await engine.collective_rpc("begin_weight_update")

    async def end_weight_update(self):
        """Flush accumulated weights via model.load_weights()."""
        engine = self._get_engine()
        return await engine.collective_rpc("end_weight_update")

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def reset_prefix_cache(self):
        engine = self._get_engine()
        await engine.reset_prefix_cache()

    async def _destroy_weights_update_group(self):
        engine = self._get_engine()
        return await engine.collective_rpc("destroy_weights_update_group")

    # ----------------------------------------
    # Methods for handling OpenAI API requests
    # ----------------------------------------

    async def _handle_openai_request(self, request_payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Handle OpenAI API request."""
        assert endpoint in ["/chat/completions", "/completions"]

        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})

        # TODO(Charlie): Hacky! We are hijacking to update the sampling params.
        # Can we allow Harbor to use customized sampling params?
        body.update({
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
        })

        # 1. Build request
        try:
            if endpoint == "/chat/completions":
                request = ChatCompletionRequest(**body)
            else:
                request = CompletionRequest(**body)
            assert request.stream is False, "Streaming is not supported in SkyRL yet, please set stream to False."
        except Exception as e:
            if version.parse(vllm.__version__) >= version.parse("0.10.0"):
                from vllm.entrypoints.openai.protocol import ErrorInfo

                return ErrorResponse(
                    error=ErrorInfo(
                        message=str(e),
                        type=HTTPStatus.BAD_REQUEST.phrase,
                        code=HTTPStatus.BAD_REQUEST.value,
                    ),
                ).model_dump()
            else:
                return ErrorResponse(
                    message=str(e),
                    type=HTTPStatus.BAD_REQUEST.phrase,
                    code=HTTPStatus.BAD_REQUEST.value,
                ).model_dump()

        # 2. Call vllm engine
        try:
            # Create a minimal request-like object with attributes used by vLLM
            minimal_request = _MinimalRequest(headers)
            if endpoint == "/chat/completions":
                generator = await self.openai_serving_chat.create_chat_completion(request, minimal_request)
                assert isinstance(generator, (ChatCompletionResponse, ErrorResponse))
            else:
                generator = await self.openai_serving_completion.create_completion(request, minimal_request)
                assert isinstance(generator, (CompletionResponse, ErrorResponse))
            return generator.model_dump()

        except Exception as e:
            # Handle it here so we can surface the error from a ray worker.
            if version.parse(vllm.__version__) >= version.parse("0.10.0"):
                from vllm.entrypoints.openai.protocol import ErrorInfo

                return ErrorResponse(
                    error=ErrorInfo(
                        message=str(e),
                        type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                        code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    ),
                ).model_dump()
            else:
                return ErrorResponse(
                    message=str(e),
                    type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                    code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                ).model_dump()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/chat/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_chat.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/chat/completions")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_completion.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/completions")

    async def abort_generation(self) -> None:
        """
        Abort all running and waiting requests, which make the ongoing requests return the
        already-generated tokens with a stop_reason of "abort".
        """
        engine = self._get_engine()
        # Collect all request IDs currently tracked by the scheduler/output processor
        unfinished_request_ids = list(engine.output_processor.request_states.keys())
        if unfinished_request_ids:
            await engine.abort(unfinished_request_ids)
        await engine.reset_prefix_cache()  # avoid KV-cache pollution
        logger.info(f"abort_generation() finished, aborted {len(unfinished_request_ids)} requests")


class _MinimalRequest:
    """
    Minimal request-like object for vLLM's openai_serving_chat and openai_serving_completion.

    We cannot use the original user Request object because it cannot be serialized and hence
    cannot be a ray method argument. Instead we take the original request's headers and
    reconstruct an instance of _MinimalRequest to mimic the FastAPI Request object.

    The fields depend on what vLLM accesses internally.
    """

    def __init__(self, headers):
        self.headers = headers  # Expect a mapping with .get support
        self.state = SimpleNamespace()  # vLLM sets raw_request.state.request_metadata


class VLLMWeightTransferReceiver:
    """Receives weights via broadcast or CUDA IPC for vLLM.

    Handles both transfer strategies based on the request contents.
    Created locally in WorkerWrap with worker-specific state.
    """

    def __init__(self, model_update_group: Any, model_config: Any, device: torch.device) -> None:
        """Initialize the receiver with worker-local state.

        Args:
            model_update_group: Torch process group for weight updates.
            model_config: vLLM model configuration.
            device: CUDA device for this worker.
        """
        self.model_update_group = model_update_group
        self.model_config = model_config
        self.device = device

    def receive_weights(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights and yield (name, tensor) tuples.

        Args:
            request: Weight update request with names, dtypes, shapes, and optionally IPC handles.
        """
        extras = request.get("extras")
        is_ipc = extras and len(extras) > 0 and "ipc_handles" in extras[0]

        if is_ipc:
            yield from self._receive_ipc(request)
        else:
            yield from self._receive_broadcast(request)

    def _receive_broadcast(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via torch.distributed.broadcast."""
        import os
        _fuse = os.environ.get("SKYRL_FUSE_WEIGHTS", "0") == "1"
        for name, dtype_str, shape in zip(request["names"], request["dtypes"], request["shapes"]):
            dtype = str_to_torch_dtype(dtype_str)
            if not _fuse:
                assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
            # Always receive in sender's dtype, load_weights handles conversion
            weight = torch.empty(shape, dtype=dtype, device="cuda")
            torch.distributed.broadcast(weight, 0, group=self.model_update_group)
            yield name, weight

    def _receive_ipc(self, request: NamedWeightsUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via CUDA IPC handles."""
        names = request["names"]
        dtypes = request["dtypes"]
        shapes = request["shapes"]
        sizes = request.get("sizes", [])
        ipc_handles = [extra["ipc_handles"] for extra in request["extras"]]
        packed = request.get("packed", False)

        if packed:
            assert len(ipc_handles) == 1, "packed weight update should receive one ipc handle for all tensors"
            assert len(set(dtypes)) == 1, "packed weight update should have all tensors with the same dtype"
            assert (
                str_to_torch_dtype(dtypes[0]) == self.model_config.dtype
            ), f"mismatch dtype: src {dtypes[0]}, dst {self.model_config.dtype}"
            assert len(sizes) == len(names), "sizes must be provided for packed weight update"
            assert all(isinstance(size, int) for size in sizes), "sizes should be a list of integers"

            cuda_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_device)
            physical_gpu_id = str(props.uuid)

            handle = ipc_handles[0][physical_gpu_id]
            device_id = self.device.index
            func, args = handle
            list_args = list(args)
            list_args[6] = device_id
            packed_tensor = func(*list_args)

            offset = 0
            for name, shape, size in zip(names, shapes, sizes):
                yield name, packed_tensor[offset : offset + size].view(*shape)
                offset += size
        else:
            cuda_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(cuda_device)
            physical_gpu_id = str(props.uuid)
            for name, dtype_str, shape, ipc_handle in zip(names, dtypes, shapes, ipc_handles):
                dtype = str_to_torch_dtype(dtype_str)
                assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

                handle = ipc_handle[physical_gpu_id]
                device_id = self.device.index
                func, args = handle
                list_args = list(args)
                list_args[6] = device_id
                weight = func(*list_args)
                yield name, weight


class VLLMWeightLoader(WeightLoader):
    """Loads weights into vLLM engine, managing RPC coordination.

    This loader encapsulates the collective_rpc calls to workers.
    Workers create VLLMWeightTransferReceiver locally for the actual weight transfer.
    """

    def __init__(self, engine: Any, is_async: bool = False) -> None:
        """Initialize the loader.

        Args:
            engine: The vLLM engine (LLM or AsyncLLMEngine).
            is_async: Whether this is for AsyncVLLMInferenceEngine.
        """
        self._engine = engine.engine if hasattr(engine, "engine") else engine
        self._is_async = is_async

    async def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights by coordinating RPC to workers.

        Sends the request to workers via collective_rpc. Workers create
        the receiver locally and use it to receive and load weights.

        Args:
            request: Weight update request containing names, dtypes, shapes,
                    and optionally IPC handles.
        """
        if self._is_async:
            await self._engine.collective_rpc(
                "load_weights",
                args=(request,),
            )
        else:
            await asyncio.to_thread(
                self._engine.collective_rpc,
                "load_weights",
                args=(request,),
            )


VLLMRayActor = ray.remote(VLLMInferenceEngine)
AsyncVLLMRayActor = ray.remote(AsyncVLLMInferenceEngine)
