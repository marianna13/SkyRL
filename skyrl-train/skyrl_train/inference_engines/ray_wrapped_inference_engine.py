import os

import ray
from packaging import version
from ray.actor import ActorHandle
from typing import Any, List, Dict
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group
from loguru import logger

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.inference_engines.utils import get_rendezvous_addr_port


class RayWrappedInferenceEngine(InferenceEngineInterface):
    """
    A thin wrapper around a Ray ActorHandle to another InferenceEngineInterface.
    This class implements the InferenceEngineInterface by delegating calls to the remote actor.
    """

    def __init__(self, inference_engine_actor: ActorHandle):
        self.inference_engine_actor = inference_engine_actor

    def tp_size(self):
        return ray.get(self.inference_engine_actor.tp_size.remote())

    def pp_size(self):
        return ray.get(self.inference_engine_actor.pp_size.remote())

    def dp_size(self):
        return ray.get(self.inference_engine_actor.dp_size.remote())

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        return await self.inference_engine_actor.generate.remote(input_batch=input_batch)

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.wake_up.remote(*args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.sleep.remote(*args, **kwargs)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        return await self.inference_engine_actor.init_weight_update_communicator.remote(
            master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self.inference_engine_actor.update_named_weights.remote(request)

    async def begin_weight_update(self):
        return await self.inference_engine_actor.begin_weight_update.remote()

    async def end_weight_update(self):
        return await self.inference_engine_actor.end_weight_update.remote()

    async def teardown(self):
        return await self.inference_engine_actor.teardown.remote()

    async def reset_prefix_cache(self):
        return await self.inference_engine_actor.reset_prefix_cache.remote()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.chat_completion.remote(request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.completion.remote(request_payload)

    async def abort_generation(self) -> None:
        return await self.inference_engine_actor.abort_generation.remote()


def create_ray_wrapped_inference_engines(
    num_inference_engines: int,
    tensor_parallel_size: int,
    model_dtype: str,
    pretrain: str,
    seed: int,
    vllm_v1_disable_multiproc: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    expert_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    async_engine=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    tokenizer=None,
    backend="vllm",
    sleep_level=2,  # we only set to 1 for unit tests that do not explicitly sync weights or for LoRA
    enable_lora=False,
    max_lora_rank=64,
    max_loras=1,
    fully_sharded_loras=False,
    engine_init_kwargs: Dict[str, Any] = {},
    rope_scaling: Dict[str, Any] = {},
    rope_theta: float | None = None,
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface instances.
    """
    from skyrl_train.utils import (
        get_all_env_variables,
        get_ray_pg_ready_with_timeout,
        get_reordered_bundle_indices,
        ray_noset_visible_devices,
    )
    from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S

    if backend == "vllm":
        import vllm
        from skyrl_train.inference_engines.vllm.vllm_engine import VLLMRayActor, AsyncVLLMRayActor

        # if a dev version is being used, skip the version check
        if "dev" not in vllm.__version__:
            assert version.parse(vllm.__version__) >= version.parse("0.8.3"), "SkyRL-Train only supports vLLM >= 0.8.3"
    elif backend == "sglang":
        # We import SGLang later to avoid importing vllm. See `get_sglang_engine` for more.
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inference_engine_actors = []
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    # NOTE: we use the ray backend for tensor parallel size > 1 or pipeline parallel size > 1 to explicitly manage resource allocation
    # TODO: we should be able to support mp backend by allocating resources at engine level
    distributed_executor_backend = "uni" if (tensor_parallel_size == 1 and pipeline_parallel_size == 1) else "ray"
    data_parallel_backend = "mp"
    use_hybrid_engine = shared_pg is not None
    num_gpus_per_actor = int(tensor_parallel_size == 1 and pipeline_parallel_size == 1)

    if use_hybrid_engine and tensor_parallel_size == 1 and pipeline_parallel_size == 1:
        # Every worker will use 0.2 GPU, so that we can schedule
        # inference and training workers on the same GPUs.
        num_gpus_per_actor = 0.2

    per_engine_gpu_count = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    # When colocated with training (use_hybrid_engine), the shared PG was
    # created with strategy="PACK" in get_colocate_pg — that's a soft hint,
    # so raw bundle indices i*tp..i*tp+tp-1 can land on different physical
    # nodes. policy workers already remap via get_reordered_bundle_indices;
    # without doing the same for engines, an engine's TP NCCL group ends up
    # cross-node, and the per-engine TCPStore handshake stalls at 8+ nodes.
    if use_hybrid_engine:
        from ray.util.placement_group import placement_group_table

        hybrid_reordered_bundles = get_reordered_bundle_indices(shared_pg)
        # Sanity-check that each engine's tp*pp slice lands on a single node —
        # this is what keeps the TP group on NVLink instead of falling back to
        # cross-node NCCL/IB. If the colocate PG didn't pack tightly enough
        # (e.g. fragmented cluster), fail loudly here instead of silently
        # taking the slow path that stalls at scale.
        tp_pp = tensor_parallel_size * pipeline_parallel_size
        bundles_to_node = placement_group_table(shared_pg)["bundles_to_node_id"]
        for eng_i in range(num_inference_engines):
            for dp_r in range(data_parallel_size):
                slice_start = eng_i * per_engine_gpu_count + dp_r * tp_pp
                slice_bundles = hybrid_reordered_bundles[slice_start:slice_start + tp_pp]
                slice_nodes = {bundles_to_node[b] for b in slice_bundles}
                assert len(slice_nodes) == 1, (
                    f"colocate_all engine {eng_i} dp_rank {dp_r} TP={tp_pp} bundles "
                    f"{slice_bundles} span {len(slice_nodes)} nodes ({slice_nodes}); "
                    f"TP group would fall back to cross-node NCCL. The colocate PG "
                    f"didn't pack each TP slice onto one node — check that the PG has "
                    f"exactly num_engines * tp * pp * dp bundles AND that each node "
                    f"can fit a full TP slice."
                )
        logger.info(
            f"[engine bundle reorder] colocate_all PG: {num_inference_engines} engine TP "
            f"slices of {tp_pp} bundles each — all confirmed same-node (NVLink path)"
        )
    else:
        hybrid_reordered_bundles = None

    if not use_hybrid_engine:
        # Create per-node STRICT_PACK placement groups to guarantee TP co-location.
        # Each PG reserves a full node's GPUs as 1-GPU bundles with STRICT_PACK,
        # ensuring all bundles land on the same physical node. Multiple engines
        # share each node PG via contiguous bundle index ranges.
        # Full-node allocation prevents scatter and leaves remaining nodes for policy/ref.
        gpu_nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
        gpus_per_node = min(int(n["Resources"]["GPU"]) for n in gpu_nodes)
        engines_per_node = gpus_per_node // per_engine_gpu_count
        assert engines_per_node > 0, (
            f"Cannot fit engine requiring {per_engine_gpu_count} GPUs on nodes with {gpus_per_node} GPUs"
        )
        num_node_pgs = (num_inference_engines + engines_per_node - 1) // engines_per_node

        node_pgs = []
        for _ in range(num_node_pgs):
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(gpus_per_node)]
            pg = placement_group(bundles, strategy="STRICT_PACK")
            node_pgs.append(pg)
        for pg in node_pgs:
            get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

        # Map each engine to its (pg, base_bundle_index)
        engine_pg_assignments = []
        for pg_idx, pg in enumerate(node_pgs):
            for j in range(engines_per_node):
                engine_idx = pg_idx * engines_per_node + j
                if engine_idx >= num_inference_engines:
                    break
                engine_pg_assignments.append((pg, j * per_engine_gpu_count))

    for i in range(num_inference_engines):
        if use_hybrid_engine:
            cur_pg = shared_pg
            base_pg_index = i * per_engine_gpu_count
        else:
            cur_pg, base_pg_index = engine_pg_assignments[i]

        # Get DP group rendezvous (addr, port) on the same node as DP rank 0 for this engine.
        data_parallel_address, data_parallel_rpc_port = get_rendezvous_addr_port(cur_pg, base_pg_index)

        if backend == "vllm":
            if async_engine:
                actor_class = AsyncVLLMRayActor
            else:
                actor_class = VLLMRayActor

            lora_kwargs = {
                "enable_lora": enable_lora,
                "max_lora_rank": max_lora_rank,
                "max_loras": max_loras,
                "fully_sharded_loras": fully_sharded_loras,
            }

            rope_engine_kwargs = {}
            if rope_scaling:
                rope_engine_kwargs["rope_scaling"] = rope_scaling
                if "max_model_len" not in engine_init_kwargs:
                    rope_factor = rope_scaling.get("factor", None)
                    rope_max_pos = rope_scaling.get("original_max_position_embeddings", None)
                    assert rope_factor is not None, "Please provide rope scaling `factor` to compute model max length"
                    assert (
                        rope_max_pos is not None
                    ), "Please provide rope `original_max_position_embeddings` to compute model max length"
                    rope_engine_kwargs["max_model_len"] = int(rope_factor * rope_max_pos)
            if rope_theta is not None:
                rope_engine_kwargs["rope_theta"] = rope_theta

            # Launch one actor per DP rank
            for dp_rank in range(data_parallel_size):

                # Contiguous TP*PP slice reserved for a single DP rank.
                tp_pp_size = tensor_parallel_size * pipeline_parallel_size
                base_dp_pg_index = base_pg_index + dp_rank * tp_pp_size
                if hybrid_reordered_bundles is not None:
                    # Use reordered bundles so engine i's tp_pp_size bundles
                    # are guaranteed to be on the same physical node (sorted
                    # by node_id, then gpu_id).
                    dp_rank_bundle_indices = [
                        hybrid_reordered_bundles[base_dp_pg_index + j]
                        for j in range(tp_pp_size)
                    ]
                else:
                    dp_rank_bundle_indices = list(
                        range(base_dp_pg_index, base_dp_pg_index + tp_pp_size)
                    )
                dp_rank_bundles = dp_rank_bundle_indices if tp_pp_size > 1 else None
                dp_rank_sched = PlacementGroupSchedulingStrategy(
                    placement_group=cur_pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=dp_rank_bundle_indices[0],
                )

                dp_kwargs = (
                    {
                        "data_parallel_backend": data_parallel_backend,
                        "data_parallel_size": data_parallel_size,
                        "data_parallel_rank": dp_rank,
                        "data_parallel_address": data_parallel_address,
                        "data_parallel_rpc_port": data_parallel_rpc_port,
                    }
                    if data_parallel_size > 1
                    else {}
                )

                engine = actor_class.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=dp_rank_sched,
                ).remote(
                    model=pretrain,
                    enforce_eager=enforce_eager,
                    worker_extension_cls="skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    enable_expert_parallel=expert_parallel_size > 1,
                    distributed_executor_backend=distributed_executor_backend,
                    seed=seed + i * data_parallel_size + dp_rank,
                    enable_prefix_caching=enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    vllm_v1_disable_multiproc=vllm_v1_disable_multiproc,
                    gpu_memory_utilization=gpu_memory_utilization,
                    bundle_indices=dp_rank_bundles,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    enable_sleep_mode=inference_engine_enable_sleep,
                    noset_visible_devices=noset_visible_devices,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                    max_logprobs=1,  # only need chosen-token logprobs
                    **dp_kwargs,
                    **engine_init_kwargs,
                    **lora_kwargs,
                    **rope_engine_kwargs,
                )
                inference_engine_actors.append(engine)

            # SKYRL_ENGINE_INIT_BATCH=N: after every N engines spawned, block
            # until their vLLM constructors complete before spawning the next
            # batch. Ray serializes calls per actor, so the first user method
            # (tp_size) returns only once __init__ finishes — including the
            # internal TP NCCL/TCPStore bringup that contends across engines at
            # 8+ nodes. Default 0 = current parallel-init behavior unchanged.
            _engine_init_batch = int(os.environ.get("SKYRL_ENGINE_INIT_BATCH", "0"))
            if _engine_init_batch > 0 and (i + 1) % _engine_init_batch == 0:
                pending = inference_engine_actors[-_engine_init_batch * data_parallel_size:]
                logger.info(
                    f"[engine init batch] waiting on engines "
                    f"{i + 1 - _engine_init_batch}..{i} __init__ to complete "
                    f"({len(pending)} actor(s))"
                )
                ray.get([a.tp_size.remote() for a in pending])
                logger.info(
                    f"[engine init batch] engines "
                    f"{i + 1 - _engine_init_batch}..{i} ready"
                )
        elif backend == "sglang":
            # NOTE: there is no async / sync engine distinction in SGLang

            bundle_indices = None
            if per_engine_gpu_count > 1:
                bundle_indices = list(range(base_pg_index, base_pg_index + per_engine_gpu_count))

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=cur_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=base_pg_index,
            )

            # NOTE(Charlie): We need `torch.cuda.is_available()` to be True to import SGLang. Otherwise, it requires
            # importing vllm. See https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/layers/quantization/utils.py#L11-L17
            # Similar comment: https://github.com/volcengine/verl/blob/9cc307767b0c787e8f5ef581dac929f7bde044ef/verl/workers/fsdp_workers.py#L520-L527
            @ray.remote
            def get_sglang_engine():
                # A workaround to avoid importing vllm is to give this task a GPU.
                import os

                before_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                from skyrl_train.inference_engines.sglang.sglang_engine import SGLangRayActor

                os.environ["CUDA_VISIBLE_DEVICES"] = before_cuda_visible_devices

                actor_class = SGLangRayActor
                engine = actor_class.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    model_path=pretrain,
                    tp_size=tensor_parallel_size,
                    mem_fraction_static=gpu_memory_utilization,
                    random_seed=seed + i,
                    disable_radix_cache=not enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    max_prefill_tokens=max_num_batched_tokens,
                    max_running_requests=max_num_seqs,
                    # Borrowed from veRL's SGLang rollout
                    mm_attention_backend="fa3",
                    attention_backend="fa3",
                    enable_memory_saver=inference_engine_enable_sleep,
                    # Will be popped before instantiating sgl.Engine
                    distributed_executor_backend=distributed_executor_backend,
                    noset_visible_devices=noset_visible_devices,
                    bundle_indices=bundle_indices,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    tokenizer=tokenizer,
                    **engine_init_kwargs,
                )
                return engine

            engine = ray.get(get_sglang_engine.remote())

            inference_engine_actors.append(engine)

    # Flush any engines past the last SKYRL_ENGINE_INIT_BATCH boundary so all
    # constructors complete before we hand the engines back. No-op when the
    # knob is off — sleep_refs below already forces materialization in that case.
    _engine_init_batch = int(os.environ.get("SKYRL_ENGINE_INIT_BATCH", "0"))
    if _engine_init_batch > 0 and backend == "vllm":
        remainder = num_inference_engines % _engine_init_batch
        if remainder:
            tail = inference_engine_actors[-remainder * data_parallel_size:]
            logger.info(
                f"[engine init batch] flushing final {len(tail)} actor(s)"
            )
            ray.get([a.tp_size.remote() for a in tail])

    engines = [RayWrappedInferenceEngine(actor_handle) for actor_handle in inference_engine_actors]

    if inference_engine_enable_sleep:
        if backend == "vllm":
            # NOTE(shu): set to 1 for LoRA
            sleep_level = 1 if enable_lora else sleep_level
            sleep_refs = [engine.inference_engine_actor.sleep.remote(level=sleep_level) for engine in engines]
        elif backend == "sglang":
            # NOTE(Charlie): we always need to sync weights after waking up: https://github.com/sgl-project/sglang/issues/7939
            assert sleep_level == 2, "SGLang always discards weights, so sleep_level is not applicable."
            sleep_refs = [engine.inference_engine_actor.sleep.remote() for engine in engines]
        ray.get(sleep_refs)

    return engines
