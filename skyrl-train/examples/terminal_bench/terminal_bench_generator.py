import asyncio
import logging
from typing import List, Optional, Dict, Any
from loguru import logger
from pathlib import Path

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from omegaconf import DictConfig

from skyrl_train.generators.base import (
    GeneratorInterface,
    GeneratorInput,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl_train.generators.utils import get_rollout_metrics
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

# Schema-driven Harbor config mapping (still used here for pool-level config reads)
from examples.terminal_bench.harbor_config import HarborConfigBuilder

# Coordinator actor + the per-trajectory output dataclass (moved there so the
# actor and the generator share it without a circular import).
from examples.terminal_bench.rollout_coordinator import (
    RolloutCoordinator,
    TerminalBenchAgentOutput,
)

# Default coordinator-pool sizing. These are the knobs from the perf fix:
#   num_coordinators     -> how many node-local Ray process managers to fan out to
#   cpus_per_coordinator -> CPUs each one reserves (so it is not throttled to 1)
DEFAULT_NUM_COORDINATORS = 4
DEFAULT_CPUS_PER_COORDINATOR = 8


class TerminalBenchGenerator(GeneratorInterface):
    """Drives Harbor terminal-bench rollouts across a pool of node-local Ray
    coordinator actors.

    Previously this class owned a *single* ``QueueOrchestrator`` that ran inside
    the ``num_cpus=1`` SkyRL training driver -- every trial's submit/gather/
    tokenize funnelled through one CPU. Now it owns ``num_coordinators``
    ``RolloutCoordinator`` actors (each ``cpus_per_coordinator`` CPUs, STRICT_SPREAD
    onto distinct nodes), shards each batch across them, and only does the
    cross-batch aggregation (group cascade, metrics, logprob assembly) itself.
    """

    def __init__(
        self,
        generator_cfg: DictConfig,
        terminal_bench_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
    ):
        """
        Args:
            generator_cfg: generator configuration
            terminal_bench_cfg: terminal bench configuration
            inference_engine_client: client for the inference engines (only the
                HTTP base_url is needed by coordinators)
            tokenizer: tokenizer (shipped to each coordinator via cloudpickle)
        """
        self.base_url = (
            f"http://{generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port}"
        )
        self.generator_cfg = generator_cfg
        self.terminal_bench_cfg = terminal_bench_cfg
        self.tokenizer = tokenizer
        self.model_name = generator_cfg.model_name
        self.trials_dir = terminal_bench_cfg.trials_dir

        # Pool-level Harbor config reads (each coordinator builds its own copy too).
        self._harbor_config_builder = HarborConfigBuilder(terminal_bench_cfg)
        harbor_log_level = self._harbor_config_builder.get_log_level(default="WARNING")
        self._configure_harbor_logging(harbor_log_level)
        self.model_info = self._harbor_config_builder.model_info

        # Global concurrency -- divided across coordinators so the *aggregate*
        # concurrency across the pool matches the configured global limit.
        self._n_concurrent_trials = self._harbor_config_builder.get_n_concurrent_trials(default=16)

        # Pre-build config (one-time image warm; sharded across coordinators).
        self._pre_build_enabled = self._harbor_config_builder.get_pre_build_images()
        self._all_task_paths: Optional[List[Path]] = None

        # --- coordinator-pool sizing (the perf-fix knobs) --------------------
        self._num_coordinators = int(
            terminal_bench_cfg.get("num_coordinators", DEFAULT_NUM_COORDINATORS)
        )
        self._cpus_per_coordinator = int(
            terminal_bench_cfg.get("cpus_per_coordinator", DEFAULT_CPUS_PER_COORDINATOR)
        )
        # STRICT_SPREAD pins one coordinator per node; "SPREAD" is a soft fallback
        # for clusters with fewer schedulable nodes than coordinators.
        self._spread_strategy = str(terminal_bench_cfg.get("coordinator_spread", "STRICT_SPREAD"))
        self._per_coordinator_concurrency = max(
            1, self._n_concurrent_trials // max(1, self._num_coordinators)
        )

        # Custom chat template (read once, shipped to coordinators).
        custom_chat_template_path = generator_cfg.engine_init_kwargs.get(
            "custom_chat_template_chat_completion_path", None
        )
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
        else:
            self.custom_chat_template_content = None

        # Coordinator pool state (populated in startup()).
        self._coordinators: List[Any] = []
        self._placement_group = None
        self._eval_session_active: bool = False
        self._permanently_shutdown: bool = False
        # Persistent round-robin cursor across generate() calls. In fully-async
        # mode each of the N generation workers calls generate() with a SINGLE
        # prompt; a within-call `i % n_coord` would send every such call to
        # coordinator 0. The cursor rotates across calls so concurrent single-item
        # batches spread evenly over the pool. Safe to mutate without a lock: all
        # generate() coroutines run in the one driver event loop (cooperative).
        self._dispatch_cursor: int = 0

        logger.info(
            f"TerminalBenchGenerator (pooled) initialized: "
            f"num_coordinators={self._num_coordinators}, "
            f"cpus_per_coordinator={self._cpus_per_coordinator}, "
            f"global n_concurrent_trials={self._n_concurrent_trials} "
            f"(per-coordinator={self._per_coordinator_concurrency}), "
            f"spread={self._spread_strategy}, pre_build={self._pre_build_enabled}"
        )

    def _configure_harbor_logging(self, level: str) -> None:
        log_level = getattr(logging, level.upper(), logging.WARNING)
        for logger_name in (
            "harbor",
            "harbor.trial",
            "harbor.agents",
            "harbor.verifier",
            "harbor.orchestrators",
            "harbor.environments",
            "harbor.utils.logger",
        ):
            logging.getLogger(logger_name).setLevel(log_level)
        logging.getLogger("harbor").setLevel(log_level)

    def set_task_paths(self, task_paths: List[Path]) -> None:
        """Set all dataset task paths for pre-building images at startup.

        Must be called before startup() so the pre-build can be sharded across
        coordinators.
        """
        self._all_task_paths = task_paths
        logger.info(f"Set {len(task_paths)} task paths for pre-build")

    # ----- coordinator pool lifecycle ------------------------------------
    async def startup(self) -> None:
        """Create the coordinator pool (placement group + actors), start each
        coordinator's orchestrator, and run the (sharded) image pre-build.
        """
        # STRICT_SPREAD bundle per coordinator -> one coordinator per node.
        bundles = [{"CPU": self._cpus_per_coordinator} for _ in range(self._num_coordinators)]
        self._placement_group = placement_group(bundles, strategy=self._spread_strategy)
        await self._placement_group.ready()
        logger.info(
            f"Coordinator placement group ready: {self._num_coordinators} bundles "
            f"x {self._cpus_per_coordinator} CPU ({self._spread_strategy})"
        )

        self._coordinators = []
        for i in range(self._num_coordinators):
            coord = RolloutCoordinator.options(
                num_cpus=self._cpus_per_coordinator,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._placement_group,
                    placement_group_bundle_index=i,
                ),
            ).remote(
                coordinator_index=i,
                generator_cfg=self.generator_cfg,
                terminal_bench_cfg=self.terminal_bench_cfg,
                model_name=self.model_name,
                base_url=self.base_url,
                tokenizer=self.tokenizer,
                custom_chat_template_content=self.custom_chat_template_content,
                n_concurrent_trials=self._per_coordinator_concurrency,
            )
            self._coordinators.append(coord)

        # Start all orchestrators in parallel.
        await asyncio.gather(*[c.startup.remote() for c in self._coordinators])

        # Sharded pre-build (each coordinator builds its slice of unique images).
        await self._pre_build_all_images()

        logger.info(
            f"TerminalBenchGenerator startup complete: {len(self._coordinators)} coordinators ready"
        )

    async def _pre_build_all_images(self) -> None:
        if not self._pre_build_enabled:
            return
        if not self._all_task_paths:
            logger.warning("Pre-build enabled but no task paths set; call set_task_paths().")
            return
        # Round-robin task paths across coordinators (builds are idempotent vs
        # the cluster image cache, so any overlap just no-ops).
        shards: List[List[str]] = [[] for _ in self._coordinators]
        for idx, tp in enumerate(self._all_task_paths):
            shards[idx % len(self._coordinators)].append(str(tp))
        counts = await asyncio.gather(
            *[
                self._coordinators[c].pre_build_images.remote(shards[c])
                for c in range(len(self._coordinators))
            ]
        )
        logger.info(f"Pre-build complete across pool: {sum(counts)} images total")

    async def shutdown(self) -> None:
        """Tear down all coordinators and the placement group. Idempotent."""
        self._permanently_shutdown = True
        if self._coordinators:
            try:
                await asyncio.gather(
                    *[c.shutdown.remote() for c in self._coordinators], return_exceptions=True
                )
            except Exception as e:
                logger.warning(f"Error during coordinator shutdown: {e}")
            for c in self._coordinators:
                try:
                    ray.kill(c)
                except Exception:
                    pass
            self._coordinators = []
        if self._placement_group is not None:
            try:
                remove_placement_group(self._placement_group)
            except Exception as e:
                logger.warning(f"Error removing coordinator placement group: {e}")
            self._placement_group = None

    # ----- eval session (fan out to all coordinators) --------------------
    async def start_eval_session(
        self, run_name: str, eval_step: int, val_set_name: Optional[str] = None
    ) -> None:
        await asyncio.gather(
            *[
                c.start_eval_session.remote(run_name, eval_step, val_set_name)
                for c in self._coordinators
            ]
        )
        self._eval_session_active = True
        logger.info(f"Eval session started on {len(self._coordinators)} coordinators (step={eval_step})")

    async def stop_eval_session(self) -> None:
        if not self._coordinators:
            return
        await asyncio.gather(
            *[c.stop_eval_session.remote() for c in self._coordinators], return_exceptions=True
        )
        self._eval_session_active = False
        logger.info("Eval session stopped on all coordinators")

    # ----- helpers --------------------------------------------------------
    def _failed_output(
        self, trajectory_id: TrajectoryID, exception_type: str
    ) -> TerminalBenchAgentOutput:
        return TerminalBenchAgentOutput(
            response_ids=[0],
            reward=0,
            stop_reason="error",
            loss_mask=[0],
            prompt_ids=[0],
            trajectory_id=trajectory_id,
            exclude_from_baseline=True,
            exception_type=exception_type,
        )

    def _create_all_failed_output(
        self, trajectory_ids: List[TrajectoryID], exception_type: str = "PoolFailure"
    ) -> GeneratorOutput:
        n = len(trajectory_ids)
        return {
            "prompt_token_ids": [[0] for _ in range(n)],
            "response_ids": [[0] for _ in range(n)],
            "rewards": [0.0 for _ in range(n)],
            "loss_masks": [[0] for _ in range(n)],
            "stop_reasons": ["error" for _ in range(n)],
            "rollout_metrics": {
                "generate/num_failed_instances": n,
                "generate/num_failed_trajectories": n,
                "generate/num_masked_trajectories": n,
                f"generate/exception_{exception_type}": n,
            },
            "rollout_logprobs": None,
            "exclude_from_baseline": [True for _ in range(n)],
        }

    # ----- the batch entrypoint ------------------------------------------
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Shard the batch across coordinators, gather per-trajectory outputs,
        then do the cross-batch cascade / metrics / logprob assembly.
        """
        if self._permanently_shutdown:
            raise RuntimeError(
                "Generator has been permanently shutdown (training loop exited). "
                "Background generation workers must stop."
            )
        if not self._coordinators:
            raise RuntimeError("Coordinator pool not started. Was startup() called?")

        prompts = input_batch["prompts"]
        trajectory_ids: List[TrajectoryID] = input_batch["trajectory_ids"]
        num_trials = len(prompts)
        is_eval = self._eval_session_active
        mode_str = "eval" if is_eval else "training"
        logger.info(
            f"Starting batch generation for {num_trials} trials across "
            f"{len(self._coordinators)} coordinators (mode={mode_str})"
        )

        # Round-robin shard: prompts/trajectory_ids -> coordinators, remembering
        # each item's original index so we can reassemble in order.
        n_coord = len(self._coordinators)
        shard_prompts: List[List[Any]] = [[] for _ in range(n_coord)]
        shard_tids: List[List[TrajectoryID]] = [[] for _ in range(n_coord)]
        shard_index_map: List[List[int]] = [[] for _ in range(n_coord)]
        cursor = self._dispatch_cursor
        for i in range(num_trials):
            c = (cursor + i) % n_coord
            shard_prompts[c].append(prompts[i])
            shard_tids[c].append(trajectory_ids[i])
            shard_index_map[c].append(i)
        # Advance the cursor so the next generate() call starts where this one
        # left off (keeps concurrent single-item calls balanced across the pool).
        self._dispatch_cursor = (cursor + num_trials) % n_coord

        # Dispatch all non-empty shards concurrently.
        dispatched = [
            (c, self._coordinators[c].run_batch.remote(shard_prompts[c], shard_tids[c]))
            for c in range(n_coord)
            if shard_prompts[c]
        ]
        shard_results = await asyncio.gather(
            *[ref for _, ref in dispatched], return_exceptions=True
        )

        # Reassemble into original order. A dead coordinator -> failed shard.
        all_outputs: List[Optional[TerminalBenchAgentOutput]] = [None] * num_trials
        for (c, _), shard_out in zip(dispatched, shard_results):
            if isinstance(shard_out, Exception):
                logger.error(
                    f"Coordinator {c} run_batch failed: "
                    f"{type(shard_out).__name__}: {shard_out}. Marking shard as failed."
                )
                for idx in shard_index_map[c]:
                    all_outputs[idx] = self._failed_output(
                        trajectory_ids[idx], f"CoordinatorFailure_{type(shard_out).__name__}"
                    )
            else:
                for local_i, idx in enumerate(shard_index_map[c]):
                    all_outputs[idx] = shard_out[local_i]

        # Safety: any unfilled slots (shouldn't happen) become failures.
        for i in range(num_trials):
            if all_outputs[i] is None:
                all_outputs[i] = self._failed_output(trajectory_ids[i], "MissingShardResult")

        # ===== cross-batch aggregation (unchanged from the single-orchestrator
        # implementation; operates on the reassembled, in-order batch) ========
        enable_error_classification = self._harbor_config_builder.get_error_handling_config().get(
            "enable_error_classification", False
        )

        failed_instance_ids = set()
        num_failed_trajectories = 0
        num_masked_trajectories = 0
        successful_outputs: List[TerminalBenchAgentOutput] = []

        for output in all_outputs:
            if output.stop_reason == "error":
                failed_instance_ids.add(output.trajectory_id.instance_id)
                num_failed_trajectories += 1
                if output.exclude_from_baseline:
                    num_masked_trajectories += 1

        if enable_error_classification:
            # RLOO-N mode: preserve exclude_from_baseline flags, don't cascade.
            for output in all_outputs:
                if output.stop_reason == "error":
                    output.response_ids = [0]
                    output.loss_mask = [0]
                    output.prompt_ids = [0]
                    output.reward = 0
                    output.rollout_logprobs = None
                else:
                    successful_outputs.append(output)
        else:
            # Legacy mode: if any trajectory in a group fails, zero the group.
            for output in all_outputs:
                if output.trajectory_id.instance_id in failed_instance_ids:
                    output.response_ids = [0]
                    output.stop_reason = "error"
                    output.loss_mask = [0]
                    output.prompt_ids = [0]
                    output.reward = 0
                    output.rollout_logprobs = None
                    output.exclude_from_baseline = False
                else:
                    successful_outputs.append(output)

        if len(successful_outputs) > 0:
            rollout_metrics = get_rollout_metrics(
                [output.response_ids for output in successful_outputs],
                [output.reward for output in successful_outputs],
            )
            rollout_metrics["generate/trajectories_summarized"] = sum(
                1 for output in successful_outputs if output.summarization_count > 0
            )
            rollout_metrics["generate/trajectories_truncated"] = sum(
                1 for output in successful_outputs if output.stop_reason == "length"
            )
        else:
            rollout_metrics = {}
        rollout_metrics["generate/num_failed_instances"] = len(failed_instance_ids)
        rollout_metrics["generate/num_failed_trajectories"] = num_failed_trajectories
        rollout_metrics["generate/num_masked_trajectories"] = num_masked_trajectories

        exception_counts: Dict[str, int] = {}
        for output in all_outputs:
            if output.exception_type:
                exception_counts[output.exception_type] = (
                    exception_counts.get(output.exception_type, 0) + 1
                )
        if exception_counts:
            logger.info(f"Exception breakdown: {exception_counts}")
            for exc_type, count in exception_counts.items():
                rollout_metrics[f"generate/exception_{exc_type}"] = count

        logger.info(
            f"Batch generation complete: {num_trials - num_failed_trajectories}/{num_trials} "
            f"successful, {len(failed_instance_ids)} failed instances, "
            f"{num_masked_trajectories} masked (excluded from baseline)"
        )

        # TIS logprob assembly (training only).
        rollout_logprobs_list = None
        if not is_eval:
            has_any_logprobs = any(o.rollout_logprobs is not None for o in all_outputs)
            missing_logprobs_count = sum(1 for o in all_outputs if o.rollout_logprobs is None)
            if has_any_logprobs:
                rollout_logprobs_list = []
                for output in all_outputs:
                    if output.rollout_logprobs is not None:
                        rollout_logprobs_list.append(output.rollout_logprobs)
                    else:
                        rollout_logprobs_list.append([0.0] * len(output.response_ids))
                if missing_logprobs_count > 0:
                    logger.warning(
                        f"TIS mode: {missing_logprobs_count}/{num_trials} trajectories missing "
                        f"logprobs. Filled with zeros."
                    )
            elif missing_logprobs_count > 0:
                logger.error(
                    f"TIS mode: ALL {num_trials} trajectories missing logprobs. "
                    f"This batch cannot be used for TIS training."
                )

        generator_output: GeneratorOutput = {
            "prompt_token_ids": [o.prompt_ids for o in all_outputs],
            "response_ids": [o.response_ids for o in all_outputs],
            "rewards": [o.reward for o in all_outputs],
            "loss_masks": [o.loss_mask for o in all_outputs],
            "stop_reasons": [o.stop_reason for o in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs_list,
            "exclude_from_baseline": [o.exclude_from_baseline for o in all_outputs],
        }
        return generator_output
