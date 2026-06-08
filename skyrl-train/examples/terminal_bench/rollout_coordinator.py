"""Ray-actor rollout coordinators for Harbor terminal-bench rollouts.

Background
----------
Previously, all Harbor rollouts for a batch were driven by a *single* shared
``QueueOrchestrator`` living inside the ``TerminalBenchGenerator``, which itself
runs inside the SkyRL training driver -- a ``@ray.remote(num_cpus=1)`` task
(see ``examples/terminal_bench/entrypoints/main_tbench.py``). Every trial's
submit / gather / hook / result-tokenization funnelled through that one CPU,
throttling the whole pipeline regardless of how many sandboxes were available.

This module fans that work out across ``num_coordinators`` Ray actors, each
requesting ``cpus_per_coordinator`` CPUs and (via a STRICT_SPREAD placement
group) pinned to its own node. Each coordinator owns its *own* QueueOrchestrator
and does its *own* trial-config building, submission, gathering and result
tokenization. The generator becomes a thin client that shards each batch across
the pool and reassembles the per-trajectory outputs.

The cross-batch logic (group-level failure cascade, metrics aggregation,
TIS logprob assembly, final ``GeneratorOutput`` packing) stays in the generator
because it needs the whole batch at once -- only the per-trial work is fanned
out.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from uuid import uuid4
from pathlib import Path

import ray
from loguru import logger
from omegaconf import DictConfig

# Harbor orchestrator and trial imports
from harbor.orchestrators.queue import QueueOrchestrator
from harbor.orchestrators.base import OrchestratorEvent
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.callbacks import create_rollback_hook

from skyrl_train.generators.base import TrajectoryID
from skyrl_train.generators.utils import (
    get_response_ids_and_loss_mask_from_messages,
    extract_logprobs_from_rollout_details,
)
from skyrl_train.utils.reward_shaping import shape_reward_from_output

# Schema-driven Harbor config mapping
from examples.terminal_bench.harbor_config import HarborConfigBuilder

# Maximum restart attempts for orchestrator recovery (per coordinator)
MAX_ORCHESTRATOR_RESTART_ATTEMPTS = 3


@dataclass
class TerminalBenchAgentOutput:
    """Per-trajectory rollout result. Picklable so it ships back from the actor.

    NOTE: moved here (from terminal_bench_generator.py) so both the coordinator
    actor and the generator can import it without a circular dependency.
    """

    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    trajectory_id: TrajectoryID
    summarization_count: Optional[int] = None
    rollout_logprobs: Optional[List[float]] = None
    # For RLOO-N: True = exclude from baseline (infrastructure failure)
    # False = include in baseline (agent failure or success)
    exclude_from_baseline: bool = False
    # Store the exception type for debugging/logging
    exception_type: Optional[str] = None


class _CoordinatorImpl:
    """Owns a single QueueOrchestrator and turns trial requests into outputs.

    This is everything that used to live (per-orchestrator) in
    ``TerminalBenchGenerator``, lifted out so it can run inside a Ray actor on a
    dedicated node. Lifecycle and result-processing are mostly verbatim from the
    original generator.
    """

    def __init__(
        self,
        coordinator_index: int,
        generator_cfg: DictConfig,
        terminal_bench_cfg: DictConfig,
        model_name: str,
        base_url: str,
        tokenizer,
        custom_chat_template_content: Optional[str],
        n_concurrent_trials: int,
    ):
        self.coordinator_index = coordinator_index
        self.generator_cfg = generator_cfg
        self.model_name = model_name
        self.base_url = base_url
        self.tokenizer = tokenizer
        self.custom_chat_template_content = custom_chat_template_content

        # Per-coordinator concurrency. The generator divides the *global*
        # n_concurrent_trials by num_coordinators before handing it down, so the
        # aggregate concurrency across the pool matches the configured global limit.
        self._n_concurrent_trials = max(1, int(n_concurrent_trials))

        # Rebuild the Harbor config machinery locally (cheap; avoids shipping
        # builder state across Ray and keeps each actor self-contained).
        self._harbor_config_builder = HarborConfigBuilder(terminal_bench_cfg)
        harbor_log_level = self._harbor_config_builder.get_log_level(default="WARNING")
        self._configure_harbor_logging(harbor_log_level)

        self._retry_config = self._harbor_config_builder.build_retry_config()
        self._reward_shaping_config = self._harbor_config_builder.get_reward_shaping_config()
        self._error_handling_config = self._harbor_config_builder.get_error_handling_config()
        self._eval_timeout_override_sec = self._harbor_config_builder.get_eval_timeout_override_sec(
            default=900
        )

        # Namespace this coordinator's trials dir so concurrent coordinators
        # never write to the same path.
        base_trials_dir = terminal_bench_cfg.trials_dir
        self.trials_dir = (
            str(Path(base_trials_dir) / f"coord{coordinator_index}")
            if base_trials_dir
            else base_trials_dir
        )

        # Training orchestrator state
        self._orchestrator: Optional[QueueOrchestrator] = None
        self._orchestrator_lock: Optional[asyncio.Lock] = None
        self._orchestrator_started: bool = False
        self._orchestrator_restart_count: int = 0
        self._permanently_shutdown: bool = False

        # Eval session state (separate orchestrator)
        self._eval_orchestrator: Optional[QueueOrchestrator] = None
        self._eval_orchestrator_lock: Optional[asyncio.Lock] = None
        self._eval_session_active: bool = False
        self._eval_session_name: Optional[str] = None
        self._eval_trials_dir: Optional[str] = None

        logger.info(
            f"[coord {coordinator_index}] initialized "
            f"(n_concurrent_trials={self._n_concurrent_trials}, trials_dir={self.trials_dir})"
        )

    # ----- logging --------------------------------------------------------
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

    # ----- pre-build (sharded slice of the dataset) -----------------------
    async def pre_build_images(self, task_paths: List[str]) -> int:
        """Pre-build this coordinator's slice of unique images (Beam only).

        The generator shards ``set_task_paths`` across coordinators so the
        build stampede is itself parallelized across nodes. Builds are
        idempotent against the cluster image cache, so overlap is harmless.
        """
        env_type = self._harbor_config_builder.get_environment_type()
        if env_type != "beam":
            logger.info(f"[coord {self.coordinator_index}] pre-build skipped (env={env_type})")
            return 0

        seen: set = set()
        env_dirs: list = []
        for tp in task_paths:
            env_dir = (Path(tp) / "environment").resolve()
            if env_dir not in seen and env_dir.exists() and (env_dir / "Dockerfile").exists():
                seen.add(env_dir)
                env_dirs.append(env_dir)
        if not env_dirs:
            return 0

        max_concurrent = self._harbor_config_builder.get_pre_build_max_concurrent()
        logger.info(
            f"[coord {self.coordinator_index}] pre-building {len(env_dirs)} images "
            f"(max_concurrent={max_concurrent})..."
        )
        from harbor.environments.beam import BeamEnvironment

        await BeamEnvironment.pre_build_images(env_dirs, max_concurrent=max_concurrent)
        logger.info(f"[coord {self.coordinator_index}] pre-build complete: {len(env_dirs)} images")
        return len(env_dirs)

    # ----- orchestrator lifecycle ----------------------------------------
    async def startup(self) -> None:
        self._orchestrator_lock = asyncio.Lock()
        await self._create_orchestrator()
        logger.info(f"[coord {self.coordinator_index}] startup complete")

    async def _create_orchestrator(self) -> None:
        self._orchestrator = QueueOrchestrator(
            trial_configs=[],
            n_concurrent_trials=self._n_concurrent_trials,
            metrics={},
            quiet=True,
            retry_config=self._retry_config,
        )
        rollback_hook = create_rollback_hook(
            exception_types={"ContextLengthExceededError", "AgentTimeoutError"},
            on_complete_failure="mark_metadata",
            preserve_partial_logprobs=False,
        )
        self._orchestrator.add_hook(OrchestratorEvent.TRIAL_COMPLETED, rollback_hook)
        await self._orchestrator.start()
        self._orchestrator_started = True

    async def _restart_orchestrator(self) -> bool:
        async with self._orchestrator_lock:
            if self._orchestrator_started and self._orchestrator is not None:
                return True
            self._orchestrator_restart_count += 1
            if self._orchestrator_restart_count > MAX_ORCHESTRATOR_RESTART_ATTEMPTS:
                logger.error(
                    f"[coord {self.coordinator_index}] max restart attempts exceeded, giving up"
                )
                return False
            logger.warning(
                f"[coord {self.coordinator_index}] restarting orchestrator "
                f"({self._orchestrator_restart_count}/{MAX_ORCHESTRATOR_RESTART_ATTEMPTS})"
            )
            if self._orchestrator is not None:
                try:
                    await self._orchestrator.shutdown(wait=False)
                except Exception as e:
                    logger.warning(f"[coord {self.coordinator_index}] error shutting down: {e}")
                finally:
                    self._orchestrator = None
                    self._orchestrator_started = False
            try:
                await self._create_orchestrator()
                return True
            except Exception as e:
                logger.error(f"[coord {self.coordinator_index}] failed to recreate orchestrator: {e}")
                self._orchestrator_started = False
                return False

    async def shutdown(self) -> None:
        self._permanently_shutdown = True
        if self._orchestrator_lock is None:
            return
        async with self._orchestrator_lock:
            if self._orchestrator is not None and self._orchestrator_started:
                try:
                    await self._orchestrator.shutdown(wait=True)
                except Exception as e:
                    logger.warning(f"[coord {self.coordinator_index}] shutdown error: {e}")
                finally:
                    self._orchestrator_started = False
                    self._orchestrator = None

    # ----- eval session ---------------------------------------------------
    async def start_eval_session(
        self, run_name: str, eval_step: int, val_set_name: Optional[str] = None
    ) -> None:
        if self._eval_orchestrator_lock is None:
            self._eval_orchestrator_lock = asyncio.Lock()
        async with self._eval_orchestrator_lock:
            if self._eval_session_active and self._eval_orchestrator is not None:
                try:
                    await self._eval_orchestrator.shutdown(wait=True)
                except Exception as e:
                    logger.warning(f"[coord {self.coordinator_index}] prev eval shutdown error: {e}")
                finally:
                    self._eval_orchestrator = None
                    self._eval_session_active = False

            val_set_suffix = f"_{val_set_name}" if val_set_name else ""
            # Include coordinator index so pooled coordinators don't collide.
            self._eval_session_name = (
                f"{run_name}_eval{val_set_suffix}_step{eval_step}_coord{self.coordinator_index}"
            )
            if self.trials_dir:
                self._eval_trials_dir = str(
                    Path(self.trials_dir) / "eval_sessions" / self._eval_session_name
                )
                Path(self._eval_trials_dir).mkdir(parents=True, exist_ok=True)
            else:
                self._eval_trials_dir = self.trials_dir

            self._eval_orchestrator = QueueOrchestrator(
                trial_configs=[],
                n_concurrent_trials=self._n_concurrent_trials,
                metrics={},
                quiet=True,
                retry_config=self._retry_config,
            )
            rollback_hook = create_rollback_hook(
                exception_types={"ContextLengthExceededError", "AgentTimeoutError"},
                on_complete_failure="mark_metadata",
                preserve_partial_logprobs=False,
            )
            self._eval_orchestrator.add_hook(OrchestratorEvent.TRIAL_COMPLETED, rollback_hook)
            await self._eval_orchestrator.start()
            self._eval_session_active = True
            logger.info(
                f"[coord {self.coordinator_index}] eval session {self._eval_session_name} started"
            )

    async def stop_eval_session(self) -> None:
        if self._eval_orchestrator_lock is None:
            return
        async with self._eval_orchestrator_lock:
            if self._eval_orchestrator is not None and self._eval_session_active:
                try:
                    await self._eval_orchestrator.shutdown(wait=True)
                except Exception as e:
                    logger.warning(f"[coord {self.coordinator_index}] eval shutdown error: {e}")
                finally:
                    self._eval_orchestrator = None
                    self._eval_session_active = False
                    self._eval_session_name = None
                    self._eval_trials_dir = None

    # ----- active-mode helpers -------------------------------------------
    def _get_active_orchestrator(self) -> Optional[QueueOrchestrator]:
        if self._eval_session_active and self._eval_orchestrator is not None:
            return self._eval_orchestrator
        return self._orchestrator

    def _get_active_trials_dir(self) -> Optional[str]:
        if self._eval_session_active and self._eval_trials_dir is not None:
            return self._eval_trials_dir
        return self.trials_dir

    def _get_active_timeout_override(self) -> Optional[int]:
        if self._eval_session_active:
            return self._eval_timeout_override_sec
        return None

    # ----- the actual fanned-out work ------------------------------------
    async def run_batch(
        self, prompts: List[Any], trajectory_ids: List[TrajectoryID]
    ) -> List[TerminalBenchAgentOutput]:
        """Build + submit + gather + process this coordinator's shard.

        Returns one ``TerminalBenchAgentOutput`` per input, in input order.
        Group-level cascade / metrics are done by the generator over the union
        of all shards.
        """
        if self._permanently_shutdown:
            raise RuntimeError(f"[coord {self.coordinator_index}] permanently shut down")

        num_trials = len(prompts)
        if num_trials == 0:
            return []

        active_trials_dir = self._get_active_trials_dir()
        timeout_override = self._get_active_timeout_override()
        model_alias = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name

        trial_configs: List[TrialConfig] = []
        for i in range(num_trials):
            session_id = uuid4().hex  # sticky routing to inference engines
            trial_configs.append(
                self._harbor_config_builder.build_trial_config(
                    task_path=prompts[i],
                    trials_dir=active_trials_dir,
                    model_name=f"hosted_vllm/{model_alias}",
                    api_base=f"{self.base_url}/v1",
                    session_id=session_id,
                    timeout_override_sec=timeout_override,
                )
            )

        is_eval = self._eval_session_active
        active = self._get_active_orchestrator()
        started = self._eval_session_active if is_eval else self._orchestrator_started
        if not started or active is None:
            if not await self._restart_orchestrator():
                return [self._failed_output(tid, "OrchestratorNotStarted") for tid in trajectory_ids]
            active = self._get_active_orchestrator()

        try:
            futures = await active.submit_batch(trial_configs)
            results = await asyncio.gather(*futures, return_exceptions=True)
        except Exception as orch_err:
            logger.error(
                f"[coord {self.coordinator_index}] orchestrator-level failure: "
                f"{type(orch_err).__name__}: {orch_err}"
            )
            if is_eval or not await self._restart_orchestrator():
                tag = f"OrchestratorFailure_{type(orch_err).__name__}"
                return [self._failed_output(tid, tag) for tid in trajectory_ids]
            try:
                active = self._get_active_orchestrator()
                futures = await active.submit_batch(trial_configs)
                results = await asyncio.gather(*futures, return_exceptions=True)
            except Exception as retry_err:
                tag = f"OrchestratorRetryFailure_{type(retry_err).__name__}"
                return [self._failed_output(tid, tag) for tid in trajectory_ids]

        return [
            self._process_trial_result(results[i], trajectory_ids[i]) for i in range(num_trials)
        ]

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
            exclude_from_baseline=True,  # infra failure
            exception_type=exception_type,
        )

    # ----- result processing (verbatim from the original generator) -------
    def _classify_exception(self, exception: Exception) -> tuple:
        exception_type = type(exception).__name__
        if not self._error_handling_config.get("enable_error_classification", False):
            return False, exception_type
        mask_exceptions = self._error_handling_config.get("mask_exceptions", set())
        zero_exceptions = self._error_handling_config.get("zero_exceptions", set())
        default_treatment = self._error_handling_config.get("default_error_treatment", "zero")
        if exception_type in mask_exceptions:
            return True, exception_type
        if exception_type in zero_exceptions:
            return False, exception_type
        return (default_treatment == "mask"), exception_type

    def _process_trial_result(
        self, result, trajectory_id: TrajectoryID
    ) -> TerminalBenchAgentOutput:
        if isinstance(result, Exception):
            exclude_from_baseline, exception_type = self._classify_exception(result)
            logger.warning(
                f"[coord {self.coordinator_index}] trajectory {trajectory_id} failed: {result} "
                f"(type={exception_type}, exclude_from_baseline={exclude_from_baseline})"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
                exclude_from_baseline=exclude_from_baseline,
                exception_type=exception_type,
            )

        if not result.verifier_result:
            exception_info = getattr(result, "exception_info", None)
            exception_type = "UnknownError"
            exclude_from_baseline = False
            if exception_info:
                if hasattr(exception_info, "exception_type"):
                    exception_type = exception_info.exception_type
                elif hasattr(exception_info, "__class__"):
                    exception_type = type(exception_info).__name__

                class MockException(Exception):
                    pass

                MockException.__name__ = exception_type
                exclude_from_baseline, _ = self._classify_exception(MockException())
            logger.warning(
                f"[coord {self.coordinator_index}] trajectory {trajectory_id} failed: "
                f"no verifier result (type={exception_type})"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
                exclude_from_baseline=exclude_from_baseline,
                exception_type=exception_type,
            )

        try:
            original_reward = result.verifier_result.rewards["reward"]
            chat_history = result.agent_result.metadata["all_messages"]
            summarization_count = result.agent_result.metadata["summarization_count"]
        except (KeyError, AttributeError, TypeError) as e:
            exception_type = type(e).__name__
            exclude_from_baseline, _ = self._classify_exception(e)
            logger.warning(
                f"[coord {self.coordinator_index}] trajectory {trajectory_id} failed: "
                f"could not extract results: {e}"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
                exclude_from_baseline=exclude_from_baseline,
                exception_type=exception_type,
            )

        if self._reward_shaping_config.get("enable_reward_shaping", True):
            verifier_stdout = getattr(result.verifier_result, "stdout", None)
            reward = shape_reward_from_output(
                stdout=verifier_stdout,
                original_reward=original_reward,
                parser_name=self._reward_shaping_config.get("reward_parser"),
                shaper_name=self._reward_shaping_config.get("reward_shaper", "pass_ratio"),
                shaper_kwargs=self._reward_shaping_config.get("shaper_kwargs", {}),
                fallback_to_original=self._reward_shaping_config.get("reward_shaping_fallback", True),
            )
        else:
            reward = original_reward

        if not chat_history or len(chat_history) < 2 or chat_history[0]["role"] != "user":
            logger.warning(
                f"[coord {self.coordinator_index}] trajectory {trajectory_id} failed: "
                f"invalid chat history structure"
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
                exclude_from_baseline=True,
                exception_type="InvalidChatHistory",
            )

        prompt = [chat_history[0]]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=False,
            tokenize=True,
            chat_template=self.custom_chat_template_content,
        )
        initial_prompt_length = len(prompt_ids)
        response_messages = chat_history[1:]

        rollout_details = getattr(result.agent_result, "rollout_details", None)
        assistant_logprobs = extract_logprobs_from_rollout_details(rollout_details)
        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs,
            custom_chat_template=self.custom_chat_template_content,
        )

        max_response_tokens = (
            self.generator_cfg.sampling_params.max_generate_length
            + self.generator_cfg.max_input_length
            - initial_prompt_length
        )
        stop_reason = "complete"
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        if rollout_logprobs is not None:
            rollout_logprobs = rollout_logprobs[:max_response_tokens]

        if stop_reason == "length" and self._reward_shaping_config.get("mask_truncated_loss", False):
            loss_mask = [0] * len(loss_mask)

        return TerminalBenchAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            rollout_logprobs=rollout_logprobs,
            summarization_count=summarization_count,
        )


# Ray actor wrapper. num_cpus is set at construction time via .options() so the
# generator can plumb cpus_per_coordinator from config.
RolloutCoordinator = ray.remote(_CoordinatorImpl)
