import asyncio
import json
import logging
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import litellm
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from harbor.models.agent.rollout_detail import RolloutDetail
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from skyrl.backends.skyrl_train.inference_servers.base import (
    ConversationType,
    InferenceEngineInterface,
)
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import get_rollout_metrics
from skyrl.train.utils.rate_limiter import create_rate_limiter

from .ttt.sampler import StateSampler, create_sampler
from .ttt.state import State, read_construction_artifact

litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

MAX_NUM_RETRIES_PER_TRIAL = 2


def adaptive_entropic_group(
    rewards: torch.Tensor,
    *,
    target_kl: float = math.log(2.0),
    beta_max: float = 1e6,
    iterations: int = 60,
) -> tuple[torch.Tensor, float, float, bool]:
    """Reference adaptive-beta solve for one reward group."""
    values = rewards.float()
    group_size = values.numel()
    if group_size < 2:
        return torch.zeros_like(values), 0.0, 0.0, False

    log_group_size = math.log(group_size)

    def kl_at(beta: float) -> float:
        logits = beta * (values - values.max())
        log_probabilities = logits - torch.logsumexp(logits, dim=0)
        probabilities = torch.exp(log_probabilities)
        kl = (probabilities * (log_probabilities + log_group_size)).sum()
        return float(kl.item())

    low, high = 0.0, 1.0
    while high < beta_max and kl_at(high) < target_kl:
        high = min(2.0 * high, beta_max)
    saturated = kl_at(high) < target_kl
    if not saturated:
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            if kl_at(middle) < target_kl:
                low = middle
            else:
                high = middle
    beta = high

    exponentials = torch.exp(beta * (values - values.max()))
    loo_normalizer = (exponentials.sum() - exponentials) / (group_size - 1)
    advantages = exponentials / (loo_normalizer + 1e-12) - 1.0
    return advantages, beta, kl_at(beta), saturated


@dataclass
class HarborTrajectoryOutput:
    """One trajectory's raw output and verifier-owned search artifact."""

    trajectory_id: TrajectoryID
    rollout_details: Optional[List[RolloutDetail]] = None
    reward: float = 0.0
    num_turns: int = 0
    stop_reason: str = "complete"
    e2e_time: Optional[float] = None
    verifier_output: str = ""
    submitted_solution: str = ""
    verifier_metrics: dict[str, object] = field(default_factory=dict)
    construction: list[float] | None = None
    selection_value: float | None = None


def build_step_wise_generator_output(
    trajectory_outputs: List[HarborTrajectoryOutput], overlong_filtering: bool
) -> GeneratorOutput:
    """Flatten per-trajectory Harbor rollout details into one entry per LLM turn."""
    timeout_instance_ids = set()
    error_instance_ids = set()
    num_timeout_trajectories = 0
    num_error_trajectories = 0
    for trajectory in trajectory_outputs:
        instance_id = trajectory.trajectory_id.instance_id
        if trajectory.stop_reason == "agent_timeout":
            num_timeout_trajectories += 1
            timeout_instance_ids.add(instance_id)
        elif trajectory.stop_reason == "error" or trajectory.rollout_details is None:
            num_error_trajectories += 1
            error_instance_ids.add(instance_id)
    masked_instance_ids = timeout_instance_ids | error_instance_ids

    prompt_token_ids: List[List[int]] = []
    response_ids: List[List[int]] = []
    rewards: List[float] = []
    loss_masks: List[List[int]] = []
    stop_reasons: List[str] = []
    is_last_step_list: List[bool] = []
    out_trajectory_ids: List[TrajectoryID] = []
    rollout_logprobs_list: List[List[float]] = []
    out_env_metrics: List[dict] = []

    successful_trajectories: List[HarborTrajectoryOutput] = []
    response_ids_for_metrics: List[List[int]] = []
    rewards_for_metrics: List[float] = []
    env_metrics_for_metrics: List[dict] = []
    trajectory_generation_times_per_prompt: List[Optional[float]] = []
    out_trajectory_generation_times: List[Optional[float]] = []

    for trajectory in trajectory_outputs:
        trajectory_id = trajectory.trajectory_id
        if trajectory_id.instance_id in masked_instance_ids:
            prompt_token_ids.append([0])
            response_ids.append([0])
            rewards.append(0.0)
            loss_masks.append([0])
            stop_reasons.append("error")
            is_last_step_list.append(True)
            out_trajectory_ids.append(trajectory_id)
            rollout_logprobs_list.append([0.0])
            out_env_metrics.append({})
            out_trajectory_generation_times.append(trajectory.e2e_time)
            continue

        successful_trajectories.append(trajectory)
        assert trajectory.rollout_details is not None
        assert len(trajectory.rollout_details) == 1, (
            "Expected exactly one rollout segment, got "
            f"{len(trajectory.rollout_details)}."
        )
        rollout_detail = trajectory.rollout_details[0]
        prompt_ids_per_turn = rollout_detail["prompt_token_ids"]
        completion_ids_per_turn = rollout_detail["completion_token_ids"]
        logprobs_per_turn = rollout_detail["logprobs"]
        num_turns = len(completion_ids_per_turn)
        assert len(prompt_ids_per_turn) == num_turns and len(logprobs_per_turn) == num_turns, (
            "Malformed rollout_details "
            f"(prompts={len(prompt_ids_per_turn)}, completions={num_turns}, "
            f"logprobs={len(logprobs_per_turn)})."
        )

        for turn in range(num_turns):
            completion_ids = completion_ids_per_turn[turn]
            turn_prompt_ids = prompt_ids_per_turn[turn]
            logprobs = logprobs_per_turn[turn]
            assert len(logprobs) == len(completion_ids), (
                "logprobs and completion token ids must have the same length"
            )
            is_last = turn == num_turns - 1
            reward = trajectory.reward if is_last else 0.0
            loss_mask = [1] * len(completion_ids)
            stop_reason = "complete"
            if trajectory.stop_reason == "context_length":
                stop_reason = "context_length"
                if overlong_filtering:
                    loss_mask = [0] * len(completion_ids)

            prompt_token_ids.append(turn_prompt_ids)
            response_ids.append(completion_ids)
            rewards.append(reward)
            loss_masks.append(loss_mask)
            stop_reasons.append(stop_reason)
            is_last_step_list.append(is_last)
            out_trajectory_ids.append(trajectory_id)
            rollout_logprobs_list.append(logprobs)
            out_env_metrics.append(dict(trajectory.verifier_metrics))
            out_trajectory_generation_times.append(trajectory.e2e_time)

        response_ids_for_metrics.append(
            prompt_ids_per_turn[-1] + completion_ids_per_turn[-1]
        )
        rewards_for_metrics.append(trajectory.reward)
        env_metrics_for_metrics.append(dict(trajectory.verifier_metrics))
        trajectory_generation_times_per_prompt.append(trajectory.e2e_time)

    if any(value is None for value in trajectory_generation_times_per_prompt):
        trajectory_generation_times_per_prompt = None  # type: ignore[assignment]
    if any(value is None for value in out_trajectory_generation_times):
        out_trajectory_generation_times = None  # type: ignore[assignment]
    if successful_trajectories:
        rollout_metrics = get_rollout_metrics(
            response_ids_for_metrics,
            rewards_for_metrics,
            env_metrics=env_metrics_for_metrics,
            trajectory_completion_times=trajectory_generation_times_per_prompt,
        )
        rollout_metrics["generate/trajectories_context_length_exceeded"] = sum(
            trajectory.stop_reason == "context_length"
            for trajectory in successful_trajectories
        )
        rollout_metrics["generate/avg_num_turns"] = sum(
            trajectory.num_turns for trajectory in successful_trajectories
        ) / len(successful_trajectories)
    else:
        rollout_metrics = {}
    rollout_metrics["generate/num_timeout_trajectories"] = num_timeout_trajectories
    rollout_metrics["generate/num_error_trajectories"] = num_error_trajectories
    rollout_metrics["generate/num_masked_instances"] = len(masked_instance_ids)

    return GeneratorOutput(
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        stop_reasons=stop_reasons,
        rollout_metrics=rollout_metrics,
        rollout_logprobs=rollout_logprobs_list,
        trajectory_ids=out_trajectory_ids,
        env_metrics=out_env_metrics,
        is_last_step=is_last_step_list,
        trajectory_generation_times=out_trajectory_generation_times,
    )


def get_sampler(
    sampler_dir: str,
    sampler_kwargs,
    sampler_type: str = "greedy",
    n_samples_per_prompt: int = 1,
) -> StateSampler:
    kwargs = dict(sampler_kwargs or {})
    kwargs.pop("sampler_dir", None)
    return create_sampler(
        sampler_type,
        sampler_dir=sampler_dir,
        group_size=n_samples_per_prompt,
        **kwargs,
    )


class TTTHarborGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        ttt_cfg: DictConfig,
        inference_engine_client: InferenceEngineInterface,
        tokenizer,
        max_seq_len: int,
    ):
        ie_cfg = generator_cfg.inference_engine
        self.base_url = inference_engine_client.get_endpoint_url()
        self.inference_engine_client = inference_engine_client
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if not getattr(generator_cfg, "step_wise_trajectories", False):
            raise ValueError(
                "HarborGenerator only supports step-wise training. "
                "Set generator.step_wise_trajectories=true."
            )
        if not getattr(generator_cfg, "merge_stepwise_output", False):
            logger.warning(
                "merge_stepwise_output=true is not set; training can be much slower"
            )

        self._harbor_trial_config_template = deepcopy(harbor_cfg)
        self._served_model_name = ie_cfg.served_model_name
        assert ie_cfg.served_model_name is not None, "served_model_name must be set"
        assert "/" not in ie_cfg.served_model_name, (
            "Harbor expects a served_model_name without '/'"
        )
        self._harbor_trial_config_template.setdefault("agent", {})[
            "model_name"
        ] = f"hosted_vllm/{ie_cfg.served_model_name}"
        self._harbor_trial_config_template["agent"].setdefault("kwargs", {})[
            "api_base"
        ] = f"{self.base_url}/v1"

        agent_kwargs = self._harbor_trial_config_template["agent"]["kwargs"]
        if not agent_kwargs.get("collect_rollout_details", False):
            logger.warning(
                "step_wise_trajectories requires collect_rollout_details; enabling it"
            )
            agent_kwargs["collect_rollout_details"] = True
        if agent_kwargs.get("enable_summarize", False):
            raise ValueError(
                "step-wise Harbor training is incompatible with enable_summarize=true"
            )

        logger.info(
            "HarborGenerator initialized with agent={} trials_dir={}",
            self._harbor_trial_config_template.get("agent", {}).get("name"),
            self._harbor_trial_config_template.get("trials_dir", "trials"),
        )
        self._rate_limiter = create_rate_limiter(
            getattr(generator_cfg, "rate_limit", None)
        )

        trials_dir = harbor_cfg["trials_dir"]
        self.temp_tasks_dir = os.path.join(trials_dir, "tmp_tasks")
        os.makedirs(self.temp_tasks_dir, exist_ok=True)
        self.sampler_dir = os.path.join(trials_dir, "sampler")
        os.makedirs(self.sampler_dir, exist_ok=True)
        self.sampler_type = ttt_cfg.sampler_type
        self.sampler_kwargs = dict(ttt_cfg.sampler_kwargs or {})
        self.sampler = get_sampler(
            self.sampler_dir,
            self.sampler_kwargs,
            self.sampler_type,
            self.generator_cfg.n_samples_per_prompt,
        )

    def _compute_cache_salt(self) -> Optional[str]:
        if not getattr(self.generator_cfg, "use_cache_salt", False):
            return None
        weight_version = getattr(self.inference_engine_client, "weight_version", None)
        if weight_version is None:
            return None
        prefix = f"{self._served_model_name}@" if self._served_model_name else ""
        return f"{prefix}{weight_version}"

    @staticmethod
    def _selection_value(parent: State, result: HarborTrajectoryOutput) -> float:
        metric_name = parent.prompt_context.get("PERFORMANCE_METRIC", "")
        raw_value = result.verifier_metrics.get(metric_name) if metric_name else None
        if isinstance(raw_value, (int, float)) and math.isfinite(float(raw_value)):
            raw_value = float(raw_value)
            result.verifier_metrics["construction_performance"] = raw_value
            mode = parent.prompt_context.get("PERFORMANCE_MODE", "maximize").lower()
            if mode not in {"maximize", "minimize"}:
                raise ValueError(f"Unknown PERFORMANCE_MODE {mode!r}")
            derived = -raw_value if mode == "minimize" else raw_value
            if result.selection_value is not None and not math.isclose(
                result.selection_value, derived, rel_tol=1e-9, abs_tol=1e-12
            ):
                logger.warning(
                    "Ignoring construction artifact selection_value={} because "
                    "task metadata derives {} from {}={}",
                    result.selection_value,
                    derived,
                    metric_name,
                    raw_value,
                )
            return derived
        if result.selection_value is not None:
            return result.selection_value
        return float(result.reward)

    async def generate(
        self, input_batch: GeneratorInput, disable_tqdm: bool = False
    ) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch.get("batch_metadata")
        if trajectory_ids is None:
            raise ValueError("trajectory_ids is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError("prompt count does not match trajectory_ids count")
        if batch_metadata is None:
            raise ValueError("TTTHarborGenerator requires input_batch.batch_metadata")
        global_step = batch_metadata.global_step
        is_training = batch_metadata.training_phase == "train"

        instance_prompts: dict[str, str] = {}
        for prompt, trajectory_id in zip(prompts, trajectory_ids, strict=True):
            if not isinstance(prompt, str):
                raise TypeError("TTT Harbor prompts must be task-directory paths")
            previous = instance_prompts.setdefault(trajectory_id.instance_id, prompt)
            if previous != prompt:
                raise ValueError(
                    f"Instance {trajectory_id.instance_id!r} has multiple task paths"
                )
        instance_ids = list(instance_prompts)
        initial_states = [
            State.initial(
                instance_prompts[instance_id],
                tasks_dir=self.temp_tasks_dir,
            )
            for instance_id in instance_ids
        ]
        sampled_states = self.sampler.sample_states(
            len(initial_states), fallback_states=initial_states
        )
        state_by_instance = dict(zip(instance_ids, sampled_states, strict=True))
        cache_salt = self._compute_cache_salt()

        all_outputs: List[HarborTrajectoryOutput] = [None] * len(prompts)  # type: ignore[list-item]
        child_states: List[Optional[State]] = [None] * len(prompts)
        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def _worker(index, trajectory_id):
            parent = state_by_instance[trajectory_id.instance_id]
            result = await self._harbor_agent_loop(
                prompt=parent.task_path,
                trajectory_id=trajectory_id,
                cache_salt=cache_salt,
            )
            all_outputs[index] = result
            if is_training and result.rollout_details and result.stop_reason == "complete":
                try:
                    construction_required = parent.prompt_context.get(
                        "CONSTRUCTION_REQUIRED", "false"
                    ).lower() in {"1", "true", "yes"}
                    if construction_required and result.construction is None:
                        raise ValueError(
                            "task requires a verified construction artifact, but none was produced"
                        )
                    decoded_rollout = self._decode_solution(result.rollout_details)
                    solution = result.submitted_solution or decoded_rollout
                    child_states[index] = parent.make_child(
                        solution=solution,
                        value=self._selection_value(parent, result),
                        timestep=global_step,
                        tasks_dir=self.temp_tasks_dir,
                        observation=result.verifier_output,
                        construction=result.construction,
                    )
                except Exception:
                    logger.exception(
                        "Failed to construct a TTT child state for {}", trajectory_id
                    )
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as task_group:
                for index, trajectory_id in enumerate(trajectory_ids):
                    task_group.create_task(_worker(index, trajectory_id))
        finally:
            progress.close()

        if is_training:
            children: List[State] = []
            parents: List[State] = []
            failed_parents: dict[str, State] = {}
            successful_parent_ids: set[str] = set()
            for child, trajectory_id in zip(child_states, trajectory_ids, strict=True):
                parent = state_by_instance[trajectory_id.instance_id]
                if child is not None:
                    children.append(child)
                    parents.append(parent)
                    successful_parent_ids.add(parent.id)
                else:
                    failed_parents[parent.id] = parent
            self.sampler.update_states(
                children, parents, save=False, step=global_step
            )
            if hasattr(self.sampler, "record_failed_rollout"):
                for parent_id, parent in failed_parents.items():
                    if parent_id in successful_parent_ids:
                        continue
                    self.sampler.record_failed_rollout(parent)
            self.sampler.flush(global_step)

        return build_step_wise_generator_output(
            all_outputs,
            overlong_filtering=self.generator_cfg.apply_overlong_filtering,
        )

    def _decode_solution(self, rollout_details: List[RolloutDetail]) -> str:
        turns: list[str] = []
        for detail in rollout_details:
            for token_ids in detail.get("completion_token_ids", []):
                text = self.tokenizer.decode(
                    token_ids, skip_special_tokens=False
                ).strip()
                if text:
                    turns.append(text)
        if not turns:
            raise ValueError("Cannot create a TTT state from an empty completion")
        return "\n\n".join(turns)

    async def _harbor_agent_loop(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str] = None,
    ) -> HarborTrajectoryOutput:
        agent_loop_start_time = time.monotonic()
        reward = None
        results = None
        rollout_details = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False
        verifier_output = ""
        submitted_solution = ""
        verifier_metrics: dict[str, object] = {}
        construction = None
        selection_value = None

        for attempt in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = (
                f"Trajectory {trajectory_id} attempt "
                f"{attempt + 1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            )
            results = None
            verifier_output = ""
            submitted_solution = ""
            verifier_metrics = {}
            construction = None
            selection_value = None
            session_id = uuid4().hex
            try:
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config["agent"]["kwargs"]["session_id"] = session_id
                if cache_salt is not None:
                    llm_kwargs = config["agent"]["kwargs"].setdefault(
                        "llm_kwargs", {}
                    )
                    extra_body = llm_kwargs.setdefault("extra_body", {})
                    if not isinstance(extra_body, dict):
                        raise TypeError(
                            "harbor agent llm_kwargs.extra_body must be a mapping"
                        )
                    extra_body["cache_salt"] = cache_salt
                trial = await Trial.create(TrialConfig.model_validate(config))
                async with self._rate_limiter:
                    results = await trial.run()

                verifier_output_path = trial.paths.test_stdout_path
                if verifier_output_path.is_file():
                    try:
                        verifier_output = verifier_output_path.read_text()
                    except OSError as exc:
                        logger.warning(
                            "Ignoring unreadable verifier output at {}: {}",
                            verifier_output_path,
                            exc,
                        )
                verifier_dir = Path(trial.paths.trial_dir) / "verifier"
                submitted_program = verifier_dir / "submitted-program.py"
                if submitted_program.is_file():
                    submitted_solution = submitted_program.read_text()
                metrics_path = verifier_dir / "metrics.json"
                if metrics_path.is_file():
                    try:
                        loaded_metrics = json.loads(metrics_path.read_text())
                        if isinstance(loaded_metrics, dict):
                            verifier_metrics = loaded_metrics
                    except (OSError, json.JSONDecodeError) as exc:
                        logger.warning(
                            "Ignoring malformed verifier metrics at {}: {}",
                            metrics_path,
                            exc,
                        )
                construction_path = verifier_dir / "construction.json"
                if construction_path.is_file():
                    try:
                        construction, selection_value = read_construction_artifact(
                            construction_path
                        )
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        logger.warning(
                            "Ignoring malformed construction artifact at {}: {}",
                            construction_path,
                            exc,
                        )

                exception_type = (
                    results.exception_info.exception_type
                    if results.exception_info
                    else None
                )
                is_context_length_error = (
                    exception_type == "ContextLengthExceededError"
                )
                is_agent_timeout_error = exception_type == "AgentTimeoutError"
                if is_agent_timeout_error:
                    logger.debug("{} hit AgentTimeoutError", prefix)
                    break
                if is_context_length_error:
                    logger.debug("{} hit context length; assigning reward 0", prefix)
                    reward = 0.0
                elif not results.verifier_result:
                    logger.warning(
                        "{} failed: exception={} results={}",
                        prefix,
                        results.exception_info,
                        results,
                    )
                    continue
                else:
                    reward = float(results.verifier_result.rewards["reward"])

                rollout_details = results.agent_result.rollout_details
                num_turns = results.agent_result.metadata["n_episodes"]
                if (
                    rollout_details
                    and len(rollout_details) >= 1
                    and len(rollout_details[0].get("completion_token_ids", [])) > 0
                ):
                    successful = True
                    logger.debug("{} successful: reward={}", prefix, reward)
                    break
                logger.warning("{} failed: empty rollout_details", prefix)
            except Exception as exc:
                logger.warning(
                    "{} failed while running Harbor trial: {}; results={}",
                    prefix,
                    exc,
                    results,
                )
            finally:
                await self.inference_engine_client.finish_session(session_id)

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            logger.warning(
                "Trajectory {} failed (stop_reason={}); results={}",
                trajectory_id,
                stop_reason,
                results,
            )
            return HarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=None,
                stop_reason=stop_reason,
                e2e_time=time.monotonic() - agent_loop_start_time,
            )

        return HarborTrajectoryOutput(
            trajectory_id=trajectory_id,
            rollout_details=rollout_details,
            reward=reward,
            num_turns=num_turns,
            stop_reason=(
                "context_length" if is_context_length_error else "complete"
            ),
            e2e_time=time.monotonic() - agent_loop_start_time,
            verifier_output=verifier_output,
            submitted_solution=submitted_solution,
            verifier_metrics=verifier_metrics,
            construction=construction,
            selection_value=selection_value,
        )
