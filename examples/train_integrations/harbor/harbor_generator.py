import asyncio
import base64
import io
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

# Suppress LiteLLM verbose logging
import litellm
import numpy as np
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
from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    decode_packed_routed_experts,
)
from skyrl.backends.skyrl_train.utils.routed_experts import (
    RoutedExpertIndices,
    compact_routed_expert_indices,
)
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import get_rollout_metrics
from skyrl.train.utils.rate_limiter import create_rate_limiter

from .reward_shaping import (
    read_bounded_verifier_output,
    score_harbor_trajectory_async,
    shape_reward_from_output,
)

litellm.suppress_debug_info = True  # Suppress the "Provider List" output
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# We have N retries for each trial, if one of the rollout (out of n_samples_per_prompt) fails
# after N attemptes, we skip this prompt altogether.
MAX_NUM_RETRIES_PER_TRIAL = 2


@dataclass
class HarborTrajectoryOutput:
    """One trajectory's raw output from Harbor.

    Holds the entire ``rollout_details`` from ``agent_result``. Per-step interpretation
    (loss-mask / reward broadcast / overlong filtering) is done downstream in
    `build_step_wise_generator_output`.
    """

    trajectory_id: TrajectoryID
    # Entire rollout_details list as returned by harbor's agent_result. None for failed trajectories
    # (agent_timeout / error) that we will mask in `build_step_wise_generator_output`.
    rollout_details: Optional[List[RolloutDetail]] = None
    reward: float = 0.0
    # Keep verifier reward separate so pass@k/eval metrics remain comparable
    # when the training reward includes a small process bonus.
    raw_reward: Optional[float] = None
    test_output_reward: Optional[float] = None
    test_output_parsed: bool = False
    test_output_passed: int = 0
    test_output_total: int = 0
    process_bonus: float = 0.0
    process_penalty: float = 0.0
    process_premature_completion: bool = False
    process_multi_tool_turns: int = 0
    process_wrote_tests: bool = False
    process_ran_tests: bool = False
    process_used_syntax_checker: bool = False
    process_reasoning_chars_per_turn: float = 0.0
    process_concise_reasoning_bonus: float = 0.0
    process_oracle_test_reward: float = 0.0
    process_n_valid_tests: int = 0
    process_n_discriminating_tests: int = 0
    process_n_fixed_tests: int = 0
    process_test_use_reward: float = 0.0
    num_turns: int = 0
    # One of: "complete", "context_length", "agent_timeout", "error". Used by
    # `build_step_wise_generator_output` to decide whether to skip the entire prompt group.
    stop_reason: str = "complete"
    # End-to-end wall-clock time (seconds) to generate this trajectory. Optional: left as None if
    # timing was not recorded.
    e2e_time: Optional[float] = None


def _decode_routed_experts(value) -> RoutedExpertIndices:
    """Decode routed experts returned by either SkyRL's or vLLM's HTTP API."""
    if isinstance(value, dict):
        return decode_packed_routed_experts(value)
    if isinstance(value, str):
        try:
            decoded = np.load(
                io.BytesIO(base64.b64decode(value, validate=True)), allow_pickle=False
            )
        except (ValueError, TypeError, EOFError) as exc:
            raise ValueError(
                "Invalid base64 .npy routed_experts payload from vLLM"
            ) from exc
        return compact_routed_expert_indices(decoded)
    if isinstance(value, list):
        value = np.asarray(value)
    return compact_routed_expert_indices(value)


def _trajectory_rollout_detail(traj: HarborTrajectoryOutput):
    """Validate and return the per-turn fields for one linear Harbor chat."""
    assert traj.rollout_details is not None
    assert len(traj.rollout_details) == 1, (
        f"Expected exactly one rollout segment, got {len(traj.rollout_details)}."
    )
    rollout_detail = traj.rollout_details[0]
    prompts = rollout_detail["prompt_token_ids"]
    completions = rollout_detail["completion_token_ids"]
    logprobs = rollout_detail["logprobs"]
    n_turns = len(completions)
    if not n_turns or len(prompts) != n_turns or len(logprobs) != n_turns:
        raise ValueError(
            f"Malformed rollout_details (prompts={len(prompts)}, completions={n_turns}, "
            f"logprobs={len(logprobs)})."
        )
    for turn, (completion, turn_logprobs) in enumerate(zip(completions, logprobs)):
        if len(turn_logprobs) != len(completion):
            raise ValueError(
                f"Turn {turn}: logprobs ({len(turn_logprobs)}) and completion token ids "
                f"({len(completion)}) must have the same length."
            )
    return rollout_detail, prompts, completions, logprobs


def _turn_routed_experts(
    rollout_detail: RolloutDetail,
    prompts: List[List[int]],
    completions: List[List[int]],
    trajectory_id: TrajectoryID,
) -> List[RoutedExpertIndices]:
    """Decode and validate the vLLM route prefix associated with every turn."""
    trajectory_label = trajectory_id.to_string()
    values = rollout_detail.get("extra", {}).get("routed_experts")
    if not isinstance(values, list) or len(values) != len(completions):
        raise ValueError(
            f"Trajectory {trajectory_label}: R3 was enabled, but Harbor did not collect one "
            "routed_experts payload per LLM turn. "
            "Check the vLLM/LiteLLM provider metadata path."
        )

    routes = []
    expected_layer_topk = None
    for turn, (value, prompt, completion) in enumerate(
        zip(values, prompts, completions)
    ):
        turn_label = f"Trajectory {trajectory_label}, turn {turn}"
        if value is None:
            raise ValueError(
                f"{turn_label}: vLLM returned no routed_experts while R3 is enabled"
            )
        try:
            route = _decode_routed_experts(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{turn_label}: invalid routed_experts payload") from exc
        if route.shape[0] > len(prompt) + len(completion):
            raise ValueError(
                f"{turn_label}: got {route.shape[0]} route rows for only "
                f"{len(prompt) + len(completion)} prompt+completion tokens"
            )
        if expected_layer_topk is None:
            expected_layer_topk = route.shape[1:]
        elif route.shape[1:] != expected_layer_topk:
            raise ValueError(
                f"{turn_label}: routed_experts shape {route.shape[1:]} differs from "
                f"the first turn's {expected_layer_topk}"
            )
        routes.append(route)
    return routes


def _failed_instances(trajectory_outputs: List[HarborTrajectoryOutput]):
    timeout_instance_ids = set()
    error_instance_ids = set()
    num_timeout_trajectories = 0
    num_error_trajectories = 0
    for traj in trajectory_outputs:
        instance_id = traj.trajectory_id.instance_id
        if traj.stop_reason == "agent_timeout":
            num_timeout_trajectories += 1
            timeout_instance_ids.add(instance_id)
        elif traj.stop_reason == "error" or traj.rollout_details is None:
            num_error_trajectories += 1
            error_instance_ids.add(instance_id)
    return (
        timeout_instance_ids | error_instance_ids,
        num_timeout_trajectories,
        num_error_trajectories,
    )


def _harbor_rollout_metrics(
    successful_trajectories: List[HarborTrajectoryOutput],
    response_ids: List[List[int]],
    rewards: List[float],
    trajectory_generation_times: List[Optional[float]],
    *,
    num_timeout_trajectories: int,
    num_error_trajectories: int,
    num_masked_instances: int,
) -> dict:
    """Build the common trajectory-level Harbor metrics for either training mode."""
    metric_times = (
        None
        if any(t is None for t in trajectory_generation_times)
        else trajectory_generation_times
    )
    if successful_trajectories:
        rollout_metrics = get_rollout_metrics(
            response_ids,
            rewards,
            trajectory_completion_times=metric_times,
        )
        denominator = len(successful_trajectories)
        rollout_metrics["generate/trajectories_context_length_exceeded"] = sum(
            1 for t in successful_trajectories if t.stop_reason == "context_length"
        )
        rollout_metrics["generate/avg_num_turns"] = (
            sum(t.num_turns for t in successful_trajectories) / denominator
        )
        fields = {
            "reward/avg_process_bonus": "process_bonus",
            "reward/avg_process_penalty": "process_penalty",
            "reward/premature_completion_rate": "process_premature_completion",
            "reward/avg_multi_tool_turns": "process_multi_tool_turns",
            "reward/wrote_tests_rate": "process_wrote_tests",
            "reward/ran_tests_rate": "process_ran_tests",
            "reward/used_syntax_checker_rate": "process_used_syntax_checker",
            "reward/avg_reasoning_chars_per_turn": "process_reasoning_chars_per_turn",
            "reward/avg_concise_reasoning_bonus": "process_concise_reasoning_bonus",
            "reward/avg_oracle_test_reward": "process_oracle_test_reward",
            "reward/avg_n_valid_tests": "process_n_valid_tests",
            "reward/avg_n_discriminating_tests": "process_n_discriminating_tests",
            "reward/avg_n_fixed_tests": "process_n_fixed_tests",
            "reward/avg_test_use_reward": "process_test_use_reward",
        }
        for metric_name, field_name in fields.items():
            rollout_metrics[metric_name] = (
                sum(getattr(t, field_name) for t in successful_trajectories)
                / denominator
            )

        shaped = [
            t for t in successful_trajectories if t.test_output_reward is not None
        ]
        if shaped:
            rollout_metrics["reward/avg_test_output_reward"] = sum(
                t.test_output_reward for t in shaped
            ) / len(shaped)
            rollout_metrics["reward/test_output_parse_rate"] = sum(
                t.test_output_parsed for t in shaped
            ) / len(shaped)
            rollout_metrics["reward/test_output_partial_credit_rate"] = sum(
                t.test_output_parsed and 0.0 < t.test_output_reward < 1.0
                for t in shaped
            ) / len(shaped)
    else:
        rollout_metrics = {}

    rollout_metrics["generate/num_timeout_trajectories"] = num_timeout_trajectories
    rollout_metrics["generate/num_error_trajectories"] = num_error_trajectories
    rollout_metrics["generate/num_masked_instances"] = num_masked_instances
    return rollout_metrics


def build_step_wise_generator_output(
    trajectory_outputs: List[HarborTrajectoryOutput],
    overlong_filtering: bool,
    return_routed_experts: bool = False,
) -> GeneratorOutput:
    """Flatten per-trajectory rollout details into one entry per LLM turn.

    Steps for one trajectory are emitted contiguously and the last step has
    ``is_last_step=True``. Failures (timeout / unknown error / empty rollout
    details) are batched per ``instance_id``: if any rollout for prompt P
    failed, all rollouts for P are replaced with single zeroed-out
    placeholder steps.
    """
    # 1. Identify failed instances. If any rollout for prompt P failed, mask all rollouts for P (conservative).
    masked_instance_ids, num_timeout_trajectories, num_error_trajectories = (
        _failed_instances(trajectory_outputs)
    )

    # 2. Walk trajectories and emit one entry of GeneratorOutput per step.
    prompt_token_ids: List[List[int]] = []
    response_ids: List[List[int]] = []
    rewards: List[float] = []
    loss_masks: List[List[int]] = []
    stop_reasons: List[str] = []
    is_last_step_list: List[bool] = []
    out_trajectory_ids: List[TrajectoryID] = []
    rollout_logprobs_list: List[List[float]] = []
    rollout_expert_indices: Optional[List[Optional[RoutedExpertIndices]]] = (
        [] if return_routed_experts else None
    )
    route_layer_topk = None

    successful_trajectories: List[HarborTrajectoryOutput] = []
    response_ids_for_metrics: List[List[int]] = []
    rewards_for_metrics: List[float] = []
    # One generation time per successful trajectory; used for completion-time metrics (avoids the
    # duplicate per-step entries below inflating the stats).
    trajectory_generation_times_per_prompt: List[Optional[float]] = []
    # One generation time per emitted step, aligned 1:1 with the flattened per-step arrays above.
    # Per trajectory we replicate its trajectory-level e2e_time across all of its steps.
    out_trajectory_generation_times: List[Optional[float]] = []
    for traj in trajectory_outputs:
        tid = traj.trajectory_id

        # 2.1. For failed trajectories, set loss mask to [0] and stop reason to "error".
        if tid.instance_id in masked_instance_ids:
            prompt_token_ids.append([0])
            response_ids.append([0])
            rewards.append(0.0)
            loss_masks.append([0])
            stop_reasons.append("error")
            is_last_step_list.append(True)
            out_trajectory_ids.append(tid)
            rollout_logprobs_list.append([0.0])
            if rollout_expert_indices is not None:
                rollout_expert_indices.append(None)
            out_trajectory_generation_times.append(traj.e2e_time)
            continue

        # 2.2. For successful trajectories, emit one entry per step.
        successful_trajectories.append(traj)

        # 2.3. Check rollout_details expected format.
        # Expect no summarization; rollout_details is a single linear chat segment from the main agent.
        # TODO(Charlie): Support summarization.
        (
            rollout_detail,
            prompt_token_ids_per_turn,
            completion_token_ids_per_turn,
            logprobs_per_turn,
        ) = _trajectory_rollout_detail(traj)
        n_turns = len(completion_token_ids_per_turn)
        turn_routes = (
            _turn_routed_experts(
                rollout_detail,
                prompt_token_ids_per_turn,
                completion_token_ids_per_turn,
                traj.trajectory_id,
            )
            if return_routed_experts
            else None
        )
        if turn_routes:
            if route_layer_topk is None:
                route_layer_topk = turn_routes[0].shape[1:]
            elif turn_routes[0].shape[1:] != route_layer_topk:
                raise ValueError(
                    f"Routed-expert shape changed across trajectories: {turn_routes[0].shape[1:]} "
                    f"vs {route_layer_topk}"
                )

        # 2.4. Emit one entry per step, following SkyRL's step-wise convention.
        for t in range(n_turns):
            comp_ids = completion_token_ids_per_turn[t]
            p_ids = prompt_token_ids_per_turn[t]
            lp = logprobs_per_turn[t]
            assert len(lp) == len(comp_ids), (
                "logprobs and completion token ids must have the same length."
            )

            # Record actual reward in last turn, and zeros for all other turns.
            is_last = t == n_turns - 1
            reward = traj.reward if is_last else 0.0

            # Loss mask.
            step_loss_mask = [1] * len(comp_ids)
            step_stop_reason = "complete"
            if traj.stop_reason == "context_length":
                step_stop_reason = "context_length"
                if overlong_filtering:
                    step_loss_mask = [0] * len(comp_ids)

            prompt_token_ids.append(p_ids)
            response_ids.append(comp_ids)
            rewards.append(reward)
            loss_masks.append(step_loss_mask)
            stop_reasons.append(step_stop_reason)
            is_last_step_list.append(is_last)
            out_trajectory_ids.append(tid)
            rollout_logprobs_list.append(lp)
            if rollout_expert_indices is not None:
                rollout_expert_indices.append(turn_routes[t])
            # For trajectory completion per turn we just use the trajectory-level e2e time.
            out_trajectory_generation_times.append(traj.e2e_time)

        # 2.5. For trajectory-level metrics, record the last turn's prompt IDs and response IDs which
        # contains the entire trajectory.
        response_ids_for_metrics.append(
            prompt_token_ids_per_turn[-1] + completion_token_ids_per_turn[-1]
        )
        rewards_for_metrics.append(
            traj.raw_reward if traj.raw_reward is not None else traj.reward
        )
        trajectory_generation_times_per_prompt.append(traj.e2e_time)

    # 3. Aggregate trajectory-level metrics for logging. Metrics use the per-prompt
    # times (not the per-step duplicates) to avoid skewing the stats.
    if any(t is None for t in out_trajectory_generation_times):
        out_trajectory_generation_times = None
    rollout_metrics = _harbor_rollout_metrics(
        successful_trajectories,
        response_ids_for_metrics,
        rewards_for_metrics,
        trajectory_generation_times_per_prompt,
        num_timeout_trajectories=num_timeout_trajectories,
        num_error_trajectories=num_error_trajectories,
        num_masked_instances=len(masked_instance_ids),
    )

    if rollout_expert_indices is not None:
        # A fully failed generation group has no model route shape to infer yet.
        # Keep a shape-less zero-row sentinel so fully-async concatenation can
        # combine it with a successful group before preprocessing.
        route_layer_topk = route_layer_topk or (0, 0)
        rollout_expert_indices = [
            route
            if route is not None
            else np.empty((0, route_layer_topk[0], route_layer_topk[1]), dtype=np.uint8)
            for route in rollout_expert_indices
        ]

    return GeneratorOutput(
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        stop_reasons=stop_reasons,
        rollout_metrics=rollout_metrics,
        rollout_logprobs=rollout_logprobs_list,
        rollout_expert_indices=rollout_expert_indices,
        trajectory_ids=out_trajectory_ids,
        is_last_step=is_last_step_list,
        # Per-step times, aligned 1:1 with the flattened per-step arrays above.
        trajectory_generation_times=out_trajectory_generation_times,
    )


def build_tito_generator_output(
    trajectory_outputs: List[HarborTrajectoryOutput],
    overlong_filtering: bool,
    return_routed_experts: bool = False,
) -> GeneratorOutput:
    """Build one exact token-in/token-out training row per Harbor trajectory.

    Each later chat prompt must contain the prior prompt, sampled completion,
    and environment observation as an exact token prefix. This lets us splice
    observations into the response with a zero loss mask without re-tokenizing
    any model action. A prompt that violates that invariant is conservatively
    masked together with the other samples for the same instance.
    """
    masked_instance_ids, num_timeout_trajectories, num_error_trajectories = (
        _failed_instances(trajectory_outputs)
    )
    assembled = [None] * len(trajectory_outputs)
    alignment_error_instances = set()
    route_layer_topk = None

    for index, traj in enumerate(trajectory_outputs):
        if traj.trajectory_id.instance_id in masked_instance_ids:
            continue
        rollout_detail, prompts, completions, logprobs = _trajectory_rollout_detail(
            traj
        )
        turn_routes = (
            _turn_routed_experts(
                rollout_detail, prompts, completions, traj.trajectory_id
            )
            if return_routed_experts
            else None
        )
        if turn_routes:
            if route_layer_topk is None:
                route_layer_topk = turn_routes[-1].shape[1:]
            elif turn_routes[-1].shape[1:] != route_layer_topk:
                raise ValueError(
                    f"Routed-expert shape changed across trajectories: {turn_routes[-1].shape[1:]} "
                    f"vs {route_layer_topk}"
                )

        initial_prompt = list(prompts[0])
        response: List[int] = []
        loss_mask: List[int] = []
        response_logprobs: List[float] = []
        alignment_error = None
        for turn, (prompt, completion, turn_logprobs) in enumerate(
            zip(prompts, completions, logprobs)
        ):
            accumulated = initial_prompt + response
            if turn and not (
                len(accumulated) <= len(prompt)
                and accumulated == prompt[: len(accumulated)]
            ):
                alignment_error = (
                    f"turn {turn} prompt is not an exact extension of the previously observed token stream "
                    f"(expected prefix length {len(accumulated)}, prompt length {len(prompt)})"
                )
                break

            observation_delta = prompt[len(accumulated) :] if turn else []
            response.extend(observation_delta)
            loss_mask.extend([0] * len(observation_delta))
            response_logprobs.extend([0.0] * len(observation_delta))

            response.extend(completion)
            completion_mask = (
                0 if traj.stop_reason == "context_length" and overlong_filtering else 1
            )
            loss_mask.extend([completion_mask] * len(completion))
            response_logprobs.extend(turn_logprobs)

        if alignment_error is not None:
            logger.warning(
                "TITO alignment failed for trajectory {}: {}. Masking instance {}.",
                traj.trajectory_id.to_string(),
                alignment_error,
                traj.trajectory_id.instance_id,
            )
            alignment_error_instances.add(traj.trajectory_id.instance_id)
            continue

        if initial_prompt + response != prompts[-1] + completions[-1]:
            raise AssertionError(
                "Internal TITO assembly error: final token stream differs from final Harbor turn"
            )
        assembled[index] = (
            initial_prompt,
            response,
            loss_mask,
            response_logprobs,
            turn_routes[-1] if turn_routes else None,
        )

    masked_instance_ids |= alignment_error_instances
    num_alignment_error_trajectories = sum(
        traj.trajectory_id.instance_id in alignment_error_instances
        for traj in trajectory_outputs
    )
    num_error_trajectories += num_alignment_error_trajectories

    prompt_token_ids: List[List[int]] = []
    response_ids: List[List[int]] = []
    rewards: List[float] = []
    loss_masks: List[List[int]] = []
    stop_reasons: List[str] = []
    rollout_logprobs: List[List[float]] = []
    trajectory_ids: List[TrajectoryID] = []
    trajectory_generation_times: List[Optional[float]] = []
    rollout_expert_indices: Optional[List[Optional[RoutedExpertIndices]]] = (
        [] if return_routed_experts else None
    )
    successful_trajectories = []
    metric_responses = []
    metric_rewards = []
    metric_times = []

    for index, traj in enumerate(trajectory_outputs):
        trajectory_ids.append(traj.trajectory_id)
        trajectory_generation_times.append(traj.e2e_time)
        if traj.trajectory_id.instance_id in masked_instance_ids:
            prompt_token_ids.append([0])
            response_ids.append([0])
            rewards.append(0.0)
            loss_masks.append([0])
            stop_reasons.append("error")
            rollout_logprobs.append([0.0])
            if rollout_expert_indices is not None:
                rollout_expert_indices.append(None)
            continue

        initial_prompt, response, loss_mask, response_logprobs, routes = assembled[
            index
        ]
        prompt_token_ids.append(initial_prompt)
        response_ids.append(response)
        rewards.append(traj.reward)
        loss_masks.append(loss_mask)
        stop_reasons.append(traj.stop_reason)
        rollout_logprobs.append(response_logprobs)
        if rollout_expert_indices is not None:
            rollout_expert_indices.append(routes)
        successful_trajectories.append(traj)
        metric_responses.append(initial_prompt + response)
        metric_rewards.append(
            traj.raw_reward if traj.raw_reward is not None else traj.reward
        )
        metric_times.append(traj.e2e_time)

    if rollout_expert_indices is not None:
        route_layer_topk = route_layer_topk or (0, 0)
        rollout_expert_indices = [
            route
            if route is not None
            else np.empty((0, route_layer_topk[0], route_layer_topk[1]), dtype=np.uint8)
            for route in rollout_expert_indices
        ]

    rollout_metrics = _harbor_rollout_metrics(
        successful_trajectories,
        metric_responses,
        metric_rewards,
        metric_times,
        num_timeout_trajectories=num_timeout_trajectories,
        num_error_trajectories=num_error_trajectories,
        num_masked_instances=len(masked_instance_ids),
    )
    rollout_metrics["generate/num_tito_alignment_error_trajectories"] = (
        num_alignment_error_trajectories
    )

    if any(t is None for t in trajectory_generation_times):
        trajectory_generation_times = None
    return GeneratorOutput(
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        stop_reasons=stop_reasons,
        rollout_metrics=rollout_metrics,
        rollout_logprobs=rollout_logprobs,
        rollout_expert_indices=rollout_expert_indices,
        trajectory_ids=trajectory_ids,
        is_last_step=None,
        trajectory_generation_times=trajectory_generation_times,
    )


class HarborGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        inference_engine_client: InferenceEngineInterface,
        tokenizer,
        max_seq_len: int,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            harbor_cfg: DictConfig object containing the Harbor configuration
            inference_engine_client: inference engine client for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
            max_seq_len: Maximum total sequence length (prompt + response). Used to truncate responses.
        """
        ie_cfg = generator_cfg.inference_engine
        self.base_url = inference_engine_client.get_endpoint_url()
        # Kept so we can notify the router when a session (trial attempt) ends,
        # which lets session-aware routing policies rebalance new trajectories.
        self.inference_engine_client = inference_engine_client
        self.generator_cfg = generator_cfg
        self.reward_shaping_cfg = getattr(generator_cfg, "reward_shaping", None)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.trajectory_mode = getattr(
            generator_cfg, "harbor_trajectory_mode", "step_wise"
        )
        self.return_routed_experts = bool(ie_cfg.enable_return_routed_experts)

        if self.trajectory_mode not in {"step_wise", "tito"}:
            raise ValueError(
                "generator.harbor_trajectory_mode must be one of: step_wise, tito; "
                f"got {self.trajectory_mode!r}"
            )
        step_wise = bool(getattr(generator_cfg, "step_wise_trajectories", False))
        merge_stepwise = bool(getattr(generator_cfg, "merge_stepwise_output", False))
        if self.trajectory_mode == "step_wise" and not step_wise:
            raise ValueError(
                "harbor_trajectory_mode=step_wise requires generator.step_wise_trajectories=true."
            )
        if self.trajectory_mode == "tito" and step_wise:
            raise ValueError(
                "harbor_trajectory_mode=tito requires generator.step_wise_trajectories=false."
            )
        if self.trajectory_mode == "tito" and merge_stepwise:
            raise ValueError(
                "harbor_trajectory_mode=tito requires generator.merge_stepwise_output=false."
            )
        if self.trajectory_mode == "step_wise" and not merge_stepwise:
            logger.warning(
                "merge_stepwise_output=true is not set; will not merge step-wise outputs. This "
                "may result in much slower training."
            )

        self._harbor_trial_config_template = deepcopy(harbor_cfg)

        # Mixed into the prefix-cache salt so distinct models / adapters don't share cache blocks.
        self._served_model_name = ie_cfg.served_model_name

        # Set model_name and api_base once (constant across all trials)
        assert ie_cfg.served_model_name is not None, "served_model_name must be set"
        assert "/" not in ie_cfg.served_model_name, (
            "served_model_name must not contain '/', Harbor expects hosted_vllm/{model_name}"
        )
        self._harbor_trial_config_template.setdefault("agent", {})["model_name"] = (
            f"hosted_vllm/{ie_cfg.served_model_name}"
        )
        self._harbor_trial_config_template["agent"].setdefault("kwargs", {})[
            "api_base"
        ] = f"{self.base_url}/v1"

        # Both modes need exact per-turn token IDs and logprobs from vLLM via Harbor.
        agent_kwargs = self._harbor_trial_config_template["agent"]["kwargs"]
        if not agent_kwargs.get("collect_rollout_details", False):
            logger.warning(
                "Harbor RL requires collect_rollout_details=true; enabling automatically."
            )
            agent_kwargs["collect_rollout_details"] = True

        # Can support summarization in future.
        if agent_kwargs.get("enable_summarize", False):
            raise ValueError(
                f"harbor_trajectory_mode={self.trajectory_mode} is incompatible with enable_summarize=true. "
                "Set harbor_trial_config.agent.kwargs.enable_summarize=false."
            )
        if (
            self.trajectory_mode == "tito"
            and agent_kwargs.get("output_length_retries", 1) != 0
        ):
            raise ValueError(
                "harbor_trajectory_mode=tito requires output_length_retries=0 because retry recovery "
                "mutates chat history without a matching sampled-token/logprob record."
            )

        logger.info(
            f"HarborGenerator initialized with Harbor config. "
            f"Agent: {self._harbor_trial_config_template.get('agent', {}).get('name')}, "
            f"Trials dir: {self._harbor_trial_config_template.get('trials_dir', 'trials')}, "
            f"trajectory_mode: {self.trajectory_mode}, R3: {self.return_routed_experts}"
        )

        rate_limit_config = getattr(generator_cfg, "rate_limit", None)
        self._rate_limiter = create_rate_limiter(rate_limit_config)

    def _compute_cache_salt(self) -> Optional[str]:
        """Derive a prefix-cache salt from the current policy version.

        Mirrors ``SkyRLGymGenerator._compute_cache_salt``: keyed on the engine's ``weight_version`` and
        served model name, called once per ``generate`` batch. Returns ``None`` when disabled or when the
        client exposes no weight version.
        """
        if not getattr(self.generator_cfg, "use_cache_salt", False):
            return None
        weight_version = getattr(self.inference_engine_client, "weight_version", None)
        if weight_version is None:
            return None
        version = (
            f"{self._served_model_name}@" if self._served_model_name is not None else ""
        )
        return f"{version}{weight_version}"

    async def generate(
        self, input_batch: GeneratorInput, disable_tqdm: bool = False
    ) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        sampling_params = dict(input_batch.get("sampling_params") or {})
        batch_metadata = input_batch.get("batch_metadata")

        logger.info(
            "Harbor sampling for {}: temperature={}, top_p={}, top_k={}, max_tokens={}",
            getattr(batch_metadata, "training_phase", "unknown"),
            sampling_params.get("temperature"),
            sampling_params.get("top_p"),
            sampling_params.get("top_k"),
            sampling_params.get("max_tokens"),
        )

        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match trajectory_ids count ({len(trajectory_ids)})"
            )

        # Captured once so every trajectory shares the policy version at the start of the batch.
        cache_salt = self._compute_cache_salt()

        all_outputs: List[HarborTrajectoryOutput] = [None] * len(prompts)  # type: ignore[list-item]
        progress = tqdm(
            disable=disable_tqdm,  # disable for fully async training
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def _worker(idx, prompt, trajectory_id):
            result = await self._harbor_agent_loop(
                prompt=prompt,
                trajectory_id=trajectory_id,
                cache_salt=cache_salt,
                sampling_params=sampling_params,
            )
            all_outputs[idx] = result
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, (prompt, trajectory_id) in enumerate(
                    zip(prompts, trajectory_ids)
                ):
                    tg.create_task(_worker(idx, prompt, trajectory_id))
        finally:
            progress.close()

        builder = (
            build_step_wise_generator_output
            if self.trajectory_mode == "step_wise"
            else build_tito_generator_output
        )
        return builder(
            all_outputs,
            overlong_filtering=self.generator_cfg.apply_overlong_filtering,
            return_routed_experts=self.return_routed_experts,
        )

    async def _harbor_agent_loop(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str] = None,
        sampling_params: Optional[dict] = None,
    ) -> HarborTrajectoryOutput:
        """Run a single Harbor trial and return the rollout details plus a trajectory-level reward.
        Retries on unknown errors; context length errors train with reward=0; agent timeouts mask the trajectory.
        """
        agent_loop_start_time = time.monotonic()
        reward = None
        results = None
        rollout_details = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False

        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i + 1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            # Each attempt is a distinct router session; track it so it can be
            # released on completion/error/cancellation.
            session_id = uuid4().hex
            try:
                # Create a fresh Trial each attempt so agent state is clean on retry.
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                agent_kwargs = config["agent"]["kwargs"]
                agent_kwargs["session_id"] = session_id

                # SkyRL supplies distinct train/eval sampling parameters in each
                # GeneratorInput. Apply them to the per-trial Harbor config rather
                # than using the fixed template values for both phases.
                if sampling_params:
                    if sampling_params.get("temperature") is not None:
                        agent_kwargs["temperature"] = sampling_params["temperature"]

                    llm_kwargs = agent_kwargs.setdefault("llm_kwargs", {})
                    for key in ("top_p", "top_k", "min_p", "stop"):
                        if sampling_params.get(key) is not None:
                            llm_kwargs[key] = sampling_params[key]

                    llm_call_kwargs = agent_kwargs.setdefault("llm_call_kwargs", {})
                    if sampling_params.get("max_tokens") is not None:
                        llm_call_kwargs["max_tokens"] = sampling_params["max_tokens"]

                    # These vLLM-only controls are OpenAI-compatible extensions.
                    # LiteLLM forwards them in extra_body.
                    extra_body = llm_call_kwargs.setdefault("extra_body", {})
                    for key in (
                        "min_tokens",
                        "skip_special_tokens",
                        "include_stop_str_in_output",
                    ):
                        if sampling_params.get(key) is not None:
                            extra_body[key] = sampling_params[key]
                # Forward the salt via llm_kwargs.extra_body -> LiteLLM -> the vLLM request's top-level
                # `cache_salt` field. vLLM rejects an empty salt, so attach only when set.
                if cache_salt is not None:
                    llm_kwargs = config["agent"]["kwargs"].setdefault("llm_kwargs", {})
                    extra_body = llm_kwargs.setdefault("extra_body", {})
                    if not isinstance(extra_body, dict):
                        raise TypeError(
                            "harbor_trial_config.agent.kwargs.llm_kwargs.extra_body must be a mapping"
                        )
                    extra_body["cache_salt"] = cache_salt
                trial_config = TrialConfig.model_validate(config)
                trial = await Trial.create(trial_config)

                async with self._rate_limiter:
                    results = await trial.run()

                # Parse exception type
                exc_type = (
                    results.exception_info.exception_type
                    if results.exception_info
                    else None
                )
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"

                # Determine reward.
                if is_agent_timeout_error:
                    # AgentTimeoutError: not successful, no retry, loss-masked
                    logger.debug(
                        f"{prefix} hit AgentTimeoutError (no retry). Results: {results}"
                    )
                    break
                elif is_context_length_error:
                    # ContextLengthExceededError: always train with reward=0.
                    logger.debug(
                        f"{prefix} hit ContextLengthExceededError, setting reward=0. Results: {results}"
                    )
                    reward = 0.0
                elif not results.verifier_result:
                    # Does not have a verifier result, so it's not successful, will retry
                    logger.warning(
                        f"{prefix} failed: Exception info: {results.exception_info}. Results: {results}"
                    )
                    continue
                else:
                    reward = float(results.verifier_result.rewards["reward"])

                raw_reward = reward
                test_output_reward = None
                test_output_result = None
                if (
                    self.reward_shaping_cfg is not None
                    and getattr(self.reward_shaping_cfg, "enable_reward_shaping", False)
                    and not is_context_length_error
                ):
                    verifier_stdout = read_bounded_verifier_output(
                        Path(trial.paths.test_stdout_path),
                        getattr(
                            self.reward_shaping_cfg,
                            "reward_shaping_max_output_bytes",
                            262_144,
                        ),
                    )
                    test_output_result = shape_reward_from_output(
                        stdout=verifier_stdout,
                        original_reward=raw_reward,
                        parser_name=getattr(
                            self.reward_shaping_cfg, "reward_parser", None
                        ),
                        shaper_name=getattr(
                            self.reward_shaping_cfg, "reward_shaper", "pass_ratio"
                        ),
                        fallback_to_original=getattr(
                            self.reward_shaping_cfg,
                            "reward_shaping_fallback",
                            True,
                        ),
                    )
                    reward = test_output_result.reward
                    test_output_reward = reward
                    logger.debug(
                        f"{prefix} test-output reward: raw={raw_reward:.4f}, "
                        f"shaped={reward:.4f}, parsed={test_output_result.parsed}, "
                        f"passed={test_output_result.passed}/{test_output_result.total}, "
                        f"parser={test_output_result.parser}"
                    )

                if self.reward_shaping_cfg is not None:
                    process_reward = await score_harbor_trajectory_async(
                        Path(trial.paths.agent_dir) / "trajectory.json",
                        verifier_reward=reward,
                        config=self.reward_shaping_cfg,
                    )
                    reward = process_reward.training_reward
                    process_bonus = process_reward.bonus
                    process_penalty = process_reward.penalty
                    if process_bonus > 0 or process_penalty > 0:
                        logger.debug(
                            f"{prefix} process reward: raw={raw_reward:.4f}, bonus={process_bonus:.4f}, "
                            f"penalty={process_penalty:.4f}, "
                            f"multi_tool_turns={process_reward.multi_tool_turns}, "
                            f"wrote_tests={process_reward.wrote_tests}, ran_tests={process_reward.ran_tests}, "
                            f"syntax_checker={process_reward.used_syntax_checker}, "
                            f"premature_completion={process_reward.premature_completion}, "
                            f"reasoning_chars_per_turn={process_reward.reasoning_chars_per_turn:.1f}, "
                            f"concision_bonus={process_reward.concise_reasoning_bonus:.4f}"
                        )
                else:
                    process_bonus = 0.0
                    process_penalty = 0.0
                    process_reward = None

                # Extract rollout details and check for success
                rollout_details = results.agent_result.rollout_details
                num_turns = results.agent_result.metadata["n_episodes"]

                if (
                    rollout_details
                    and len(rollout_details) >= 1
                    and len(rollout_details[0].get("completion_token_ids", [])) > 0
                ):
                    successful = True
                    logger.debug(f"{prefix} successful: reward={reward}.")
                    break
                else:
                    logger.warning(
                        f"{prefix} failed: empty/missing rollout_details. Results: {results}"
                    )
            except Exception as e:
                logger.warning(
                    f"{prefix} failed: Error running trial: {e}. Results: {results}"
                )
                continue
            finally:
                await self.inference_engine_client.finish_session(session_id)

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            error_message = f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), will set loss mask to [0]."
            if stop_reason == "error":
                error_message += f" Results: {results}"
            logger.warning(error_message)
            return HarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=None,
                stop_reason=stop_reason,
                e2e_time=time.monotonic() - agent_loop_start_time,
            )
        else:
            return HarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=rollout_details,
                reward=reward,
                raw_reward=raw_reward,
                test_output_reward=test_output_reward,
                test_output_parsed=(
                    test_output_result.parsed
                    if test_output_result is not None
                    else False
                ),
                test_output_passed=(
                    test_output_result.passed if test_output_result is not None else 0
                ),
                test_output_total=(
                    test_output_result.total if test_output_result is not None else 0
                ),
                process_bonus=process_bonus,
                process_penalty=process_penalty,
                process_premature_completion=(
                    process_reward.premature_completion
                    if process_reward is not None
                    else False
                ),
                process_multi_tool_turns=process_reward.multi_tool_turns
                if process_reward is not None
                else 0,
                process_wrote_tests=process_reward.wrote_tests
                if process_reward is not None
                else False,
                process_ran_tests=process_reward.ran_tests
                if process_reward is not None
                else False,
                process_used_syntax_checker=(
                    process_reward.used_syntax_checker
                    if process_reward is not None
                    else False
                ),
                process_reasoning_chars_per_turn=(
                    process_reward.reasoning_chars_per_turn
                    if process_reward is not None
                    else 0.0
                ),
                process_concise_reasoning_bonus=(
                    process_reward.concise_reasoning_bonus
                    if process_reward is not None
                    else 0.0
                ),
                process_oracle_test_reward=(
                    process_reward.oracle_test_reward
                    if process_reward is not None
                    else 0.0
                ),
                process_n_valid_tests=(
                    process_reward.n_valid_tests if process_reward is not None else 0
                ),
                process_n_discriminating_tests=(
                    process_reward.n_discriminating_tests
                    if process_reward is not None
                    else 0
                ),
                process_n_fixed_tests=(
                    process_reward.n_fixed_tests if process_reward is not None else 0
                ),
                process_test_use_reward=(
                    process_reward.test_use_reward
                    if process_reward is not None
                    else 0.0
                ),
                num_turns=num_turns,
                stop_reason="context_length" if is_context_length_error else "complete",
                e2e_time=time.monotonic() - agent_loop_start_time,
            )
