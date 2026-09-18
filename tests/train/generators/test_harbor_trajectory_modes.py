import numpy as np

from examples.train_integrations.harbor.harbor_generator import (
    HarborTrajectoryOutput,
    build_step_wise_generator_output,
    build_tito_generator_output,
)
from skyrl.train.generators.base import TrajectoryID


def _trajectory(
    instance_id="task", repetition_id=0, *, prompts=None, completions=None, routes=None
):
    prompts = prompts or [[10], [10, 20, 30]]
    completions = completions or [[20], [40]]
    detail = {
        "prompt_token_ids": prompts,
        "completion_token_ids": completions,
        "logprobs": [[-0.1] * len(completion) for completion in completions],
    }
    if routes is not None:
        detail["extra"] = {"routed_experts": routes}
    return HarborTrajectoryOutput(
        trajectory_id=TrajectoryID(
            instance_id=instance_id, repetition_id=repetition_id
        ),
        rollout_details=[detail],
        reward=1.25,
        raw_reward=1.0,
        num_turns=len(completions),
        e2e_time=2.0,
    )


def test_tito_assembles_exact_multiturn_stream():
    output = build_tito_generator_output([_trajectory()], overlong_filtering=True)

    assert output["prompt_token_ids"] == [[10]]
    assert output["response_ids"] == [[20, 30, 40]]
    assert output["loss_masks"] == [[1, 0, 1]]
    assert output["rollout_logprobs"] == [[-0.1, 0.0, -0.1]]
    assert output["rewards"] == [1.25]
    assert output["is_last_step"] is None


def test_tito_masks_all_samples_for_instance_on_alignment_failure():
    malformed = _trajectory(
        repetition_id=0,
        prompts=[[10], [99, 20, 30]],
    )
    otherwise_valid = _trajectory(repetition_id=1)

    output = build_tito_generator_output(
        [malformed, otherwise_valid],
        overlong_filtering=True,
    )

    assert output["prompt_token_ids"] == [[0], [0]]
    assert output["response_ids"] == [[0], [0]]
    assert output["loss_masks"] == [[0], [0]]
    assert output["rewards"] == [0.0, 0.0]
    assert (
        output["rollout_metrics"]["generate/num_tito_alignment_error_trajectories"] == 2
    )


def test_both_modes_propagate_vllm_routes():
    first_routes = np.asarray([[[1, 2]]], dtype=np.uint8)
    final_routes = np.asarray([[[3, 4]], [[5, 6]], [[7, 8]]], dtype=np.uint8)
    trajectory = _trajectory(routes=[first_routes, final_routes])

    stepwise = build_step_wise_generator_output(
        [trajectory],
        overlong_filtering=True,
        return_routed_experts=True,
    )
    tito = build_tito_generator_output(
        [trajectory],
        overlong_filtering=True,
        return_routed_experts=True,
    )

    np.testing.assert_array_equal(stepwise["rollout_expert_indices"][0], first_routes)
    np.testing.assert_array_equal(stepwise["rollout_expert_indices"][1], final_routes)
    np.testing.assert_array_equal(tito["rollout_expert_indices"][0], final_routes)


def test_r3_builds_zero_length_routes_for_failed_placeholder():
    failed = HarborTrajectoryOutput(
        trajectory_id=TrajectoryID(instance_id="failed", repetition_id=0),
        rollout_details=None,
        stop_reason="error",
    )
    routes = [
        np.asarray([[[1, 2]]], dtype=np.uint8),
        np.asarray([[[3, 4]], [[5, 6]], [[7, 8]]], dtype=np.uint8),
    ]
    valid = _trajectory(instance_id="valid", routes=routes)

    output = build_tito_generator_output(
        [failed, valid],
        overlong_filtering=True,
        return_routed_experts=True,
    )

    assert output["rollout_expert_indices"][0].shape == (0, 1, 2)
    np.testing.assert_array_equal(output["rollout_expert_indices"][1], routes[-1])


def test_all_failed_r3_group_uses_shape_less_route_sentinel():
    failed = HarborTrajectoryOutput(
        trajectory_id=TrajectoryID(instance_id="failed", repetition_id=0),
        rollout_details=None,
        stop_reason="error",
    )

    output = build_tito_generator_output(
        [failed],
        overlong_filtering=True,
        return_routed_experts=True,
    )

    assert output["rollout_expert_indices"][0].shape == (0, 0, 0)
