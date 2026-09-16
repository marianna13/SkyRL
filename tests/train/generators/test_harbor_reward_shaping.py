import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.train_integrations.harbor.reward_shaping import (
    HarborRewardShapingConfig,
    score_harbor_trajectory,
)


def _write_trajectory(path, commands_by_turn, messages_by_turn=None):
    steps = [{"step_id": 1, "source": "user", "message": "task"}]
    for turn_index, commands in enumerate(commands_by_turn):
        step_id = turn_index + 2
        steps.append(
            {
                "step_id": step_id,
                "source": "agent",
                "message": (
                    messages_by_turn[turn_index]
                    if messages_by_turn is not None
                    else "working"
                ),
                "tool_calls": [
                    {
                        "tool_call_id": f"call_{step_id}_{i}",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": command, "duration": 1},
                    }
                    for i, command in enumerate(commands)
                ],
            }
        )
    path.write_text(json.dumps({"steps": steps}))


class TestHarborRewardShaping(unittest.TestCase):
    def test_scores_capped_success_gated_process_bonus(self):
        with TemporaryDirectory() as directory:
            trajectory_path = Path(directory) / "trajectory.json"
            _write_trajectory(
                trajectory_path,
                [
                    [
                        "cat > test_solution.py <<'EOF'\nassert True\nEOF",
                        "python -m pytest -q",
                    ],
                    ["python -m py_compile solution.py", "echo done"],
                ],
            )
            config = HarborRewardShapingConfig(enabled=True)

            result = score_harbor_trajectory(trajectory_path, 1.0, config)

            self.assertAlmostEqual(result.bonus, 0.05)
            self.assertAlmostEqual(result.training_reward, 1.05)
            self.assertEqual(result.multi_tool_turns, 2)
            self.assertTrue(result.wrote_tests)
            self.assertTrue(result.ran_tests)
            self.assertTrue(result.used_syntax_checker)

            failed_result = score_harbor_trajectory(trajectory_path, 0.0, config)
            self.assertEqual(failed_result.bonus, 0.0)
            self.assertEqual(failed_result.penalty, 0.05)
            self.assertEqual(failed_result.training_reward, -0.05)

    def test_disabled_or_malformed_trace_is_safe(self):
        with TemporaryDirectory() as directory:
            trajectory_path = Path(directory) / "trajectory.json"
            trajectory_path.write_text("not json")

            enabled = score_harbor_trajectory(
                trajectory_path, 1.0, HarborRewardShapingConfig(enabled=True)
            )
            disabled = score_harbor_trajectory(
                trajectory_path, 1.0, HarborRewardShapingConfig(enabled=False)
            )

            self.assertEqual(enabled.training_reward, 1.0)
            self.assertEqual(disabled.training_reward, 1.0)

    def test_success_gated_concision_bonus(self):
        config = HarborRewardShapingConfig(
            enabled=True,
            max_bonus=0.1,
            multi_tool_turn_bonus=0.0,
            wrote_tests_bonus=0.0,
            ran_tests_bonus=0.0,
            syntax_checker_bonus=0.0,
            concise_reasoning_bonus=0.02,
            concise_reasoning_target_chars_per_turn=10,
            concise_reasoning_max_chars_per_turn=30,
        )

        with TemporaryDirectory() as directory:
            trajectory_path = Path(directory) / "trajectory.json"

            _write_trajectory(trajectory_path, [[]], ["short"])
            short = score_harbor_trajectory(trajectory_path, 1.0, config)
            self.assertAlmostEqual(short.concise_reasoning_bonus, 0.02)
            self.assertAlmostEqual(short.training_reward, 1.02)

            _write_trajectory(trajectory_path, [[]], ["x" * 20])
            medium = score_harbor_trajectory(trajectory_path, 1.0, config)
            self.assertAlmostEqual(medium.concise_reasoning_bonus, 0.01)
            self.assertAlmostEqual(medium.reasoning_chars_per_turn, 20.0)

            _write_trajectory(trajectory_path, [[]], ["x" * 30])
            long = score_harbor_trajectory(trajectory_path, 1.0, config)
            self.assertEqual(long.concise_reasoning_bonus, 0.0)

            failed = score_harbor_trajectory(trajectory_path, 0.0, config)
            self.assertEqual(failed.concise_reasoning_bonus, 0.0)
            self.assertEqual(failed.training_reward, -0.05)

    def test_penalizes_completion_with_an_unresolved_observed_failure(self):
        with TemporaryDirectory() as directory:
            trajectory_path = Path(directory) / "trajectory.json"
            trajectory_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {"source": "user", "message": "task"},
                            {
                                "source": "agent",
                                "message": "Check the sample.",
                                "tool_calls": [
                                    {
                                        "function_name": "bash_command",
                                        "arguments": {
                                            "keystrokes": "python /app/solution.py"
                                        },
                                    }
                                ],
                                "observation": {
                                    "results": [
                                        {"content": "FAILED! Expected: 7, Actual: 5"}
                                    ]
                                },
                            },
                            {
                                "source": "agent",
                                "message": "The output is wrong, but submit.",
                                "tool_calls": [
                                    {
                                        "function_name": "mark_task_complete",
                                        "arguments": {},
                                    }
                                ],
                            },
                        ]
                    }
                )
            )

            result = score_harbor_trajectory(
                trajectory_path, 0.0, HarborRewardShapingConfig(enabled=True)
            )

            self.assertTrue(result.premature_completion)
            self.assertEqual(result.penalty, 0.1)
            self.assertEqual(result.training_reward, -0.1)

    def test_solution_edit_clears_observed_failure_before_completion(self):
        with TemporaryDirectory() as directory:
            trajectory_path = Path(directory) / "trajectory.json"
            trajectory_path.write_text(
                json.dumps(
                    {
                        "steps": [
                            {"source": "user", "message": "task"},
                            {
                                "source": "agent",
                                "message": "Check the sample.",
                                "tool_calls": [],
                                "observation": {
                                    "results": [{"content": "AssertionError: mismatch"}]
                                },
                            },
                            {
                                "source": "agent",
                                "message": "Fix and submit.",
                                "tool_calls": [
                                    {
                                        "function_name": "bash_command",
                                        "arguments": {
                                            "keystrokes": "cat > /app/solution.py <<'PY'\nprint(7)\nPY"
                                        },
                                    },
                                    {
                                        "function_name": "mark_task_complete",
                                        "arguments": {},
                                    },
                                ],
                            },
                        ]
                    }
                )
            )

            result = score_harbor_trajectory(
                trajectory_path, 0.0, HarborRewardShapingConfig(enabled=True)
            )

            self.assertFalse(result.premature_completion)
            self.assertEqual(result.penalty, 0.05)
            self.assertEqual(result.training_reward, -0.05)

    def test_negative_shaping_can_be_disabled_with_zero_weights(self):
        with TemporaryDirectory() as directory:
            trajectory_path = Path(directory) / "trajectory.json"
            _write_trajectory(trajectory_path, [[]])
            config = HarborRewardShapingConfig(
                enabled=True,
                verifier_failure_penalty=0.0,
                premature_completion_penalty=0.0,
                max_penalty=0.0,
            )

            result = score_harbor_trajectory(trajectory_path, 0.0, config)

            self.assertEqual(result.penalty, 0.0)
            self.assertEqual(result.training_reward, 0.0)


if __name__ == "__main__":
    unittest.main()
