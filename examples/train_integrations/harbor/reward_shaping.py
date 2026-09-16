"""Conservative, trajectory-derived reward shaping for Harbor training."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any


# Keep blocking oracle subprocesses off the rollout event loop and its default
# executor (also used for bridge I/O). One worker preserves serial cache access
# and bounds subprocess concurrency even when an awaiting rollout is cancelled.
_SCORING_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="harbor-reward")


async def score_harbor_trajectory_async(
    trajectory_path: Path,
    verifier_reward: float,
    config: HarborRewardShapingConfig | Mapping[str, Any] | Any,
) -> ProcessRewardResult:
    """Score in a dedicated worker so logging and router requests can progress."""
    return await asyncio.get_running_loop().run_in_executor(
        _SCORING_EXECUTOR,
        partial(score_harbor_trajectory, trajectory_path, verifier_reward, config),
    )


@dataclass
class HarborRewardShapingConfig:
    """Weights for optional Harbor process reward shaping.

    Positive bonuses are verifier-success-gated by default, so unsuccessful
    trajectories cannot farm reward by issuing irrelevant commands. Penalties
    remain active on verifier failures to distinguish ordinary failures from
    premature submissions after an observed test failure.
    """

    enabled: bool = False

    # Verifier-output shaping from the original SkyRL Harbor integration.
    # This is independent from the trajectory/process bonuses below: when
    # enabled, the verifier's binary reward is replaced by passed / total for
    # multi-test tasks.  Single-test and unparseable outputs retain the binary
    # verifier reward.
    enable_reward_shaping: bool = False
    reward_parser: str | None = None
    reward_shaper: str = "pass_ratio"
    reward_shaping_fallback: bool = True
    reward_shaping_max_output_bytes: int = 262_144

    require_verifier_success: bool = True
    max_bonus: float = 0.05
    multi_tool_turn_bonus: float = 0.005
    max_multi_tool_turns: int = 2
    wrote_tests_bonus: float = 0.015
    ran_tests_bonus: float = 0.015
    syntax_checker_bonus: float = 0.01
    concise_reasoning_bonus: float = 0.02
    concise_reasoning_target_chars_per_turn: int = 600
    concise_reasoning_max_chars_per_turn: int = 1800
    verifier_failure_penalty: float = 0.05
    premature_completion_penalty: float = 0.05
    max_penalty: float = 0.1

    # --- oracle-checked discriminating-test reward (see oracle_test_reward.py) ---
    # Rewards MEANINGFUL tests (correct expected outputs + bug-catching inputs),
    # graded against verified oracle solutions -- not the gameable wrote_tests regex.
    # Added ON TOP of `max_bonus` (independent budget), verifier-success-gated like
    # the other bonuses.
    oracle_test_enabled: bool = False
    oracle_test_dir: str = ""          # dir of <task>.py verified oracles
    oracle_test_task_dir: str = ""     # dataset dir (<task>/tests/test_data.json) for shipped I/O
    oracle_test_bonus: float = 0.05    # scale: final add = oracle_test_bonus * reward[0,1]
    oracle_test_timeout: float = 6.0
    oracle_test_target: int = 3
    # test-USE (caught-and-fixed) reward: credit when a later solution version fixes a valid
    # test an earlier version failed -> the causal self-correction signal for pass@1. 0 = off.
    oracle_test_use_bonus: float = 0.0
    oracle_test_use_target: int = 2


@dataclass(frozen=True)
class ProcessRewardResult:
    raw_reward: float
    bonus: float = 0.0
    multi_tool_turns: int = 0
    wrote_tests: bool = False
    ran_tests: bool = False
    used_syntax_checker: bool = False
    reasoning_chars: int = 0
    reasoning_chars_per_turn: float = 0.0
    concise_reasoning_bonus: float = 0.0
    penalty: float = 0.0
    premature_completion: bool = False
    oracle_test_reward: float = 0.0
    n_valid_tests: int = 0
    n_discriminating_tests: int = 0
    n_fixed_tests: int = 0
    test_use_reward: float = 0.0

    @property
    def training_reward(self) -> float:
        return self.raw_reward + self.bonus - self.penalty


@dataclass(frozen=True)
class TestOutputRewardResult:
    """Result of shaping a verifier reward from its test-runner output."""

    reward: float
    parsed: bool = False
    passed: int = 0
    total: int = 0
    parser: str | None = None

    @property
    def is_partial_credit(self) -> bool:
        return self.parsed and 0.0 < self.reward < 1.0


# These reproduce the existing SkyRL Harbor pass-ratio semantics while parsing
# one line at a time.  The older whole-output SUMMARY_PATTERN can backtrack for
# a very long time on unusually large verifier logs.
_PYTEST_SUMMARY_LINE_RE = re.compile(
    r"=+\s*(?P<results>(?:\d+\s+\w+(?:,\s*)?)+)"
    r"\s+in\s+(?P<duration>[\d.]+)s?\s*=+",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(r"(\d+)\s+(\w+)", re.IGNORECASE)
_PYTEST_COLLECTION_ERRORS = (
    "error during collection",
    "collection error",
    "no tests ran",
    "import error",
)
_UNITTEST_RAN_RE = re.compile(
    r"Ran\s+(\d+)\s+tests?\s+in\s+[\d.]+s", re.IGNORECASE
)
_UNITTEST_FAILED_RE = re.compile(r"FAILED\s*\(([^)]+)\)", re.IGNORECASE)
_UNITTEST_OK_RE = re.compile(r"^OK(?:\s*\(([^)]+)\))?", re.MULTILINE | re.IGNORECASE)
_KEY_VALUE_RE = re.compile(r"(\w+)=(\d+)")


def read_bounded_verifier_output(path: Path, max_bytes: int = 262_144) -> str | None:
    """Read the tail of verifier stdout without loading arbitrarily large logs."""

    try:
        max_bytes = max(1, int(max_bytes))
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            return stream.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return None


def _pytest_counts(output: str) -> tuple[int, int] | None:
    lowered = output.lower()
    if any(marker in lowered for marker in _PYTEST_COLLECTION_ERRORS) or re.search(
        r"interrupted:.*error", output, re.IGNORECASE
    ):
        return None

    for line in reversed(output.splitlines()):
        if " in " not in line or "=" not in line:
            continue
        match = _PYTEST_SUMMARY_LINE_RE.search(line)
        if not match:
            continue
        counts = {
            "passed": 0,
            "failed": 0,
            "error": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
            "skipped": 0,
        }
        for count_match in _COUNT_RE.finditer(match.group("results")):
            status = count_match.group(2).lower()
            if status in counts:
                counts[status] = int(count_match.group(1))
        total = sum(counts.values())
        if total:
            return counts["passed"], total

    statuses = {
        "passed": len(re.findall(r"^PASSED\s+", output, re.MULTILINE)),
        "failed": len(re.findall(r"^FAILED\s+", output, re.MULTILINE)),
        "error": len(re.findall(r"^ERROR\s+", output, re.MULTILINE)),
        "xfailed": len(re.findall(r"^XFAIL\s+", output, re.MULTILINE)),
        "xpassed": len(re.findall(r"^XPASS\s+", output, re.MULTILINE)),
        "skipped": len(re.findall(r"^SKIPPED\s+", output, re.MULTILINE)),
    }
    total = sum(statuses.values())
    return (statuses["passed"], total) if total else None


def _unittest_counts(output: str) -> tuple[int, int] | None:
    ran_match = _UNITTEST_RAN_RE.search(output)
    if not ran_match:
        return None
    total = int(ran_match.group(1))
    counts = {"failures": 0, "errors": 0, "skipped": 0}
    result_match = _UNITTEST_FAILED_RE.search(output) or _UNITTEST_OK_RE.search(output)
    if result_match and result_match.group(1):
        for match in _KEY_VALUE_RE.finditer(result_match.group(1)):
            if match.group(1).lower() in counts:
                counts[match.group(1).lower()] = int(match.group(2))
    passed = max(0, total - sum(counts.values()))
    return passed, total


def _generic_counts(output: str) -> tuple[int, int] | None:
    passed = len(re.findall(r"\bPASS(?:ED)?\b|\bOK\b|\bSUCCESS\b|\[PASS\]|✓", output, re.IGNORECASE))
    failed = len(re.findall(r"\bFAIL(?:ED|URE)?\b|\[FAIL\]|✗|✘", output, re.IGNORECASE))
    errors = len(re.findall(r"\bERROR\b|\[ERROR\]", output, re.IGNORECASE))
    total = passed + failed + errors
    return (passed, total) if total else None


def shape_reward_from_output(
    stdout: str | None,
    original_reward: float,
    parser_name: str | None = None,
    shaper_name: str = "pass_ratio",
    fallback_to_original: bool = True,
) -> TestOutputRewardResult:
    """Apply SkyRL Harbor's test pass-ratio reward to verifier stdout.

    ``pass_ratio`` deliberately falls back for a one-test task, matching the
    original implementation: a single test provides no useful partial signal.
    """

    original_reward = float(original_reward)
    if not stdout:
        return TestOutputRewardResult(original_reward if fallback_to_original else 0.0)

    selected = parser_name.lower() if parser_name else None
    if selected not in {None, "pytest", "unittest", "generic"}:
        raise ValueError(f"Unknown reward parser: {parser_name}")

    parsed: tuple[int, int] | None = None
    used_parser: str | None = None
    candidates = [selected] if selected else ["pytest", "unittest", "generic"]
    for candidate in candidates:
        if candidate == "pytest":
            parsed = _pytest_counts(stdout)
        elif candidate == "unittest":
            parsed = _unittest_counts(stdout)
        else:
            parsed = _generic_counts(stdout)
        if parsed is not None:
            used_parser = candidate
            break

    if parsed is None:
        return TestOutputRewardResult(original_reward if fallback_to_original else 0.0)

    passed, total = parsed
    if shaper_name != "pass_ratio":
        raise ValueError(
            f"This Harbor integration currently supports reward_shaper=pass_ratio, got {shaper_name!r}"
        )
    reward = original_reward if total <= 1 else passed / total
    return TestOutputRewardResult(
        reward=reward,
        parsed=True,
        passed=passed,
        total=total,
        parser=used_parser,
    )


_TEST_PATH = r"(?:^|[/\\\s'\"])(?:test_[^/\\\s]+|[^/\\\s]+_test\.[^/\\\s]+|[^/\\\s]+\.(?:spec|test)\.[^/\\\s]+)"
_WRITE_TEST_RE = re.compile(
    rf"(?:\b(?:cat|tee|touch|cp|mv|install)\b[^\n]*{_TEST_PATH}|(?:>>?|\btee\b)\s*[^\n]*{_TEST_PATH})",
    re.IGNORECASE,
)
_RUN_TEST_RE = re.compile(
    r"(?:^|[;&|]\s*|\b)(?:"
    r"(?:python(?:3)?\s+-m\s+)?pytest|"
    r"python(?:3)?\s+-m\s+unittest|"
    r"(?:python(?:3)?\s+)?[^\s;&|]*test_[^\s;&|]*\.py|"
    r"ctest|cargo\s+test|go\s+test|npm\s+(?:run\s+)?test|"
    r"pnpm\s+(?:run\s+)?test|yarn\s+test|mvn\s+test|gradle\s+test"
    r")\b",
    re.IGNORECASE,
)
_SYNTAX_CHECK_RE = re.compile(
    r"(?:^|[;&|]\s*|\b)(?:"
    r"python(?:3)?\s+-m\s+(?:py_compile|compileall)|"
    r"ruff(?:\s+check)?|mypy|pyright|shellcheck|eslint|tsc|"
    r"cargo\s+check|go\s+vet|"
    r"(?:gcc|g\+\+|clang|clang\+\+)\b[^\n;&|]*-fsyntax-only"
    r")\b",
    re.IGNORECASE,
)

_OBSERVED_FAILURE_RE = re.compile(
    r"(?:FAILED!\s+Expected:|\bAssertionError\b|\b(?:Syntax|Indentation)Error\b|"
    r"Traceback \(most recent call last\)|\bcommand timed out\b|"
    r"\bsolution timed out\b|\bExec failed\b)",
    re.IGNORECASE,
)
_ACKNOWLEDGED_FAILURE_RE = re.compile(
    r"(?:\b(?:solution|code|algorithm|approach|answer|result|output)\b.{0,100}"
    r"\b(?:is|was|seems?|looks?)?\s*(?:wrong|incorrect|failing|failed)\b|"
    r"\bexpected\b.{0,120}\b(?:but got|instead got|rather than)\b|"
    r"\b(?:does not|doesn't|did not|didn't) match\b)",
    re.IGNORECASE | re.DOTALL,
)
_SOLUTION_EDIT_RE = re.compile(
    r"(?:\b(?:cat|tee|touch|cp|mv|install)\b[^\n]*|(?:>>?|\btee\b)\s*[^\n]*|"
    r"\bsed\b[^\n]*\s-i\b[^\n]*)/app/solution\.py\b",
    re.IGNORECASE,
)


def _observation_text(step: Mapping[str, Any]) -> str:
    observation = step.get("observation")
    if not isinstance(observation, Mapping):
        return ""
    results = observation.get("results")
    if not isinstance(results, list):
        return ""
    return "\n".join(
        result.get("content", "")
        for result in results
        if isinstance(result, Mapping) and isinstance(result.get("content"), str)
    )


def _completed_with_unresolved_failure(steps: list[Any]) -> bool:
    """Detect submission after observed failure without a subsequent solution edit."""
    unresolved_failure = False
    premature_completion = False
    for step in steps:
        if not isinstance(step, Mapping) or step.get("source") != "agent":
            continue

        message = step.get("message")
        if isinstance(message, str) and _ACKNOWLEDGED_FAILURE_RE.search(message):
            unresolved_failure = True

        tool_calls = step.get("tool_calls") or []
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            if call.get("function_name") == "bash_command":
                arguments = call.get("arguments") or {}
                command = (
                    arguments.get("keystrokes")
                    if isinstance(arguments, Mapping)
                    else None
                )
                if isinstance(command, str) and _SOLUTION_EDIT_RE.search(command):
                    unresolved_failure = False

        if _OBSERVED_FAILURE_RE.search(_observation_text(step)):
            unresolved_failure = True

        if (
            any(
                isinstance(call, Mapping)
                and call.get("function_name") == "mark_task_complete"
                for call in tool_calls
            )
            and unresolved_failure
        ):
            premature_completion = True

    return premature_completion


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def score_harbor_trajectory(
    trajectory_path: Path,
    verifier_reward: float,
    config: HarborRewardShapingConfig | Mapping[str, Any] | Any,
) -> ProcessRewardResult:
    """Return capped process shaping, falling back safely on malformed traces."""

    raw_reward = float(verifier_reward)
    if not bool(_config_value(config, "enabled", False)):
        return ProcessRewardResult(raw_reward=raw_reward)

    failure_penalty = 0.0
    if raw_reward <= 0:
        failure_penalty = max(
            0.0, float(_config_value(config, "verifier_failure_penalty", 0.05))
        )

    try:
        trajectory = json.loads(trajectory_path.read_text())
        steps = trajectory.get("steps", [])
        if not isinstance(steps, list):
            return ProcessRewardResult(raw_reward=raw_reward, penalty=failure_penalty)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return ProcessRewardResult(raw_reward=raw_reward, penalty=failure_penalty)

    premature_completion = _completed_with_unresolved_failure(steps)
    penalty = failure_penalty
    if premature_completion:
        penalty += max(
            0.0,
            float(_config_value(config, "premature_completion_penalty", 0.05)),
        )
    max_penalty = max(0.0, float(_config_value(config, "max_penalty", 0.1)))
    penalty = min(penalty, max_penalty)

    multi_tool_turns = 0
    agent_turns = 0
    reasoning_chars = 0
    commands: list[str] = []
    for step in steps:
        if not isinstance(step, dict) or step.get("source") != "agent":
            continue
        agent_turns += 1
        message = step.get("message")
        if isinstance(message, str):
            reasoning_chars += len(message.strip())
        tool_calls = step.get("tool_calls") or []
        bash_calls = [
            call
            for call in tool_calls
            if isinstance(call, dict) and call.get("function_name") == "bash_command"
        ]
        if len(bash_calls) >= 2:
            multi_tool_turns += 1
        for call in bash_calls:
            arguments = call.get("arguments") or {}
            if isinstance(arguments, dict):
                command = arguments.get("keystrokes")
                if isinstance(command, str):
                    commands.append(command)

    command_text = "\n".join(commands)
    wrote_tests = bool(_WRITE_TEST_RE.search(command_text))
    ran_tests = bool(_RUN_TEST_RE.search(command_text))
    used_syntax_checker = bool(_SYNTAX_CHECK_RE.search(command_text))

    capped_multi_turns = min(
        multi_tool_turns,
        max(0, int(_config_value(config, "max_multi_tool_turns", 2))),
    )
    reasoning_chars_per_turn = reasoning_chars / agent_turns if agent_turns else 0.0
    max_concision_bonus = max(
        0.0, float(_config_value(config, "concise_reasoning_bonus", 0.02))
    )
    target_chars = max(
        0,
        int(_config_value(config, "concise_reasoning_target_chars_per_turn", 600)),
    )
    max_chars = max(
        target_chars + 1,
        int(_config_value(config, "concise_reasoning_max_chars_per_turn", 1800)),
    )
    if not agent_turns or reasoning_chars_per_turn >= max_chars:
        concision_bonus = 0.0
    elif reasoning_chars_per_turn <= target_chars:
        concision_bonus = max_concision_bonus
    else:
        concision_bonus = max_concision_bonus * (
            (max_chars - reasoning_chars_per_turn) / (max_chars - target_chars)
        )
    eligible_for_bonus = (
        not bool(_config_value(config, "require_verifier_success", True))
        or raw_reward > 0
    )
    bonus = 0.0
    if eligible_for_bonus:
        bonus = (
            capped_multi_turns
            * float(_config_value(config, "multi_tool_turn_bonus", 0.005))
            + float(_config_value(config, "wrote_tests_bonus", 0.015)) * wrote_tests
            + float(_config_value(config, "ran_tests_bonus", 0.015)) * ran_tests
            + float(_config_value(config, "syntax_checker_bonus", 0.01))
            * used_syntax_checker
            + concision_bonus
        )
    max_bonus = max(0.0, float(_config_value(config, "max_bonus", 0.05)))
    bonus = max(0.0, min(bonus, max_bonus))
    reported_concision_bonus = concision_bonus if eligible_for_bonus else 0.0

    # --- oracle-checked discriminating-test reward (independent budget) ---
    oracle_test_reward = 0.0
    n_valid_tests = n_discriminating_tests = n_fixed_tests = 0
    test_use_reward_val = 0.0
    if eligible_for_bonus and bool(_config_value(config, "oracle_test_enabled", False)):
        try:
            from .oracle_test_reward import discriminating_test_bonus, load_shipped_io_for_trajectory
            oracle_dir = str(_config_value(config, "oracle_test_dir", ""))
            task_data_dir = str(_config_value(config, "oracle_test_task_dir", ""))
            task, shipped_io = load_shipped_io_for_trajectory(trajectory_path, task_data_dir)
            if oracle_dir and task:
                add, tr = discriminating_test_bonus(
                    steps, task, oracle_dir, shipped_io,
                    timeout=float(_config_value(config, "oracle_test_timeout", 6.0)),
                    bonus_scale=float(_config_value(config, "oracle_test_bonus", 0.05)),
                    target_discriminating=int(_config_value(config, "oracle_test_target", 3)),
                    trajectory_path=str(trajectory_path),
                    test_use_scale=float(_config_value(config, "oracle_test_use_bonus", 0.0)),
                    test_use_target=int(_config_value(config, "oracle_test_use_target", 2)),
                )
                oracle_test_reward = add
                n_valid_tests = tr.n_valid
                n_discriminating_tests = tr.n_discriminating
                n_fixed_tests = tr.n_fixed
                test_use_reward_val = tr.test_use_reward
                bonus += add
        except Exception:
            pass  # never let shaping break the reward path

    return ProcessRewardResult(
        raw_reward=raw_reward,
        bonus=bonus,
        multi_tool_turns=multi_tool_turns,
        wrote_tests=wrote_tests,
        ran_tests=ran_tests,
        used_syntax_checker=used_syntax_checker,
        reasoning_chars=reasoning_chars,
        reasoning_chars_per_turn=reasoning_chars_per_turn,
        concise_reasoning_bonus=reported_concision_bonus,
        penalty=penalty,
        premature_completion=premature_completion,
        oracle_test_reward=oracle_test_reward,
        n_valid_tests=n_valid_tests,
        n_discriminating_tests=n_discriminating_tests,
        n_fixed_tests=n_fixed_tests,
        test_use_reward=test_use_reward_val,
    )
