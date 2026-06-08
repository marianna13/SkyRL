"""
Reward shaping utilities for parsing test outputs and computing shaped rewards.

This module provides a flexible framework for:
1. Parsing test output from various frameworks (pytest, unittest, etc.)
2. Computing shaped rewards based on test pass/fail ratios

Usage:
    from skyrl_train.utils.reward_shaping import (
        get_output_parser,
        get_reward_shaper,
        shape_reward_from_output,
    )

    # Parse and shape in one call
    shaped_reward = shape_reward_from_output(
        stdout=verifier_stdout,
        original_reward=0.0,
        parser_name="pytest",
        shaper_name="pass_ratio",
    )

    # Or use components separately
    parser = get_output_parser("pytest")
    shaper = get_reward_shaper("pass_ratio")

    parsed = parser.parse(stdout)
    if parsed is not None:
        shaped_reward = shaper.shape(parsed, original_reward)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from loguru import logger


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ParsedTestResult:
    """
    Structured representation of test results from any test framework.

    Attributes:
        passed: Number of tests that passed
        failed: Number of tests that failed (assertions failed)
        errors: Number of tests that errored (couldn't run, setup issues)
        xfailed: Expected failures (test failed as expected, counts as success)
        xpassed: Unexpected passes (test passed when expected to fail)
        skipped: Tests that were skipped
        warnings: Number of warnings (informational)
        total: Total number of tests (computed if not set)
        duration_sec: Test duration in seconds (if available)
        raw_output: Original output string for debugging
        metadata: Additional framework-specific data
    """

    passed: int = 0
    failed: int = 0
    errors: int = 0
    xfailed: int = 0
    xpassed: int = 0
    skipped: int = 0
    warnings: int = 0
    total: int = 0
    duration_sec: Optional[float] = None
    raw_output: str = ""
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self):
        # Compute total if not explicitly set
        if self.total == 0:
            self.total = (
                self.passed
                + self.failed
                + self.errors
                + self.xfailed
                + self.xpassed
                + self.skipped
            )

    @property
    def effective_passed(self) -> int:
        """Tests that behaved as expected (passed + xfailed)."""
        return self.passed + self.xfailed

    @property
    def effective_failed(self) -> int:
        """Tests that did not behave as expected (failed + errors + xpassed)."""
        return self.failed + self.errors + self.xpassed

    @property
    def runnable_total(self) -> int:
        """Total tests excluding skipped (tests that actually ran)."""
        return self.total - self.skipped

    @property
    def pass_ratio(self) -> float:
        """Simple pass ratio: passed / total (0.0 if no tests)."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def effective_pass_ratio(self) -> float:
        """Effective pass ratio: effective_passed / runnable_total."""
        if self.runnable_total == 0:
            return 0.0
        return self.effective_passed / self.runnable_total


# =============================================================================
# Output Parsers
# =============================================================================


class OutputParser(ABC):
    """
    Abstract base class for parsing test output strings.

    Subclasses implement parsing logic for specific test frameworks.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the parser name for registry lookup."""
        pass

    @abstractmethod
    def parse(self, output: str) -> Optional[ParsedTestResult]:
        """
        Parse test output and extract structured results.

        Args:
            output: Raw test output string (stdout/stderr)

        Returns:
            ParsedTestResult if parsing succeeded, None if output format
            not recognized or parsing failed.
        """
        pass

    def can_parse(self, output: str) -> bool:
        """
        Quick check if this parser can handle the output.

        Override for more efficient detection before full parsing.
        """
        return self.parse(output) is not None


class PytestOutputParser(OutputParser):
    """
    Parser for pytest output.

    Recognizes pytest summary lines like:
        ============== 1 failed, 62 passed, 2 xfailed, 66 errors in 2.39s ==============
        ============================= 5 passed in 0.12s ==============================

    Also counts individual test result lines:
        PASSED test_file.py::test_name
        FAILED test_file.py::test_name - AssertionError
        ERROR test_file.py::test_name
        XFAIL test_file.py::test_name - reason
        XPASS test_file.py::test_name
        SKIPPED test_file.py::test_name - reason

    Collection errors (where pytest can't load tests) are detected and
    return None to signal unparseable output, falling back to original reward.
    """

    # Regex for the summary line at the end of pytest output
    # Matches: "=== 1 failed, 62 passed, 2 xfailed in 2.39s ==="
    SUMMARY_PATTERN = re.compile(
        r"=+\s*"
        r"(?P<results>(?:\d+\s+\w+(?:,\s*)?)+)"
        r"\s+in\s+"
        r"(?P<duration>[\d.]+)s?"
        r"\s*=+",
        re.IGNORECASE,
    )

    # Pattern to extract individual counts from summary
    COUNT_PATTERN = re.compile(r"(\d+)\s+(\w+)", re.IGNORECASE)

    # Patterns for individual test result lines
    RESULT_LINE_PATTERNS = {
        "passed": re.compile(r"^PASSED\s+", re.MULTILINE),
        "failed": re.compile(r"^FAILED\s+", re.MULTILINE),
        "error": re.compile(r"^ERROR\s+", re.MULTILINE),
        "xfail": re.compile(r"^XFAIL\s+", re.MULTILINE),
        "xpass": re.compile(r"^XPASS\s+", re.MULTILINE),
        "skipped": re.compile(r"^SKIPPED\s+", re.MULTILINE),
    }

    # Patterns that indicate pytest couldn't collect/load tests
    # These are infrastructure failures, not agent failures
    COLLECTION_ERROR_PATTERNS = [
        re.compile(r"error during collection", re.IGNORECASE),
        re.compile(r"Interrupted:.*error", re.IGNORECASE),
        re.compile(r"no tests ran", re.IGNORECASE),
        re.compile(r"collection error", re.IGNORECASE),
        re.compile(r"import error", re.IGNORECASE),
    ]

    @classmethod
    def name(cls) -> str:
        return "pytest"

    def _is_collection_error(self, output: str) -> bool:
        """Check if output indicates a pytest collection/import error."""
        for pattern in self.COLLECTION_ERROR_PATTERNS:
            if pattern.search(output):
                return True
        return False

    def parse(self, output: str) -> Optional[ParsedTestResult]:
        """Parse pytest output to extract test counts.

        Returns None for collection errors (where pytest couldn't load tests),
        which causes fallback to original reward rather than treating as 0/N failures.
        """
        if not output:
            return None

        # Check for collection errors FIRST - these indicate pytest couldn't
        # even load the tests, which is typically an infrastructure issue
        # (bad test file, missing dependencies) not an agent failure.
        # Return None to fall back to original verifier reward.
        if self._is_collection_error(output):
            logger.debug(
                "Detected pytest collection error - skipping reward shaping. "
                "Falling back to original verifier reward."
            )
            return None

        # Try to find the summary line first (most reliable)
        summary_match = self.SUMMARY_PATTERN.search(output)

        if summary_match:
            return self._parse_from_summary(output, summary_match)

        # Fall back to counting individual result lines
        return self._parse_from_lines(output)

    def _parse_from_summary(
        self, output: str, summary_match: re.Match
    ) -> Optional[ParsedTestResult]:
        """Parse from the pytest summary line."""
        results_str = summary_match.group("results")
        duration_str = summary_match.group("duration")

        counts = {
            "passed": 0,
            "failed": 0,
            "error": 0,
            "errors": 0,
            "xfailed": 0,
            "xpassed": 0,
            "skipped": 0,
            "warnings": 0,
            "warning": 0,
            "deselected": 0,
        }

        for count_match in self.COUNT_PATTERN.finditer(results_str):
            count = int(count_match.group(1))
            status = count_match.group(2).lower()
            if status in counts:
                counts[status] = count

        # Combine error/errors (pytest uses both)
        errors = counts["error"] + counts["errors"]
        warnings = counts["warning"] + counts["warnings"]

        try:
            duration = float(duration_str)
        except (ValueError, TypeError):
            duration = None

        return ParsedTestResult(
            passed=counts["passed"],
            failed=counts["failed"],
            errors=errors,
            xfailed=counts["xfailed"],
            xpassed=counts["xpassed"],
            skipped=counts["skipped"],
            warnings=warnings,
            duration_sec=duration,
            raw_output=output,
            metadata={"parse_method": "summary", "deselected": counts["deselected"]},
        )

    def _parse_from_lines(self, output: str) -> Optional[ParsedTestResult]:
        """Parse by counting individual test result lines."""
        counts = {}
        found_any = False

        for status, pattern in self.RESULT_LINE_PATTERNS.items():
            matches = pattern.findall(output)
            counts[status] = len(matches)
            if counts[status] > 0:
                found_any = True

        if not found_any:
            return None

        return ParsedTestResult(
            passed=counts.get("passed", 0),
            failed=counts.get("failed", 0),
            errors=counts.get("error", 0),
            xfailed=counts.get("xfail", 0),
            xpassed=counts.get("xpass", 0),
            skipped=counts.get("skipped", 0),
            raw_output=output,
            metadata={"parse_method": "line_count"},
        )

    def can_parse(self, output: str) -> bool:
        """Quick check for pytest indicators."""
        if not output:
            return False
        # Look for pytest-specific markers
        return (
            self.SUMMARY_PATTERN.search(output) is not None
            or "PASSED " in output
            or "FAILED " in output
            or "pytest" in output.lower()
        )


class UnittestOutputParser(OutputParser):
    """
    Parser for Python unittest output.

    Recognizes unittest summary lines like:
        Ran 5 tests in 0.003s
        OK
        FAILED (failures=2, errors=1)
        OK (skipped=3)
    """

    # "Ran X tests in Y.YYs"
    RAN_PATTERN = re.compile(r"Ran\s+(\d+)\s+tests?\s+in\s+([\d.]+)s", re.IGNORECASE)

    # "FAILED (failures=2, errors=1)"
    FAILED_PATTERN = re.compile(
        r"FAILED\s*\(([^)]+)\)",
        re.IGNORECASE,
    )

    # "OK" or "OK (skipped=3)"
    OK_PATTERN = re.compile(r"^OK(?:\s*\(([^)]+)\))?", re.MULTILINE | re.IGNORECASE)

    # Extract key=value pairs
    KV_PATTERN = re.compile(r"(\w+)=(\d+)")

    @classmethod
    def name(cls) -> str:
        return "unittest"

    def parse(self, output: str) -> Optional[ParsedTestResult]:
        """Parse unittest output."""
        if not output:
            return None

        # Find "Ran X tests"
        ran_match = self.RAN_PATTERN.search(output)
        if not ran_match:
            return None

        total = int(ran_match.group(1))
        try:
            duration = float(ran_match.group(2))
        except (ValueError, TypeError):
            duration = None

        counts = {
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "expected failures": 0,
            "unexpected successes": 0,
        }

        # Check for FAILED line
        failed_match = self.FAILED_PATTERN.search(output)
        if failed_match:
            details = failed_match.group(1)
            for kv_match in self.KV_PATTERN.finditer(details):
                key = kv_match.group(1).lower()
                value = int(kv_match.group(2))
                if key in counts:
                    counts[key] = value

        # Check for OK line (may have skipped, etc.)
        ok_match = self.OK_PATTERN.search(output)
        if ok_match and ok_match.group(1):
            details = ok_match.group(1)
            for kv_match in self.KV_PATTERN.finditer(details):
                key = kv_match.group(1).lower()
                value = int(kv_match.group(2))
                if key in counts:
                    counts[key] = value

        failed = counts["failures"]
        errors = counts["errors"]
        skipped = counts["skipped"]
        xfailed = counts["expected failures"]
        xpassed = counts["unexpected successes"]
        passed = total - failed - errors - skipped - xfailed - xpassed

        return ParsedTestResult(
            passed=max(0, passed),
            failed=failed,
            errors=errors,
            xfailed=xfailed,
            xpassed=xpassed,
            skipped=skipped,
            total=total,
            duration_sec=duration,
            raw_output=output,
            metadata={"parse_method": "unittest"},
        )

    def can_parse(self, output: str) -> bool:
        """Quick check for unittest indicators."""
        if not output:
            return False
        return self.RAN_PATTERN.search(output) is not None


class GenericOutputParser(OutputParser):
    """
    Generic fallback parser that counts PASS/FAIL/ERROR keywords.

    Less accurate but works as a fallback for unknown formats.
    """

    PASS_PATTERNS = [
        re.compile(r"\bPASS(?:ED)?\b", re.IGNORECASE),
        re.compile(r"\bOK\b"),
        re.compile(r"\bSUCCESS\b", re.IGNORECASE),
        re.compile(r"\[PASS\]", re.IGNORECASE),
        re.compile(r"✓"),
    ]

    FAIL_PATTERNS = [
        re.compile(r"\bFAIL(?:ED|URE)?\b", re.IGNORECASE),
        re.compile(r"\[FAIL\]", re.IGNORECASE),
        re.compile(r"✗"),
        re.compile(r"✘"),
    ]

    ERROR_PATTERNS = [
        re.compile(r"\bERROR\b", re.IGNORECASE),
        re.compile(r"\[ERROR\]", re.IGNORECASE),
    ]

    @classmethod
    def name(cls) -> str:
        return "generic"

    def parse(self, output: str) -> Optional[ParsedTestResult]:
        """Count pass/fail/error keywords in output."""
        if not output:
            return None

        passed = sum(len(p.findall(output)) for p in self.PASS_PATTERNS)
        failed = sum(len(p.findall(output)) for p in self.FAIL_PATTERNS)
        errors = sum(len(p.findall(output)) for p in self.ERROR_PATTERNS)

        # Only return result if we found something
        if passed == 0 and failed == 0 and errors == 0:
            return None

        return ParsedTestResult(
            passed=passed,
            failed=failed,
            errors=errors,
            raw_output=output,
            metadata={"parse_method": "generic_keywords"},
        )


# =============================================================================
# Reward Shapers
# =============================================================================


class RewardShaper(ABC):
    """
    Abstract base class for computing shaped rewards from parsed test results.

    Shapers convert ParsedTestResult into a reward value in [0, 1].
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the shaper name for registry lookup."""
        pass

    @abstractmethod
    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """
        Compute shaped reward from parsed test results.

        Args:
            parsed: Structured test results from an OutputParser
            original_reward: The original reward from the verifier

        Returns:
            Shaped reward value in [0, 1]
        """
        pass


class PassRatioShaper(RewardShaper):
    """
    Simple pass ratio shaper: reward = passed / total.

    This is the most straightforward approach - partial credit proportional
    to the fraction of tests passed.
    """

    def __init__(self, **kwargs):
        # Accept and ignore any kwargs for uniform interface
        pass

    @classmethod
    def name(cls) -> str:
        return "pass_ratio"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Return simple pass ratio, falling back to original_reward for single-test tasks."""
        if parsed.total <= 1:
            return original_reward
        return parsed.pass_ratio


class EffectivePassRatioShaper(RewardShaper):
    """
    Effective pass ratio shaper: reward = effective_passed / runnable_total.

    Treats xfailed (expected failures) as passes since the test behaved
    as expected. Excludes skipped tests from the denominator.
    """

    def __init__(self, **kwargs):
        # Accept and ignore any kwargs for uniform interface
        pass

    @classmethod
    def name(cls) -> str:
        return "effective_pass_ratio"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Return effective pass ratio."""
        return parsed.effective_pass_ratio


class WeightedShaper(RewardShaper):
    """
    Weighted shaper with configurable weights for different outcomes.

    reward = (w_pass * passed + w_xfail * xfailed - w_fail * failed - w_error * errors) / total

    Allows penalizing errors more heavily than failures, or giving
    partial credit for xfailed tests.
    """

    def __init__(
        self,
        weight_pass: float = 1.0,
        weight_xfail: float = 1.0,
        weight_fail: float = 0.0,
        weight_error: float = 0.0,
        weight_xpass: float = 0.5,  # Unexpected pass - partial credit
        **kwargs,  # Accept and ignore extra kwargs
    ):
        self.weight_pass = weight_pass
        self.weight_xfail = weight_xfail
        self.weight_fail = weight_fail
        self.weight_error = weight_error
        self.weight_xpass = weight_xpass

    @classmethod
    def name(cls) -> str:
        return "weighted"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Compute weighted reward."""
        if parsed.runnable_total == 0:
            return 0.0

        score = (
            self.weight_pass * parsed.passed
            + self.weight_xfail * parsed.xfailed
            + self.weight_fail * parsed.failed
            + self.weight_error * parsed.errors
            + self.weight_xpass * parsed.xpassed
        )

        # Normalize to [0, 1]
        max_score = self.weight_pass * parsed.runnable_total
        if max_score <= 0:
            return 0.0

        return max(0.0, min(1.0, score / max_score))


class ThresholdShaper(RewardShaper):
    """
    Threshold-based shaper with configurable pass threshold.

    Returns 1.0 if pass_ratio >= threshold, else returns scaled pass_ratio.
    Useful for "almost passing" scenarios where you want to reward
    getting close to full success.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        below_threshold_scale: float = 0.5,
        **kwargs,  # Accept and ignore extra kwargs
    ):
        """
        Args:
            threshold: Pass ratio threshold for full reward (default 1.0 = all tests)
            below_threshold_scale: Scale factor for rewards below threshold
        """
        self.threshold = threshold
        self.below_threshold_scale = below_threshold_scale

    @classmethod
    def name(cls) -> str:
        return "threshold"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Apply threshold-based shaping."""
        ratio = parsed.effective_pass_ratio

        if ratio >= self.threshold:
            return 1.0

        # Scale the ratio for below-threshold results
        return ratio * self.below_threshold_scale


class BinaryWithPartialCreditShaper(RewardShaper):
    """
    Binary reward with optional partial credit for near-successes.

    - If all tests pass: reward = 1.0
    - If >= partial_threshold pass: reward = partial_credit
    - Otherwise: reward = 0.0

    Useful when you want mostly binary rewards but give some credit
    for getting close.
    """

    def __init__(
        self,
        partial_threshold: float = 0.9,
        partial_credit: float = 0.5,
        **kwargs,  # Accept and ignore extra kwargs
    ):
        self.partial_threshold = partial_threshold
        self.partial_credit = partial_credit

    @classmethod
    def name(cls) -> str:
        return "binary_partial"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Apply binary with partial credit shaping."""
        ratio = parsed.effective_pass_ratio

        if ratio >= 1.0:
            return 1.0
        elif ratio >= self.partial_threshold:
            return self.partial_credit
        else:
            return 0.0


class OriginalRewardShaper(RewardShaper):
    """
    Pass-through shaper that returns the original reward unchanged.

    Useful as a no-op option when reward shaping is disabled.
    """

    def __init__(self, **kwargs):
        # Accept and ignore any kwargs for uniform interface
        pass

    @classmethod
    def name(cls) -> str:
        return "original"

    def shape(
        self,
        parsed: ParsedTestResult,
        original_reward: float,
    ) -> float:
        """Return original reward unchanged."""
        return original_reward


# =============================================================================
# Registry
# =============================================================================

# Parser registry
_PARSER_REGISTRY: Dict[str, Type[OutputParser]] = {}

# Shaper registry
_SHAPER_REGISTRY: Dict[str, Type[RewardShaper]] = {}


def register_parser(parser_cls: Type[OutputParser]) -> Type[OutputParser]:
    """Register a parser class in the registry."""
    _PARSER_REGISTRY[parser_cls.name()] = parser_cls
    return parser_cls


def register_shaper(shaper_cls: Type[RewardShaper]) -> Type[RewardShaper]:
    """Register a shaper class in the registry."""
    _SHAPER_REGISTRY[shaper_cls.name()] = shaper_cls
    return shaper_cls


# Register built-in parsers
register_parser(PytestOutputParser)
register_parser(UnittestOutputParser)
register_parser(GenericOutputParser)

# Register built-in shapers
register_shaper(PassRatioShaper)
register_shaper(EffectivePassRatioShaper)
register_shaper(WeightedShaper)
register_shaper(ThresholdShaper)
register_shaper(BinaryWithPartialCreditShaper)
register_shaper(OriginalRewardShaper)


def get_output_parser(name: str) -> OutputParser:
    """
    Get an output parser by name.

    Args:
        name: Parser name ("pytest", "unittest", "generic")

    Returns:
        Instantiated OutputParser

    Raises:
        ValueError: If parser name not found
    """
    if name not in _PARSER_REGISTRY:
        available = ", ".join(_PARSER_REGISTRY.keys())
        raise ValueError(f"Unknown parser '{name}'. Available: {available}")
    return _PARSER_REGISTRY[name]()


def get_reward_shaper(name: str, **kwargs) -> RewardShaper:
    """
    Get a reward shaper by name.

    Args:
        name: Shaper name ("pass_ratio", "effective_pass_ratio", "weighted", etc.)
        **kwargs: Additional arguments passed to shaper constructor

    Returns:
        Instantiated RewardShaper

    Raises:
        ValueError: If shaper name not found
    """
    if name not in _SHAPER_REGISTRY:
        available = ", ".join(_SHAPER_REGISTRY.keys())
        raise ValueError(f"Unknown shaper '{name}'. Available: {available}")
    return _SHAPER_REGISTRY[name](**kwargs)


def list_parsers() -> List[str]:
    """List all registered parser names."""
    return list(_PARSER_REGISTRY.keys())


def list_shapers() -> List[str]:
    """List all registered shaper names."""
    return list(_SHAPER_REGISTRY.keys())


# =============================================================================
# Convenience Functions
# =============================================================================


def auto_detect_parser(output: str) -> Optional[OutputParser]:
    """
    Auto-detect the appropriate parser for the given output.

    Tries parsers in order of specificity (pytest, unittest, generic).

    Args:
        output: Test output string

    Returns:
        Appropriate OutputParser or None if no parser matches
    """
    # Try in order of specificity
    parser_order = ["pytest", "unittest", "generic"]

    for parser_name in parser_order:
        parser = get_output_parser(parser_name)
        if parser.can_parse(output):
            return parser

    return None


def parse_test_output(
    output: str,
    parser_name: Optional[str] = None,
) -> Optional[ParsedTestResult]:
    """
    Parse test output using specified or auto-detected parser.

    Args:
        output: Test output string
        parser_name: Parser to use, or None for auto-detection

    Returns:
        ParsedTestResult or None if parsing failed
    """
    if parser_name:
        parser = get_output_parser(parser_name)
    else:
        parser = auto_detect_parser(output)
        if parser is None:
            return None

    return parser.parse(output)


def shape_reward_from_output(
    stdout: Optional[str],
    original_reward: float,
    parser_name: Optional[str] = None,
    shaper_name: str = "pass_ratio",
    shaper_kwargs: Optional[Dict] = None,
    fallback_to_original: bool = True,
) -> float:
    """
    Parse test output and compute shaped reward in one call.

    This is the main entry point for reward shaping.

    Args:
        stdout: Test output string (verifier stdout)
        original_reward: Original reward from verifier
        parser_name: Parser to use (None for auto-detection)
        shaper_name: Shaper to use (default: "pass_ratio")
        shaper_kwargs: Additional kwargs for shaper
        fallback_to_original: If True, return original_reward on parse failure

    Returns:
        Shaped reward value in [0, 1]
    """
    # Handle missing output
    if not stdout:
        if fallback_to_original:
            return original_reward
        return 0.0

    # Parse output
    parsed = parse_test_output(stdout, parser_name)

    if parsed is None:
        logger.debug(
            f"Could not parse test output with parser={parser_name or 'auto'}. "
            f"Falling back to original reward: {original_reward}"
        )
        if fallback_to_original:
            return original_reward
        return 0.0

    # Log parse results
    logger.debug(
        f"Parsed test results: passed={parsed.passed}, failed={parsed.failed}, "
        f"errors={parsed.errors}, total={parsed.total}, "
        f"effective_pass_ratio={parsed.effective_pass_ratio:.3f}"
    )

    # Shape reward
    shaper = get_reward_shaper(shaper_name, **(shaper_kwargs or {}))
    shaped = shaper.shape(parsed, original_reward)

    logger.debug(
        f"Shaped reward: {original_reward:.3f} -> {shaped:.3f} "
        f"(shaper={shaper_name})"
    )

    return shaped
