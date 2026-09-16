"""Task-independent state representation for Harbor test-time training."""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

LAST_SOLUTION_PLACEHOLDER = "<<<LAST_CODE>>>"
INITIAL_CONSTRUCTION_PLACEHOLDERS = (
    "<<<INITIAL_CONSTRUCTION>>>",
    # Backward-compatible task-specific alias used by the Erdős prompt.
    "<<<INITIAL_H_VALUES>>>",
)
VALUE_CONTEXT_PLACEHOLDERS = ("<<<VALUE_CONTEXT>>>", "<<<VALUE_CTX>>>")
TTT_STATE_RELATIVE_PATH = Path("tests") / "ttt_state.json"


def _normalize_construction(values: Sequence[float] | None) -> list[float] | None:
    """Return a JSON-safe, finite copy of a one-dimensional construction."""
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError("construction must be a sequence of finite numbers")
    normalized = [float(value) for value in values]
    if not normalized:
        raise ValueError("construction must not be empty")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("construction contains NaN or infinity")
    return normalized


def read_construction_artifact(path: Path) -> tuple[list[float], float | None]:
    """Validate the verifier-owned construction transport artifact."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("unsupported or missing construction schema_version")
    construction = _normalize_construction(raw.get("construction"))
    if construction is None:
        raise ValueError("construction must be a non-empty list")
    selection_value = raw.get("selection_value")
    if selection_value is not None:
        selection_value = float(selection_value)
        if not math.isfinite(selection_value):
            raise ValueError("selection_value is not finite")
    return construction, selection_value


def to_json_serializable(value: Any) -> Any:
    """Compatibility helper for the legacy best-sequence utility."""
    if isinstance(value, dict):
        return {str(key): to_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _read_materialized_construction(task: Path) -> list[float] | None:
    state_path = task / TTT_STATE_RELATIVE_PATH
    if not state_path.is_file():
        return None
    raw = json.loads(state_path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"Malformed TTT state artifact: {state_path}")
    # ``initial_h_values`` is accepted for migration from the first Erdős-only
    # schema. New state artifacts are task-independent.
    return _normalize_construction(
        raw.get("construction", raw.get("initial_h_values"))
    )


def _write_materialized_construction(
    task: Path, construction: Sequence[float] | None
) -> None:
    """Replace inherited state so a child never sees an ancestor's construction."""
    state_path = task / TTT_STATE_RELATIVE_PATH
    normalized = _normalize_construction(construction)
    if normalized is None:
        state_path.unlink(missing_ok=True)
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "verified_construction",
                "construction": normalized,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    os.replace(temporary, state_path)


def _source_symbol(source: str, symbol: str, source_path: Path) -> str:
    tree = ast.parse(source, filename=str(source_path))
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and node.name == symbol:
            segment = ast.get_source_segment(source, node)
            if segment is not None:
                return segment
    raise ValueError(f"Could not find top-level symbol {symbol!r} in {source_path}")


def _load_prompt_context(task: Path) -> dict[str, str]:
    """Load literal or file-backed placeholder values owned by a Harbor task."""
    context_path = task / "prompt_context.json"
    if not context_path.is_file():
        return {}
    raw = json.loads(context_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt context must be a JSON object: {context_path}")
    context: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            # Optional task-owned placeholders may be intentionally unavailable.
            context[key] = ""
            continue
        if isinstance(value, (str, int, float, bool)):
            context[key] = str(value)
            continue
        if not isinstance(value, dict) or "path" not in value:
            raise ValueError(
                f"Prompt context {key!r} must be a scalar or a mapping with a path"
            )
        source_path = task / value["path"]
        source = source_path.read_text()
        context[key] = (
            _source_symbol(source, value["symbol"], source_path)
            if value.get("symbol")
            else source
        )
    return context


def _construction_prompt_context(construction: Sequence[float] | None) -> str:
    if construction is None:
        return "not available for this initial rollout"
    return (
        f"available (length={len(construction)}); the task verifier exposes an "
        "exact fresh copy through its task-specific interface"
    )


def _instruction_with_state(
    template: str,
    solution: str,
    value_context: str,
    prompt_context: dict[str, str],
    construction: Sequence[float] | None = None,
) -> str:
    """Render task-owned placeholders plus generic TTT state."""
    instruction = template
    if LAST_SOLUTION_PLACEHOLDER in instruction:
        instruction = instruction.replace(LAST_SOLUTION_PLACEHOLDER, solution)
    else:
        instruction += (
            "\n\n--- Previous step solution ---\n"
            + solution.strip()
            + "\n--- End previous step solution ---\n"
        )
    for placeholder in INITIAL_CONSTRUCTION_PLACEHOLDERS:
        instruction = instruction.replace(
            placeholder, _construction_prompt_context(construction)
        )
    for placeholder in VALUE_CONTEXT_PLACEHOLDERS:
        instruction = instruction.replace(placeholder, value_context)
    for key, value in prompt_context.items():
        instruction = instruction.replace(f"<<<{key}>>>", value)
    return instruction


def _build_value_context(
    static_context: str,
    value: float | None,
    parent_value: float | None = None,
    observation: str = "",
) -> str:
    parts: list[str] = []
    if parent_value is not None and value is not None:
        parts.append(
            "Previous and current search values (higher is better): "
            f"{parent_value:.6f} -> {value:.6f}."
        )
    elif value is not None:
        parts.append(f"Current search value (higher is better): {value:.6f}.")
    if static_context.strip():
        parts.append(static_context.strip())
    if observation.strip():
        output = observation.strip()
        if len(output) > 2000:
            output = "...(truncated)...\n" + output[-2000:]
        parts.append(
            "--- Previous verifier output ---\n"
            + output
            + "\n--- End previous verifier output ---"
        )
    return ("\n" + "\n".join(parts)) if parts else ""


@dataclass
class State:
    """A search state whose executable representation is a Harbor task directory."""

    task_path: str
    base_task_path: str
    instruction_template: str
    prompt_context: dict[str, str] = field(default_factory=dict)
    static_value_context: str = ""
    solution: str = ""
    construction: list[float] | None = None
    observation: str = ""
    timestep: int = -1
    value: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_values: list[float] = field(default_factory=list)
    parents: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.construction = _normalize_construction(self.construction)

    @property
    def task_key(self) -> str:
        return os.path.realpath(self.base_task_path)

    @classmethod
    def initial(
        cls,
        task_path: str,
        tasks_dir: str | None = None,
        *,
        materialize_solution: bool = True,
    ) -> "State":
        task = Path(task_path).expanduser().resolve()
        instruction_path = task / "instruction.md"
        if not instruction_path.is_file():
            raise ValueError(f"Harbor task is missing instruction.md: {task}")
        template = instruction_path.read_text()
        prompt_context = _load_prompt_context(task)
        # The construction placeholder is controlled by State, not static task JSON.
        prompt_context.pop("INITIAL_H_VALUES", None)
        static_value_context = prompt_context.pop("VALUE_CONTEXT", "")
        if not static_value_context:
            static_value_context = prompt_context.pop("VALUE_CTX", "")
        solution_path = task / "solution" / "solution.py"
        solution = solution_path.read_text() if solution_path.is_file() else ""
        initial_value_path = task / "initial_value.txt"
        if initial_value_path.is_file():
            value = float(initial_value_path.read_text().strip())
        else:
            initial_performance = prompt_context.pop("INITIAL_PERFORMANCE", "")
            if initial_performance:
                raw_value = float(initial_performance)
                performance_mode = prompt_context.get(
                    "PERFORMANCE_MODE", "maximize"
                ).lower()
                if performance_mode not in {"maximize", "minimize"}:
                    raise ValueError(
                        f"Unknown PERFORMANCE_MODE {performance_mode!r}"
                    )
                value = -raw_value if performance_mode == "minimize" else raw_value
            else:
                value = None
        construction = _read_materialized_construction(task)
        state_id = uuid.uuid4().hex
        materialized_path = task
        if materialize_solution and tasks_dir is not None and (
            LAST_SOLUTION_PLACEHOLDER in template
            or any(
                placeholder in template
                for placeholder in INITIAL_CONSTRUCTION_PLACEHOLDERS
            )
        ):
            materialized_path = Path(tasks_dir) / state_id
            shutil.copytree(task, materialized_path)
            _write_materialized_construction(materialized_path, construction)
            (materialized_path / "instruction.md").write_text(
                _instruction_with_state(
                    template,
                    solution,
                    _build_value_context(static_value_context, value),
                    prompt_context,
                    construction,
                )
            )
        return cls(
            id=state_id,
            task_path=str(materialized_path),
            base_task_path=str(task),
            instruction_template=template,
            prompt_context=prompt_context,
            static_value_context=static_value_context,
            solution=solution,
            construction=construction,
            value=value,
        )

    def make_child(
        self,
        solution: str,
        value: float,
        timestep: int,
        tasks_dir: str,
        observation: str = "",
        construction: Sequence[float] | None = None,
    ) -> "State":
        """Copy this task and inject the latest verified construction."""
        normalized_construction = _normalize_construction(construction)
        child_id = uuid.uuid4().hex
        child_path = Path(tasks_dir) / child_id
        shutil.copytree(self.task_path, child_path)
        _write_materialized_construction(child_path, normalized_construction)
        value_context = _build_value_context(
            self.static_value_context,
            float(value),
            self.value,
            observation,
        )
        (child_path / "instruction.md").write_text(
            _instruction_with_state(
                self.instruction_template,
                solution,
                value_context,
                self.prompt_context,
                normalized_construction,
            )
        )
        return State(
            id=child_id,
            task_path=str(child_path),
            base_task_path=self.base_task_path,
            instruction_template=self.instruction_template,
            prompt_context=dict(self.prompt_context),
            static_value_context=self.static_value_context,
            solution=solution,
            construction=normalized_construction,
            observation=observation,
            timestep=timestep,
            value=float(value),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "HarborTaskState",
            "id": self.id,
            "task_path": self.task_path,
            "base_task_path": self.base_task_path,
            "instruction_template": self.instruction_template,
            "prompt_context": self.prompt_context,
            "static_value_context": self.static_value_context,
            "solution": self.solution,
            "construction": self.construction,
            "observation": self.observation,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        return cls(
            id=data["id"],
            task_path=data["task_path"],
            base_task_path=data.get("base_task_path", data["task_path"]),
            instruction_template=data["instruction_template"],
            prompt_context=dict(data.get("prompt_context", {})),
            static_value_context=data.get("static_value_context", ""),
            solution=data.get("solution", ""),
            # Schema-v1/source-only snapshots legitimately have no construction.
            construction=data.get("construction"),
            observation=data.get("observation", ""),
            timestep=int(data.get("timestep", -1)),
            value=data.get("value"),
            parent_values=list(data.get("parent_values", [])),
            parents=list(data.get("parents", [])),
        )


def state_from_dict(data: dict[str, Any]) -> State:
    return State.from_dict(data)
