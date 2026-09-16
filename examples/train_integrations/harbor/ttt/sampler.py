"""Task-independent greedy and PUCT state samplers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import struct
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from .state import State, state_from_dict

SAMPLER_TYPES = {"greedy", "puct", "puct_backprop", "fixed"}
SNAPSHOT_SCHEMA_VERSION = 2


def _step_path(path: str, step: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_step_{step:06d}{ext or '.json'}"


def _write_json(path: str, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, target)


class StateSampler(ABC):
    """Sampler over valued states, partitioned by each state's task key."""

    @abstractmethod
    def sample_states(
        self,
        num_states: int,
        *,
        fallback_states: Sequence[State] | None = None,
    ) -> list[State]:
        pass

    @abstractmethod
    def update_states(
        self,
        states: Sequence[State],
        parent_states: Sequence[State],
        *,
        save: bool = True,
        step: int | None = None,
    ) -> None:
        pass

    @abstractmethod
    def flush(self, step: int | None = None) -> None:
        pass

    @staticmethod
    def _set_parent_info(child: State, parent: State) -> None:
        child.parent_values = (
            ([float(parent.value)] if parent.value is not None else [])
            + parent.parent_values
        )
        child.parents = [
            {"id": parent.id, "timestep": parent.timestep}
        ] + parent.parents

    @staticmethod
    def _key(state: State) -> tuple[str, str]:
        """Deduplicate verified constructions, falling back to source for legacy states."""
        if state.construction is not None:
            payload = struct.pack("<Q", len(state.construction)) + struct.pack(
                f"<{len(state.construction)}d", *state.construction
            )
            return state.task_key, "construction:" + hashlib.sha256(payload).hexdigest()
        return state.task_key, "solution:" + hashlib.sha256(
            state.solution.encode()
        ).hexdigest()

    @staticmethod
    def _topk(
        states: Sequence[State], parents: Sequence[State], k: int
    ) -> list[tuple[State, State]]:
        grouped: dict[str, list[tuple[State, State]]] = {}
        for child, parent in zip(states, parents, strict=True):
            grouped.setdefault(parent.id, []).append((child, parent))
        kept: list[tuple[State, State]] = []
        for pairs in grouped.values():
            pairs.sort(
                key=lambda pair: (
                    pair[0].value if pair[0].value is not None else -math.inf
                ),
                reverse=True,
            )
            kept.extend(pairs if k <= 0 else pairs[:k])
        return kept


class GreedySampler(StateSampler):
    def __init__(
        self,
        sampler_dir: str,
        resume_step: int | None = None,
        topk_children: int = 1,
        epsilon: float = 0.0,
        max_buffer_size: int = 1000,
        **_: object,
    ):
        self.file_path = os.path.join(sampler_dir, "greedy_sampler.json")
        self.topk_children = int(topk_children)
        self.epsilon = float(epsilon)
        self.max_buffer_size = int(max_buffer_size)
        self._states: list[State] = []
        self._current_step = int(resume_step or 0)
        self._lock = threading.Lock()
        if resume_step is not None:
            self._load(resume_step)

    def _load(self, step: int) -> None:
        path = _step_path(self.file_path, step)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cannot resume sampler: {path}")
        with open(path) as handle:
            data = json.load(handle)
        self._states = [state_from_dict(item) for item in data.get("states", [])]

    def _candidates(self, fallback: State | None) -> list[State]:
        return [
            state
            for state in self._states
            if fallback is None or state.task_key == fallback.task_key
        ]

    def sample_states(
        self,
        num_states: int,
        *,
        fallback_states: Sequence[State] | None = None,
    ) -> list[State]:
        fallbacks = list(fallback_states or [])
        if fallbacks and len(fallbacks) != num_states:
            raise ValueError("fallback_states must have num_states entries")
        sampled: list[State] = []
        for index in range(num_states):
            fallback = fallbacks[index] if fallbacks else None
            candidates = self._candidates(fallback)
            if not candidates:
                if fallback is None:
                    raise ValueError("Sampler is empty and no fallback state was supplied")
                sampled.append(fallback)
                continue
            candidates.sort(
                key=lambda state: (
                    state.value if state.value is not None else -math.inf
                ),
                reverse=True,
            )
            use_random = (
                self.epsilon > 0
                and len(candidates) > 1
                and random.random() < self.epsilon
            )
            sampled.append(random.choice(candidates) if use_random else candidates[0])
        return sampled

    def update_states(
        self,
        states: Sequence[State],
        parent_states: Sequence[State],
        *,
        save: bool = True,
        step: int | None = None,
    ) -> None:
        pairs = self._topk(states, parent_states, self.topk_children)
        with self._lock:
            existing = {self._key(state) for state in self._states}
            for child, parent in pairs:
                key = self._key(child)
                if child.value is None or key in existing:
                    continue
                self._set_parent_info(child, parent)
                self._states.append(child)
                existing.add(key)
            if save:
                self._finalize(step)

    def _finalize(self, step: int | None) -> None:
        if step is not None:
            self._current_step = int(step)
        self._states.sort(
            key=lambda state: (
                state.value if state.value is not None else -math.inf
            ),
            reverse=True,
        )
        if self.max_buffer_size > 0:
            self._states = self._states[: self.max_buffer_size]
        self._save()

    def _save(self) -> None:
        _write_json(
            _step_path(self.file_path, self._current_step),
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "step": self._current_step,
                "states": [state.to_dict() for state in self._states],
            },
        )

    def flush(self, step: int | None = None) -> None:
        with self._lock:
            self._finalize(step)


class FixedSampler(StateSampler):
    def __init__(self, **_: object):
        pass

    def sample_states(
        self,
        num_states: int,
        *,
        fallback_states: Sequence[State] | None = None,
    ) -> list[State]:
        if fallback_states is None or len(fallback_states) != num_states:
            raise ValueError("FixedSampler requires one fallback state per sample")
        return list(fallback_states)

    def update_states(
        self, states, parent_states, *, save=True, step=None
    ) -> None:
        pass

    def flush(self, step: int | None = None) -> None:
        pass


class PUCTSampler(GreedySampler):
    """Reference-style PUCT over verified construction states."""

    def __init__(
        self,
        sampler_dir: str,
        puct_c: float = 1.0,
        group_size: int = 1,
        topk_children: int = 2,
        min_puct_scale: float = 1e-6,
        **kwargs: object,
    ):
        self.puct_c = float(puct_c)
        # Retained for old launch configurations; reference PUCT counts parents,
        # not individual rollouts, so group_size does not enter its equation.
        self.group_size = max(1, int(group_size))
        self.min_puct_scale = float(min_puct_scale)
        if self.min_puct_scale <= 0:
            raise ValueError("min_puct_scale must be positive")
        self._n: dict[str, int] = {}
        self._m: dict[str, float] = {}
        self._T = 0
        self._initial_states: list[State] = []
        resume_step = kwargs.pop("resume_step", None)
        super().__init__(
            sampler_dir=sampler_dir,
            resume_step=None,
            topk_children=topk_children,
            **kwargs,
        )
        self.file_path = os.path.join(sampler_dir, "puct_sampler.json")
        self._current_step = int(resume_step or 0)
        if resume_step is not None:
            self._load(int(resume_step))

    def _load(self, step: int) -> None:
        path = _step_path(self.file_path, step)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cannot resume sampler: {path}")
        with open(path) as handle:
            data = json.load(handle)
        self._states = [state_from_dict(item) for item in data.get("states", [])]
        self._initial_states = [
            state_from_dict(item) for item in data.get("initial_states", [])
        ]
        self._n = {str(key): int(value) for key, value in data.get("puct_n", {}).items()}
        self._m = {str(key): float(value) for key, value in data.get("puct_m", {}).items()}
        self._T = int(data.get("puct_T", 0))

    @staticmethod
    def _lineage_ids(state: State) -> set[str]:
        return {state.id} | {
            str(parent["id"]) for parent in state.parents if parent.get("id")
        }

    def _full_lineage_ids(
        self, selected: State, candidates: Sequence[State]
    ) -> set[str]:
        blocked = self._lineage_ids(selected)
        blocked.update(
            candidate.id
            for candidate in candidates
            if selected.id in self._lineage_ids(candidate)
        )
        return blocked

    def _score_entries(
        self,
        candidates: Sequence[State],
        *,
        virtual_visits: dict[str, int] | None = None,
        virtual_total: int = 0,
    ) -> list[tuple[float, float, State]]:
        values = [
            float(state.value) if state.value is not None else 0.0
            for state in candidates
        ]
        initial_ids = {state.id for state in self._initial_states}
        non_initial_values = [
            value
            for state, value in zip(candidates, values, strict=True)
            if state.id not in initial_ids
        ]
        scale_values = non_initial_values or values
        scale = max(max(scale_values) - min(scale_values), self.min_puct_scale)
        order = sorted(range(len(candidates)), key=lambda index: values[index], reverse=True)
        rank = {candidate_index: position for position, candidate_index in enumerate(order)}
        weights = [len(candidates) - rank[index] for index in range(len(candidates))]
        weight_sum = sum(weights)
        root = math.sqrt(1.0 + self._T + virtual_total)
        virtual_visits = virtual_visits or {}

        entries: list[tuple[float, float, State]] = []
        for index, state in enumerate(candidates):
            visits = self._n.get(state.id, 0) + virtual_visits.get(state.id, 0)
            value = values[index]
            q_value = self._m.get(state.id, value) if visits else value
            bonus = (
                self.puct_c
                * scale
                * (weights[index] / weight_sum)
                * root
                / (1.0 + visits)
            )
            entries.append((q_value + bonus, value, state))
        return sorted(entries, key=lambda entry: (entry[0], entry[1]), reverse=True)

    def _select_diverse(
        self, candidates: list[State], count: int
    ) -> list[State]:
        picked: list[State] = []
        blocked: set[str] = set()
        for _, _, state in self._score_entries(candidates):
            if state.id in blocked:
                continue
            picked.append(state)
            blocked.update(self._full_lineage_ids(state, candidates))
            if len(picked) == count:
                return picked

        # A small archive can contain fewer independent lineages than the batch.
        # Virtual visits provide a deterministic, explicit exploration fallback.
        virtual_visits: dict[str, int] = {}
        while len(picked) < count:
            entries = self._score_entries(
                candidates,
                virtual_visits=virtual_visits,
                virtual_total=sum(virtual_visits.values()),
            )
            state = entries[0][2]
            picked.append(state)
            virtual_visits[state.id] = virtual_visits.get(state.id, 0) + 1
        return picked

    def _register_initial_fallbacks(self, fallbacks: Sequence[State]) -> None:
        by_task: dict[str, list[State]] = {}
        for fallback in fallbacks:
            by_task.setdefault(fallback.task_key, []).append(fallback)
        existing_tasks = {state.task_key for state in self._states}
        for task_key, task_fallbacks in by_task.items():
            if task_key in existing_tasks:
                continue
            self._states.extend(task_fallbacks)
            self._initial_states.extend(task_fallbacks)

    def sample_states(
        self,
        num_states: int,
        *,
        fallback_states: Sequence[State] | None = None,
    ) -> list[State]:
        fallbacks = list(fallback_states or [])
        if fallbacks and len(fallbacks) != num_states:
            raise ValueError("fallback_states must have num_states entries")
        if fallbacks:
            self._register_initial_fallbacks(fallbacks)

        if not fallbacks:
            if not self._states:
                raise ValueError("Sampler is empty and no fallback state was supplied")
            return self._select_diverse(list(self._states), num_states)

        sampled: list[State | None] = [None] * num_states
        positions_by_task: dict[str, list[int]] = {}
        for index, fallback in enumerate(fallbacks):
            positions_by_task.setdefault(fallback.task_key, []).append(index)
        for positions in positions_by_task.values():
            candidates = self._candidates(fallbacks[positions[0]])
            picked = self._select_diverse(candidates, len(positions))
            for index, state in zip(positions, picked, strict=True):
                sampled[index] = state
        return [state for state in sampled if state is not None]

    def update_states(
        self,
        states: Sequence[State],
        parent_states: Sequence[State],
        *,
        save: bool = True,
        step: int | None = None,
    ) -> None:
        if len(states) != len(parent_states):
            raise ValueError("states and parent_states must have the same length")

        # Reference accounting performs one backup per distinct sampled parent.
        best_by_parent: dict[str, tuple[float, State]] = {}
        for child, parent in zip(states, parent_states, strict=True):
            if child.value is None:
                continue
            value = float(child.value)
            previous = best_by_parent.get(parent.id)
            if previous is None or value > previous[0]:
                best_by_parent[parent.id] = (value, parent)
        for parent_id, (value, parent) in best_by_parent.items():
            self._m[parent_id] = max(self._m.get(parent_id, -math.inf), value)
            for state_id in self._lineage_ids(parent):
                self._n[state_id] = self._n.get(state_id, 0) + 1
            self._T += 1

        super().update_states(states, parent_states, save=save, step=step)

    def record_failed_rollout(self, parent: State) -> None:
        for state_id in self._lineage_ids(parent):
            self._n[state_id] = self._n.get(state_id, 0) + 1
        self._T += 1

    def _finalize(self, step: int | None) -> None:
        if step is not None:
            self._current_step = int(step)
        initial_ids = {state.id for state in self._initial_states}
        initial = [state for state in self._states if state.id in initial_ids]
        others = [state for state in self._states if state.id not in initial_ids]
        others.sort(
            key=lambda state: (
                state.value if state.value is not None else -math.inf
            ),
            reverse=True,
        )
        if self.max_buffer_size > 0:
            capacity = max(0, self.max_buffer_size - len(initial))
            others = others[:capacity]
        self._states = initial + others
        self._save()

    def _save(self) -> None:
        _write_json(
            _step_path(self.file_path, self._current_step),
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "step": self._current_step,
                "states": [state.to_dict() for state in self._states],
                "initial_states": [
                    state.to_dict() for state in self._initial_states
                ],
                "puct_n": self._n,
                "puct_m": self._m,
                "puct_T": self._T,
            },
        )


def create_sampler(
    sampler_type: str, sampler_dir: str, **kwargs: object
) -> StateSampler:
    if sampler_type not in SAMPLER_TYPES:
        raise ValueError(
            f"Unknown sampler_type {sampler_type!r}; "
            f"expected one of {sorted(SAMPLER_TYPES)}"
        )
    if sampler_type == "greedy":
        return GreedySampler(sampler_dir=sampler_dir, **kwargs)
    if sampler_type in {"puct", "puct_backprop"}:
        return PUCTSampler(sampler_dir=sampler_dir, **kwargs)
    return FixedSampler(**kwargs)
