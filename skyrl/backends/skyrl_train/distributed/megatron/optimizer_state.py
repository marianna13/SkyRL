"""Helpers for memory-safe Megatron optimizer checkpoint loading."""

from contextlib import contextmanager
from types import MethodType
from typing import Any, Iterator

_MISSING = object()


@contextmanager
def reuse_preinitialized_state_buffers(optimizer: Any) -> Iterator[None]:
    """Make a TE-style optimizer reattach existing state buffers on load.

    Transformer Engine's ``FusedAdam.load_state_dict`` clears each parameter's
    state and calls ``_initialize_state`` before copying the incoming value.  In
    Megatron distributed-checkpoint loading, the incoming values are templates
    backed by optimizer buffers that were allocated immediately beforehand.
    Allocating replacements therefore doubles the Adam-state peak temporarily.

    This scoped patch keeps shallow references to the initialized buffers and
    makes ``_initialize_state`` reattach a matching buffer.  Missing state names
    retain the optimizer's normal allocation behavior.  The original method is
    restored even if checkpoint loading raises.
    """

    reusable = {param: dict(state) for param, state in optimizer.state.items()}
    original_initialize_state = optimizer._initialize_state
    had_instance_override = "_initialize_state" in vars(optimizer)
    previous_instance_override = vars(optimizer).get("_initialize_state")

    def _reuse_or_initialize(
        self: Any,
        param: Any,
        state_name: str,
        zero_buffer: bool,
        store_param_remainders: bool = False,
    ) -> Any:
        buffer = reusable.get(param, {}).get(state_name, _MISSING)
        if buffer is _MISSING:
            return original_initialize_state(
                param,
                state_name,
                zero_buffer,
                store_param_remainders,
            )

        self.state[param][state_name] = buffer
        if zero_buffer:
            buffer.zero_()
        return None

    optimizer._initialize_state = MethodType(_reuse_or_initialize, optimizer)
    try:
        yield
    finally:
        if had_instance_override:
            optimizer._initialize_state = previous_instance_override
        else:
            del optimizer._initialize_state
