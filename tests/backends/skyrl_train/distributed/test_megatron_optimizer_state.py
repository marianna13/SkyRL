from collections import defaultdict

from skyrl.backends.skyrl_train.distributed.megatron.optimizer_state import (
    reuse_preinitialized_state_buffers,
)


class _FakeBuffer:
    def __init__(self, value):
        self.value = value
        self.zero_calls = 0

    def zero_(self):
        self.value = 0
        self.zero_calls += 1


class _FakeOptimizer:
    def __init__(self):
        self.state = defaultdict(dict)
        self.allocations = 0

    def _initialize_state(
        self, param, state_name, zero_buffer, store_param_remainders=False
    ):
        self.allocations += 1
        buffer = _FakeBuffer(-1)
        self.state[param][state_name] = buffer
        if zero_buffer:
            buffer.zero_()


def test_reuses_preinitialized_state_buffer_and_restores_method():
    optimizer = _FakeOptimizer()
    param = object()
    original_buffer = _FakeBuffer(7)
    optimizer.state[param]["exp_avg"] = original_buffer

    with reuse_preinitialized_state_buffers(optimizer):
        # Simulate TE FusedAdam.load_state_dict clearing the state mapping.
        optimizer.state[param] = {}
        optimizer._initialize_state(param, "exp_avg", False)

        assert optimizer.state[param]["exp_avg"] is original_buffer
        assert optimizer.allocations == 0

    assert "_initialize_state" not in vars(optimizer)
    optimizer.state[param] = {}
    optimizer._initialize_state(param, "exp_avg", False)
    assert optimizer.allocations == 1
    assert optimizer.state[param]["exp_avg"] is not original_buffer


def test_falls_back_for_missing_state_and_preserves_zero_semantics():
    optimizer = _FakeOptimizer()
    param = object()
    original_buffer = _FakeBuffer(11)
    optimizer.state[param]["exp_avg"] = original_buffer

    with reuse_preinitialized_state_buffers(optimizer):
        optimizer.state[param] = {}
        optimizer._initialize_state(param, "exp_avg", True)
        optimizer._initialize_state(param, "new_state", True)

        assert optimizer.state[param]["exp_avg"] is original_buffer
        assert original_buffer.value == 0
        assert original_buffer.zero_calls == 1
        assert optimizer.allocations == 1
        assert optimizer.state[param]["new_state"].value == 0
