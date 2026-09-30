"""LR schedules built by get_megatron_optimizer_param_scheduler (Megatron OptimizerParamScheduler)."""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (  # noqa: E402
    get_megatron_optimizer_param_scheduler,
)


def _optimizer():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)


def _config(scheduler, lr=1e-5, min_lr=0.0, warmup=10):
    return SimpleNamespace(scheduler=scheduler, lr=lr, min_lr=min_lr, num_warmup_steps=warmup, weight_decay=0.01)


def _lrs(scheduler, steps):
    optimizer = scheduler.optimizer
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        scheduler.step(1)
        lrs.append(optimizer.param_groups[0]["lr"])
    return lrs


def test_cosine_warms_up_then_decays_to_min_lr():
    total, warmup, lr, min_lr = 235, 10, 1e-5, 1e-6
    sched = get_megatron_optimizer_param_scheduler(_optimizer(), _config("cosine", lr, min_lr, warmup), total)
    lrs = _lrs(sched, total)
    assert lrs[0] == pytest.approx(0.0, abs=1e-12)
    assert lrs[warmup] == pytest.approx(lr)
    mid = warmup + (total - warmup) // 2
    progress = (mid - warmup) / (total - warmup)
    assert lrs[mid] == pytest.approx(min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress)), rel=1e-6)
    assert lrs[total] == pytest.approx(min_lr)
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1 :]))


def test_linear_decays_to_min_lr():
    sched = get_megatron_optimizer_param_scheduler(_optimizer(), _config("linear", warmup=0), 100)
    lrs = _lrs(sched, 100)
    assert lrs[50] == pytest.approx(0.5e-5)
    assert lrs[100] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("scheduler", ["constant_with_warmup", "constant"])
def test_constant_after_warmup(scheduler):
    sched = get_megatron_optimizer_param_scheduler(_optimizer(), _config(scheduler), 100)
    lrs = _lrs(sched, 100)
    assert lrs[10] == pytest.approx(1e-5) and lrs[100] == pytest.approx(1e-5)


@pytest.mark.parametrize("num_training_steps", [None, int(1e9)])
def test_unknown_step_count_gives_constant_placeholder(num_training_steps):
    sched = get_megatron_optimizer_param_scheduler(_optimizer(), _config("cosine"), num_training_steps)
    lrs = _lrs(sched, 1000)
    assert lrs[10] == pytest.approx(1e-5) and lrs[1000] == pytest.approx(1e-5)


def test_rebuilt_scheduler_restores_position_from_state_dict():
    """SFT rebuilds the scheduler with the real step count, then a resume loads its state."""
    config = _config("cosine")
    first = get_megatron_optimizer_param_scheduler(_optimizer(), config, 235)
    _lrs(first, 100)
    optimizer = _optimizer()
    resumed = get_megatron_optimizer_param_scheduler(optimizer, config, 235)
    resumed.load_state_dict(first.state_dict())
    assert optimizer.param_groups[0]["lr"] == pytest.approx(first.optimizer.param_groups[0]["lr"])


def test_rejects_unsupported_scheduler_and_warmup_past_end():
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        get_megatron_optimizer_param_scheduler(_optimizer(), _config("polynomial"), 100)
    with pytest.raises(ValueError, match="num_warmup_steps"):
        get_megatron_optimizer_param_scheduler(_optimizer(), _config("cosine", warmup=100), 100)
