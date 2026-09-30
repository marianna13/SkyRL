"""Strict tiny Grug parity against the committed Levanter oracle.

The fixture freezes Levanter behavior at its documented producing commit. This
test detects PyTorch drift from that snapshot; regenerating the fixture is what
detects intentional Levanter-side changes.
"""

from __future__ import annotations

import pytest
import torch

from tests.backends.skyrl_train.models.grug_training_parity import run_grug_training_parity


def test_grug_training_matches_levanter_oracle():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (tolerances were locked on H100 FP32)")

    run_grug_training_parity()
