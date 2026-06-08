"""
Proactive Beam cluster cleanup at epoch boundaries.

Between epochs, the GCP cluster's host disks fill with image cache layers.
This callback drains workers and clears image caches at the end of each epoch
so that the next epoch starts with clean disk and fresh workers.
"""

import logging
from typing import Optional

from skyrl_train.callbacks.base import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)


class BeamEpochCleanupCallback(TrainerCallback):
    """Drains Beam workers and clears image caches between epochs."""

    def __init__(self, wait_seconds: int = 60):
        super().__init__()
        self.wait_seconds = wait_seconds

    def on_epoch_end(
        self,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> Optional[TrainerControl]:
        try:
            from harbor.environments.beam import BeamEnvironment
        except ImportError:
            logger.debug("BeamEpochCleanupCallback: harbor.environments.beam not available, skipping")
            return None

        logger.info(
            "BeamEpochCleanupCallback: proactive cleanup at epoch boundary (epoch %s)",
            state.epoch,
        )
        BeamEnvironment.cleanup_between_steps(wait_seconds=self.wait_seconds)
        return None
