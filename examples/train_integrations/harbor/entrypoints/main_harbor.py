"""
Main entrypoint for training on Harbor tasks.
"""

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import ray
import torch
import yaml

from skyrl.backends.skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry
from skyrl.train.config import GeneratorConfig, SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.rate_limiter import RateLimiterConfig
from skyrl.train.utils.utils import initialize_ray

from ..dataset import HarborTaskDataset, TTTHarborOneTaskDataset
from ..harbor_generator import HarborGenerator
from ..harbor_generator_ttt import TTTHarborGenerator, adaptive_entropic_group
from ..reward_shaping import HarborRewardShapingConfig

# NOTE (sumanthrh): We use a YAML to store the defaults for the Harbor trial configuration
# TODO: Convert to a dataclass
HARBOR_DEFAULT_CONFIG = Path(__file__).parent.parent / "harbor_trial_config" / "default.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class HarborGeneratorConfig(GeneratorConfig):
    """GeneratorConfig with Harbor-specific rate limiting."""

    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)
    reward_shaping: HarborRewardShapingConfig = field(default_factory=HarborRewardShapingConfig)


@dataclass
class TTTConfig(GeneratorConfig):
    enable: bool = False
    budget_s: int = 1000
    sampler_type: str = "greedy"
    sampler_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"puct_c": 1.0}
    )


@dataclass
class HarborSkyRLConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig with Harbor trial configuration."""

    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)
    ttt_cfg: TTTConfig = field(default_factory=TTTConfig)


class HarborExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborGenerator.
        """
        if cfg.ttt_cfg.enable:
            return TTTHarborGenerator(
                generator_cfg=cfg.generator,
                harbor_cfg=cfg.harbor_trial_config,  # Pass harbor config to the generator
                ttt_cfg=cfg.ttt_cfg,
                inference_engine_client=inference_engine_client,
                tokenizer=tokenizer,
                max_seq_len=cfg.trainer.algorithm.max_seq_len,
            )

        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,  # Pass harbor config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        if self.cfg.ttt_cfg.enable:
            prompts_dataset = TTTHarborOneTaskDataset(
                data_files=self.cfg.data.train_data,
                virtual_size=self.cfg.trainer.train_batch_size,
            )
        else:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.train_data,
            )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None


def entropic_adv_estimator(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    beta: float = 2.0,
    **kwargs,
):
    """
    Entropic objective favors maximum reward actions

    Matches the reference TTT implementation's leave-one-out entropic weight.
    """

    with torch.no_grad():
        rewards = token_level_rewards.sum(dim=-1)
        advantages = torch.zeros_like(rewards)
        id2positions: dict[object, list[int]] = defaultdict(list)
        for position, group_id in enumerate(index):
            id2positions[group_id].append(position)

        # SkyRL forwards the AlgorithmConfig through kwargs. A custom config can
        # therefore expose adv_estimator_beta; otherwise the function argument
        # (2.0 by default) is used.
        config = kwargs.get("config")
        beta = float(getattr(config, "adv_estimator_beta", beta))

        for positions in id2positions.values():
            group_size = len(positions)
            if group_size == 1:
                # No leave-one-out comparison is available.
                advantages[positions[0]] = 0.0
                continue

            position_tensor = torch.as_tensor(positions, device=rewards.device)
            group_rewards = rewards[position_tensor]

            centered = group_rewards - group_rewards.max()
            exponentials = torch.exp(beta * centered)
            loo_normalizer = (
                exponentials.sum() - exponentials
            ) / (group_size - 1)
            group_advantages = exponentials / (loo_normalizer + 1e-12) - 1.0

            advantages[position_tensor] = group_advantages

        advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages.clone()

def entropic_adaptive_beta_adv_estimator(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    beta: float = 2.0,
    **kwargs,
):
    """
    Match Discover's adaptive entropic estimator with target KL log(2).

    ``beta`` is accepted for registry compatibility but intentionally unused.
    """

    with torch.no_grad():
        rewards = token_level_rewards.sum(dim=-1)
        advantages = torch.zeros_like(rewards)
        id2positions: dict[object, list[int]] = defaultdict(list)
        for position, group_id in enumerate(index):
            id2positions[group_id].append(position)

        beta_values: list[float] = []
        achieved_kls: list[float] = []
        saturated_groups = 0

        for positions in id2positions.values():
            group_size = len(positions)
            if group_size == 1:
                # No leave-one-out comparison is available.
                advantages[positions[0]] = 0.0
                continue

            position_tensor = torch.as_tensor(positions, device=rewards.device)
            group_advantages, solved_beta, achieved_kl, saturated = (
                adaptive_entropic_group(rewards[position_tensor])
            )
            advantages[position_tensor] = group_advantages.to(advantages.dtype)
            beta_values.append(solved_beta)
            achieved_kls.append(achieved_kl)
            saturated_groups += int(saturated)

        advantages = advantages.unsqueeze(-1) * response_mask
        if beta_values:
            logging.getLogger(__name__).info(
                "adaptive_beta groups=%d beta=[%.6g, %.6g] kl=[%.6g, %.6g] "
                "reward=[%.6g, %.6g] advantage=[%.6g, %.6g] saturated=%d",
                len(beta_values),
                min(beta_values),
                max(beta_values),
                min(achieved_kls),
                max(achieved_kls),
                float(rewards.min().item()),
                float(rewards.max().item()),
                float(advantages.min().item()),
                float(advantages.max().item()),
                saturated_groups,
            )
    return advantages, advantages.clone()



AdvantageEstimatorRegistry.register("entropic", entropic_adv_estimator)
AdvantageEstimatorRegistry.register("entropic_adaptive_beta", entropic_adaptive_beta_adv_estimator)

@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = HarborExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    # Load harbor defaults and merge CLI overrides on top
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor training; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
