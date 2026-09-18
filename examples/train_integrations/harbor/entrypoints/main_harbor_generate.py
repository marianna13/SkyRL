"""Evaluation-only entrypoint for Harbor tasks.

This starts only the rollout inference engines and Harbor environments. It does
not construct a policy/ref model, optimizer, or Megatron trainer. Results are
aggregated by :mod:`skyrl.train.evaluate` and, when
``trainer.dump_eval_results=true``, written below
``trainer.export_path/dumped_evals/eval_only``.
"""

import asyncio
import sys
from typing import Any

import ray
import yaml
from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInterface,
)
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.evaluate import evaluate, evaluate_step_wise
from skyrl.train.utils.trainer_utils import build_dataloader
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg

from ..dataset import HarborTaskDataset
from ..harbor_generator import HarborGenerator
from .main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborSkyRLConfig,
    _deep_merge,
)


class HarborEvalOnlyEntrypoint(EvalOnlyEntrypoint):
    """Run a complete Harbor evaluation without initializing a trainer."""

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_eval_dataset(self):
        if not self.cfg.data.val_data:
            return None
        return HarborTaskDataset(data_files=self.cfg.data.val_data)

    async def run(
        self, inference_engine_client: InferenceEngineInterface
    ) -> dict[str, Any]:
        assert self.eval_dataset is not None, (
            "The Harbor evaluation-only entrypoint requires data.val_data"
        )

        await inference_engine_client.wake_up()
        generator = self.get_generator(
            self.cfg, self.tokenizer, inference_engine_client
        )
        eval_fn = (
            evaluate_step_wise
            if self.cfg.generator.step_wise_trajectories
            else evaluate
        )
        return await eval_fn(
            eval_dataloader=build_dataloader(
                self.cfg, self.eval_dataset, is_train=False
            ),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: HarborSkyRLConfig) -> dict:
    exp = HarborEvalOnlyEntrypoint(cfg)
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_generator_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor "
            "evaluation; it is required to bound generated trajectories."
        )
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info("Metrics from Harbor eval-only run: {}", metrics)


if __name__ == "__main__":
    main()
