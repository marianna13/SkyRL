# GrugMoE vLLM compatibility pin

This out-of-tree model integration is validated with **vLLM 0.27.0**. The
model module rejects every other vLLM version before importing version-specific
APIs. This pin is intentionally local to GrugMoE: SkyRL's general dependency
lock still targets vLLM 0.26.0, while Jupiter supplies the newer compiled
Torch/vLLM stack through its runtime virtual environment.

The pin must not be relaxed without revalidating the attention-window
semantics. Grug has local 2,048-token layers and full-attention layers 3, 7,
11, 15, 19, 23, and 25 (zero-based). Under vLLM 0.27,
`GrugMoeConfig.layer_types` remains
attention-only so schema-1 checkpoints are not mistaken for Mamba hybrids;
the actual schedule is stored in `grug_attention_layer_types`. Consequently,
`CacheConfig.sliding_window` must be `None`, and the model passes 2,048 only to
the local layers. Model construction rejects a non-`None` cache-level window.

JURECA's validated vLLM 0.26 plugin reaches the same effective schedule by
declaring mixed `layer_types`. Do not copy that configuration verbatim into
this integration: vLLM 0.26 and 0.27 derive the cache-level window differently.

After a vLLM or Grug integration change, repeat the frozen Snowball canary with
the same checkpoint, task IDs, prompt/tool contract, token budget, special-token
handling, and sampler. Before comparing task reward, verify that:

- the startup log reports `cache_window=None`, `local_window=2048`, and
  `global_layers=[3, 7, 11, 15, 19, 23, 25]`;
- accepted-turn rate remains flat across prompt-length buckets; and
- the paired score and trajectory-health metrics do not regress against the
  last accepted canary.

The JURECA reference procedure and analysis commands are documented in
`monorepo/rl/scripts/eval_jureca/README.md`.
