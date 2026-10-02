# Code and artifact structure

```text
dicm/
  models/             SD1.5 loader
  module/
    sparse_delta.py   Sparse parametrization, budget projection, serialization, composition
    backbone_partition.py  Retain-sensitivity coordinate protection
    latents.py        Frozen embeddings, latent caching, diffusion noise
    library.py        Named module selection, installation, exact restoration
    conflict_bench.py Object benchmark and retain evaluation
    concept_bench.py  Object/style/content benchmark
    backbone.py       SD1.5 / SDXL denoiser interface
    library16.py      Larger-library concepts and prompts
  experiments/        Training, ablation, baseline, and evaluation engines
  evaluation/         CLIP, DINO, and subset result aggregation
  utils/              Model resolution and experiment record I/O
experiments/          RQ1–RQ4 executable JSON plans
scripts/              CLI entry points, smoke checks, reporting, plots
tests/                CPU regression checks
```

## One module

For each selected weight tensor, the parametrization computes `W + delta`. The base weights are frozen. Trainable deltas are float32 and projected onto a global coordinate budget after every optimizer step. Retain-sensitive locked coordinates are zeroed before projection. The projection selects an exact number of entries when magnitudes tie.

The main training loop creates frozen base predictions for the target, retain, and non-target inputs. It optimizes erasure, retain, and cohesion losses, then evaluates additivity against a selected frozen partner subset. Only the new module receives gradients. Acceptance checks evaluate the new module on its own.

## Sparse artifact format

A module is a PyTorch dictionary loaded using `weights_only=True`:

```python
{
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight": {
        "shape": (320, 768),
        "indices": torch.tensor([...], dtype=torch.int64),  # flattened nonzero indices
        "values": torch.tensor([...], dtype=torch.float32),
    },
    # Remaining selected tensors follow the same format.
}
```

All values and indices are exported on CPU. The module owns its update values; a support mask alone does not define the update. Overlapping module coordinates are allowed and their values add. Composition does not retrain the base or existing modules.

## Assembly and removal

`ModuleLibrary.activate(unet, concepts)` validates parameter shapes, sorts concept names, sums sparse deltas in float32, and materializes edited weights. On exit it copies the saved base tensors back, including when generation raises an exception. Empty selection leaves the model unchanged. The context edits a model in place; serialize concurrent access to that model and start each selection from the base.

## Baselines and measurements

`train_baselines.py` builds singleton, sequential, and joint ESD/UCE artifacts from a fresh checkout. The LoRA engine compiles independent or discipline-trained adapters. `evaluate_subsets.py` applies the shared fixed subset plan. `measure_modularity.py` measures behavior in denoiser-output space; `conflict_bench.py` measures generated images.

The migration index in [source-map.json](source-map.json) maps the release engines to the numbered experiment scripts from the research workspace.
