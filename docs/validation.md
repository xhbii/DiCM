# Release validation

Validated on 2026-10-02.

| Environment | Version |
|---|---|
| Python | 3.14.4 |
| PyTorch | 2.12.0+cu130 |
| torchvision | 0.27.0+cu130 |
| Diffusers | 0.38.0 |
| Transformers | 5.11.0 |
| Accelerate | 1.13.0 |
| pytest | 9.0.3 |
| GPU | NVIDIA GeForce RTX 5080, 16 GB |

## Checks performed

- **17 CPU tests passed.** Coverage includes global sparse budgets with ties and locks, zero budgets, overlapping/canceling deltas, frozen-library gradients, nested disable contexts, exception-safe restoration, canonical assembly, protocol/base-model resume checks, complete subset aggregation, missing scores, main preset flags, seed overrides, and fresh baseline-prefix construction.
- **78 CLI checks passed.** Every entry point accepts `--help`; every RQ job's argument list was passed through its actual parser without launching training.
- **Static checks passed.** Python compilation and Ruff's undefined-name checks completed.
- **Editable installation and wheel build passed.** The wheel contains the SD1.5 K/V layer specification.
- **Real SD1.5 training smoke passed.** Two concepts, two optimizer steps each, all 18 target tensors, 1,024-coordinate module budgets, retained-coordinate protection, sibling cohesion, and a frozen existing partner module. Both modules had 1,024 nonzero coordinates and no updates to protected coordinates. The first saved module remained unchanged when training the second.
- **Real composition and restoration passed.** Reversing module selection produced identical generated-image bytes. Six subsequent selection/removal checks recovered the base weight hash and fixed-input denoiser prediction exactly.
- **Metric smoke passed.** CLIP and Faster R-CNN produced finite scores on the generated image; DINO self-similarity was 1.0.
- **Reporting and SVG generation passed.** The reporter was checked against existing final-configuration results for both construction seeds, covering 39 configurations each. The resulting full-library erasure/DINO values matched the study records, and both had 0/39 thresholded conflicts.

The full RQ training runs were not repeated for this release. The GPU checks exercise the actual compiler, model loader, composition, and evaluation components on a short run. Run the checked-in experiment plans for complete evaluations.

## Release changes tested

The release removes research-workspace paths and prior-run dependencies, makes sparse projection respect its budget under tied values, restores nested disable states, and preserves acceptance/capacity records when resuming. It also guards cached outputs against a changed protocol or base model and stores retrospective-refinement artifacts separately from first-pass modules.

The JSON plans select the final sibling-cohesion/wide-gate configuration explicitly and keep ablations in separate directories. The 39-subset specification and image-score definitions are retained from the experiment code.
