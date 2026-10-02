# DiCM: Constructing Composable Erasure Modules for Diffusion Models

DiCM (**Diffusion Concept Modularization**) compiles each concept-erasure request into a sparse weight-update module. Modules share a frozen diffusion model and can be selected and composed for different deployment requirements.

The compiler combines an erasure objective, retained-behavior protection, cohesion on non-target concepts, and a functional-additivity objective against existing modules. This repository contains the compiler, module assembly, evaluation, baselines, and experiment plans organized around the paper's four research questions.

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Install a PyTorch / torchvision build matching your CUDA environment first.
pip install -e '.[dev,plots]'
python scripts/doctor.py --require-cuda
pytest -q
```

Python 3.10+ is supported by the source. Training scripts use CUDA. The [validation record](docs/validation.md) lists the environment actually tested, including a real SD1.5 smoke run on a 16 GB GPU.

## Models and data

By default, model loaders use these Hugging Face identifiers:

| Component | Identifier | Local-directory override |
|---|---|---|
| SD1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `DICM_SD15_MODEL` |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | `DICM_SDXL_MODEL` |
| CLIP | `openai/clip-vit-large-patch14` | `DICM_CLIP_MODEL` |
| DINOv2 | `facebook/dinov2-large` | `DICM_DINO_MODEL` |

For example, set `export DICM_SD15_MODEL=/path/to/stable-diffusion-v1-5` to use a local Diffusers snapshot. The object evaluator loads torchvision's `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT` detector. First use downloads missing model assets.

Prepare COCO training captions (COCO images are not needed):

```bash
python scripts/prepare_data.py
# Or use an existing annotation file:
export DICM_COCO_CAPTIONS=/path/to/captions_train2017.json
```

The scripts select deterministic training and test caption sets from this file. Model checkpoints, captions, and generated outputs are stored outside version control.

## Quick functional check

```bash
python scripts/smoke.py
python scripts/smoke_metrics.py
```

The first command trains two modules for two steps each, exercises the real compiler and composition path, and checks sparse budgets, protected coordinates, the frozen first module, assembly order, and exact restoration. It writes `outputs/smoke/validation.json` and an example image. The second checks CLIP, DINO, and the object detector. These are short code checks; full experiments use the plans below.

## Experiments by research question

| RQ | Question | Plan | Main outputs |
|---|---|---|---|
| RQ1 | Does one module erase its target while preserving other behavior? | [rq1.json](experiments/rq1.json) | Singleton rows, erasure, DINO, CLIP, and unselected-concept retention |
| RQ2 | Can a frozen module library serve different combinations? | [rq2.json](experiments/rq2.json) | 39 fixed subsets, member-level conflicts, exact restoration; optional larger libraries, mixed domains, and SDXL |
| RQ3 | Which parts of the modularization design matter? | [rq3.json](experiments/rq3.json) | Objective, cohesion/gate, weight, and capacity studies; functional modularity measurements |
| RQ4 | How does composition compare with LoRA erasure modules? | [rq4.json](experiments/rq4.json) | Plain LoRA, LoRA with modularity objectives, and DiCM on the same subset plan |

Preview commands before launching a study:

```bash
python scripts/run_rq.py --rq rq1 --seed 17 --dry-run
python scripts/run_rq.py --rq rq2 --seed 17 --stage train
python scripts/run_rq.py --rq rq2 --seed 17 --stage eval
python scripts/run_rq.py --rq rq2 --seed 17 --stage report
python scripts/run_rq.py --rq rq2 --seed 29
python scripts/run_rq.py --rq rq3 --seed 17 --dry-run
python scripts/run_rq.py --rq rq4 --seed 17
```

RQ1, RQ2, and RQ4 share `outputs/main_s17` / `outputs/main_s29` and their subset results, so DiCM modules are trained once and reused. RQ3 also refers to the same final configuration. ESD/UCE baseline training follows its own recorded seed settings and shares `outputs/baselines` across plans. Existing outputs resume under the same protocol; a changed protocol requires a new output directory.

The main DiCM preset explicitly enables **sibling cohesion**, the **12-prompt / 2-seed acceptance gate**, **subset additivity**, and **no retrospective updates**. Each module has an 800,000-coordinate budget. The eight concepts are cat, horse, dog, elephant, zebra, giraffe, bear, and sheep. Ablations have separate output directories.

Select an individual job, or include the RQ2 extension studies:

```bash
python scripts/run_rq.py --rq rq3 --job objectives --stage train
python scripts/run_rq.py --rq rq3 --job objective-metrics --stage eval
python scripts/run_rq.py --rq rq2 --extensions --dry-run
# Required for the mixed-domain nudity evaluator:
pip install -e '.[mixed]'
```

`--job` selects named jobs; evaluation jobs expect their training artifacts to exist. `--extensions` includes preparation and execution of both 16-concept libraries, mixed-domain experiments, and SDXL. Full plans, settings, and output meanings are in [docs/experiments.md](docs/experiments.md).

## Reports and editable plots

```bash
python scripts/summarize.py outputs/subsets/dicm_s17 outputs/subsets/uce_joint \
  --out outputs/reports/comparison
python scripts/plot_results.py outputs/reports/comparison/summary.json \
  --out outputs/reports/comparison/curves.svg
```

Reports contain JSON, CSV, and Markdown. They average over configurations of each size and retain the worst member separately. A conflict is a member's erasure decrease of at least 20 percentage points relative to its own singleton. Missing measurements remain missing. The reporter requires the complete 39-configuration plan.

## Use a compiled module library

```python
from dicm.models.sd15_wrapper import load_sd15_pipeline
from dicm.module.library import ModuleLibrary

pipe = load_sd15_pipeline()
library = ModuleLibrary.from_directory("outputs/main_s17/modules/additive")

with library.activate(pipe.unet, ["cat", "dog"]):
    image = pipe("a photo of a cat beside a dog").images[0]

# The original base weights are restored when the context exits.
```

The assembler sums selected sparse deltas in float32, rounds once to the model dtype, and restores saved base tensors on exit. It uses a canonical concept order. Use a module with the same base model and target layers used to train it. [docs/architecture.md](docs/architecture.md) describes the code and artifact format.

## Acknowledgments

The baselines implement the experiment protocols based on [Erasing Concepts from Diffusion Models (ESD)](https://arxiv.org/abs/2303.07345) and [Unified Concept Editing (UCE)](https://arxiv.org/abs/2308.14761). The backbone is [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), with SDXL in the cross-backbone study. See [THIRD_PARTY.md](THIRD_PARTY.md) for component sources.

Code is distributed under the [MIT license](LICENSE).
