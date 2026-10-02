# Experiment guide

All commands are run from the checkout after `pip install -e .`. The JSON plans in `experiments/` are executable specifications: `scripts/run_rq.py` expands the seed and executes them in training, evaluation, then reporting order. `--dry-run` prints the exact commands without importing model libraries. Each engine also has a standalone script with `--help`.

## Shared main configuration

| Setting | Value |
|---|---|
| Backbone | SD1.5, DDIM |
| Library order | cat, horse, dog, elephant, zebra, giraffe, bear, sheep |
| Module target | 18 selected cross-attention K/V tensors, 14,745,600 coordinates total |
| Per-module budget | 800,000 nonzero coordinates |
| Protected coordinates | Most retain-sensitive half, based on 16 gradient probes |
| Loss weights | retain 8, cohesion 4, interaction 4 |
| Erasure guidance | eta = 3 |
| Optimizer | Adam, learning rate 0.0005 |
| Retain training | 96 captions, batch size 8 |
| Cohesion | 10 external object classes plus the other 7 library concepts; one sample from each stratum per step |
| Partner selection | Full installed library with probability 0.5; otherwise a random nonempty proper subset when one exists |
| Acceptance | Erasure >= 0.83; checked every 100 steps; at most 600 steps |
| Main validation | 12 prompts × 2 generation seeds |
| Construction seeds | 17 and 29 |
| Evaluation | 12 prompts × 2 seeds per concept; 24 fixed retain captions |
| Generation | 512 × 512, 20 DDIM steps, guidance scale 7.5 |

The checked-in main presets always pass `--cohesion-siblings all --wide-val --subset-additivity`. The lower-level compiler exposes defaults needed by the ablations; use the RQ presets for the main configuration.

`protocol.json` records hyperparameters, inputs, and source hashes. Base-weight fingerprint records prevent a resume from mixing a different base model into cached artifacts. Output paths distinguish the seed, method, and ablation. The two main seeds use separate module libraries and shared evaluation prompts.

## RQ1: one module

```bash
python scripts/run_rq.py --rq rq1 --seed 17
python scripts/run_rq.py --rq rq1 --seed 29
```

RQ1 uses the eight singleton rows (`k=1`) of the same 39-configuration evaluation used by RQ2. The report gives mean target erasure, retain DINO/CLIP, and preservation of the other seven concepts. ESD-x and UCE singleton modules are evaluated with the same object detector. The full subset pass is retained so its images and scores can serve RQ2 directly.

## RQ2: combinations and restoration

```bash
python scripts/run_rq.py --rq rq2 --seed 17
python scripts/run_rq.py --rq rq2 --seed 29
```

The fixed plan contains 8 singletons, 10 pairs, 10 four-member subsets, 10 six-member subsets, and the full eight-member library. It is generated with subset seed `20260921`. DiCM, ESD merge, and UCE merge reuse independently saved modules or edits. UCE joint solves a new joint edit for each selected set. The additional prefix-baseline job evaluates sequential ESD/UCE and joint UCE along the fixed construction order.

**Merge** means adding per-concept weight deltas trained or solved against the base. **Sequential** means editing the current already-edited model. **Joint UCE** solves all concepts requested by a configuration together.

Object erasure is measured on evaluation images where the base model detects the target. The target detector threshold is 0.5. Unselected-concept retention is one minus erasure on the unselected concepts. Retain DINO compares edited and base images generated from the same caption and seed. Retain CLIP is raw image/text cosine similarity. A member has an erasure conflict if its score decreases by at least 0.20 relative to its own singleton; the report counts configurations containing at least one such member.

`verify_restore.py` exercises repeated selections, reversed selection orders, and the empty set. It checks weight fingerprints and denoiser predictions against a fixed base. The assembly context restores saved tensors instead of subtracting deltas from rounded weights.

### Extension studies

```bash
python scripts/run_rq.py --rq rq2 --extensions --dry-run
python scripts/run_rq.py --rq rq2 --job prepare-library16 --stage train
python scripts/run_rq.py --rq rq2 --job library16-A-reserve --stage train
python scripts/run_rq.py --rq rq2 --job library16-B-adaptive --stage train
python scripts/run_rq.py --rq rq2 --job sdxl --stage train
pip install -e '.[mixed]'
python scripts/run_rq.py --rq rq2 --job mixed-domain --stage train
```

The preparation job creates detector-filtered caption sets for the two 16-object libraries in `data/library16/`. Run it before either library study. The full extension plan includes both libraries, both capacity policies, their ESD/UCE baselines, mixed-domain erasure, and the four-object SDXL experiment. These studies keep their own recorded protocols and seed 17; the main preset's wide gate and sibling cohesion are not implicitly added to them.

## RQ3: mechanisms and alternative configurations

```bash
python scripts/run_rq.py --rq rq3 --seed 17 --dry-run
python scripts/run_rq.py --rq rq3 --seed 17 --job objectives --stage train
python scripts/run_rq.py --rq rq3 --seed 17 --job objective-metrics --stage eval
```

| Named jobs | What changes | Engine |
|---|---|---|
| `objectives` / `objective-metrics` | Four-concept independent, lock, library-aware, cohesion, additivity, and reverse-construction arms | `ablate_objectives.py` / `measure_modularity.py` |
| `objectives8` / `objectives8-metrics` | Eight-concept lock/additivity/reverse arms | `ablate_library.py` / `measure_modularity.py` |
| `weights-1-1`, `weights-8-4`, `weights-16-8` | Preservation strength, retain batch size, and caption coverage | `ablate_library.py` |
| `external-narrow` | External cohesion, 6 validation prompts × 1 seed | `train_library.py` |
| `siblings-narrow` | Adds the other library concepts to cohesion, keeps the narrow gate | `train_library.py` |
| `dicm` | Sibling cohesion and the 12-prompt × 2-seed gate | `train_library.py` |
| `capacity-fixed/adaptive/reserve/disjoint` | Capacity policies under external cohesion and the narrow gate | `train_library.py` |
| `global-accept` | Full installed-library additivity plus acceptance | `train_library.py` |
| `global-subset-refine` | Subset additivity plus one 150-step retrospective refinement pass | `train_library.py` |

The four-concept and eight-concept objective engines retain their respective neutral concept sets. The preservation-strength configurations use fixed 200-step training; `(1,1)` uses 24 retain captions and batch size 2, while `(8,4)` and `(16,8)` use 96 captions and batch size 8. `global-accept` uses the recorded maximum of 600 steps for seed 17 and 800 for seed 29. The main `dicm` preset uses 600 for both.

The capacity study starts with the same eight-concept order. **Fixed** allows 800,000 coordinates per module. **Adaptive** starts at 100,000 and doubles after a failed acceptance check up to 800,000. **Reserve** assigns half the remaining nominal 3.2M allowance, with a 100,000-coordinate floor. This floor allows the summed allowances to exceed 3.2M. **Disjoint** partitions each tensor's flattened indices modulo eight, then applies the shared lock.

The modularity measurement script exports per-concept cohesion, pairwise coupling, and self-consistency in `results.json` and `report.md`. Cohesion compares own-concept effect energy with non-target effect energy. Coupling measures the pair's deviation from the sum of isolated effects. Self-consistency compares a module's effect inside the full library with its isolated effect. Training minimizes unnormalized residual losses; reporting normalizes these effects.

Retrospective variants store their refined modules under `refinement/pass1/modules/additive/`. The original first-pass modules remain available in `modules/additive/`. The corresponding metrics job explicitly reads the refinement directory. The main presets do not run this pass.

## RQ4: adapter comparison

```bash
python scripts/run_rq.py --rq rq4 --seed 17
```

The plan compiles rank-4 LoRA modules in `plain` and `discipline` arms, then evaluates them and the main DiCM library on the same 39 subsets. `plain` uses fixed-step ESD-style training. `discipline` adds retained-behavior, cohesion, functional-additivity objectives, and acceptance testing. Its validation and cohesion settings are those of the adapter study. DiCM additionally uses coordinate protection and its final cohesion/gate settings. Detailed adapter defaults are in `dicm/experiments/train_lora.py` and saved into its protocol.

LoRA deltas are exported as dense `(alpha/r) BA` tensors to evaluate additive composition. DiCM artifacts retain their sparse format. The report includes all k values for DINO and CLIP and preserves target erasure alongside preservation.

## Artifacts and reporting

- `modules/<arm>/<concept>.pt`: a sparse DiCM module.
- `deltas/<method>_<concept>.pt`: a baseline weight delta.
- `protocol.json`: run configuration and source/input references.
- `history/`, `acceptance.json`: optimization traces and acceptance outcomes.
- `results.json`: raw per-concept and per-image scores, plus retain metrics.
- `summary.json`, `final_status.json`: engine-level summaries and completion state.
- `outputs/reports/`: cross-method JSON/CSV/Markdown reports produced by the RQ runner.

Prefix/objective/capacity engines score their outputs during training. Their native `summary.json` and metric `report.md` complement the 39-subset reports. `summarize.py` is specifically for complete subset evaluations, not prefix results.
