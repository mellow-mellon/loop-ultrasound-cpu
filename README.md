# Loop Transformer × breast ultrasound: CPU pilot

This repository starts a research project about **parameter sharing, segmentation supervision, and benign/malignant prediction** from full-frame breast ultrasound images. It includes original runnable code, real BUS-BRA data acquisition, and executed CPU engineering reports.

**Status: engineering pilot, not clinical validation.** Tiny-set training checks that the implementation can learn. It does not establish that looping improves diagnosis. The released `Case` grouping has not yet been verified against patient identity, so the current split cannot support patient-independent clinical claims.

## What the loop actually does

```text
Full image (no ground-truth crop), 224×224 letterbox
  → frozen ImageNet DeiT-Tiny encoder → 196 spatial tokens × 192
  → refinement block → h1 → refinement block → h2 → … → h4
                         ↓                         ↓
             shared classification head + shared segmentation head
```

The S arms reuse **one Transformer block four times**, with updated hidden tokens each time. The U arms apply **four independent blocks**, initially equal in value. This study loops an added refinement module after a frozen image encoder; it does not loop all of DeiT or host an LLM.

| Arm | Refinement parameters | Segmentation gradient into representation |
|---|---|---|
| SC | Shared | Detached; mask head is a diagnostic probe |
| SJ | Shared | Joint classification + segmentation supervision |
| UC | Independent at each depth | Detached probe |
| UJ | Independent at each depth | Joint supervision |

Both heads read every trained depth 1–4. All arms use the same losses, initialization seed and Case/view sampling rules. The C arms also train the mask decoder, but its loss cannot update the representation or classifier. Gradient clipping is separate for the representation/classifier and mask head.

Research contrasts: `D = Brier(step 2) − Brier(step 4)` and `S = Dice(step 4) − Dice(step 2)`. Positive values indicate improvement. The proposed interaction is `(D_SJ − D_SC) − (D_UJ − D_UC)`. Better segmentation alone is not evidence of better benign/malignant prediction. The tiny pilot only verifies that these measurements can be computed and aligned.

## Reproduce the five startup steps

Use Python 3.12. The executed environment was macOS arm64, CPU only, with two PyTorch threads. There are no CUDA/MPS calls.

### 1. Isolated environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation .
```

`requirements-lock.txt` records the actual installed package versions. The lock is an observed macOS environment, not a guarantee that all wheels exist for every OS. For other systems, install the current CPU-compatible PyTorch build, then `python -m pip install '.[test]'`. A normal package install is used above; after modifying source, reinstall or set `PYTHONPATH=src` during development.

### 2. Acquire real data, split and audit

```bash
python -m loop_ultrasound.prepare --data-dir data
python -m loop_ultrasound.prepare_encoder --data-dir data
```

The first command downloads the fixed [BUS-BRA archive](https://zenodo.org/records/8231412), validates size and MD5, safely extracts it, builds a local manifest and audits image/mask pixels. The pretrained encoder comes from [timm DeiT-Tiny](https://huggingface.co/timm/deit_tiny_patch16_224.fb_in1k); download failures never fall back to random weights. Local encoder state SHA-256 and source are recorded. The upstream default checkpoint can change; the executed hash is in the reports.

The official outer test fold `kFold=1` is retained. Internal partitions are rebuilt by Case, stratified by pathology; the authors' `valid_i` assignments are ignored because paired views can cross those partitions. Seeds and source checksums are recorded locally. Approximate Case fractions are 55% train, 15% tune, 10% calibration and 20% test. Two views of a Case stay together. Each training epoch samples one randomly chosen view per Case.

Input is the full image. Ground-truth masks, bounding boxes, BI-RADS and pathology are never inference inputs. Masks are supervision/evaluation targets. Aspect ratio is preserved; padding is excluded from segmentation loss. Released binary masks are decoded before nearest-neighbor resizing. Actual PNG geometry is used, with inconsistent CSV dimensions reported as warnings rather than silently changing the source CSV.

The full pixel audit includes test/calibration geometry and exact duplicates, but the **trainer uses only training pixels and tune evaluation pixels**. Calibration and test prediction remain sealed.

### 3. Four-arm implementation checks

```bash
python -m pytest -q
mkdir -p outputs
python -m loop_ultrasound.selfcheck --pretrained \
  --weights-path data/deit_tiny_encoder.pt > outputs/model-selfcheck.json
```

Checks cover block reuse/independent storage, batch/depth shapes, gradient routing, separate clipping groups, frozen encoder behavior and equality of full-trajectory versus terminal-only predictions. Default selfchecks may use random encoder weights for shape checks; the command above explicitly loads real pretrained weights.

### 4. Real-image tiny CPU training

```bash
for arm in SC SJ UC UJ; do
  python -m loop_ultrasound.train --arm "$arm" \
    --max-cases 8 --epochs 30 --batch-size 2 --threads 2 \
    --cache-features --eval-max-cases 16 \
    --run-dir "outputs/cpu-pilot-$arm-seed17"
done
```

The pilot uses 4 benign + 4 malignant training Cases, all their available views, seed 17 and no augmentation. Frozen unaugmented encoder features are cached. The loss is averaged across all four trained readouts: classification BCE plus `0.5 × valid-pixel BCE + 0.5 × soft Dice loss`. A detached mask branch still trains its decoder. AdamW uses pilot learning rate 0.001. The loader has zero workers and one/two CPU threads are allowed. Completed runs are never overwritten.

Each run saves a local checkpoint, loss history and per-image predictions, plus an aggregate summary. The fixed-seed 16-Case tune subset is label-blind. In this tiny-run command, a different `--seed` also changes subset selection; the expanded experiment below instead fixes an independent `--selection-seed` across training seeds. Passing `--eval-max-cases 0` disables tune evaluation. An online encoder/no-cache run is also supported; augmentation and feature caching cannot be combined.

### 5. Metrics, alignment and CPU timing

```bash
python -m loop_ultrasound.analyze --run-dirs \
  outputs/cpu-pilot-SC-seed17 outputs/cpu-pilot-SJ-seed17 \
  outputs/cpu-pilot-UC-seed17 outputs/cpu-pilot-UJ-seed17 \
  --output outputs/pilot-analysis.json
python -m loop_ultrasound.benchmark --iterations 100 --warmup 5 \
  --batch-size 2 --threads 2 --output outputs/cpu-benchmark.json
```

Predictions are averaged across images within a Case before classification metrics. Dice is an equal-Case average, computed on valid 224×224 letterboxed pixels. Brier/NLL/AUROC and threshold-specific sensitivity/specificity are available; a constant threshold 0.5 is used in the pilot. The calibration utility can select a threshold on calibration data, but no threshold calibration was executed in this startup. The proposed 95% sensitivity target is a research setting, not a clinical standard.

Analysis explicitly aligns Case/image sets across arms/depths and validates training/development separation. Its paired Case bootstrap is a descriptive measurement check on a tiny development set, not a formal mechanism test.

The benchmark includes the frozen encoder, four refinement applications, all-step heads/loss, backward, separate clipping and AdamW. Terminal inference at depths 1/2/4 includes both heads but skips earlier heads. PNG preprocessing is timed separately. These are warm-process batch latencies on this machine; they do not predict GPU speed. Timing models are initialized for the benchmark and updated on one fixed mini-batch, with no accuracy claims.

## Executed results and next research gate

See [CPU results](docs/CPU_RESULTS.md), [implementation selfcheck](reports/model_selfcheck.json), aggregate pilot summaries and timing/analysis JSON in `reports/`. Images, case-level manifests, individual predictions, model weights and environment files remain local and are ignored by Git.

## Expanded CPU screen: 128 training Cases, three seeds

This experiment fixes a proportional training cohort (87 benign / 41 malignant Cases, 227 images) and all 159 tune Cases (281 images). Cohort selection seed 20261004 is independent of training seeds 17/29/43. Four arms each train 50 epochs at learning rate 0.0003, batch 2, without augmentation. All models report the final epoch; periodic development evaluation every five epochs is diagnostic and does not select a checkpoint.

```bash
python -m pip install '.[experiment,plot]'
python -m loop_ultrasound.expanded --output-dir outputs/cpu-expanded128 --threads 2
python -m loop_ultrasound.analyze_expanded --run-dirs \
  outputs/cpu-expanded128/SC-seed17 outputs/cpu-expanded128/SJ-seed17 \
  outputs/cpu-expanded128/UC-seed17 outputs/cpu-expanded128/UJ-seed17 \
  outputs/cpu-expanded128/SC-seed29 outputs/cpu-expanded128/SJ-seed29 \
  outputs/cpu-expanded128/UC-seed29 outputs/cpu-expanded128/UJ-seed29 \
  outputs/cpu-expanded128/SC-seed43 outputs/cpu-expanded128/SJ-seed43 \
  outputs/cpu-expanded128/UC-seed43 outputs/cpu-expanded128/UJ-seed43 \
  --baseline-dir outputs/cpu-expanded128/baselines \
  --output outputs/cpu-expanded128/expanded_analysis.json
```

The launcher writes an immutable protocol before training and reuses completed runs only if their config, source-code, encoder, manifest and cohort digests match. Cached features are bound to ordered image IDs, source image bytes, encoder weights and preprocessing. Caching is restricted to train/tune pixels. Evaluation uses its own DataLoader RNG so monitoring does not change the training random sequence. The recorded training-loop wall time includes periodic development evaluation.

Two controls fit only training Cases: a constant probability equal to the training malignancy proportion, and standardized frozen features plus fixed `C=1` logistic regression. The latter averages all view features before Case classification; the loop models train on one sampled view and average view probabilities at evaluation. It is a simple reference with a documented aggregation difference, not a precisely matched architecture control.

Analysis retains every training seed, validates the same Cases/views/labels across twelve runs, and averages seed-specific effects within each Case before a paired Case bootstrap. It does not treat three predictions of the same Case as three independent people. Mean AUROC/Brier across runs is distinct from evaluating a prediction ensemble.

The planned resource decision is to inspect both diagnostic ranking and probability quality, plus the consistency of depth benefits and the sharing×supervision interaction. Positive development results still require patient grouping verification, external evaluation and novelty review before a clinical or scientific claim. The full proposed 100-epoch protocol remains separate from this no-augmentation CPU screen.

Executed results are in [the expanded CPU report](docs/CPU_EXPANDED128_RESULTS.md), with all twelve aggregate run summaries, paired analysis and scientific figures in [reports/expanded128](reports/expanded128/). To regenerate figures from these public aggregate records:

```bash
python -m loop_ultrasound.plot_expanded --report-dir reports/expanded128 \
  --output-dir outputs/expanded128-figures
```

Before a formal study:

1. Verify Case-to-patient mapping, merge any repeated patient groups, and remake partitions if needed. Exact hashes do not replace near-duplicate or patient identity review.
2. Review the nearest loop/classification literature and preregister the actual contribution. Reusing a loop for ultrasound is not by itself novel.
3. Fix cohorts across training seeds and implement the proposed full training schedule/model selection. `configs/research_protocol.json` is a **proposal**, not an executable or completed experiment; the pilot has no warmup/cosine schedule.
4. Add original-resolution contour metrics, clinical annotation QA, label-policy sensitivity analyses and audited external evaluation. Better contours and better calibrated malignancy predictions must be evaluated separately.
5. Calibrate thresholds on calibration Cases, then evaluate the sealed test once under a locked protocol. This repository currently demonstrates benchmark-label prediction, not clinical usefulness or a deployment decision aid.

## CPU follow-up: separately trained depth × supervision

The completed follow-up held the shared+joint architecture fixed and freshly trained 2-step/4-step models under all-readout/terminal-only supervision, at seeds 17/29/43. It retained the same 128/159 Case cohorts and the fixed 50-epoch schedule. See [the locked experiment plan](docs/CPU_DEPTH_SUPERVISION_PLAN.md), [executed configuration](configs/cpu_depth_supervision128.json), and [results and interpretation](docs/CPU_DEPTH_SUPERVISION_RESULTS.md).

Terminal supervision uses the identical classification and mask loss only on the final readout, without dividing by configured depth; gradients still traverse the full recurrence. Every-step supervision averages losses over configured readouts. All four settings have identical trainable parameter counts and paired initial weights. Equal optimizer updates and Case exposure allow the deeper model extra compute; this is not a compute-matched experiment.

```bash
python -m loop_ultrasound.depth_supervision --prepare-only \
  --reference-protocol outputs/cpu-expanded128/protocol.json --threads 2
python -m loop_ultrasound.depth_supervision \
  --reference-protocol outputs/cpu-expanded128/protocol.json --threads 2
```

Preparation must finish before training starts. The default single worker executes all twelve runs sequentially. The executed experiment instead starts two separate CPU processes after preparation, each with `--workers 2` and a distinct `--worker-index 0` or `1`; their six-run partitions are disjoint. An unfinished claimed directory is never silently retrained or overwritten.

On another machine, first reproduce the expanded phase and use its locally produced reference protocol as above: manifest hashes include local paths, while cohort fingerprints exclude paths. The checked-in configurations record this machine's executed provenance. Previously validated feature caches may be reused; fitted model weights are always fresh.

After every run completes:

```bash
python -m loop_ultrasound.analyze_depth_supervision \
  --protocol outputs/cpu-depth-supervision128/protocol.json \
  --run-dirs outputs/cpu-depth-supervision128/SJ-depth*-*-seed* \
  --baseline-dir outputs/cpu-expanded128/baselines \
  --output outputs/cpu-depth-supervision128/depth_supervision_analysis.json
```

The primary outcome compares separately fitted 2-step/4-step **endpoints** within each supervision mode. Within-model intermediate changes are secondary; terminal-only intermediate readouts were not directly supervised. This follow-up uses an already observed development cohort, so it is exploratory. Both classification and mask supervision locations change together; their individual contributions are not isolated.

Terminal-only supervision gave positive 2-to-4-step Brier benefits in all three seeds (mean benefit 0.00928), while every-step supervision gave mixed directions (mean benefit -0.00189). Both paired Case intervals include zero. The supervision interaction also varies in direction across seeds; this does not identify a causal mechanism or establish clinical utility. The earlier negative within-model intermediate change reproduced exactly, but that comparison differs from training a short model separately.

All twelve aggregate summaries, paired analysis, execution checks and PNG/SVG figures are in [reports/depth_supervision128](reports/depth_supervision128/). Regenerate the figures from these public records into a fresh output directory:

```bash
python -m loop_ultrasound.plot_depth_supervision \
  --report-dir reports/depth_supervision128 \
  --output-dir outputs/depth-supervision-figures
```

## CPU replication: five new training seeds

The completed terminal-only screen paired separately trained depths 2/4 at five new seeds: 53, 71, 89, 107 and 131. It preserved the earlier 128/159 Case cohorts, core training code and fixed final epoch 50. See [the prespecified plan](docs/CPU_SEED_REPLICATION_PLAN.md), [locked configuration](configs/cpu_seed_replication128.json), and [results and next-experiment decision](docs/CPU_SEED_REPLICATION_RESULTS.md).

The primary resource screen uses only the five new seed pairs: at least four positive Brier benefits, positive mean benefit after removing any one seed, and no worsening of mean NLL. This is a screening rule for proposing a separate data-expansion experiment, not significance, clinical validation, a GPU gate, or automatic authorization to run that experiment. Old seeds may appear only in a secondary combined summary.

All three conditions passed. Five out of five new pairs improved Brier, AUROC and NLL; mean Brier benefit was 0.04618 with a fitted-model-conditional Case interval [0.01682, 0.07611]. Removing any one seed left positive mean benefit (minimum 0.03695). This supports reviewing a separate 256-training-Case CPU experiment. New two-step endpoints showed schedule sensitivity; Dice did not improve on average, and sensitivity at threshold 0.5 decreased while specificity increased. These results do not establish clinical utility or an internal correction mechanism. All ten aggregate summaries, strict paired analysis, execution checks and scientific figures are in [reports/seed_replication128](reports/seed_replication128/).

```bash
python -m loop_ultrasound.seed_replication --prepare-only \
  --reference-protocol outputs/cpu-depth-supervision128/protocol.json --threads 2
python -m loop_ultrasound.seed_replication \
  --reference-protocol outputs/cpu-depth-supervision128/protocol.json --threads 2
```

Finish preparation before training. The default single worker executes all ten fits sequentially; two workers use disjoint `--workers 2 --worker-index 0` and `--workers 2 --worker-index 1` commands, each with two CPU threads. The previous feature cache can be reused, but model weights are freshly initialized. Source and initial-parameter digests are fixed before the first new fit. Calibration and test predictions remain absent.

After all ten fits complete, analyze the new pairs as primary and the earlier six terminal fits as a secondary eight-seed description:

```bash
python -m loop_ultrasound.analyze_seed_replication \
  --protocol outputs/cpu-seed-replication128/protocol.json \
  --run-dirs outputs/cpu-seed-replication128/SJ-depth*-terminal-seed* \
  --reference-protocol outputs/cpu-depth-supervision128/protocol.json \
  --reference-run-dirs outputs/cpu-depth-supervision128/SJ-depth*-terminal-seed* \
  --baseline-dir outputs/cpu-expanded128/baselines \
  --output outputs/cpu-seed-replication128/seed_replication_analysis.json
```

The analyzer validates every Case, view, label, source, paired initialization and final training budget before recomputing metrics from local predictions. It also reports high-confidence mistakes, exact probability saturation and paired classification corrections separately from Brier improvement. Aggregate-only plots use only the five new seeds:

```bash
python -m loop_ultrasound.plot_seed_replication \
  --report-dir reports/seed_replication128 \
  --protocol configs/cpu_seed_replication128.json \
  --output-dir outputs/seed-replication-figures
```

## CPU Case expansion: nested 128/256 Cases at fixed optimizer updates

The completed terminal-only experiment trained six fresh 256-Case models at depths 2/4 and seeds 17/29/43, matched to the original six 128-Case terminal fits. The 256-Case cohort contains every earlier Case and view. Both sizes finished 3,200 updates: 256 Cases used 25 epochs, while the reference 128 Cases used 50 epochs. This trades repeated Case exposure for more distinct training evidence; it is not a pure data-size causal contrast or an equal-FLOPs depth comparison. See [the locked plan](docs/CPU_CASE_EXPANSION_PLAN.md), [configuration](configs/cpu_case_expansion256.json), and [completed results](docs/CPU_CASE_EXPANSION_RESULTS.md).

Both depths improved Brier, AUROC and Dice at all three matched seeds after expansion; the data-size Brier intervals still include zero. At 256 Cases, four steps improved Brier over two steps in two out of three seeds, with mean benefit 0.01064 and a fitted-model-conditional Case 95% interval [-0.01277, 0.03377]. The paired change in depth benefit from 128 to 256 Cases was 0.00136, interval [-0.02973, 0.03226]. More data gives a positive exploratory signal, while extra-loop benefit remains mixed. These results do not establish clinical utility, calibration or an internal correction mechanism. All six new aggregate summaries, refitted baseline results, paired analysis, execution checks and scientific figures are in [reports/case_expansion256](reports/case_expansion256/).

```bash
python -m loop_ultrasound.case_expansion --prepare-only \
  --reference-protocol outputs/cpu-depth-supervision128/protocol.json --threads 2
python -m loop_ultrasound.case_expansion \
  --reference-protocol outputs/cpu-depth-supervision128/protocol.json --threads 2
```

Preparation verifies the nested cohort, builds the expanded frozen-feature cache and refits the same fixed train-only constant/logistic controls. Finish it before training. Two disjoint workers can use `--workers 2 --worker-index 0` and `1`, each with two CPU threads. Same-size depth pairs share initial parameters and Case/view sampling; initial parameters also match across sizes, but batch/view sequences differ across sizes. Calibration and test remain unused.

After six fits complete:

```bash
python -m loop_ultrasound.analyze_case_expansion \
  --protocol outputs/cpu-case-expansion256/protocol.json \
  --run-dirs outputs/cpu-case-expansion256/SJ-depth*-terminal-seed* \
  --reference-protocol outputs/cpu-depth-supervision128/protocol.json \
  --reference-run-dirs outputs/cpu-depth-supervision128/SJ-depth*-terminal-seed* \
  --baseline-dir outputs/cpu-case-expansion256/baselines \
  --reference-baseline-dir outputs/cpu-expanded128/baselines \
  --output outputs/cpu-case-expansion256/case_expansion_analysis.json
```

The primary contrast is `G256 = Brier_depth2 - Brier_depth4`; the paired data-size contrast is `G256 - G128`. There is no newly invented success gate or automatic larger/GPU run. Learning curves match optimizer updates 640/1280/1920/2560/3200, corresponding to new epochs 5/10/15/20/25 and old epochs 10/20/30/40/50. Aggregate-only figure regeneration:

```bash
python -m loop_ultrasound.plot_case_expansion \
  --report-dir reports/case_expansion256 \
  --reference-report-dir reports/depth_supervision128 \
  --protocol configs/cpu_case_expansion256.json \
  --output-dir outputs/case-expansion-figures
```

## CPU 256-Case replication: five additional paired seeds

The completed fixed-budget replication added seeds 53, 71, 89, 107 and 131 at 256 Cases, with separately trained terminal-only depths 2/4. All ten fresh fits used the same 25 epochs and 3,200 updates as the original 256-Case experiment, paired to their own existing 128-Case, 50-epoch fits. The five new seeds are primary; the original three and an explicitly separated eight-seed combined description are secondary. See [the prespecified plan](docs/CPU_CASE_EXPANSION_REPLICATION_PLAN.md), [locked configuration](configs/cpu_case_expansion256_replication.json), and [completed results](docs/CPU_CASE_EXPANSION_REPLICATION_RESULTS.md).

Four steps improved Brier at only two of the five additional seeds. Mean G256 was -0.00976 with a fitted-model-conditional Case 95% interval [-0.02967, 0.01060]; all leave-one-seed-out means remained negative. The paired depth-benefit change from 128 to 256 Cases was -0.05595, interval [-0.08686, -0.02616], negative at all five seeds. The eight-seed secondary G256 mean was -0.00211, interval [-0.01757, 0.01382], with four positive/four negative seeds. Depth2 benefited more from the fixed-update expansion; diagnostic probability improvements at depth4 were mixed despite better segmentation. This supports a separate CPU study of training endpoint stability, without establishing a pure data-size cause, a general recurrence failure or clinical utility. All ten aggregate summaries, paired analysis, reused baseline aggregate, execution checks and three scientific figures are in [reports/case_expansion256_replication](reports/case_expansion256_replication/).

```bash
python -m loop_ultrasound.case_expansion_replication --prepare-only --threads 2
python -m loop_ultrasound.case_expansion_replication --threads 2
```

The defaults use `outputs/cpu-seed-replication128` for paired references and `outputs/cpu-case-expansion256` for the original expansion protocol and fixed 256-Case controls. Preparation verifies all ten reference summaries and reuses the identical frozen-feature cache and byte-bound baseline artifacts. Two disjoint workers can use `--workers 2 --worker-index 0` and `1`, each with two CPU threads. Fresh trainable weights match each seed's earlier initialization; the core training source stays unchanged. There is no new success gate, model-selection change or automatic larger/GPU experiment.

After all ten new fits complete, analyze the five new pairs as primary, with the earlier three pairs and the eight-seed combination explicitly secondary:

```bash
python -m loop_ultrasound.analyze_case_expansion \
  --protocol outputs/cpu-case-expansion256-replication/protocol.json \
  --run-dirs outputs/cpu-case-expansion256-replication/SJ-depth*-terminal-seed* \
  --reference-protocol outputs/cpu-seed-replication128/protocol.json \
  --reference-run-dirs outputs/cpu-seed-replication128/SJ-depth*-terminal-seed* \
  --baseline-dir outputs/cpu-case-expansion256/baselines \
  --reference-baseline-dir outputs/cpu-expanded128/baselines \
  --secondary-protocol outputs/cpu-case-expansion256/protocol.json \
  --secondary-run-dirs outputs/cpu-case-expansion256/SJ-depth*-terminal-seed* \
  --secondary-reference-protocol outputs/cpu-depth-supervision128/protocol.json \
  --secondary-reference-run-dirs outputs/cpu-depth-supervision128/SJ-depth*-terminal-seed* \
  --output outputs/cpu-case-expansion256-replication/case_expansion_analysis.json

python -m loop_ultrasound.plot_case_expansion \
  --report-dir reports/case_expansion256_replication \
  --reference-report-dir reports/seed_replication128 \
  --protocol configs/cpu_case_expansion256_replication.json \
  --output-dir outputs/case-expansion-replication-figures
```

The figures show only the five new seeds: endpoint means, optimizer-update learning curves, and each paired seed's depth benefit, interaction and direct data-size benefit. Conditional Case intervals and leave-one-seed-out means describe the locked fitted models and do not establish clinical utility.

## CPU probability diagnostic and tail learning-rate control

The two completed prespecified checks use the previously observed tune cohort and keep calibration/test predictions sealed. See [the plan](docs/CPU_PROBABILITY_AND_TAIL_PLAN.md), [completed results](docs/CPU_PROBABILITY_AND_TAIL_RESULTS.md), [temperature configuration](configs/cpu_temperature_diagnostic.json), and [tail-LR configuration](configs/cpu_tail_lr256.json).

Temperature cross-fitting reduced mean probability losses but left the 256-Case depth benefit near zero: OOF G=-0.00164, conditional Case interval [-0.01237, 0.00916]. The separate fixed tail-LR intervention produced H=+0.02725, interval [0.00979, 0.04596], positive at all five paired seeds; absolute low-tail G=+0.01749, interval [0.00171, 0.03376], positive at four of five. All ten constant branches exactly replayed earlier models. These are exploratory optimizer-policy and probability-scale findings on observed tune, without independent patient validation or a general recurrence claim. Aggregate-only summaries and figures are in [reports/temperature_diagnostic](reports/temperature_diagnostic/) and [reports/tail_lr256](reports/tail_lr256/).

The temperature diagnostic cross-fits one positive Case-level temperature on five common stratified tune folds, separately for each fixed depth/seed model. The primary models are the five 256-Case seed pairs; matched 128-Case models are secondary. Original probabilities remain unchanged, and out-of-fold predictions stay local. Fold-specific temperatures can change pooled AUROC, so this is a probability-scale diagnostic rather than representation learning or untouched validation.

```bash
python -m loop_ultrasound.temperature_diagnostic --prepare \
  --run-dirs outputs/cpu-case-expansion256-replication/SJ-depth*-terminal-seed* \
  --source-protocol outputs/cpu-case-expansion256-replication/protocol.json \
  --reference-run-dirs outputs/cpu-seed-replication128/SJ-depth*-terminal-seed* \
  --reference-protocol outputs/cpu-seed-replication128/protocol.json
python -m loop_ultrasound.temperature_diagnostic --execute \
  --run-dirs outputs/cpu-case-expansion256-replication/SJ-depth*-terminal-seed* \
  --source-protocol outputs/cpu-case-expansion256-replication/protocol.json \
  --reference-run-dirs outputs/cpu-seed-replication128/SJ-depth*-terminal-seed* \
  --reference-protocol outputs/cpu-seed-replication128/protocol.json
```

The training control starts ten fresh seed/depth prefixes at learning rate 3e-4 through update 2,560. Complete model, AdamW, RNG and sampler/loader states are restored into constant 3e-4 and fixed 3e-5 tails through update 3,200. Both branches share identical tail Case/view sequences. Twenty endpoint trajectories use 38,400 actual new updates because their prefixes are shared. Each constant endpoint must exactly replay its earlier fitted parameters, predictions and histories; these replays are controls, not additional independent seeds.

```bash
python -m loop_ultrasound.tail_lr --prepare-only --threads 2
python -m loop_ultrasound.tail_lr --threads 2
python -m loop_ultrasound.analyze_tail_lr \
  --protocol outputs/cpu-tail-lr256/protocol.json \
  --run-dirs outputs/cpu-tail-lr256/SJ-depth*-seed* \
  --reference-protocol outputs/cpu-case-expansion256-replication/protocol.json \
  --reference-run-dirs outputs/cpu-case-expansion256-replication/SJ-depth*-terminal-seed* \
  --output outputs/cpu-tail-lr256/tail_lr_analysis.json
```

Two disjoint training workers can use `--workers 2 --worker-index 0` and `1`, each with two CPU threads. Completed/claimed trajectories are never overwritten. The primary contrast is `H = G_low_tail - G_constant`, with `G = Brier_depth2 - Brier_depth4`. Report H together with absolute low-tail G, NLL, AUROC, Dice and prefix-to-endpoint changes. Changing AdamW learning rate changes both gradient and decoupled weight-decay step magnitudes. There is no learning-rate sweep, best-epoch selection or automatic next stage.

## CPU exploratory sharing-by-depth architecture comparison

The completed comparison reused twenty validated shared endpoints and added twenty fresh ordinary/untied depth2 and depth4 endpoints at the same terminal-only joint supervision, 256 training/159 tune Cases and five paired seeds. Both fixed tail learning-rate policies applied symmetrically to both architectures. See [the plan](docs/CPU_ARCHITECTURE256_PLAN.md), [locked protocol](configs/cpu_architecture256.json), [completed results and lesion-phase decision](docs/CPU_ARCHITECTURE256_RESULTS.md), and [aggregate records/figures](reports/architecture256/).

Each untied layer starts at the corresponding shared block values but owns independent parameter storage; readout/head initial values and Case-view sampling are paired. Ten fresh untied prefixes plus twenty tails require38,400 new updates. The full-state fork uses the same published training helper; core model/data/training sources are unchanged. Existing shared endpoints are bound references, not extra independent fits.

```bash
python -m loop_ultrasound.architecture_comparison --prepare-only --threads 2
python -m loop_ultrasound.architecture_comparison --threads 2
```

The completed analysis validates all forty models against raw Case records and the bound shared replay chain:

```bash
python -m loop_ultrasound.analyze_architecture \
  --protocol outputs/cpu-architecture256/protocol.json \
  --run-dirs outputs/cpu-architecture256/UJ-depth*-seed* \
  --shared-protocol outputs/cpu-tail-lr256/protocol.json \
  --shared-run-dirs outputs/cpu-tail-lr256/SJ-depth*-seed* \
  --shared-reference-protocol outputs/cpu-case-expansion256-replication/protocol.json \
  --shared-reference-run-dirs outputs/cpu-case-expansion256-replication/SJ-depth*-terminal-seed* \
  --output outputs/cpu-architecture256/architecture_analysis.json

python -m loop_ultrasound.plot_architecture \
  --report-dir reports/architecture256 \
  --protocol configs/cpu_architecture256.json \
  --output-dir outputs/architecture256-figures
```

Two disjoint workers can use `--workers 2 --worker-index 0` and `1`, each with two CPU threads. Report same-depth sharing benefits (`Brier_UJ - Brier_SJ`), each architecture's trained-depth benefit, and their interaction separately for both policies. Malignant and benign errors and cross-model prediction changes use the fixed0.5 threshold plus inclusive high-confidence cutoffs0.05/0.95. This is a frozen-feature refiner comparison at matched width/depth, not matched parameters, fully tuned architectures, internal inference correction or independent clinical validation. The completed review defers actual lesion intervention. With the lower tail LR, shared4 had a positive raw Brier advantage over untied4 at all five seeds (mean +0.00930; conditional Case interval [-0.00638, +0.02408]), while false negatives/positives were nearly unchanged. With constant LR, sharing strongly traded fewer benign false positives for more malignant false negatives. Depth gains in both architectures depended on the policy, and their interaction remains uncertain. These are exploratory recipe-dependent signals; no lesion intervention, new temperature fit, threshold search or calibration/test prediction was executed.

## Data and code attribution

BUS-BRA is described in [the dataset publication](https://pubmed.ncbi.nlm.nih.gov/37937827/) and distributed under CC-BY-4.0 on Zenodo. Download it from the official source; it is not bundled here. Follow the authors' citation requirements when reporting experiments. The encoder's upstream license applies to its weights. This repository's original code is MIT; the authors' MATLAB training implementation was not copied. Do not upload `data/`, `outputs/`, weights or individual prediction files to this public repository.
