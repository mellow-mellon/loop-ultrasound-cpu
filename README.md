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

Each run saves a local checkpoint, loss history and per-image predictions, plus an aggregate summary. The fixed-seed 16-Case tune subset is label-blind. A different `--seed` currently also changes subset selection; multi-seed scientific analysis must fix the cohort separately before comparing seeds. Passing `--eval-max-cases 0` disables tune evaluation. An online encoder/no-cache run is also supported; augmentation and feature caching cannot be combined.

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

The next experiment fixes a proportional training cohort (87 benign / 41 malignant Cases, 227 images) and all 159 tune Cases (281 images). Cohort selection seed 20261004 is independent of training seeds 17/29/43. Four arms each train 50 epochs at learning rate 0.0003, batch 2, without augmentation. All models report the final epoch; periodic development evaluation every five epochs is diagnostic and does not select a checkpoint.

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

Before a formal study:

1. Verify Case-to-patient mapping, merge any repeated patient groups, and remake partitions if needed. Exact hashes do not replace near-duplicate or patient identity review.
2. Review the nearest loop/classification literature and preregister the actual contribution. Reusing a loop for ultrasound is not by itself novel.
3. Fix cohorts across training seeds and implement the proposed full training schedule/model selection. `configs/research_protocol.json` is a **proposal**, not an executable or completed experiment; the pilot has no warmup/cosine schedule.
4. Add original-resolution contour metrics, clinical annotation QA, label-policy sensitivity analyses and audited external evaluation. Better contours and better calibrated malignancy predictions must be evaluated separately.
5. Calibrate thresholds on calibration Cases, then evaluate the sealed test once under a locked protocol. This repository currently demonstrates benchmark-label prediction, not clinical usefulness or a deployment decision aid.

## Data and code attribution

BUS-BRA is described in [the dataset publication](https://pubmed.ncbi.nlm.nih.gov/37937827/) and distributed under CC-BY-4.0 on Zenodo. Download it from the official source; it is not bundled here. Follow the authors' citation requirements when reporting experiments. The encoder's upstream license applies to its weights. This repository's original code is MIT; the authors' MATLAB training implementation was not copied. Do not upload `data/`, `outputs/`, weights or individual prediction files to this public repository.
