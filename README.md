# r-cnn-soil-mapping

<p align="center">
  <img src="https://img.shields.io/badge/R-%3E%3D4.1-276DC3?style=flat-square&logo=r&logoColor=white"/>
  <img src="https://img.shields.io/badge/torch-deep%20learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/domain-digital%20soil%20mapping-4CAF50?style=flat-square"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/status-in%20development-orange?style=flat-square"/>
</p>

<p align="center">
  A <strong>caret-style hyperparameter tuning framework</strong> for dual-branch CNNs applied to
  <strong>digital soil mapping</strong> with raster predictors — in R.
</p>

---

## Overview

Traditional ML frameworks like `caret` make spatial prediction straightforward: provide a raster stack, a target variable, and a `tuneLength` — the framework handles the rest.

This project brings the same philosophy to **convolutional neural networks**, covering the full pipeline from raw rasters to a ranked comparison of CNN architectures.

```
Rasters (TIF stack)                 Soil profiles (GPKG)
        │                                    │
        └────────────┬───────────────────────┘
                     ▼
           01_prepare_dataset.R
           Extract · QC · predictor types.  Decides no roles.
                     │
                     ▼
           02_extract_patches.R
           Patch store  N × C × H × W, stored RAW, one file per window
                     │
                     ▼
           03_run_tuning.R
           plan <- spatial_folds(meta, k, test_frac, block_size, buffer)
           make_tune_grid(tune_length = 30)  ←──  like caret's tuneLength
                     │
            ┌────────┴──────────┐
            │  config × fold × seed │   the unit of work
            └────────┬──────────┘
                     ▼
           comparison_by_config.csv
           mean ± sd over repetitions, next to the seed noise floor
                     │
                     ▼
           03b_run_baselines.R
           rf(centre) · rf(centre+window means) · mlp(centre) · cnn
           SAME folds, SAME seeds.  The gap between the context RF and the
           CNN is what the convolution is worth.
                     │
                     ▼
           04_final_model.R
           Refit on everything but the test set, by the tuning plan's own
           criterion.  The scaling is written next to the weights.
                     │
                     ▼
           05_predict_spatial.R      ◄── single-tile worker, not run alone at scale
           Block-streaming · seed ensemble · median map + uncertainty layers
                     │
                     ▼
           05a_run_parallel.R
           Orchestrates many 05 workers over a row × col tile grid, with
           resume-on-restart. (05a_test.R: cheap dry-run on a few tiles first.)
                     │
                     ▼
           05b_merge_spatial_parts.R
           Mosaics all tiles into the final wall-to-wall rasters
                     │
                     ▼
           07_area_of_applicability.R
           Where the map should be believed at all
```

`05_predict_spatial.R` predicts a single rectangular tile — that's the whole
point of the 2D-tiled design (bounded RAM per process, see
[`docs/design_decisions.md`](docs/design_decisions.md)). For anything beyond
a one-off test tile, `05a_run_parallel.R` is what you actually run: it splits
the raster into a `n_row_shards × n_col_shards` grid and launches many `05`
workers concurrently, tracking progress so an interrupted job resumes instead
of restarting. `05c_estimate_eta.R` can be run at any time while a job is in
flight to check progress. See the [applied example](#applied-example) below
for the full script-by-script breakdown.

---

## What the framework is careful about

Three things cost this project real time before they were made structural.

**The split is not stored with the data.** Stage 01 writes points, coordinates,
the target and the predictor types — and nothing about who trains. A fold plan
decides that in stage 03, from coordinates, in seconds:

```r
plan <- holdout(meta, validation_frac = 0.15, test_frac = 0.15)
plan <- random_folds(meta, k = 5, test_frac = 0.15)
plan <- spatial_folds(meta, k = 5, test_frac = 0.15, block_size = 2, buffer = 0.034)
plan <- region_folds(meta, group = meta$biome, test_frac = 0.15)
```

One criterion carves the test set **and** the folds, so the two numbers a run
reports answer the same question. Changing the strategy costs seconds; before,
it cost a five-hour re-extraction.

**A repetition is a seed, and the noise is measured.** The unit of work is
`(config, fold, seed)`, and the seed is shared across configs on purpose, so
two configs within a repetition start from the same draw. What the spread
across repetitions then measures is luck — reported as a noise floor, because a
gap between configs smaller than it is not evidence. `one_se()` is available
for when it is not: among configs within one standard error of the best, take
the simplest.

**A row is not always an independent observation.** In 3-D soil mapping one
profile yields several rows — 0–5, 5–15, 15–30 cm — at identical coordinates,
from the same pit. Split those across training and validation and the model is
scored on a depth of a profile it already learned. `holdout()` and
`random_folds()` keep a profile together by default (`group = "auto"`), and
`check_fold_plan()` **proves** no group was split rather than trusting the
constructor that was meant to prevent it.
[Wang et al. 2025, *Geoderma*](https://www.sciencedirect.com/science/article/pii/S0016706125000618)

**A map needs to say where it should be believed.** A prediction exists at
every pixel, including pixels whose predictor combination the model never saw,
and the cross-validated CCC does not describe those. `R/aoa.R` computes a
dissimilarity index and the area of applicability — the DI expressed in units
of the training set's own mean pairwise distance, and the threshold derived
from distances **across** folds, which is what makes it mean "as dissimilar as
something cross-validation coped with".
[Meyer & Pebesma 2021, *MEE*](https://arxiv.org/pdf/2005.07939)

**Identical inputs are a defect; nearby ones are not.** Two points in the same
raster cell give the network the same patch, bit for bit, and one can be scored
on what the other trained on — under any plan. Two *neighbouring* points
sharing some surrounding pixels is not a defect: under a random split it is the
condition being measured. The leakage report keeps the two apart.

---

## The dual-branch idea

Each soil profile is represented by **two spatial patches** extracted from a stack of raster layers:

| Branch | Window | Processes captured |
|--------|--------|--------------------|
| Small  | 3 × 3 cells | Local topography, land cover, proximity effects |
| Large  | 9 × 9 or 15 × 15 cells | Landscape position, parent material, local climate |

A window's physical extent is `window_size × raster resolution`, so pixel sizes are chosen per resolution. At the example's 250 m they span ~0.75 km (3 × 3) to ~3.75 km (15 × 15).

Patches are stored **raw** and scaled when a fold's tensors are built — z-score for continuous predictors, /100 for proportions, identity for dummies — from the training rows **of that fold**. This equalises gradient flow across channels of very different magnitude (elevation in thousands against vegetation indices in 0–1), and it is what lets one 16 GB patch store serve any number of folds: the alternative is one re-extraction per fold.

The scaling therefore belongs to the **fitted model**, not to the dataset. Stage 04 writes it next to the weights, and prediction reads it from there.

A learned **gate** fuses the two embeddings per sample — letting each location draw from whichever spatial scale is more informative for the target variable.

<details>
<summary>Architecture diagram (click to expand)</summary>

```
Input patches
│
├── Branch 1 (small)          ├── Branch 2 (large)
│   N × C × w₁ × w₁          │   N × C × w₂ × w₂
│   Conv blocks + SE          │   Conv blocks + SE
│   → embedding (N × E)       │   → embedding (N × E)
│                             │
│           f₁                │           f₂
│            └────────┬───────┘
│                     │
│           Gate network
│           input: [f₁, f₂, |f₁−f₂|, f₁⊙f₂]
│           output: gate ∈ (0,1)ᴱ
│
│           fused = gate·f₁ + (1−gate)·f₂
│
│           Head: [fused, |f₁−f₂|] → FC → prediction
```

See [`docs/architecture.md`](docs/architecture.md) for full detail.
For the reasoning behind every architectural and training choice see [`docs/design_decisions.md`](docs/design_decisions.md).
</details>

---

## Framework files

| File | Purpose |
|------|---------|
| [`R/utils.R`](R/utils.R) | Safe I/O helpers, torch device setup |
| [`R/metrics.R`](R/metrics.R) | `ccc()` · R² · MAE · NSE · RMSE · MQI, per split and per quantile group |
| [`R/cnn_architecture.R`](R/cnn_architecture.R) | Conv blocks, residual connections, SE attention, gate types, full model |
| [`R/tune_grid.R`](R/tune_grid.R) | `make_tune_grid()` · `make_manual_tune_grid()` with documented parameter ranges |
| [`R/patches.R`](R/patches.R) | One patch-indexing path, shared by extraction and prediction |
| [`R/preprocess.R`](R/preprocess.R) | QC (fold-independent) split from scaling (fold-dependent) |
| [`R/dataset.R`](R/dataset.R) | The patch store: one file per window, the split as an index |
| [`R/resample.R`](R/resample.R) | Fold plans · distance buffering · `summarise_resamples()` · `seed_noise_floor()` · `one_se()` |
| [`R/diagnostics.R`](R/diagnostics.R) | Checks about THIS RUN on real data: patch centres, overlap between splits, run snapshots |
| [`R/train_cnn.R`](R/train_cnn.R) | `train_one_cnn()` · `run_cnn_tuning()` · `run_cnn_resample()` |
| [`R/model_registry.R`](R/model_registry.R) | `model_spec()` · `register_model()` · `list_models()` |
| [`R/baselines.R`](R/baselines.R) | `rf` · `mlp` · `cnn`, registered |
| [`R/train_table.R`](R/train_table.R) | `run_table_resample()` — tabular models, same comparison table |
| [`R/caret_adapter.R`](R/caret_adapter.R) | `caret_spec()` — borrow ~230 models, never caret's resampling |
| [`R/aoa.R`](R/aoa.R) | Dissimilarity index · area of applicability |
| [`R/conformal.R`](R/conformal.R) | `conformal_calibrate()` · `picp_report()` — intervals with a coverage guarantee, and the check that they keep it |
| [`R/occlusion.R`](R/occlusion.R) | `spatial_occlusion()` — does the trained network use the neighbourhood, or only the centre pixel? |
| [`R/test_optimism.R`](R/test_optimism.R) | `freeze_selection()` · `score_test_grid()` — the test set, scored only after the choice is locked |
| [`R/api.R`](R/api.R) | **The front end**: `dsm_load()` · `spatial_cv()` · `dsm_train()` |
| [`R/load_all.R`](R/load_all.R) | One `source()` for every module, in dependency order |

---

## Quickstart

```r
source("R/load_all.R")          # the whole framework, in dependency order

data <- dsm_load(
  patch_dir    = "outputs/patches/.../",
  points       = "data/processed/.../full_modeling_dataset_raw.csv",
  type_table   = "outputs/metadata/.../predictor_type_table.csv",
  raster_table = "outputs/metadata/.../raster_table_used.csv",
  windows      = c(3L, 9L, 15L)
)

fit <- dsm_train(
  data,
  model       = "cnn",
  resampling  = spatial_cv(k = 5, block_size = "auto", buffer = "auto"),
  tune_length = 30,            # a budget, like caret's
  n_seeds     = 3,             # a claim without repetitions has no error bar
  transform   = expm1          # the target was trained on log1p
)

fit$by_config                            # mean ± sd, one row per config
print_noise_floor(seed_noise_floor(fit$comparison))
print_one_se(one_se(fit$by_config))      # the simplest config within 1 SE
```

`dsm_load()` opens the store, reads the points and predictor types, aligns
them, reads the raster resolution — and **refuses** if the store was built
under a different predictor set, window set, target or resolution.

### One line decides who trains and who scores

```r
spatial_cv(k = 5)                        # blocks of ground, buffered
random_cv(k = 10)                        # ignores geography, on purpose
holdout_cv(validation_frac = 0.2)        # a single split
region_cv(group = points$biome)          # leave-one-region-out
```

Swap it and nothing else changes. One criterion carves the test set **and** the
folds, so the two numbers a run reports answer the same question.

### `"auto"` means measured, never guessed

| argument | what it resolves to |
|---|---|
| `block_size = "auto"` | the **largest** block whose worst case still fits the balance constraint, measured on *these* points |
| `buffer = "auto"` | `max(window) × cell_size` — the exact distance at which two patches stop sharing a pixel, under the Chebyshev metric |

Both print what they chose, and `"auto"` without a raster resolution **refuses**
rather than inventing one.

This is not a style preference. A block size measured on the full point set and
carried into a 10% subsample once left a single block holding a third of the
data — because block-subsampling keeps *whole* blocks, so a smaller draw has
fewer blocks of the **same** width.

### Any registered model, same folds, same tables

```r
list_models()

rf  <- dsm_train(data, model = "rf", resampling = plan,
                 features = c("centre", "window_mean"))
mlp <- dsm_train(data, model = "mlp", resampling = plan, features = "centre")

register_model(caret_spec("xgbTree"))    # ~230 methods, borrowed from caret
xgb <- dsm_train(data, model = "xgbTree", resampling = plan)
```

Every family produces the same comparison table, so `summarise_resamples()`,
`seed_noise_floor()` and `one_se()` work across all of them. **The gap between
the context RF and the CNN is what the convolution is worth** — and if it is
smaller than the noise floor, the convolution is doing averaging.

caret is borrowed for its model *library*, never for its resampling: the fold
plan stays here, with its blocks and its buffer. `caret_spec()` calls
`train(method = "none")` with a one-row grid, so two objects never both believe
they own the split.

See the full worked example in [`examples/soc_stock_0_5cm/`](examples/soc_stock_0_5cm/)
and the tour in [`examples/quickstart.R`](examples/quickstart.R).

---

## Numbers you can defend

Three things a soil-mapping paper is normally asked for and normally cannot
give: an honest test score, evidence that the architecture earns its cost, and
an uncertainty map that covers what it claims.

### The test set stays frozen, and the optimism is measured

Tuning never scores the test set. Once the choice is locked — recorded on disk,
with a timestamp and a commit — the whole grid *can* be scored on it, and that
measures something worth publishing:

```r
freeze_selection(run_dir, "cfg_014", rule = "one_se")   # stage 04 does this
score_test_grid(run_dir, data, device = device)         # afterwards, any time
```

```
chosen by validation : cfg_014   test CCC 0.4612   (rank 6 of 24 on test)
best on test         : cfg_003   test CCC 0.4980
SELECTION OPTIMISM   : +0.0368 CCC
```

That gap is how much a test-selected number would have been overstated. It is
not a reason to switch config — switching is what the measurement is measuring.
`score_test_grid()` refuses to run before the selection is frozen, and
`freeze_selection()` refuses to be overwritten with a different config, because
"score, dislike, re-freeze, re-score" is the loop the ordering exists to
prevent. Nothing is retrained: the checkpoints are already on disk.

### Does the convolution earn its cost?

Two independent routes to the same question, which is the point — if they
disagree, one of the measurements is wrong and that is worth knowing.

*From the outside*, stage 03b races the CNN against a forest fed the same
neighbourhood with the arrangement thrown away. *From the inside*,
`spatial_occlusion()` hides part of the patch of a trained network and
re-predicts:

```r
occlusion_report(run_dir, data, config_id = "cfg_014", device = device)
```

```
scope                n_pixels_hidden    ccc   delta_ccc
baseline                           0  0.489
context_all                      224  0.487      -0.002
centre_only_hidden                 1  0.331      -0.158
ring_01                            8  0.488      -0.001
...
-> THE CENTRE PIXEL CARRIES MORE THAN THE WHOLE NEIGHBOURHOOD.
```

The hidden region is **permuted from another sample**, not zeroed. After scaling
zero is the training mean, and a patch whose rim is the mean everywhere is a
landscape that does not exist — the drop would then mix "this region mattered"
with "this input is impossible", and the second grows with the area hidden,
which is exactly the comparison being made.

### Calibrated uncertainty, and a check that it is calibrated

```r
cal <- conformal_calibrate(val$obs, val$pred, alpha = 0.1)   # 90%
iv  <- conformal_interval(cal, test$pred, lower_limit = 0)
picp_report(test$obs, iv$lower, iv$upper, group = test$block, alpha = 0.1)
```

Split conformal gives `P(y ∈ interval) ≥ 1 − α` with no distributional
assumption, from one pass over held-out residuals. Stage 04 calibrates on the
validation rows and checks coverage on the **test** rows — a coverage measured
on the points that calibrated it comes out right by arithmetic, not by evidence
— and stage 05 reads that calibration to write `soc_pi90_lower` / `_upper`
bands. It never recomputes: two code paths producing "the interval" is how a map
ends up claiming a coverage nobody measured.

**PICP** turns uncertainty from an adjective into a number that can be wrong.
Promise 90%, deliver 61%, and you can see it. And because the conformal
guarantee is *marginal*, not conditional, coverage is broken down by group and
the worst is printed first: 90% overall is compatible with 99% over the easy
half and 60% over the hard half, and the hard half is where anyone needs an
interval at all. When a group falls far below, that is where exchangeability —
the theorem's only assumption — is breaking, which is the same place the area of
applicability is pointing at.

---

## Tuneable parameters

The table below summarises the search space. See [`docs/tuning_guide.md`](docs/tuning_guide.md) for the rationale behind every range and its connection to digital soil mapping.

| Parameter | Options | Controls |
|-----------|---------|---------|
| `window_sizes` | `c(3)` · `c(9)` · `c(15)` · `c(3,9)` · `c(3,15)` · `c(9,15)` | Spatial scale(s) |
| `conv_channels` | `c(32,64)` to `c(128,256,256)` | Network depth & width |
| `use_residual` | `TRUE` · `FALSE` | Skip connections (ResNet-style) |
| `use_se_block` | `TRUE` · `FALSE` | Channel attention |
| `gate_type` | `vector_featurewise` · `scalar_per_sample` · `no_gate_concat` | Branch fusion strategy |
| `embedding_dim` | 128 · 256 · 384 · 512 | Representation size |
| `embed_pool` | `flatten` · `gap` | Pre-embedding reduction: keep every cell (params grow with window²) vs. global average pool (window-independent, ~25× lighter for 15×15) |
| `dropout` | 0.0 · 0.1 · 0.2 · 0.3 | Overall regularisation (maps to 5 internal sites) |
| `base_lr` | 1e-4 → 3e-3 | Peak learning rate (Adam + warmup) |
| `loss_fn` | `smooth_l1` · `mse` · `mae` | Training objective |
| `weight_decay` | 0 → 1e-3 | L2 regularisation |
| `batch_size` | 128 · 256 · 512 | Mini-batch size |

---

## Evaluation metrics

All splits (train · validation · test) are evaluated with six metrics, also broken down by **quantile group** of the observed values (Q0–Q25, …, Q99–Q100):

| Metric | Description |
|--------|-------------|
| **CCC** | Lin's Concordance Correlation Coefficient — accuracy + precision combined |
| **R²** | Coefficient of determination |
| **MAE** | Mean Absolute Error (native target units) |
| **NSE** | Nash-Sutcliffe Efficiency — 0 = mean-only model, 1 = perfect |
| **RMSE** | Root Mean Squared Error |
| **RPD** | Ratio of Performance to Deviation = sd(obs) / RMSE — standard pedometric benchmark (<1.4 poor, 1.4–2.0 fair, >2.0 good) |
| **MQI** | Model Quality Index = (CCC × NSE) / (MAE / mean(obs)) |

Model selection across configs ranks by **validation CCC** (descending), then **validation MAE** (ascending) as a tiebreaker — or by `one_se()`, which takes the simplest config within one standard error of the best and is the default in stage 04.

**The test set is not scored during tuning at all** (`evaluate_test = FALSE`). The columns exist and hold `NA`, so the table has one shape either way. An earlier version computed them "as diagnostic reference"; there is no such thing. A test score sitting beside the selection metric is selection on the test set performed by whoever reads the table, and with 24 configs × 9 repetitions the *best* of 216 noisy test scores is higher than any one of them by construction — before anyone chooses anything.

The test is scored once, in stage 04, on the config chosen without it.

Early stopping uses **validation SmoothL1 loss** — keeping the stopping criterion consistent with the training objective.

### Multi-seed ensemble

After architecture selection, the top config(s) are re-trained with N independent seeds (different weight initialisation + batch shuffling). Sources of run-to-run variance:

| Source | Effect |
|--------|--------|
| Weight initialisation | Different local minima after convergence |
| Batch shuffle order | Different gradient path through the loss landscape |
| Dropout masks | Different regularisation per forward pass |

Spatial prediction aggregates all seed models per pixel. The **median** is the recommended headline map: it is invariant to the monotone `expm1` back-transform (`median(expm1(z)) = expm1(median(z))`), robust to divergent seeds, and consistent with what SmoothL1 learns (a conditional median).

SD and MAD are written too — but **they are not a prediction interval**, and the framework says so rather than letting a reader assume otherwise. They measure how much the answer moves when the initialisation moves: a property of the optimiser, not of the soil. In this project's numbers the seed spread is 0.038 CCC while the MAE is ~17 t/ha on a median stock of 29.3. A map drawn from that spread would promise an order of magnitude more certainty than it has, and a map that understates is worse than no map, because somebody acts on it.

What the interval bands come from instead is [calibrated uncertainty](#calibrated-uncertainty-and-a-check-that-it-is-calibrated).

---

## Dependencies

```r
install.packages(c(
  "torch", "coro",                          # deep learning
  "terra", "sf",                            # geospatial
  "dplyr", "tidyr", "readr", "tibble",      # data wrangling
  "purrr", "janitor", "ggplot2", "stringr", # utilities
  "DescTools",                              # CCC calculation
  "matrixStats",                            # rowMedians / rowSds for ensemble aggregation
  "ps", "processx"                          # spatial prediction: RSS monitoring, worker orchestration (05a_run_parallel.R)
))
```

A CUDA-capable GPU is strongly recommended. CPU training is supported but ~10–20× slower.

---

## Applied example

The [`examples/soc_stock_0_5cm/`](examples/soc_stock_0_5cm/) directory contains a complete end-to-end run predicting **soil organic carbon stock (0–5 cm, ton/ha)** from 181 global raster predictors and ~37,000 WOSIS profiles.

| Script | What it does |
|--------|-------------|
| [`01_prepare_dataset.R`](examples/soc_stock_0_5cm/01_prepare_dataset.R) | Read GPKG + rasters · QC · predictor types. Decides no roles, and owns no scaling |
| [`02_extract_patches.R`](examples/soc_stock_0_5cm/02_extract_patches.R) | Extract 3×3, 9×9, 15×15 patches band-by-band, stored RAW, one file per window |
| [`03_run_tuning.R`](examples/soc_stock_0_5cm/03_run_tuning.R) | Choose a fold plan · generate the grid · train every (config, fold, seed) · report mean ± sd against the seed noise floor |
| [`04_final_model.R`](examples/soc_stock_0_5cm/04_final_model.R) | Refit the selected config(s) on everything but the test set, by the tuning plan's own criterion · writes the scaling next to the weights |
| [`05_predict_spatial.R`](examples/soc_stock_0_5cm/05_predict_spatial.R) | Worker: predicts **one** row × col tile · seed ensemble · median + uncertainty layers |
| [`05a_test.R`](examples/soc_stock_0_5cm/05a_test.R) | Cheap dry-run on a handful of tiles — sanity-check geometry/RAM/throughput before committing to the full job |
| [`05a_run_parallel.R`](examples/soc_stock_0_5cm/05a_run_parallel.R) | Orchestrator: splits the raster into a tile grid, runs many `05` workers concurrently, resumes on restart |
| [`05b_merge_spatial_parts.R`](examples/soc_stock_0_5cm/05b_merge_spatial_parts.R) | Mosaics all finished tiles into the final wall-to-wall rasters |
| [`05c_estimate_eta.R`](examples/soc_stock_0_5cm/05c_estimate_eta.R) | Re-runnable at any time while `05a_run_parallel.R` is in flight — reports progress and ETA |

---

## License

MIT © [moquedace](https://github.com/moquedace)
