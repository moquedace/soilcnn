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

---

## Quickstart

```r
for (m in c("utils", "patches", "preprocess", "dataset", "metrics",
            "diagnostics", "cnn_architecture", "tune_grid", "resample",
            "train_cnn")) source(file.path("R", paste0(m, ".R")))

device <- setup_torch_device(n_threads = 8, use_cuda = FALSE)

# The patch store is raw: one file per window, and only the windows the grid
# asks for are read.
store  <- load_patch_store("outputs/patches/.../", windows = c(3L, 9L, 15L))
points <- align_points_to_meta(readr::read_csv2("…/dataset_raw.csv"), store$meta)

# WHO TRAINS AND WHO SCORES -- one line, and one criterion for both the test
# set and the folds.
plan <- spatial_folds(store$meta, k = 5, test_frac = 0.15,
                      block_size = 2,              # in the units of x/y
                      buffer     = 15 * cell_size) # window x resolution
print(plan)

# Random grid -- like caret's tuneLength
grid <- make_tune_grid(tune_length = 30, seed = 42,
                       fixed = list(loss_fn = "smooth_l1"))

results <- run_cnn_resample(
  tune_grid  = grid,
  store      = store,
  points     = points,
  type_table = type_table,
  plan       = plan,
  n_seeds    = 3,              # a claim without repetitions has no error bar
  transform  = expm1,          # inverse of the log1p applied to the target
  output_dir = "outputs/tuning",
  device     = device,
  n_epochs   = 500L,
  patience   = 60L
)

results$by_config                       # mean +/- sd, one row per config
print_noise_floor(seed_noise_floor(results$comparison))
print_one_se(one_se(results$by_config)) # optional: simplest within 1 SE
```

The fold plan is the only line that changes to ask a different question. The
loop order is fold outside, configs and seeds inside, because the per-fold
cache — scaling fitted on that fold's training rows and broadcast over every
patch — is the expensive object, while training one config is minutes.

See the full worked example in [`examples/soc_stock_0_5cm/`](examples/soc_stock_0_5cm/).

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

Model selection across configs ranks by **validation CCC** (descending), then **validation MAE** (ascending) as a tiebreaker. Test metrics are computed and written to the comparison CSV for every config during tuning, but strictly as diagnostic reference — they are never read to choose between configurations. Only after the winning architecture is locked in does a human actually look at test performance. This avoids the common mistake of tuning toward test performance.

Early stopping uses **validation SmoothL1 loss** — keeping the stopping criterion consistent with the training objective.

### Multi-seed ensemble

After architecture selection, the top config(s) are re-trained with N independent seeds (different weight initialisation + batch shuffling). Sources of run-to-run variance:

| Source | Effect |
|--------|--------|
| Weight initialisation | Different local minima after convergence |
| Batch shuffle order | Different gradient path through the loss landscape |
| Dropout masks | Different regularisation per forward pass |

Spatial prediction aggregates all seed models per pixel. The **median** is the recommended headline map: it is invariant to the monotone `expm1` back-transform (`median(expm1(z)) = expm1(median(z))`), robust to divergent seeds, and consistent with what SmoothL1 learns (a conditional median). SD and MAD are written as epistemic uncertainty layers.

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
