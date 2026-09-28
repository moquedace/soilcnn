# SoilCNN

<p align="center">
  <strong>Convolutional Neural Networks for Digital Soil Mapping</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/R-%3E%3D4.5-276DC3?style=flat-square&logo=r&logoColor=white"/>
  <img src="https://img.shields.io/badge/torch-deep%20learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/domain-digital%20soil%20mapping-4CAF50?style=flat-square"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/status-in%20development-orange?style=flat-square"/>
</p>

<p align="center">
  A <strong>caret-style hyperparameter tuning framework</strong> for dual-branch CNNs applied to
  <strong>digital soil mapping</strong> with raster predictors — in R, as the
  package <code>soilcnn</code>.
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
           01_prepare_dataset.R  →  dsm_prepare()
           Extract · QC · predictor types · patch store, stored RAW,
           one file per window, with the recipe inside.  Decides no roles.
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

For the global map, [`05_dsm_predict_global.R`](examples/soc_stock_0_5cm/05_dsm_predict_global.R)
does all of this -- and 07's DI and AOA -- in one call to `dsm_predict()`.

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
| [`R/utils.R`](R/utils.R) | Safe I/O helpers, torch device setup, `env_*()` overrides, `latest_run_dir()` — the newest *finished* run, by time |
| [`R/checks.R`](R/checks.R) | `check_ledger()` · `ledger_check()` · `ledger_verdict()` — a ledger whose verdict refuses to pass while a promised check is missing |
| [`R/metrics.R`](R/metrics.R) | `ccc()` · R² · MAE · NSE · RMSE · MQI · **signed bias**, per split and per quantile group |
| [`R/cnn_architecture.R`](R/cnn_architecture.R) | Conv blocks, residual connections, SE attention, gate types, full model |
| [`R/final.R`](R/final.R) | `dsm_final()`: the selected config refitted under N seeds, side by side with fixed threads per seed; the ensemble, the conformal interval, the smearing factor; and `final_report.md`, which declares every hyperparameter of the chosen CNN and whether the search chose it |
| [`R/predict.R`](R/predict.R) | `dsm_predict()`: the map, for a grid as large as the world at 250 m -- row bands read once through a buffer that keeps its halo, the network fully convolutional where that is exact (`fcn_supported()`), workers side by side and resumable; the ensemble bands, the smeared mean, the constant and the level-and-DI conformal intervals and the AOA for every calibration source given (block, kNNDM), one VRT per band; and, first, a probe that must reproduce the final model's stored predictions at the profiles |
| [`R/tune_grid.R`](R/tune_grid.R) | `make_tune_grid()` · `make_manual_tune_grid()` with documented parameter ranges |
| [`R/patches.R`](R/patches.R) | One patch-indexing path, shared by extraction and prediction |
| [`R/preprocess.R`](R/preprocess.R) | QC (fold-independent) split from scaling (fold-dependent) |
| [`R/dataset.R`](R/dataset.R) | The patch store: one file per window, the split as an index |
| [`R/prepare.R`](R/prepare.R) | `dsm_prepare()` — a point table and a folder of aligned rasters become a patch store, with the target transform, the predictor types and the QC rules written into it as a recipe; `dsm_load(store)` then needs nothing else |
| [`R/resample.R`](R/resample.R) | Fold plans · distance buffering · `summarise_resamples()` · `seed_noise_floor()` · `one_se()` |
| [`R/diagnostics.R`](R/diagnostics.R) | Checks about THIS RUN on real data: patch centres, overlap between splits, run snapshots |
| [`R/train_cnn.R`](R/train_cnn.R) | `train_one_cnn()` · `run_cnn_tuning()` · `run_cnn_resample()` |
| [`R/model_registry.R`](R/model_registry.R) | `model_spec()` · `register_model()` · `list_models()` |
| [`R/baselines.R`](R/baselines.R) | `rf` · `mlp` · `cnn`, registered |
| [`R/train_table.R`](R/train_table.R) | `run_table_resample()` — tabular models, same comparison table |
| [`R/caret_adapter.R`](R/caret_adapter.R) | `caret_spec()` — borrow ~230 models, never caret's resampling |
| [`R/aoa.R`](R/aoa.R) | Dissimilarity index · area of applicability |
| [`R/knndm.R`](R/knndm.R) | `knndm_folds()` · `prediction_sample()` — folds whose geometry matches what prediction faces, not what a block grid happens to give |
| [`R/conformal.R`](R/conformal.R) | `conformal_calibrate()` · `picp_report()` — intervals with a coverage guarantee, and the check that they keep it |
| [`R/occlusion.R`](R/occlusion.R) | `spatial_occlusion()` — does the trained network use the neighbourhood, or only the centre pixel? |
| [`R/smearing.R`](R/smearing.R) | `smearing_factor()` · `smear()` — the back-transform of a log-trained median, and the one surface that may be summed |
| [`R/test_optimism.R`](R/test_optimism.R) | `freeze_selection()` · `score_test_grid()` — the test set, scored only after the choice is locked |
| [`R/api.R`](R/api.R) | **The front end**: `dsm_load()` · `spatial_cv()` · `dsm_train()` |
| [`R/zzz.R`](R/zzz.R) | `.onLoad()`: registers the built-in models, and fingerprints the code it loaded — every worker compares its own with it before it starts |

Beside `R/`:

| Where | What |
|------|---------|
| [`DESCRIPTION`](DESCRIPTION) · [`NAMESPACE`](NAMESPACE) | The package, `soilcnn`: what it imports, and the 64 functions it exports — the `dsm_*()` front end, the resampling specs and fold constructors, the model registry, and the tools applied to results (AOA, conformal intervals, smearing, metrics, noise floor, occlusion). The runners underneath `dsm_train()`, the patch store's plumbing and the scripts' helpers are internal (`soilcnn:::`); the scripts load the source tree, where every function is visible. NAMESPACE is what roxygen2 writes from the `@export` tags, and `tests/test_package_metadata.R` checks that it still is |
| [`tests/run_all.R`](tests/run_all.R) | 31 files: 25 fast, then 6 slow ones that train, map, and build and install the package. The package is loaded once for the suite. Every accumulator is named and `.report()` refuses an empty, unnamed, NA-bearing or non-logical one. `test_sources_parse.R` runs first and is the authority on syntax. |
| [`tools/`](tools/) | Three Python checks that need no R: `r_lint.py` (a top-level `else`, the native pipe — the two mistakes that have cost a round trip here; has a `--selftest`), `r_calls.py` (every project function a script calls exists, `do.call` targets included; named arguments match formals), `r_skeleton.py` (an edit touched only comments and strings). Run them after any edit made without an R session. |
| [`utils/install_load_pkg.R`](utils/install_load_pkg.R) | Installs what is missing, then **stops** if a package will not load |

---

## Quickstart

```r
pkgload::load_all(".")          # the package, from this source tree (or library(soilcnn))

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
  n_seeds     = 3              # a claim without repetitions has no error bar
)
# What it did not have to be told, and read instead:
#   the grid's windows    every window the store holds, alone and in pairs
#   the batch sizes       those that give the smallest fold >= 4 steps an epoch
#   the inverse           the store's (log1p -> expm1); one that disagrees is refused
#   the cores             n_cores = NULL is the physical cores minus one

fit$by_config                            # mean ± sd, one row per config
print_noise_floor(seed_noise_floor(fit$comparison))
print_one_se(one_se(fit$by_config))      # the simplest config within 1 SE
```

`dsm_load()` opens the store, reads the points and predictor types, aligns
them, reads the raster resolution — and **refuses** if the store was built
under a different predictor set, window set, target or resolution. A store
written by `dsm_prepare()` carries its own tables and recipe, so
`dsm_load(store)` needs nothing else.

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

### A median surface and a mean surface, and they are not the same map

```r
sm <- smearing_from_run(tuning_dir, config_id)   # Duan (1983), one scalar
mean_surface <- smear(log1p(median_surface), sm, lower_limit = 0)
```

Train on `log1p` with a SmoothL1 loss and the network estimates a conditional
**median** in log space. `expm1()` of a median is the median of the stock — not
its mean. On a right-skewed target the two are far apart, and the gap lands
entirely on anyone who sums the map:

| surface | bias on the frozen test set | may be summed for a total? |
|---|---|---|
| median (`expm1`) | **−24.4%** | **no** |
| mean (smeared) | +2.7% | yes |

Duan's smearing estimator corrects it with one scalar, calibrated on the same
out-of-fold **ensemble** residuals the conformal interval uses — the deployed
prediction is the ensemble median, so the calibrated residual has to be the
ensemble's and not one seed's.

**Nothing is replaced.** Stage 05 writes `soc_smeared_mean_ton_ha` *beside* the
median band and labels both, because they answer different questions: the median
is the typical stock at a pixel and minimises absolute error; the mean is the
only one you may add up. Silently swapping one for the other trades a known bias
for an unknown one.

`print.smearing_cal()` also reports S by quintile of the prediction, because
Duan's derivation assumes the residual is independent of the prediction and that
is checkable — in the worked example it runs 1.80 at the low end to 1.15 at the
high end, which the print warns about rather than silently averaging away.

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

All splits (train · validation · test) are evaluated with nine metrics, also broken down by **quantile group** of the observed values (Q0–Q25, …, Q99–Q100):

| Metric | Description |
|--------|-------------|
| **CCC** | Lin's Concordance Correlation Coefficient — accuracy + precision combined |
| **R²** | Coefficient of determination |
| **MAE** | Mean Absolute Error (native target units) |
| **NSE** | Nash-Sutcliffe Efficiency — 0 = mean-only model, 1 = perfect |
| **RMSE** | Root Mean Squared Error |
| **RPD** | Ratio of Performance to Deviation = sd(obs) / RMSE — standard pedometric benchmark (<1.4 poor, 1.4–2.0 fair, >2.0 good) |
| **MQI** | Model Quality Index = (CCC × NSE) / (MAE / mean(obs)) |
| **bias** | `mean(pred − obs)`, **signed**, in native units |
| **bias_pct** | the same relative to `mean(obs)`, so it compares across targets and depths |

The last two were added late, and the reason is worth stating: every other metric on this list is blind to the *sign* of the error. MAE and RMSE are unsigned by construction; R², NSE and RPD are unmoved by a constant offset in the right circumstances; CCC penalises bias but mixes it with scatter, so a low CCC never says which one it is. This framework's own final model was under-predicting its test set by **24.4%** — more than half its MAE — and nothing in the tables could see it.

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

The framework's own are `DESCRIPTION`'s Imports:

```r
install.packages(c(
  "torch", "coro",                          # deep learning
  "dplyr", "readr", "tibble", "purrr",      # data wrangling
  "terra", "matrixStats", "janitor",        # rasters, ensemble aggregation, names
  "callr", "ps",                            # worker processes, and their memory
  "pkgload"                                 # to load the package from its source tree
))
torch::install_torch()                      # ONCE: the C++ backend, ~200 MB
```

`install.packages("torch")` puts the R package in place; the backend is a
separate download that `library(torch)` asks for the first time. Until
`install_torch()` has run, nothing here trains.

Its Suggests, each needed only by the part that says so when called:
`sf` (points from a spatial file; kNNDM), `CAST` (kNNDM folds), `FNN` (a
faster exact nearest neighbour for the AOA), `randomForest` or `ranger` (the
RF baseline; ranger is far faster), `caret` (its model library).

The worked example (`examples/`) adds:

```r
install.packages(c(
  "sf", "ggplot2", "stringr", "tidyr",      # 01 and 06
  "DescTools", "processx",                  # 04b, 05a
  "randomForest", "ranger", "caret"         # baselines (03b); ranger optional
))
```

Every example script begins with `install_load_pkg(...)`, which installs what
is missing and then **stops** if a package will not load — it used to say
"completed" either way.

### Loading it

The framework is an R package, `soilcnn`:

```r
pkgload::load_all("<project root>")         # the source tree, as it stands -- what the scripts do
library(soilcnn)                            # an installed copy
```

To install a copy, build the tarball first: `R CMD INSTALL` of the directory
itself copies `data/` into the library, subdirectories and all.

```r
tgz <- pkgbuild::build("<project root>", dest_path = tempdir(), vignettes = FALSE)
install.packages(tgz, repos = NULL, type = "source")
```

`dsm_final()` and `dsm_predict()` start worker processes, and each loads the
framework the way its session did — the source tree, or that same installed
copy — and stops before its first unit if the code there is not the code its
session loaded.

A session that loaded the framework the old way, by `source()`ing `R/`, still
holds those copies in its global environment — RStudio's *Restart R* keeps
it — and they would answer every call in place of the package. The package
refuses to attach while one is there, and says how to remove them:
`rm(list = soilcnn:::.pkg_stale_copies(), envir = globalenv())`.

A CUDA-capable GPU is strongly recommended. CPU training is supported but ~10–20× slower; `setup_torch_device()` uses every physical core but one unless told otherwise.

---

## Applied example

The [`examples/soc_stock_0_5cm/`](examples/soc_stock_0_5cm/) directory contains a complete end-to-end run predicting **soil organic carbon stock (0–5 cm, ton/ha)** from 181 global raster predictors and WOSIS profiles: 4,154 rows extracted, 3,766 surviving QC, 3,728 reaching the patch store.

| Script | What it does |
|--------|-------------|
| [`01_prepare_dataset.R`](examples/soc_stock_0_5cm/01_prepare_dataset.R) | The SOC settings, and one call to `dsm_prepare()`: GPKG + rasters → QC · predictor types · a patch store at 3×3, 9×9, 15×15, stored RAW, with its recipe. Decides no roles, and owns no scaling |
| [`02_extract_patches.R`](examples/soc_stock_0_5cm/02_extract_patches.R) | Folded into 01 on 2026-09-26; now only says so |
| [`03_run_tuning.R`](examples/soc_stock_0_5cm/03_run_tuning.R) | Choose a fold plan · generate the grid · train every (config, fold, seed) · report mean ± sd against the seed noise floor |
| [`04_final_model.R`](examples/soc_stock_0_5cm/04_final_model.R) | Refit the selected config(s) on everything but the test set, by the tuning plan's own criterion · writes the scaling next to the weights |
| [`05_predict_spatial.R`](examples/soc_stock_0_5cm/05_predict_spatial.R) | Worker: predicts **one** row × col tile · seed ensemble · median + uncertainty layers |
| [`05a_test.R`](examples/soc_stock_0_5cm/05a_test.R) | Cheap dry-run on a handful of tiles — sanity-check geometry/RAM/throughput before committing to the full job |
| [`05a_run_parallel.R`](examples/soc_stock_0_5cm/05a_run_parallel.R) | Orchestrator: splits the raster into a tile grid, runs many `05` workers concurrently, resumes on restart |
| [`05b_merge_spatial_parts.R`](examples/soc_stock_0_5cm/05b_merge_spatial_parts.R) | Mosaics all finished tiles into the final wall-to-wall rasters |
| [`05c_estimate_eta.R`](examples/soc_stock_0_5cm/05c_estimate_eta.R) | Re-runnable at any time while `05a_run_parallel.R` is in flight — reports progress and ETA |
| [`05_dsm_predict_global.R`](examples/soc_stock_0_5cm/05_dsm_predict_global.R) | The global 250 m map in one `dsm_predict()` call: every band for both calibration sources (block, kNNDM), 249 units of 256 rows, resumable, the probe first; ~33 h at 2 workers x 7 threads (T3). For the global map it replaces 05, 05a, 05b, 05c and 07's rasters |
| [`06_avaliacao_grafica.R`](examples/soc_stock_0_5cm/06_avaliacao_grafica.R) | Graphical evaluation of the final model · it was this script, computing the bias itself, that first exposed the −24.4% back-transform defect |
| [`07_area_of_applicability.R`](examples/soc_stock_0_5cm/07_area_of_applicability.R) | Dissimilarity index and AOA mask over the prediction grid |
| [`99_check_pipeline.R`](examples/soc_stock_0_5cm/99_check_pipeline.R) | Numeric consistency across every artefact the pipeline wrote, against a saved snapshot |
| [`99b_check_pipeline_visual.R`](examples/soc_stock_0_5cm/99b_check_pipeline_visual.R) | The same, but showing the actual thing on screen: where the profiles are, whether tuning improved anything, what the patches look like |
| [`_capability_sweep.R`](examples/soc_stock_0_5cm/_capability_sweep.R) | Exercises the framework paths a second user would reach for first — reports *ran*, *asserted* and *measured* separately, because a path that ran without being asserted is the interesting row |
| `_b1` … `_b6`, `_c1` | The tier B and C capability checks of [`docs/test_plan.md`](docs/test_plan.md): kNNDM on the real points, two configs in stage 04, augmentation on/off, sharded prediction and merge, resume after an interruption, the same grid under two validation designs |

### Running it

Nothing needs editing first. Every script finds the project for itself — it
asks `Rscript --file`, then the `source()` frame, then the working directory,
and climbs to the directory holding `R/cnn_architecture.R` — and loads the
package from there with `pkgload::load_all()`, so a clone anywhere runs as it
is, with no network access needed to start. The one thing a new user
must set is `predictor_raster_dir` in `01_prepare_dataset.R`, which is
where *their* rasters are.

Every script is run with `source("<full path>")` from an R console, in this
order, with `tests/run_all.R` before anything expensive:

```
01 → 99 → 03 → 03b → 99 → 04 → 05 (or 05a_test → 05a → 05b) → 07 → 06 → 99 / 99b
```

Each script clears the workspace, so a parameter cannot be passed as a
variable; it is passed as an **environment variable**, read through
`env_chr()` / `env_int()` / `env_csv()`, which refuse a value that does not
parse instead of turning it into `NA`:

| Variable | Read by | Meaning |
|---|---|---|
| `soc_tuning_design` | 03 | `spatial` (block folds, default) or `knndm` — needs `_b1`'s `predpoints.csv` |
| `soc_tune_length`, `soc_tuning_n_seeds`, `soc_tuning_run_id` | 03 | grid size, seeds per unit, run directory name |
| `soc_final_tuning_run_id`, `soc_final_config_ids`, `soc_final_seeds` | 04 | which tuning run to refit from, which config(s), which seeds |
| `soc_row_shard_id`, `soc_col_shard_id`, `soc_n_row_shards`, `soc_n_col_shards`, `soc_max_concurrent` | 05, 05a, 05c | the tile grid and the concurrency |
| `soc_predict_raster_dir` | 05, 05a, 05a_test, 07, `_b1` | predict over another raster directory (the 20 km wiring grid) |
| `soc_b6_phase` | `_b6` | `prepare` or `verify` |

`Sys.setenv(soc_tune_length = "8")` before the `source()`; `Sys.unsetenv()`
after, or the next run inherits it. Every override announces itself with
"(from the environment)" when it is read.

---

## License

MIT © [moquedace](https://github.com/moquedace)
