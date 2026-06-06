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
           Extract + QC + scale + stratified split
                     │
                     ▼
           02_extract_patches.R
           Spatial patch arrays  N × C × H × W
                     │
                     ▼
           03_run_tuning.R
           make_tune_grid(tune_length = 30)  ←──  like caret's tuneLength
                     │
            ┌────────┴────────┐
            │  cfg_001 ... cfg_N  │   train → validate → evaluate
            └────────┬────────┘
                     ▼
           comparison_ranked.csv
           Best architecture selected by CCC + MAE
```

---

## The dual-branch idea

Each soil profile is represented by **two spatial patches** extracted from a stack of raster layers:

| Branch | Window | Processes captured |
|--------|--------|--------------------|
| Small  | 3 × 3 cells | Local topography, land cover, proximity effects |
| Large  | 5 × 5 or 7 × 7 cells | Landscape climate, parent material, watershed position |

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
</details>

---

## Framework files

| File | Purpose |
|------|---------|
| [`R/utils.R`](R/utils.R) | Safe I/O helpers, torch device setup |
| [`R/metrics.R`](R/metrics.R) | CCC · R² · MAE · NSE · RMSE · MQI per split and per quantile group |
| [`R/cnn_architecture.R`](R/cnn_architecture.R) | Conv blocks, residual connections, SE attention, gate types, full model |
| [`R/tune_grid.R`](R/tune_grid.R) | `make_tune_grid()` · `make_manual_tune_grid()` with documented parameter ranges |
| [`R/train_cnn.R`](R/train_cnn.R) | `train_one_cnn()` · `run_cnn_tuning()` — single config and full grid loop |

---

## Quickstart

```r
source("R/utils.R"); source("R/metrics.R")
source("R/cnn_architecture.R")
source("R/tune_grid.R"); source("R/train_cnn.R")

device <- setup_torch_device(n_threads = 8, use_cuda = TRUE)
patches <- readRDS("outputs/patches/.../patches_all_splits.rds")

# Random grid — like caret's tuneLength
grid <- make_tune_grid(tune_length = 30, seed = 42,
                       fixed = list(loss_fn = "smooth_l1"))

results <- run_cnn_tuning(
  tune_grid    = grid,
  n_channels   = 187,
  patches      = patches,
  points_valid = list(train      = patches$train$meta,
                      validation = patches$validation$meta,
                      test       = patches$test$meta),
  transform    = expm1,        # inverse of log1p applied to target
  output_dir   = "outputs/tuning",
  device       = device,
  n_epochs     = 500L,
  patience     = 60L
)
```

See the full worked example in [`examples/soc_stock_0_5cm/`](examples/soc_stock_0_5cm/).

---

## Tuneable parameters

The table below summarises the search space. See [`docs/tuning_guide.md`](docs/tuning_guide.md) for the rationale behind every range and its connection to digital soil mapping.

| Parameter | Options | Controls |
|-----------|---------|---------|
| `window_sizes` | `c(3)` · `c(5)` · `c(7)` · `c(3,5)` · `c(3,7)` · `c(5,7)` | Spatial scale(s) |
| `conv_channels` | `c(32,64)` to `c(128,256,256)` | Network depth & width |
| `use_residual` | `TRUE` · `FALSE` | Skip connections (ResNet-style) |
| `use_se_block` | `TRUE` · `FALSE` | Channel attention |
| `gate_type` | `vector_featurewise` · `scalar_per_sample` · `no_gate_concat` | Branch fusion strategy |
| `embedding_dim` | 128 · 256 · 384 · 512 | Representation size |
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
| **MQI** | Model Quality Index = (CCC × NSE) / (MAE / mean(obs)) |

Model selection across configs ranks by **test CCC** (descending), then **test MAE** (ascending).  
Early stopping uses **validation SmoothL1 loss** — keeping the stopping criterion consistent with the training objective.

---

## Dependencies

```r
install.packages(c(
  "torch", "coro",                          # deep learning
  "terra", "sf",                            # geospatial
  "dplyr", "tidyr", "readr", "tibble",      # data wrangling
  "purrr", "janitor", "ggplot2",            # utilities
  "DescTools"                               # CCC calculation
))
```

A CUDA-capable GPU is strongly recommended. CPU training is supported but ~10–20× slower.

---

## Applied example

The [`examples/soc_stock_0_5cm/`](examples/soc_stock_0_5cm/) directory contains a complete end-to-end run predicting **soil organic carbon stock (0–5 cm, ton/ha)** from 187 global raster predictors and ~37 000 WOSIS profiles.

| Script | What it does |
|--------|-------------|
| [`01_prepare_dataset.R`](examples/soc_stock_0_5cm/01_prepare_dataset.R) | Read GPKG + rasters · QC · scale · stratified split |
| [`02_extract_patches.R`](examples/soc_stock_0_5cm/02_extract_patches.R) | Extract 3×3, 5×5, 7×7 patch arrays for all splits |
| [`03_run_tuning.R`](examples/soc_stock_0_5cm/03_run_tuning.R) | Generate grid · train all configs · rank results |

---

## License

MIT © [moquedace](https://github.com/moquedace)
