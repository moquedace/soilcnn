# r-cnn-soil-mapping

A **caret-style tuning framework for dual-branch CNNs** applied to digital soil mapping (DSM) using raster predictors.

## Why this exists

Traditional ML frameworks like `caret` and `tidymodels` make it easy to explore hyperparameter spaces — you specify a `tuneLength` or a `tuneGrid`, the framework samples configurations, trains each one, evaluates on a validation set, and returns a ranked comparison table.

Deep learning frameworks in R (`torch`, `keras`) give you full control over architecture and training, but no standardised tuning interface. This project fills that gap for the specific case of **spatial regression with raster patches**: predicting continuous soil attributes (carbon, clay, pH, …) at georeferenced profile locations using surrounding raster cells as input.

## Core idea

Each soil profile location is represented as a spatial **patch** extracted from a stack of raster layers (predictors). A **dual-branch CNN** processes two patches of different sizes simultaneously:

- **Small branch** (e.g., 3 × 3 cells): captures local-scale processes — micro-relief, land cover, proximity effects.
- **Large branch** (e.g., 5 × 5 or 7 × 7 cells): captures landscape-scale processes — climate gradients, parent material, watershed position.

A learned **gate** fuses the two embeddings, letting each sample draw from whichever scale is most informative for that location and target variable.

## Framework components

| File | Purpose |
|---|---|
| `R/utils.R` | Safe I/O, device setup, helper functions |
| `R/metrics.R` | CCC, R², MAE, NSE, RMSE, MQI — per split and per quantile group |
| `R/cnn_architecture.R` | Generic CNN: conv blocks, residual connections, SE blocks, gate types |
| `R/tune_grid.R` | Random and manual hyperparameter grid generation |
| `R/train_cnn.R` | Single-config trainer + full tuning loop (`run_cnn_tuning`) |

## Quick start

```r
library(torch)
library(here)

source(here("R", "utils.R"))
source(here("R", "metrics.R"))
source(here("R", "cnn_architecture.R"))
source(here("R", "tune_grid.R"))
source(here("R", "train_cnn.R"))

device <- setup_torch_device(n_threads = 8, use_cuda = TRUE)

# 1. Generate a random tuning grid (like caret's tuneLength)
grid <- make_tune_grid(tune_length = 20, seed = 42)

# 2. Run tuning — patches and points_valid come from your patch extraction step
results <- run_cnn_tuning(
  tune_grid    = grid,
  n_channels   = 204,        # number of raster layers
  patches      = patches,    # list: train / validation / test, each with x arrays + y
  points_valid = meta,       # list: train / validation / test metadata tibbles
  transform    = expm1,      # inverse of log1p applied to target before training
  output_dir   = "./outputs/tuning",
  device       = device,
  n_epochs     = 500,
  patience     = 60
)
```

See `examples/soc_stock_0_5cm/` for a complete worked example.

## Metrics

All splits (train, validation, test) are evaluated with:

| Metric | Description |
|---|---|
| **CCC** | Lin's Concordance Correlation Coefficient — accuracy + precision in one number |
| **R²** | Coefficient of determination |
| **MAE** | Mean Absolute Error (in original target units) |
| **NSE** | Nash-Sutcliffe Efficiency — 0 = mean-only model, 1 = perfect |
| **RMSE** | Root Mean Squared Error |
| **MQI** | Model Quality Index = (CCC × NSE) / (MAE / mean(obs)) |

Metrics are also broken down by **quantile group** of the observed values (Q0–Q25, Q25–Q50, …, Q99–Q100), helping diagnose model behaviour in extreme value ranges.

## Hyperparameter guide

See [`docs/tuning_guide.md`](docs/tuning_guide.md) for a full explanation of every tuneable parameter: what it controls, why the range was chosen, and how it connects to digital soil mapping.

See [`docs/architecture.md`](docs/architecture.md) for a diagram and explanation of the CNN architecture.

## Target variable transformation

Soil carbon (and many other soil attributes) has a right-skewed distribution with extreme values. Training is done on `log1p(target)` and predictions are back-transformed with `expm1()`. Pass `transform = expm1` to `run_cnn_tuning()`. For other targets you can pass any invertible function.

## Dependencies

```r
install.packages(c("torch", "coro", "dplyr", "tidyr", "readr",
                   "tibble", "purrr", "ggplot2", "DescTools",
                   "terra", "sf", "here"))
```

GPU (CUDA) is strongly recommended. CPU training is supported but significantly slower.

## License

MIT
