# soilcnn

**Convolutional Neural Networks for Digital Soil Mapping**

[![R CMD
check](https://github.com/moquedace/soilcnn/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/moquedace/soilcnn/actions/workflows/R-CMD-check.yaml)
![R 4.5 or
later](https://img.shields.io/badge/R-%3E%3D4.5-276DC3?style=flat-square&logo=r&logoColor=white)![torch](https://img.shields.io/badge/torch-deep%20learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)![digital
soil
mapping](https://img.shields.io/badge/domain-digital%20soil%20mapping-4CAF50?style=flat-square)![MIT
licence](https://img.shields.io/badge/license-MIT-blue?style=flat-square)![in
development](https://img.shields.io/badge/status-in%20development-orange?style=flat-square)

A **caret-style hyperparameter tuning framework** for dual-branch CNNs
applied to **digital soil mapping** with raster predictors — in R, as
the package `soilcnn`.

[Website](https://moquedace.github.io/soilcnn/) · [Getting
started](https://moquedace.github.io/soilcnn/articles/soilcnn.html) ·
[Reference](https://moquedace.github.io/soilcnn/reference/)

------------------------------------------------------------------------

## Overview

Traditional ML frameworks like `caret` make spatial prediction
straightforward: provide a raster stack, a target variable, and a
`tuneLength` — the framework handles the rest.

This project brings the same philosophy to **convolutional neural
networks**, from a folder of rasters and a table of soil profiles to a
map with its uncertainty and what the model learned — five calls:

    Soil profiles (GPKG)               Rasters (a folder of aligned TIFs)
            │                                       │
            └───────────────┬───────────────────────┘
                            ▼
       dsm_prepare()    Extract · QC · predictor types → a patch store, stored RAW,
                        one file per window, with its recipe.  Decides no roles.
                            │
                            ▼
       dsm_train()      A fold plan — spatial_cv() · knndm_cv() · random_cv() ·
                        region_cv() · holdout_cv() — and a grid (tune_length, like
                        caret's tuneLength).  Every (config × fold × seed), mean ± sd
                        against the seed noise floor.  rf, mlp or any caret model
                        under the SAME folds.
                            │
                            ▼
       dsm_final()      The config the run supports (one_se), refitted under N seeds
                        side by side; the ensemble, the conformal intervals (each
                        checked on the test set), the smearing factor, and a
                        declaration of every hyperparameter.
                            │
                            ▼
       dsm_predict()    The map: median, mean, spread, intervals, DI and AOA bands —
                        probed at the profiles first, resumable, as large as the
                        world at 250 m.
                            │
                            ▼
       dsm_importance() What the final model learned — permutation, context, SHAP,
                        SAGE, refits, ALE — by variable or by theme; SHAP also as
                        maps over the grid (importance_map()).

The package was developed on a global model of soil organic carbon
stock, and tested on a continental one; the numbers quoted below come
from the first unless they name the second (see [where it was
developed](#where-it-was-developed)).

------------------------------------------------------------------------

## Installation

``` r

# install.packages("remotes")
remotes::install_github("moquedace/soilcnn")
torch::install_torch()          # once: the C++ backend the torch package needs
```

The tour,
[`vignette("soilcnn")`](https://moquedace.github.io/soilcnn/articles/soilcnn.md),
is there only when the install builds it, which needs knitr, rmarkdown
and pandoc (RStudio ships pandoc):

``` r

remotes::install_github("moquedace/soilcnn", build_vignettes = TRUE)
vignette("soilcnn")
```

Over a copy already installed from the same commit, add `force = TRUE`:
remotes skips a commit it has already installed, and the vignette with
it. To work on the source tree instead, clone the repository and load it
with `pkgload::load_all("<clone>")`.

------------------------------------------------------------------------

## Quickstart

The whole chain, one call per step. The vignette ([Getting
started](https://moquedace.github.io/soilcnn/articles/soilcnn.html))
walks it with the reasons, and with the figures of a real application.

``` r

library(soilcnn)                # or pkgload::load_all(".") on the source tree

store <- dsm_prepare(points = "profiles.gpkg", target = "soc_stock",
                     raster_dir = "predictors/", windows = c(3, 9, 15),
                     out_dir = "outputs", transform = "log1p")
data  <- dsm_load(store)

fit <- dsm_train(
  data,
  model       = "cnn",
  resampling  = spatial_cv(k = 5, block_size = "auto", buffer = "auto"),
  tune_length = 30,            # a budget, like caret's
  n_seeds     = 3,             # a claim without repetitions has no error bar
  output_dir  = "outputs/tuning"   # required: nothing is written where nobody said
)
# What it did not have to be told, and read instead:
#   the grid's windows    every window the store holds, alone and in pairs
#   the batch sizes       those that give the smallest fold >= 4 steps an epoch
#   the inverse           the store's (log1p -> expm1); one that disagrees is refused
#   the cores             n_cores = NULL is the physical cores minus one

fit$by_config                            # mean ± sd, one row per config
print_noise_floor(seed_noise_floor(fit$comparison))
print_one_se(one_se(fit$by_config))      # the simplest config within 1 SE

final <- dsm_final(fit, seeds = 10)      # the selected config, under ten seeds
map   <- dsm_predict(final, data)        # median, mean, intervals, DI and AOA bands

imp   <- dsm_importance(final, data)     # what the model learned: permutation, by default
plot(imp)
```

[`dsm_load()`](https://moquedace.github.io/soilcnn/reference/dsm_load.md)
opens the store, reads the points and predictor types, aligns them,
reads the raster resolution — and **refuses** if the store was built
under a different predictor set, window set, target or resolution. A
store written by
[`dsm_prepare()`](https://moquedace.github.io/soilcnn/reference/dsm_prepare.md)
carries its own tables and recipe, so `dsm_load(store)` needs nothing
else.

### Without data of your own

The package ships a small synthetic landscape – 80 x 80 cells, eight
predictors, 160 profiles in clusters, a stock drawn from vegetation,
clay, temperature, geology and the topographic position of each cell –
and a run fitted on it in about a minute, kept for the session:

``` r

ex  <- example_landscape()     # the rasters' folder and the profiles
run <- example_run()           # store, tuning and final model, in tempdir()
run$final
dsm_importance(run$final, run$data)
```

Every help page’s example starts from one of the two.

### A smaller run first

`dsm_prepare(subsample = list(frac = 0.1, block_size = 0.5, strata = 5))`
keeps a tenth of the points in whole blocks, so the clusters the
validation must face stay whole, and `strata` keeps every region: each
5-degree square keeps its points up to one common quota, so the cut
falls where the data are densest. The store records it, and a subsampled
run is never mistaken for a full one.

### One line decides who trains and who scores

``` r

spatial_cv(k = 5)                        # blocks of ground, buffered
knndm_cv(k = 5, predpoints = pts)        # folds at the distances the map predicts at
random_cv(k = 10)                        # ignores geography, on purpose
holdout_cv(validation_frac = 0.2)        # a single split
region_cv(group = points$biome)          # leave-one-region-out
```

Swap it and nothing else changes. One criterion carves the test set
**and** the folds, so the two numbers a run reports answer the same
question.

### `"auto"` means measured, never guessed

| argument | what it resolves to |
|----|----|
| `block_size = "auto"` | the **largest** block whose worst case still fits the balance constraint, measured on *these* points |
| `buffer = "auto"` | `max(window) × cell_size` — the exact distance at which two patches stop sharing a pixel, under the Chebyshev metric |

Both print what they chose, and `"auto"` without a raster resolution
**refuses** rather than inventing one.

This is not a style preference. A block size measured on the full point
set and carried into a 10% subsample once left a single block holding a
third of the data — because block-subsampling keeps *whole* blocks, so a
smaller draw has fewer blocks of the **same** width.

### Any registered model, same folds, same tables

``` r

list_models()

rf  <- dsm_train(data, model = "rf", resampling = plan,
                 features = c("centre", "window_mean"), output_dir = "outputs/tuning")
mlp <- dsm_train(data, model = "mlp", resampling = plan, features = "centre",
                 output_dir = "outputs/tuning")

register_model(caret_spec("xgbTree"))    # ~230 methods, borrowed from caret
xgb <- dsm_train(data, model = "xgbTree", resampling = plan, output_dir = "outputs/tuning")
```

Every family produces the same comparison table, so
[`summarise_resamples()`](https://moquedace.github.io/soilcnn/reference/summarise_resamples.md),
[`seed_noise_floor()`](https://moquedace.github.io/soilcnn/reference/seed_noise_floor.md)
and
[`one_se()`](https://moquedace.github.io/soilcnn/reference/one_se.md)
work across all of them. **The gap between the context RF and the CNN is
what the convolution is worth** — and if it is smaller than the noise
floor, the convolution is doing averaging.

caret is borrowed for its model *library*, never for its resampling: the
fold plan stays here, with its blocks and its buffer.
[`caret_spec()`](https://moquedace.github.io/soilcnn/reference/caret_spec.md)
calls `train(method = "none")` with a one-row grid, so two objects never
both believe they own the split.

------------------------------------------------------------------------

## What the framework is careful about

Three things cost this project real time before they were made
structural.

**The split is not stored with the data.**
[`dsm_prepare()`](https://moquedace.github.io/soilcnn/reference/dsm_prepare.md)
writes points, coordinates, the target and the predictor types — and
nothing about who trains. A fold plan decides that when training starts,
from coordinates, in seconds:

``` r

plan <- holdout(meta, validation_frac = 0.15, test_frac = 0.15)
plan <- random_folds(meta, k = 5, test_frac = 0.15)
plan <- spatial_folds(meta, k = 5, test_frac = 0.15, block_size = 2, buffer = 0.034)
plan <- region_folds(meta, group = meta$biome, test_frac = 0.15)
```

One criterion carves the test set **and** the folds, so the two numbers
a run reports answer the same question. Changing the strategy costs
seconds; before, it cost a five-hour re-extraction.

**A repetition is a seed, and the noise is measured.** The unit of work
is `(config, fold, seed)`, and the seed is shared across configs on
purpose, so two configs within a repetition start from the same draw.
What the spread across repetitions then measures is luck — reported as a
noise floor, because a gap between configs smaller than it is not
evidence.
[`one_se()`](https://moquedace.github.io/soilcnn/reference/one_se.md) is
available for when it is not: among configs within one standard error of
the best, take the simplest.

**A row is not always an independent observation.** In 3-D soil mapping
one profile yields several rows — 0–5, 5–15, 15–30 cm — at identical
coordinates, from the same pit. Split those across training and
validation and the model is scored on a depth of a profile it already
learned.
[`holdout()`](https://moquedace.github.io/soilcnn/reference/holdout.md)
and
[`random_folds()`](https://moquedace.github.io/soilcnn/reference/random_folds.md)
keep a profile together by default (`group = "auto"`), and
[`check_fold_plan()`](https://moquedace.github.io/soilcnn/reference/check_fold_plan.md)
**proves** no group was split rather than trusting the constructor that
was meant to prevent it. [Wang et al. 2025,
*Geoderma*](https://www.sciencedirect.com/science/article/pii/S0016706125000618)

**A map needs to say where it should be believed.** A prediction exists
at every pixel, including pixels whose predictor combination the model
never saw, and the cross-validated CCC does not describe those.
`R/aoa.R` computes a dissimilarity index and the area of applicability —
the DI expressed in units of the training set’s own mean pairwise
distance, and the threshold derived from distances **across** folds,
which is what makes it mean “as dissimilar as something cross-validation
coped with”.
[`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md)
writes both as bands. [Meyer & Pebesma 2021,
*MEE*](https://arxiv.org/pdf/2005.07939)

**Identical inputs are a defect; nearby ones are not.** Two points in
the same raster cell give the network the same patch, bit for bit, and
one can be scored on what the other trained on — under any plan. Two
*neighbouring* points sharing some surrounding pixels is not a defect:
under a random split it is the condition being measured. The leakage
report keeps the two apart.

------------------------------------------------------------------------

## The dual-branch idea

Each soil profile is represented by **two spatial patches** extracted
from a stack of raster layers:

| Branch | Window | Processes captured |
|----|----|----|
| Small | 3 × 3 cells | Local topography, land cover, proximity effects |
| Large | 9 × 9 or 15 × 15 cells | Landscape position, parent material, local climate |

A window’s physical extent is `window_size × raster resolution`, so
pixel sizes are chosen per resolution. At 250 m they span ~0.75 km (3 ×
3) to ~3.75 km (15 × 15).

Patches are stored **raw** and scaled when a fold’s tensors are built —
z-score for continuous predictors, /100 for proportions, identity for
dummies — from the training rows **of that fold**. This equalises
gradient flow across channels of very different magnitude (elevation in
thousands against vegetation indices in 0–1), and it is what lets one
patch store serve any number of folds: the alternative is one
re-extraction per fold.

The scaling therefore belongs to the **fitted model**, not to the
dataset.
[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)
writes it next to the weights, and
[`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md)
reads it from there.

A learned **gate** fuses the two embeddings per sample — letting each
location draw from whichever spatial scale is more informative for the
target variable.

Architecture diagram (click to expand)

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

See
[`docs/architecture.md`](https://moquedace.github.io/soilcnn/docs/architecture.md)
for full detail. For the reasoning behind every architectural and
training choice see
[`docs/design_decisions.md`](https://moquedace.github.io/soilcnn/docs/design_decisions.md).

------------------------------------------------------------------------

## Numbers you can defend

Three things a soil-mapping paper is normally asked for and normally
cannot give: an honest test score, evidence that the architecture earns
its cost, and an uncertainty map that covers what it claims.

### The test set stays frozen, and the optimism is measured

Tuning never scores the test set. Once the choice is locked — recorded
on disk, with a timestamp and a commit — the whole grid *can* be scored
on it, and that measures something worth publishing:

``` r

freeze_selection(run_dir, "cfg_014", rule = "one_se")   # dsm_final() does this
score_test_grid(run_dir, data, device = device)         # afterwards, any time
```

    chosen by validation : cfg_014   test CCC 0.4612   (rank 6 of 24 on test)
    best on test         : cfg_003   test CCC 0.4980
    SELECTION OPTIMISM   : +0.0368 CCC

That gap is how much a test-selected number would have been overstated.
It is not a reason to switch config — switching is what the measurement
is measuring.
[`score_test_grid()`](https://moquedace.github.io/soilcnn/reference/score_test_grid.md)
refuses to run before the selection is frozen, and
[`freeze_selection()`](https://moquedace.github.io/soilcnn/reference/freeze_selection.md)
refuses to be overwritten with a different config, because “score,
dislike, re-freeze, re-score” is the loop the ordering exists to
prevent. Nothing is retrained: the checkpoints are already on disk.

### Does the convolution earn its cost?

Two independent routes to the same question, which is the point — if
they disagree, one of the measurements is wrong and that is worth
knowing.

*From the outside*, the baselines race the CNN against a forest fed the
same neighbourhood with the arrangement thrown away. *From the inside*,
[`occlusion_report()`](https://moquedace.github.io/soilcnn/reference/occlusion_report.md)
hides part of the patch of a trained network and re-predicts:

``` r

occlusion_report(run_dir, data, config_id = "cfg_014", device = device)
```

    scope                n_pixels_hidden    ccc   delta_ccc
    baseline                           0  0.489
    context_all                      224  0.487      -0.002
    centre_only_hidden                 1  0.331      -0.158
    ring_01                            8  0.488      -0.001
    ...
    -> THE CENTRE PIXEL CARRIES MORE THAN THE WHOLE NEIGHBOURHOOD.

The hidden region is **permuted from another sample**, not zeroed. After
scaling zero is the training mean, and a patch whose rim is the mean
everywhere is a landscape that does not exist — the drop would then mix
“this region mattered” with “this input is impossible”, and the second
grows with the area hidden, which is exactly the comparison being made.

### Calibrated uncertainty, and a check that it is calibrated

``` r

cal <- conformal_calibrate(val$obs, val$pred, alpha = 0.1)   # 90%
iv  <- conformal_interval(cal, test$pred, lower_limit = 0)
picp_report(test$obs, iv$lower, iv$upper, group = test$block, alpha = 0.1)
```

Split conformal, on a calibration set nothing else touched, gives
`P(y ∈ interval) ≥ 1 − α` with no distributional assumption. Where the
residuals come from decides what the interval can promise, and the
framework offers three sources, each with its own trade:

| `method` | the residuals | guarantee | cost |
|----|----|----|----|
| `"cv"` | the tuning run’s cross-validated residuals | none: the configuration was chosen on those folds, and the final model is refitted on more data | nothing |
| `"split"` | a calibration set the plan carves beside the test set (`spatial_cv(calibration_frac = 0.15)`): in no fold, behind the same buffer, predicted by the final model and trained on by none | `≥ 1 − α` for points exchangeable with it (Lei et al. 2018) | the points leave the training |
| `"cv_plus"` | the cross-validated residuals, with each fold’s own model at the new point — CV+ (Barber et al. 2021) | `≥ 1 − 2α − √(2/n)` for an algorithm fixed in advance, ~`1 − α` in practice | every fold model predicts every pixel |

Each comes at a constant width or one that grows with the predicted
level and the dissimilarity index, and with points or whole groups —
blocks, regions, profiles — weighing alike: pooled point by point, a
quantile is the dense surveys’; every group weighing the same (Dunn,
Wasserman & Ramdas 2023), it speaks for a new place. Profiles cluster,
and a configuration is chosen on its folds, so no guarantee holds here
as stated.
[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)
therefore calibrates every method the runs allow and checks every one on
the **test** rows — overall, by group, inside and outside the area of
applicability, and by fifth of the predicted level — in
`<config>/intervals/coverage_test.csv` and the report. A coverage
measured on the points that calibrated it comes out right by arithmetic,
not by evidence.

`dsm_predict(intervals = c("cv", "split", "cv_plus"), weighting = "point")`
writes the bands a map carries:
`pi90_<method>_<width>_lower/upper[_<source>]` — `constant` or
`level_di` — for every calibration source it is given (block folds,
kNNDM folds); “split” is the final run’s own. Every number that
calibrated a band is in the run’s `calibration.csv`, and CV+’s fold
models pass the same probe as the seeds before anything is mapped.

**PICP** turns uncertainty from an adjective into a number that can be
wrong. Promise 90%, deliver 61%, and you can see it. And because the
conformal guarantee is *marginal*, not conditional, coverage is broken
down by group and the worst is printed first: 90% overall is compatible
with 99% over the easy half and 60% over the hard half, and the hard
half is where anyone needs an interval at all. When a group falls far
below, that is where exchangeability — the theorem’s only assumption —
is breaking, which is the same place the area of applicability is
pointing at.

### A median surface and a mean surface, and they are not the same map

``` r

sm <- smearing_from_run(tuning_dir, config_id)   # Duan (1983), one scalar
mean_surface <- smear(log1p(median_surface), sm, lower_limit = 0)
```

Train on `log1p` with a SmoothL1 loss and the network estimates a
conditional **median** in log space.
[`expm1()`](https://rdrr.io/r/base/Log.html) of a median is the median
of the stock — not its mean. On a right-skewed target the two are far
apart, and the gap lands entirely on anyone who sums the map:

| surface          | bias on the frozen test set | may be summed for a total? |
|------------------|-----------------------------|----------------------------|
| median (`expm1`) | **−24.4%**                  | **no**                     |
| mean (smeared)   | +2.7%                       | yes                        |

Duan’s smearing estimator corrects it with one scalar, calibrated on the
same out-of-fold **ensemble** residuals the cross-validated interval
uses — the deployed prediction is the ensemble median, so the calibrated
residual has to be the ensemble’s and not one seed’s. With a calibration
set, a second factor comes from its residuals (`smeared_mean_split`).

**Nothing is replaced.**
[`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md)
writes `smeared_mean_<source>` *beside* `ensemble_median` and labels
both, because they answer different questions: the median is the typical
stock at a pixel and minimises absolute error; the mean is the only one
you may add up. Silently swapping one for the other trades a known bias
for an unknown one.

[`print.smearing_cal()`](https://moquedace.github.io/soilcnn/reference/print.smearing_cal.md)
also reports S by quintile of the prediction, because Duan’s derivation
assumes the residual is independent of the prediction and that is
checkable — in the SOC model it runs 1.80 at the low end to 1.15 at the
high end, which the print warns about rather than silently averaging
away.

Two refinements are options of
[`smear()`](https://moquedace.github.io/soilcnn/reference/smear.md), not
its default: `method = "level"`, a factor by the prediction’s level from
bins of it, and `method = "total"`, the exp(f)-weighted scalar that
unbiases a sum. On the SOC model both made the held-out bias worse
(−3.9% and −5.8%, against +2.7%), because its three folds saw half the
points the deployed model saw. A run whose folds see about as many may
find otherwise, so
[`smearing_check()`](https://moquedace.github.io/soilcnn/reference/smearing_check.md)
measures all three on its test set, and
[`smear_map()`](https://moquedace.github.io/soilcnn/reference/smear_map.md)
makes the mean map by the factor chosen from a median map already made,
without running the network again.

### Two models compared on clustered test points

``` r

d   <- abs(test$pred_a - test$obs) - abs(test$pred_b - test$obs)   # what is compared
spatial_correlogram(test$x, test$y, d)            # does it resemble itself nearby, and how far?
blk <- equal_area_blocks(test$x, test$y, size_km = 100)
block_bootstrap(test$obs, test$pred_a, blk, against = test$pred_b)
block_bootstrap_by_size(test$x, test$y, test$obs, test$pred_a, against = test$pred_b)
```

Soil profiles come in surveys, farms and transects. In the SOC 0-30 cm
trial, the common test set’s 3,900 profiles sit in 303 one-degree
blocks, half of them in 11. A bootstrap that draws the profiles one by
one counts a block of 603 as 603 independent pieces of evidence, and
where two models’ errors are alike within a block its interval is too
narrow.
[`block_bootstrap()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap.md)
draws whole blocks, and answers two questions:

| weights | the question | the estimate |
|----|----|----|
| `"profile"` | which model errs less on a profile of this data base? | the mean over the points; the interval from whole blocks |
| `"block"` | which model errs less in a region? | every block one vote (cell declustering); the estimate moves too |

Where the two disagree, the advantage rests on the densely sampled
regions. `n_effective` says how many independent points the profile
interval is worth. The blocks are of equal area — a degree of longitude
is 111 km at the equator and 71 km at 50° — and
[`block_bootstrap_by_size()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap_by_size.md)
shows where the interval stops widening as they grow, which
[`spatial_correlogram()`](https://moquedace.github.io/soilcnn/reference/spatial_correlogram.md)
says from the differences themselves.

------------------------------------------------------------------------

## What the model learned

[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md)
asks a final model which variables it uses, and the method is an
argument, the way the validation design is one in
[`dsm_train()`](https://moquedace.github.io/soilcnn/reference/dsm_train.md):

``` r

dsm_importance(final, data)                                      # permutation, on the test set
dsm_importance(final, data, permutation_importance(within = 5))  # donors from the same 5-unit block
dsm_importance(final, data, context_importance())                # how far from the point it reads
dsm_importance(final, data, shap_importance())                   # what pushes each prediction, and how
dsm_importance(final, data, sage_importance(), groups = themes)  # the skill, shared among themes
dsm_importance(final, data, refit_importance(), groups = themes, seeds = 42:44)
dsm_importance(final, data, ale_effect())                        # the prediction along each range
```

Each answers its own question, and they are meant to be read together:

| method | the question it answers | how |
|----|----|----|
| [`permutation_importance()`](https://moquedace.github.io/soilcnn/reference/permutation_importance.md) | how much does the fitted model rely on the variable? | each point takes the variable’s whole patch, every window, from another point |
| … `within =` a block size or a class | what does it add beyond the place? | the donor comes from the same block or class |
| [`context_importance()`](https://moquedace.github.io/soilcnn/reference/context_importance.md) | how far from the point, and through which window, does it read? | rings of the patch, or a window’s whole input, permuted; the cost per pixel |
| [`shap_importance()`](https://moquedace.github.io/soilcnn/reference/shap_importance.md) | how does each variable push each prediction, up or down? | expected gradients (the default), integrated gradients, or exact Shapley values over themes (`"kernel"`) |
| [`sage_importance()`](https://moquedace.github.io/soilcnn/reference/sage_importance.md) | how much of the model’s skill does each variable carry, shared fairly? | Shapley values of the loss, over themes |
| [`refit_importance()`](https://moquedace.github.io/soilcnn/reference/refit_importance.md) | is the variable needed at all? | the final run’s seeds trained again with the variable at its training mean |
| [`ale_effect()`](https://moquedace.github.io/soilcnn/reference/ale_effect.md) | how does the prediction change along the variable’s range? | accumulated local effects, the whole patch moved |

A variable is a channel, or the channels of one categorical — the
one-hot sets are inferred from the names and checked against the data —
or a group of yours. With many correlated predictors, themes (climate,
relief, vegetation, …) are the unit a reader can use. Of the signal two
near-copies share, a permutation credits neither, a refit without one
finds it in the other, and SAGE splits it between them. `rows = "folds"`
scores the tuning run’s fold models on their own validation rows instead
of the final seeds on the test set — the way two validation designs are
compared on what their models learned.
[`compare_importance()`](https://moquedace.github.io/soilcnn/reference/compare_importance.md)
sets importances side by side and says how far their rankings agree, and
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) draws the
figure each one answers with.

**Nothing is measured against a model that is not the run’s.** Before
anything is perturbed, each model must give back, point by point, the
predictions its run wrote. SHAP values must add up to the prediction
less the reference, and SAGE values to the loss the model explains. A
refit first trains a seed again with nothing left out and must get the
run’s predictions back: in the SOC 0-30 cm trial’s test run, a seed
trained again five days later did, to a difference of 0.

SHAP runs at points of the map too, and becomes maps:

``` r

pts <- importance_points(final, data, extent = c(-55, -53, -31, -29))   # a tile, every cell
imp <- dsm_importance(final, data, shap_importance(), at = pts)
plot(importance_map(imp, output_dir = "shap_tile"))  # per variable, the dominant one, the prediction
```

And an importance can weight the area of applicability (Meyer & Pebesma
2021): with `dsm_predict(final, data, aoa_weights = imp)`, a predictor
the model ignores no longer pushes a pixel out of the AOA.

------------------------------------------------------------------------

## Tunable parameters

The table below summarises the search space
[`make_tune_grid()`](https://moquedace.github.io/soilcnn/reference/make_tune_grid.md)
draws from. See
[`docs/tuning_guide.md`](https://moquedace.github.io/soilcnn/docs/tuning_guide.md)
for the rationale behind every range and its connection to digital soil
mapping.

| Parameter | Options | Controls |
|----|----|----|
| `window_sizes` | the store’s windows, alone and in pairs — e.g. `c(3)` · `c(15)` · `c(3,15)` | Spatial scale(s) |
| `conv_channels` | `c(32,64)` · `c(64,128)` · `c(64,128,128)` · `c(128,256)` · `c(128,256,256)` | Network depth & width |
| `use_residual` | `TRUE` · `FALSE` | Skip connections (ResNet-style) |
| `use_se_block` | `TRUE` · `FALSE` | Channel attention |
| `se_reduction` | 16 (fixed by default) | The SE block’s bottleneck ratio |
| `conv_padding` | `same` · `valid_large` | `same` pads with zeros; `valid_large` keeps only measured pixels on the large branch |
| `gate_type` | `vector_featurewise` · `scalar_per_sample` · `no_gate_concat` | Branch fusion strategy |
| `embedding_dim` | 128 · 256 · 384 · 512 | Representation size |
| `embed_pool` | `flatten` · `gap` | Pre-embedding reduction: keep every cell (params grow with window²) vs. global average pool (window-independent, ~25× lighter for 15×15) |
| `dropout` | 0.0 · 0.1 · 0.2 · 0.3 | Overall regularisation (maps to 5 internal sites) |
| `base_lr` | 1e-4 · 3e-4 · 5e-4 · 1e-3 · 2e-3 · 3e-3 | Peak learning rate (Adam + warmup) |
| `warmup_epochs` | 5 (fixed by default) | Epochs of linear warmup to `base_lr` |
| `loss_fn` | `smooth_l1` · `mse` · `mae` | Training objective |
| `weight_decay` | 0 · 1e-5 · 1e-4 · 1e-3 | L2 regularisation |
| `batch_size` | 128 · 256 · 512 — those that give the smallest fold ≥ 4 steps an epoch | Mini-batch size |

[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)
writes, for the configuration it refits, every one of these with its
value, whether the search chose it or the grid fixed it, and the values
the grid tried (`final_report.md`, `selected_hyperparameters.csv`).

------------------------------------------------------------------------

## Evaluation metrics

All splits (train · validation · test) are evaluated with nine metrics,
also broken down by **quantile group** of the observed values (Q0–Q25,
…, Q99–Q100):

| Metric | Description |
|----|----|
| **CCC** | Lin’s Concordance Correlation Coefficient — accuracy + precision combined |
| **R²** | Coefficient of determination |
| **MAE** | Mean Absolute Error (native target units) |
| **NSE** | Nash-Sutcliffe Efficiency — 0 = mean-only model, 1 = perfect |
| **RMSE** | Root Mean Squared Error |
| **RPD** | Ratio of Performance to Deviation = sd(obs) / RMSE — standard pedometric benchmark (\<1.4 poor, 1.4–2.0 fair, \>2.0 good) |
| **MQI** | Model Quality Index = (CCC × NSE) / (MAE / mean(obs)) |
| **bias** | `mean(pred − obs)`, **signed**, in native units |
| **bias_pct** | the same relative to `mean(obs)`, so it compares across targets and depths |

The last two were added late, and the reason is worth stating: every
other metric on this list is blind to the *sign* of the error. MAE and
RMSE are unsigned by construction; R², NSE and RPD are unmoved by a
constant offset in the right circumstances; CCC penalises bias but mixes
it with scatter, so a low CCC never says which one it is. This
framework’s own final model was under-predicting its test set by
**24.4%** — more than half its MAE — and nothing in the tables could see
it.

Model selection across configs ranks by **validation CCC** (descending),
then **validation MAE** (ascending) as a tiebreaker — or by
[`one_se()`](https://moquedace.github.io/soilcnn/reference/one_se.md),
which takes the simplest config within one standard error of the best
and is the default of
[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md).

**The test set is not scored during tuning at all**
(`evaluate_test = FALSE`). The columns exist and hold `NA`, so the table
has one shape either way. An earlier version computed them “as
diagnostic reference”; there is no such thing. A test score sitting
beside the selection metric is selection on the test set performed by
whoever reads the table, and with 24 configs × 9 repetitions the *best*
of 216 noisy test scores is higher than any one of them by construction
— before anyone chooses anything.

The test is scored once, by
[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md),
on the config chosen without it.

Early stopping uses **validation SmoothL1 loss** — keeping the stopping
criterion consistent with the training objective.

### Multi-seed ensemble

After architecture selection, the chosen config(s) are re-trained with N
independent seeds (different weight initialisation + batch shuffling).
Sources of run-to-run variance:

| Source                | Effect                                             |
|-----------------------|----------------------------------------------------|
| Weight initialisation | Different local minima after convergence           |
| Batch shuffle order   | Different gradient path through the loss landscape |
| Dropout masks         | Different regularisation per forward pass          |

A seed’s numbers depend on its seed **and on its thread count**, and on
nothing else: not on which process trains it, nor on what trains beside
it.
[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)
trains the seeds side by side with a fixed number of threads each, and
records it.

Spatial prediction aggregates all seed models per pixel. The **median**
is the recommended headline map: it is invariant to the monotone `expm1`
back-transform (`median(expm1(z)) = expm1(median(z))`), robust to
divergent seeds, and consistent with what SmoothL1 learns (a conditional
median).

SD and MAD are written too — but **they are not a prediction interval**,
and the framework says so rather than letting a reader assume otherwise.
They measure how much the answer moves when the initialisation moves: a
property of the optimiser, not of the soil. In the SOC model’s numbers
the seed spread is 0.038 CCC while the MAE is ~17 t/ha on a median stock
of 29.3. A map drawn from that spread would promise an order of
magnitude more certainty than it has, and a map that understates is
worse than no map, because somebody acts on it.

What the interval bands come from instead is [calibrated
uncertainty](#calibrated-uncertainty-and-a-check-that-it-is-calibrated).

------------------------------------------------------------------------

## Framework files

| File | Purpose |
|----|----|
| [`R/api.R`](https://moquedace.github.io/soilcnn/R/api.R) | **The front end**: [`dsm_load()`](https://moquedace.github.io/soilcnn/reference/dsm_load.md) · the resampling specs ([`spatial_cv()`](https://moquedace.github.io/soilcnn/reference/spatial_cv.md) and the rest) · [`dsm_train()`](https://moquedace.github.io/soilcnn/reference/dsm_train.md) |
| [`R/prepare.R`](https://moquedace.github.io/soilcnn/R/prepare.R) | [`dsm_prepare()`](https://moquedace.github.io/soilcnn/reference/dsm_prepare.md) — a point table and a folder of aligned rasters become a patch store, with the target transform, the predictor types and the QC rules written into it as a recipe; `dsm_load(store)` then needs nothing else |
| [`R/final.R`](https://moquedace.github.io/soilcnn/R/final.R) | [`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md): the selected config refitted under N seeds, side by side with fixed threads per seed; the ensemble, the conformal intervals (cv, split, CV+), each checked on the test set, the smearing factor; and `final_report.md`, which declares every hyperparameter of the chosen CNN and whether the search chose it. A resume is held to the settings its run started with |
| [`R/predict.R`](https://moquedace.github.io/soilcnn/R/predict.R) | [`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md): the map, for a grid as large as the world at 250 m – row bands read once through a buffer that keeps its halo, the network fully convolutional where that is exact, workers side by side and resumable; the ensemble bands, the smeared mean, the conformal intervals of every calibration given (cv, split, CV+), each at a constant width and one that follows the level and the DI, and the AOA, one VRT per band; and, first, a probe that must reproduce the final model’s stored predictions at the profiles |
| [`R/utils.R`](https://moquedace.github.io/soilcnn/R/utils.R) | Safe I/O helpers, torch device setup, the internal `.env_*()` readers of environment overrides, [`latest_run_dir()`](https://moquedace.github.io/soilcnn/reference/latest_run_dir.md) — the newest *finished* run, by time |
| [`R/checks.R`](https://moquedace.github.io/soilcnn/R/checks.R) | `check_ledger()` · `ledger_check()` · `ledger_verdict()` — a ledger whose verdict refuses to pass while a promised check is missing |
| [`R/metrics.R`](https://moquedace.github.io/soilcnn/R/metrics.R) | [`ccc()`](https://moquedace.github.io/soilcnn/reference/ccc.md) · R² · MAE · NSE · RMSE · MQI · **signed bias**, per split and per quantile group |
| [`R/cnn_architecture.R`](https://moquedace.github.io/soilcnn/R/cnn_architecture.R) | Conv blocks, residual connections, SE attention, gate types, full model |
| [`R/tune_grid.R`](https://moquedace.github.io/soilcnn/R/tune_grid.R) | [`make_tune_grid()`](https://moquedace.github.io/soilcnn/reference/make_tune_grid.md) · [`make_manual_tune_grid()`](https://moquedace.github.io/soilcnn/reference/make_manual_tune_grid.md) with documented parameter ranges |
| [`R/patches.R`](https://moquedace.github.io/soilcnn/R/patches.R) | One patch-indexing path, shared by extraction and prediction |
| [`R/preprocess.R`](https://moquedace.github.io/soilcnn/R/preprocess.R) | QC (fold-independent) split from scaling (fold-dependent) |
| [`R/dataset.R`](https://moquedace.github.io/soilcnn/R/dataset.R) | The patch store: one file per window, the split as an index; a fold’s tensors cut in slabs, with no whole-window copy |
| [`R/resample.R`](https://moquedace.github.io/soilcnn/R/resample.R) | Fold plans · distance buffering · [`summarise_resamples()`](https://moquedace.github.io/soilcnn/reference/summarise_resamples.md) · [`seed_noise_floor()`](https://moquedace.github.io/soilcnn/reference/seed_noise_floor.md) · [`one_se()`](https://moquedace.github.io/soilcnn/reference/one_se.md) |
| [`R/knndm.R`](https://moquedace.github.io/soilcnn/R/knndm.R) | [`knndm_folds()`](https://moquedace.github.io/soilcnn/reference/knndm_folds.md) · [`prediction_sample()`](https://moquedace.github.io/soilcnn/reference/prediction_sample.md) — folds whose geometry matches what prediction faces, not what a block grid happens to give |
| [`R/diagnostics.R`](https://moquedace.github.io/soilcnn/R/diagnostics.R) | Checks about THIS RUN on real data: patch centres, overlap between splits, run snapshots |
| [`R/train_cnn.R`](https://moquedace.github.io/soilcnn/R/train_cnn.R) | `train_one_cnn()` · `run_cnn_tuning()` · `run_cnn_resample()` |
| [`R/train_workers.R`](https://moquedace.github.io/soilcnn/R/train_workers.R) | The tuning units `(config, fold, seed)` trained side by side in worker processes, with fixed threads each, resumable; what [`dsm_train()`](https://moquedace.github.io/soilcnn/reference/dsm_train.md) runs underneath |
| [`R/model_registry.R`](https://moquedace.github.io/soilcnn/R/model_registry.R) | [`model_spec()`](https://moquedace.github.io/soilcnn/reference/model_spec.md) · [`register_model()`](https://moquedace.github.io/soilcnn/reference/register_model.md) · [`list_models()`](https://moquedace.github.io/soilcnn/reference/list_models.md) |
| [`R/baselines.R`](https://moquedace.github.io/soilcnn/R/baselines.R) | `rf` · `mlp` · `cnn`, registered |
| [`R/train_table.R`](https://moquedace.github.io/soilcnn/R/train_table.R) | `run_table_resample()` — tabular models, same comparison table |
| [`R/caret_adapter.R`](https://moquedace.github.io/soilcnn/R/caret_adapter.R) | [`caret_spec()`](https://moquedace.github.io/soilcnn/reference/caret_spec.md) — borrow ~230 models, never caret’s resampling |
| [`R/aoa.R`](https://moquedace.github.io/soilcnn/R/aoa.R) | Dissimilarity index · area of applicability |
| [`R/conformal.R`](https://moquedace.github.io/soilcnn/R/conformal.R) | [`conformal_calibrate()`](https://moquedace.github.io/soilcnn/reference/conformal_calibrate.md) · [`conformal_scaled_calibrate()`](https://moquedace.github.io/soilcnn/reference/conformal_scaled_calibrate.md) · [`conformal_scale_fit()`](https://moquedace.github.io/soilcnn/reference/conformal_scale_fit.md) · [`cv_plus_calibrate()`](https://moquedace.github.io/soilcnn/reference/cv_plus_calibrate.md) · [`picp_report()`](https://moquedace.github.io/soilcnn/reference/picp_report.md) — split conformal and CV+, with points or whole groups weighing alike, at a constant width or one fitted to the level and the DI; and the check that they keep their coverage |
| [`R/occlusion.R`](https://moquedace.github.io/soilcnn/R/occlusion.R) | [`occlusion_report()`](https://moquedace.github.io/soilcnn/reference/occlusion_report.md) — does the trained network use the neighbourhood, or only the centre pixel? |
| [`R/smearing.R`](https://moquedace.github.io/soilcnn/R/smearing.R) | [`smearing_factor()`](https://moquedace.github.io/soilcnn/reference/smearing_factor.md) · [`smear()`](https://moquedace.github.io/soilcnn/reference/smear.md) — the back-transform of a log-trained median, and the one surface that may be summed; [`smearing_check()`](https://moquedace.github.io/soilcnn/reference/smearing_check.md) measures its factors on held-out points, and [`smear_map()`](https://moquedace.github.io/soilcnn/reference/smear_map.md) applies one to a median map |
| [`R/block_bootstrap.R`](https://moquedace.github.io/soilcnn/R/block_bootstrap.R) | [`block_bootstrap()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap.md) · [`equal_area_blocks()`](https://moquedace.github.io/soilcnn/reference/equal_area_blocks.md) · [`spatial_correlogram()`](https://moquedace.github.io/soilcnn/reference/spatial_correlogram.md) — intervals on clustered test points from whole blocks of equal area, per profile and per block |
| [`R/test_optimism.R`](https://moquedace.github.io/soilcnn/R/test_optimism.R) | [`freeze_selection()`](https://moquedace.github.io/soilcnn/reference/freeze_selection.md) · [`score_test_grid()`](https://moquedace.github.io/soilcnn/reference/score_test_grid.md) — the test set, scored only after the choice is locked |
| [`R/importance.R`](https://moquedace.github.io/soilcnn/R/importance.R) | [`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md) and its methods — permutation, context, SHAP (expected and integrated gradients, the kernel), SAGE, ALE; the groups and the one-hot sets; SHAP at points of the map ([`importance_points()`](https://moquedace.github.io/soilcnn/reference/importance_points.md), [`importance_map()`](https://moquedace.github.io/soilcnn/reference/importance_map.md)); [`compare_importance()`](https://moquedace.github.io/soilcnn/reference/compare_importance.md); [`importance_weights()`](https://moquedace.github.io/soilcnn/reference/importance_weights.md) for the AOA. Every model is held to its run’s predictions first |
| [`R/importance_refit.R`](https://moquedace.github.io/soilcnn/R/importance_refit.R) | [`refit_importance()`](https://moquedace.github.io/soilcnn/reference/refit_importance.md) — leave one covariate out: the final run’s seeds trained again without each variable, in [`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)’s own workers, once a seed trained again with nothing left out has given the run’s predictions back |
| [`R/importance_plot.R`](https://moquedace.github.io/soilcnn/R/importance_plot.R) | [`plot()`](https://rdrr.io/r/graphics/plot.default.html) for an importance, a map of SHAP values and a comparison of importances — in base graphics |
| [`R/example_landscape.R`](https://moquedace.github.io/soilcnn/R/example_landscape.R) | [`example_landscape()`](https://moquedace.github.io/soilcnn/reference/example_landscape.md) · [`example_run()`](https://moquedace.github.io/soilcnn/reference/example_run.md) — the small synthetic landscape the package ships, and a run fitted on it once per session, for the examples, the fast tests and anyone without data |
| [`R/globals.R`](https://moquedace.github.io/soilcnn/R/globals.R) | The column names dplyr resolves at run time, declared for `R CMD check` — each checked against its use |
| [`R/zzz.R`](https://moquedace.github.io/soilcnn/R/zzz.R) | `.onLoad()`: registers the built-in models, and fingerprints the code it loaded — every worker compares its own with it before it starts |

Beside `R/`:

| Where | What |
|----|----|
| [`DESCRIPTION`](https://moquedace.github.io/soilcnn/DESCRIPTION) · [`NAMESPACE`](https://moquedace.github.io/soilcnn/NAMESPACE) | The package, `soilcnn`: what it imports, and the 85 functions it exports — the `dsm_*()` front end, the resampling specs and fold constructors, the model registry, the importance methods, and the tools applied to results (AOA, conformal intervals, smearing, metrics, noise floor, occlusion, intervals by blocks). The runners underneath [`dsm_train()`](https://moquedace.github.io/soilcnn/reference/dsm_train.md), the patch store’s plumbing and the helpers are internal (`soilcnn:::`); [`pkgload::load_all()`](https://pkgload.r-lib.org/reference/load_all.html) on the source tree makes every function visible. NAMESPACE is what roxygen2 writes from the `@export` tags, and `tests/test_package_metadata.R` checks that it still is |
| [`tests/run_all.R`](https://moquedace.github.io/soilcnn/tests/run_all.R) | 35 files: 27 fast, then 8 slow ones that train, map, prepare a store, build and install the package, and run every example of the help pages. The package is loaded once for the suite. Every accumulator is named and `.report()` refuses an empty, unnamed, NA-bearing or non-logical one. `test_sources_parse.R` runs first and is the authority on syntax. |
| [`tests/testthat/`](https://moquedace.github.io/soilcnn/tests/testthat) | The fast subset `R CMD check` runs, CRAN’s too: metrics, folds, intervals, smearing, the AOA, blocks and a store of the example landscape – no network trained, no libtorch needed. The tarball holds it and none of the suite above |
| [`tools/check_package.R`](https://moquedace.github.io/soilcnn/tools/check_package.R) | Runs `R CMD check` on a staged copy of the package’s own files; `options(soilcnn.check_as_cran = TRUE)` first adds `--as-cran`, and with it the examples inside `\donttest{}`; `options(soilcnn.check_without_libtorch = TRUE)` checks as on CRAN’s machines, where torch has no backend; `options(soilcnn.check_manual = TRUE)` builds the PDF manual too, with TinyTeX, installing what LaTeX misses |
| [`docs/`](https://moquedace.github.io/soilcnn/docs/) | [`architecture.md`](https://moquedace.github.io/soilcnn/docs/architecture.md) (the network), [`design_decisions.md`](https://moquedace.github.io/soilcnn/docs/design_decisions.md) (the reason for each choice), [`tuning_guide.md`](https://moquedace.github.io/soilcnn/docs/tuning_guide.md) (the search space) |

------------------------------------------------------------------------

## Dependencies

The package’s own are `DESCRIPTION`’s Imports:

``` r

install.packages(c(
  "torch", "coro",                          # deep learning
  "dplyr", "readr", "tibble", "purrr",      # data wrangling
  "terra", "matrixStats", "janitor",        # rasters, ensemble aggregation, names
  "callr", "ps"                             # worker processes, and their memory
))
torch::install_torch()                      # ONCE: the C++ backend, ~200 MB
```

`install.packages("torch")` puts the R package in place; the backend is
a separate download that
[`library(torch)`](https://torch.mlverse.org/docs) asks for the first
time. Until `install_torch()` has run, nothing here trains.

Its Suggests, each needed only by the part that says so when called:
`sf` (points from a spatial file; kNNDM), `CAST` (kNNDM folds), `FNN` (a
faster exact nearest neighbour for the AOA), `randomForest` or `ranger`
(the RF baseline; ranger is far faster), `caret` (its model library),
`pkgload` (the source tree, loaded as it stands).

The test suite also needs `DescTools`, which
`tests/test_metrics_reporting.R` holds
[`ccc()`](https://moquedace.github.io/soilcnn/reference/ccc.md) against.

Everything here has run on a CPU.
[`dsm_train()`](https://moquedace.github.io/soilcnn/reference/dsm_train.md)
takes a torch device and can use a CUDA GPU;
[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)
and
[`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md)
train and map in worker processes on the CPU, several side by side, each
with a fixed number of threads
([`setup_torch_device()`](https://moquedace.github.io/soilcnn/reference/setup_torch_device.md)
and `n_cores = NULL` use the physical cores minus one).

### Loading it

``` r

pkgload::load_all("<clone>")                # the source tree, as it stands
library(soilcnn)                            # an installed copy
```

To install a copy from a clone, build the tarball first:

``` r

tgz <- pkgbuild::build("<clone>", dest_path = tempdir())
install.packages(tgz, repos = NULL, type = "source")
vignette("soilcnn")                         # the tour, built into the copy
```

Building the vignette needs knitr, rmarkdown and pandoc (RStudio ships
pandoc). Without them, `vignettes = FALSE` builds the package without
it.

`tests/test_package_install.R` builds, installs and loads a copy in a
fresh process; `tools/check_package.R` runs `R CMD check` on one. Both
build from a staged copy of the package’s own files, so whatever else
sits in a working directory is never read.

[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)
and
[`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md)
start worker processes, and each loads the framework the way its session
did — the source tree, or that same installed copy — and stops before
its first unit if the code there is not the code its session loaded.

A session that loaded the framework the old way, by
[`source()`](https://rdrr.io/r/base/source.html)ing `R/`, still holds
those copies in its global environment — RStudio’s *Restart R* keeps it
— and they would answer every call in place of the package. The package
refuses to attach while one is there, and says how to remove them:
`rm(list = soilcnn:::.pkg_stale_copies(), envir = globalenv())`.

------------------------------------------------------------------------

## Where it was developed

On a global model of soil organic carbon stock, 0–5 cm (t/ha): WoSIS
profiles and 181 raster predictors at 250 m, with 4,154 rows extracted,
3,766 surviving QC and 3,728 reaching the patch store. The numbers
quoted above come from it, unless they name the 0–30 cm trial.

It was then tested on a continental application, the one the vignette
follows: soil organic carbon stock at 0–30 cm over Latin America and the
Caribbean, 25,887 profiles and 174 predictors at 250 m, under five
validation designs that share one test set and one calibration set. The
vignette’s results come from a run on a tenth of the profiles (2,872),
in whole blocks. That project’s scripts, data and records are kept apart
from this repository, which holds the package alone.

------------------------------------------------------------------------

## Citation

``` r

citation("soilcnn")
```

> Moquedace CM, Baldi CGO (2026). *soilcnn: Convolutional Neural
> Networks for Digital Soil Mapping*. R package,
> <https://github.com/moquedace/soilcnn>.

------------------------------------------------------------------------

## License

MIT © 2026 Cássio Marques Moquedace and Clara Glória Oliveira Baldi. See
[`LICENSE.md`](https://moquedace.github.io/soilcnn/LICENSE.md).
