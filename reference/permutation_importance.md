# Permutation importance: how much a fitted model relies on each variable.

Each point takes one variable's whole patch – every pixel, every window
– from another point; the model predicts again, and the loss of skill is
the variable's importance to that model. Nothing is retrained.

## Usage

``` r
permutation_importance(
  draws = 5L,
  within = NULL,
  fill = c("permute", "mean"),
  metric = c("ccc", "rmse", "rmse_transform"),
  seed = 42L
)
```

## Arguments

- draws:

  Permutations per variable and per model. The importance is their mean;
  their spread is reported beside the spread between models.

- within:

  Where a donor may come from. NULL (the default): any point of the rows
  scored. A number: square blocks of that side, in the coordinate units
  of the store's `x` and `y` – the importance beyond the region. A
  column name of the point table (an ecoregion, a soil region): the same
  class. Or a vector with one label per point of the store. A point
  alone in its block or class keeps its own values; how many did is
  reported.

- fill:

  "permute" (the default), or "mean": the variable is set to its
  training mean everywhere in the patch – an ablation, which puts the
  patch off the data (see Details) and is read against the permutation.

- metric:

  What the table is ranked by: "ccc" (the drop in Lin's concordance,
  native units), "rmse" (the rise in RMSE, native units) or
  "rmse_transform" (the rise in RMSE in the space the model was trained
  in, e.g. log1p). All three are computed and kept.

- seed:

  Seed of the permutations. The same draws are given to every variable
  and every model.

## Value

An `importance_spec`, for
[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md).

## Details

A variable whose permutation costs nothing is one the model does not
use. It may still carry information other variables also carry:
permutation measures reliance, not necessity, and a model that can find
the same thing elsewhere leans on whichever it found first.

The mean of a z-scored channel is 0 and of a dummy its class frequency,
so `fill = "mean"` gives every point the average landscape, flat. A
large drop under it with a small one under permutation says the model
reacts to the input being unusual rather than to its content.

## Examples

``` r
permutation_importance()
#> <importance_spec> permutation over all rows, 5 draw(s) | ranked by ccc
permutation_importance(draws = 3, within = 0.05)   # donors from the same 0.05 block
#> <importance_spec> permutation within blocks of 0.05, 3 draw(s) | ranked by ccc
```
