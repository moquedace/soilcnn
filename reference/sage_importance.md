# SAGE: how much of the model's skill each variable carries, shared fairly.

Shapley Additive Global importancE (Covert, Lundberg & Lee 2020): the
Shapley values of the loss the model explains. The value of a set of
variables is the loss of the model's predictions when only they are
known – the others taken from background points of the model's own
training data, the whole patch, the same point in every window, as in
[`shap_importance()`](https://moquedace.github.io/soilcnn/reference/shap_importance.md)`("kernel")`.
A variable's SAGE value is the loss it takes away; variables that carry
the same information split it, where a permutation credits the shared
part to neither and a refit without one gives each nothing (the other
stands in). The values add up exactly to the loss of the mean prediction
– no variable known – less the model's own.

## Usage

``` r
sage_importance(
  loss = c("mse", "mae"),
  background = 16L,
  max_points = 200L,
  permutations = 64L,
  seed = 42L
)
```

## Arguments

- loss:

  "mse" (the default) or "mae", in the space the model was trained in
  (log1p for a log1p target).

- background:

  How many of the model's training points the absent variables are drawn
  from; each coalition costs one pass per point each.

- max_points:

  How many of the rows the loss is taken over, drawn by the seed – the
  kernel's default, so the two read the same points at the same cost.
  More points, a steadier loss, and a longer run.

- permutations:

  Above 14 variables: permutations sampled, in pairs.

- seed:

  Seed of the draws, shared by every model.

## Value

An `importance_spec`, for
[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md).

## Details

Exact over every coalition up to 14 variables, by sampled permutations
up to 40: meant for a few variables – give `groups` themes.

## Examples

``` r
sage_importance(background = 8, max_points = 100)
#> <importance_spec> SAGE, mse, 8 background point(s), at most 100 point(s)
```
