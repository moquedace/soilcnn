# SHAP importance: how each variable pushes each prediction, up or down.

Shapley values (Lundberg & Lee 2017) of the network's predictions,
estimated from its gradients. Each point's prediction, less a reference
prediction, is shared among the variables, with a sign: this one raised
it, that one lowered it. Averaged in absolute value over the points they
are a global importance – of how much each variable MOVES the
predictions, which is not how much it improves them
([`permutation_importance()`](https://moquedace.github.io/soilcnn/reference/permutation_importance.md)
measures that); kept point by point (`$points`) they are the map of what
drives the prediction where (Padarian et al. 2020; Wadoux et al. 2023
for soil carbon).

## Usage

``` r
shap_importance(
  estimator = c("expected_gradients", "integrated_gradients", "kernel"),
  samples = 100L,
  background = NULL,
  steps = 50L,
  permutations = 64L,
  max_points = NULL,
  seed = 42L
)
```

## Arguments

- estimator:

  "expected_gradients" (the default; Erion et al. 2021): the reference
  is a background of the model's own training points, drawn at random,
  so no baseline is chosen by hand – the values are SHAP values of the
  network, and the same estimator as SHAP's GradientExplainer.
  "integrated_gradients" (Sundararajan et al. 2017): one baseline, the
  training mean of every channel – the average landscape, flat, which no
  point is – kept because its sum is exact rather than sampled.
  "kernel": the Shapley values of the variables themselves, computed
  from coalitions, with no gradient (Lundberg & Lee 2017's KernelSHAP
  game): the value of a set of variables is the mean prediction, over a
  background of training points, with the variables outside it taken
  from the background point – the whole patch, the same point in every
  window. Exact over every coalition up to 14 variables; by sampled
  permutations above, up to 40. Meant for a few variables – give
  `groups` themes – and a few points. Read against the gradient
  estimators: they share the total and split it alike when the variables
  do not interact, so where they part is how much the variables do.

- samples:

  For expected gradients: draws per point, each a background point and a
  place on the straight path to it. The noise they leave is reported.

- background:

  How many of the model's training points the references are drawn from.
  NULL: 200 for expected gradients, 16 for the kernel – each coalition
  costs one pass per background point.

- steps:

  For integrated gradients: points on the path.

- permutations:

  For the kernel above 14 variables: permutations sampled (in pairs,
  each with its reverse).

- max_points:

  Explain this many of the rows only, drawn at random – the same rows
  for every estimator, so two can be compared point by point. NULL:
  every row for the gradient estimators, 200 for the kernel.

- seed:

  Seed of the draws, shared by every model.

## Value

An `importance_spec`, for
[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md).

## Details

The values are in the units the network predicts in: for a log1p target,
a SHAP value of +0.1 raises the predicted value by about 10%. Each
point's values add up to its prediction less the background's mean
prediction (expected gradients) or less the prediction at the baseline
(integrated gradients);
[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md)
checks that they do and stops if they do not. SHAP by deep-network rules
(DeepSHAP) is not offered: it needs a propagation rule written for every
layer of this architecture.

## Examples

``` r
shap_importance()
#> <importance_spec> SHAP by expected gradients, 100 sample(s) over 200 background point(s)
shap_importance("kernel", background = 8, max_points = 50)
#> <importance_spec> SHAP by kernel, Shapley over the variables, 8 background point(s), at most 50 point(s)
```
