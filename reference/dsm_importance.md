# Variable importance of a fitted model, by the method you choose.

The method is an argument, as the validation design is in
[`dsm_train()`](https://moquedace.github.io/soilcnn/reference/dsm_train.md):
[`permutation_importance()`](https://moquedace.github.io/soilcnn/reference/permutation_importance.md)
measures how much the model relies on each variable, over all rows or
within blocks or classes;
[`context_importance()`](https://moquedace.github.io/soilcnn/reference/context_importance.md)
how far from the point, and through which window, it reads;
[`shap_importance()`](https://moquedace.github.io/soilcnn/reference/shap_importance.md)
how each variable pushes each prediction, up or down;
[`sage_importance()`](https://moquedace.github.io/soilcnn/reference/sage_importance.md)
how much of the model's skill each variable carries, shared fairly among
variables that carry the same information;
[`refit_importance()`](https://moquedace.github.io/soilcnn/reference/refit_importance.md)
how much skill a network trained without the variable loses;
[`ale_effect()`](https://moquedace.github.io/soilcnn/reference/ale_effect.md)
how the prediction changes along each variable's range. Each variable is
a channel, or the channels of one categorical (see
[`importance_groups()`](https://moquedace.github.io/soilcnn/reference/importance_groups.md)),
or a group of yours.

## Usage

``` r
dsm_importance(
  final,
  data,
  method = permutation_importance(),
  rows = c("test", "folds"),
  groups = "auto",
  config = NULL,
  threads = NULL,
  batch_size = 512L,
  verbose = TRUE,
  at = NULL,
  rasters = NULL,
  seeds = NULL,
  chunk_points = 5000L
)
```

## Arguments

- final:

  A `dsm_final`, or the directory of a final run.

- data:

  The `dsm_data` the run was fitted on, from
  [`dsm_load()`](https://moquedace.github.io/soilcnn/reference/dsm_load.md).
  Its windows need not be loaded: one that is not is read from the
  store.

- method:

  An importance method:
  [`permutation_importance()`](https://moquedace.github.io/soilcnn/reference/permutation_importance.md),
  [`context_importance()`](https://moquedace.github.io/soilcnn/reference/context_importance.md),
  [`shap_importance()`](https://moquedace.github.io/soilcnn/reference/shap_importance.md),
  [`sage_importance()`](https://moquedace.github.io/soilcnn/reference/sage_importance.md),
  [`refit_importance()`](https://moquedace.github.io/soilcnn/reference/refit_importance.md)
  or
  [`ale_effect()`](https://moquedace.github.io/soilcnn/reference/ale_effect.md).

- rows:

  "test" (the default): the final run's seeds, on the test set the whole
  run held out – the importance of the model that draws the map.
  "folds": the tuning run's models of the same configuration, each on
  its own fold's validation rows – the importance as that validation
  design sees it; run it for each design to compare them.

- groups:

  Which channels form one variable: "auto", "channel", or your grouping,
  as in
  [`importance_groups()`](https://moquedace.github.io/soilcnn/reference/importance_groups.md).

- config:

  Which of the final run's configurations, when it fitted more than one.
  NULL: the selected one.

- threads:

  torch threads for this session; NULL leaves them as set.

- batch_size:

  Points per forward pass.

- verbose:

  Say what is done, model by model.

- at:

  Points of the map instead of rows of the store: a data frame of `x`,
  `y`, from
  [`importance_points()`](https://moquedace.github.io/soilcnn/reference/importance_points.md)
  or of your own. Their patches are cut from the rasters as the
  profiles' were, and only
  [`shap_importance()`](https://moquedace.github.io/soilcnn/reference/shap_importance.md)
  runs there – a map point has no observation to score against. The
  values become maps with
  [`importance_map()`](https://moquedace.github.io/soilcnn/reference/importance_map.md).

- rasters:

  With `at`: where the rasters are, as in
  [`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md);
  NULL for the store's own raster table.

- seeds:

  Which of the final run's seeds to use; NULL for all. A map of many
  points may take a few: each seed is one model explained.

- chunk_points:

  With `at`: points cut and explained at a time. Their patches are held
  in memory together.

## Value

A `dsm_importance`: `table` (one row per target – a variable, a band, a
variable at a band or a window – ranked), `by_model` (one row per target
and model), `baseline` (each model's unperturbed score), `groups`,
`targets` (what each target is), `gate` (by window, for a model with a
gate: the gate point by point), `units` (the models scored) and the
settings. For SHAP: `table` (mean \|SHAP\|, its share and direction per
variable), `points` (every point's values), `patch` and `pixels` (where
in the patch they sit) and `completeness` (how well they add up). For
ALE: `table` (the spread, trend and extremes of each effect), `curves`
(each continuous variable's curve, with its spread between models) and
`classes` (each categorical's class effects). For SAGE: `table` (the
loss each variable takes away, and its share), `by_model` and `losses`
(each model's loss, and the mean prediction's, which SAGE splits). For a
refit: `table`, `by_model`, `raw` (each unit's scores), `check` (the
seeds trained again with nothing left out, against the run), `noise`
(what a refit moves a score by with nothing to lose) and `refit_dir`.

## Details

Before anything is perturbed, each model's own score is computed again
and must reproduce the one its run wrote; if it does not, nothing is
measured.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
imp <- dsm_importance(run$final, run$data, permutation_importance(draws = 2),
                      verbose = FALSE)
imp
plot(imp)
# }
}
```
