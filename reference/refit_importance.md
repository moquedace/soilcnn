# Refit importance: what the model loses when trained without a variable.

Leave one covariate out (LOCO; Lei et al. 2018, Williamson et al. 2023):
each variable's channels are set to their training mean – every pixel,
every window, every row – and the final run's seeds are trained again,
everything else as the run trained them: its split, scaling, schedule,
thread count, and the seed. The test skill lost is the variable's
importance. A permutation asks how much the fitted network relies on a
variable; this asks whether the variable is needed at all: variables
that carry the same information each cost little, because a network
trained without one finds it in the other.

## Usage

``` r
refit_importance(
  metric = c("ccc", "rmse", "rmse_transform"),
  check_seeds = 1L,
  run_id = NULL,
  resume = TRUE,
  output_dir = NULL,
  n_cores = NULL,
  max_ram_gb = NULL
)
```

## Arguments

- metric:

  What the table is ranked by: "ccc" (the drop in Lin's concordance,
  native units), "rmse" (the rise in RMSE, native units) or
  "rmse_transform" (in the space the model was trained in). All three
  are kept.

- check_seeds:

  How many of the seeds are trained again with nothing left out first,
  which must reproduce the run. 1 or more: a seed's numbers depend only
  on its seed and its thread count, so one checks the path.

- run_id:

  NULL for `refit_<timestamp>`. An existing one, with resume = TRUE,
  finishes a refit that stopped: its finished units are kept.

- resume:

  Keep the units an existing run_id already trained.

- output_dir:

  Where refits go. NULL: refit_importance/ in the final run's directory.

- n_cores:

  Cores for the refit; NULL is the physical cores minus one. Each unit
  trains with the final run's threads per unit – part of its numbers –
  so n_cores must be at least that.

- max_ram_gb:

  RAM the workers may use in total, as in
  [`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md).

## Value

An `importance_spec`, for
[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md).

## Details

Before any variable is left out, the first `check_seeds` seeds are
trained again with nothing left out, and must give the run's own
predictions – or the refit is not the run's training, and it stops. It
trains (variables x seeds + check_seeds) networks, side by side as
[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)
does: give
[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md)
themes as `groups`, and a few of the run's `seeds`.

## Examples

``` r
refit_importance(check_seeds = 1)
#> <importance_spec> Refit without each variable (LOCO), 1 seed(s) checked first | ranked by ccc
```
