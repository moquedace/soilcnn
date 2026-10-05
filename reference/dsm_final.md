# Refit the configuration a tuning run selects, under N seeds.

Refit the configuration a tuning run selects, under N seeds.

## Usage

``` r
dsm_final(
  tuning,
  data = NULL,
  config = "auto",
  rule = c("one_se", "rank1"),
  metric = "val_ccc",
  seeds = 10L,
  validation_frac = 0.15,
  predpoints = NULL,
  training = list(),
  transform = NULL,
  n_cores = NULL,
  threads_per_unit = 5L,
  max_ram_gb = NULL,
  conformal_alpha = c(0.1, 0.05),
  output_dir = NULL,
  run_id = NULL,
  resume = TRUE,
  verbose = TRUE
)
```

## Arguments

- tuning:

  A `dsm_fit` from dsm_train(), or the directory of a tuning run (it
  holds fold_plan.rds, tune_grid.rds and comparison/).

- data:

  From dsm_load(): the store the tuning ran on. Taken from the dsm_fit
  when `tuning` is one. Any windows may be loaded – the workers read the
  ones the selected configuration needs.

- config:

  "auto" applies `rule`; config id(s) choose by hand, and the frozen
  record then says "manual".

- rule:

  "one_se" (the simplest config within one standard error of the best)
  or "rank1" (the best mean).

- metric:

  The selection metric, as in the tuning table.

- seeds:

  A count (seeds 42, 43, ...) or the seeds themselves.

- validation_frac:

  Share of the non-test rows the refit stops on, carved by the tuning
  plan's own criterion (refit_split()): blocks, random rows, whole
  regions, or kNNDM against the same prediction points.

- predpoints:

  For a kNNDM tuning run made before its plan kept its prediction points
  (2026-09-28): those points, a data frame with x and y. NULL takes the
  plan's.

- training:

  Overrides of the refit schedule (.final_training_defaults) – any
  argument of train_one_cnn().

- transform:

  NULL for the store's own inverse (see dsm_train()).

- n_cores:

  Cores for the whole fit. NULL is the physical cores minus one.

- threads_per_unit:

  Threads each seed trains with. Part of the result (see the header), so
  it is recorded; 5 was measured best here (T1, T2).

- max_ram_gb:

  RAM the workers may use in total. NULL for 70% of what is available at
  the start (read with the ps package).

- conformal_alpha:

  Miscoverage levels of the intervals: 0.1 is 90%.

- output_dir:

  Where runs go. NULL for final_model/ beside the tuning run's
  directory.

- run_id:

  NULL for `final_<timestamp>`. Give an existing one with resume = TRUE
  to finish an interrupted fit.

- resume:

  Skip seeds whose checkpoint and record are already there. A resumed
  run is held to the settings it started with (run_spec.rds): another
  thread count, schedule, grid, split or scaling is refused, and so is a
  directory dsm_final() did not start.

- verbose:

  Report progress, and print the result.

## Value

A `dsm_final`, printed with the report.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
final <- dsm_final(run$fit, seeds = 2, n_cores = 1, threads_per_unit = 1,
                   training = list(n_epochs = 10, patience = 5), output_dir = tempdir(),
                   run_id = "final_example", verbose = FALSE)
final
# }
}
```
