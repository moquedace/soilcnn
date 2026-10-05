# Cross-validated residuals of a configuration, from any tuning run.

Cross-validated residuals of a configuration, from any tuning run.

## Usage

``` r
cv_residuals_for_config(run_dir, cfg_row, required = TRUE)
```

## Arguments

- run_dir:

  A tuning run directory (tune_grid.rds, predictions/).

- cfg_row:

  One row of a grid: the configuration, whatever it is called in that
  run.

- required:

  TRUE stops, saying why, when the run cannot serve the configuration;
  FALSE says why and returns NULL.

## Value

What cv_residuals() returns, with the run's own config_id attached as
attribute "config_id".

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
grid <- readRDS(file.path(run$fit$run_dir, "tune_grid.rds"))
cfg  <- grid[grid$config_id == run$final$selected_config_ids, ]
# found by its hyperparameters, so the same call reads any other tuning run
head(cv_residuals_for_config(run$fit$run_dir, cfg))
# }
}
```
