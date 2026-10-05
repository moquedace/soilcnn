# Score every trained unit of a tuning run on the held-out test set.

Score every trained unit of a tuning run on the held-out test set.

## Usage

``` r
score_test_grid(
  run_dir,
  data,
  transform = NULL,
  device,
  config_ids = NULL,
  allow_unfrozen = FALSE
)
```

## Arguments

- run_dir:

  Tuning run directory.

- data:

  A dsm_data (from dsm_load()), or a list with store, points and
  type_table.

- transform:

  NULL (the default) for the inverse of the transform the store was
  built under, as in dsm_train(); a function is the inverse to use
  instead, and is refused if it disagrees with the store's.

- device:

  torch device.

- config_ids:

  Which configs to score. NULL means every config in the grid.

- allow_unfrozen:

  Escape hatch for teaching or for a run whose selection was recorded
  elsewhere. Not a default, and the report says it was used.

## Value

An object of class "test_optimism".

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
# every configuration on the test set, once the choice is frozen
score_test_grid(run$fit$run_dir, run$data, device = torch::torch_device("cpu"))
# }
}
```
