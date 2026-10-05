# What each part of the patch is worth to one trained unit of a tuning run.

Hides a part of every patch – the centre pixel, each ring around it, the
whole context – and measures what the unit loses for it. The caller
names the unit and nothing more: the checkpoint is found, the
architecture rebuilt from the grid and the fold's cache from the plan.

## Usage

``` r
occlusion_report(
  run_dir,
  data,
  config_id,
  fold = 1L,
  seed_i = 1L,
  role = "validation",
  transform = NULL,
  device,
  ...
)
```

## Arguments

- run_dir:

  Tuning run directory.

- data:

  dsm_data, or a list with store, points, type_table.

- config_id:

  Which config.

- fold, seed_i:

  Which unit of it.

- role:

  Which split to measure on. Validation by default: the test set is
  frozen, and occlusion is a diagnostic, not a result.

- transform:

  NULL (the default) for the inverse of the transform the store was
  built under, as in dsm_train(); a function to use instead, refused if
  it disagrees with the store's.

- device:

  torch device.

- ...:

  `method`: "permute" (the default), each hidden pixel takes another
  point's value – real terrain, somebody else's – or "zero"; `seed`, of
  the permutation; `clamp`, the plausible range of the target, c(0, Inf)
  by default.

## Value

A `spatial_occlusion`: `table`, one row per part hidden, with the pixels
hidden, the CCC and MAE without them and the change in CCC, and
`baseline_ccc`, the unit's own. Printed with what it does and does not
mean.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
occlusion_report(run$fit$run_dir, run$data, config_id = run$final$selected_config_ids,
                 device = torch::torch_device("cpu"))
# }
}
```
