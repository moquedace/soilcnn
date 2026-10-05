# Cross-validated residuals from a tuning run, for calibration.

Every point appears once as validation, pooled over folds and averaged
over seeds, so the residual distribution covers the whole study area
instead of one fold's worth of it.

## Usage

``` r
cv_residuals(run_dir, config_id, role = "validation")
```

## Arguments

- run_dir:

  Tuning run directory (the one holding predictions/).

- config_id:

  Which config's predictions to read.

- role:

  Which role to keep. "validation" is the point of this.

## Value

A tibble with sample_id, obs, pred (the seed ensemble's median), and
n_seeds; or NULL when the run wrote no usable predictions.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
res <- cv_residuals(run$fit$run_dir, run$final$selected_config_ids)
head(res)
conformal_calibrate(res$obs, res$pred, alpha = 0.1)
# }
}
```
