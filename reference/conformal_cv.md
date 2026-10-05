# Calibrate on the folds of a resampling run, then report coverage.

The natural calibration set in this framework is the VALIDATION side of
each fold: it is held out of training, it is separated spatially with a
buffer, and it already exists. This walks the per-unit predictions a run
wrote, calibrates on one fold and measures coverage on the others, so
the reported PICP is always out-of-calibration.

## Usage

``` r
conformal_cv(pred_obs, alpha = 0.1, difficulty = NULL, group = NULL)
```

## Arguments

- pred_obs:

  A table with columns fold, obs, pred (as the runners write).

- alpha:

  Miscoverage rate.

- difficulty:

  Optional column name holding a difficulty score.

- group:

  Optional column name to break coverage down by.

## Value

A picp_report over the pooled out-of-calibration points.

## Examples

``` r
set.seed(1)
pred_obs <- data.frame(fold = rep(1:5, each = 60), obs = rlnorm(300, 3, 0.4))
pred_obs$pred <- pred_obs$obs * exp(rnorm(300, 0, 0.25))
conformal_cv(pred_obs, alpha = 0.1)
#> 
#> Interval coverage (PICP)
#> ---------------------------------------------------------- 
#>   nominal   : 90%
#>   observed  : 90.3%  (300 point(s))
#>   width     : mean 20.54 | median 20.33
#> 
#>   -> Coverage matches the promise. If these are points the
#>      calibration did not see, the uncertainty is calibrated.
```
