# Calibrate a conformal interval from held-out residuals.

Calibrate a conformal interval from held-out residuals.

## Usage

``` r
conformal_calibrate(obs, pred, alpha = 0.1, difficulty = NULL, group = NULL)
```

## Arguments

- obs, pred:

  Observations and predictions on the CALIBRATION set – data the model
  did not train on. Using training residuals produces intervals that are
  too narrow by exactly the amount the model overfits, and the result
  looks like a much better map.

- alpha:

  Miscoverage rate: 0.1 asks for 90% coverage.

- difficulty:

  Optional per-point difficulty score (e.g. the ensemble spread). When
  given, intervals scale with it. Must be positive.

- group:

  Optional group of each point – a spatial block, a region, a profile.
  Given, every group weighs the same in the quantile, whatever its
  number of points: q is the smallest score at which the mean of the
  groups' empirical distributions reaches (1 - alpha)(1 + 1/m), m groups
  (Dunn, Wasserman & Ramdas 2023). The interval then speaks for a new
  point of a new group rather than for a point drawn from the pooled
  sample.

## Value

An object of class "conformal_cal".

## Examples

``` r
set.seed(1)
obs  <- rlnorm(300, 3, 0.4)
pred <- obs * exp(rnorm(300, 0, 0.25))
cal  <- conformal_calibrate(obs[1:150], pred[1:150], alpha = 0.1)
cal
#> 
#> <conformal_cal> 90% intervals (constant width)
#>   calibration points : 150
#>   rank used          : 136 of 150  (the (n+1) correction)
#>   q                  : 10.4933
# the same residuals, every one of 30 groups weighing alike
conformal_calibrate(obs[1:150], pred[1:150], alpha = 0.1, group = rep(1:30, 5))
#> 
#> <conformal_cal> 90% intervals (constant width)
#>   calibration points : 150
#>   weighting          : by group -- 30 group(s), each weighing the same
#>   level              : the groups' mean distribution at 0.9300  (the (1 + 1/m) correction)
#>   q                  : 11.7362
```
