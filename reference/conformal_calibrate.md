# Calibrate a conformal interval from held-out residuals.

Calibrate a conformal interval from held-out residuals.

## Usage

``` r
conformal_calibrate(obs, pred, alpha = 0.1, difficulty = NULL)
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
```
