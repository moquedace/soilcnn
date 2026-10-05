# The smearing factor, from held-out residuals in transform space.

The smearing factor, from held-out residuals in transform space.

## Usage

``` r
smearing_factor(
  obs_transform,
  pred_transform,
  transform = c("log1p", "log"),
  bins = 5L
)
```

## Arguments

- obs_transform, pred_transform:

  Observed and predicted values IN TRANSFORM SPACE (log1p here), on data
  the model did not train on. Training residuals are smaller than the
  ones new points will produce, so a factor calibrated on them
  under-corrects by exactly the amount the model overfits. ONE ROW PER
  POINT: see smearing_from_run() for why seeds are collapsed first.

- transform:

  Name of the forward transform. Only "log1p" and "log" are supported,
  and an unknown one is refused rather than assumed: the algebra above
  is specific to an exponential back-transform, and applying it to a
  square root or an identity would scale a number that needs no scaling.

- bins:

  Bins of the prediction for the factor by level
  (`smear(method = "level")`), cut at its quantiles; each needs 50
  residuals.

## Value

An object of class "smearing_cal".

## Examples

``` r
set.seed(1)
f <- runif(500, 2, 5)                    # predictions, in log1p space
z <- f + rnorm(500, 0, 0.4)              # what was observed, in log1p space
cal <- smearing_factor(z, f)
cal
#> 
#> <smearing_cal> log1p back-transform
#>   held-out points    : 500
#>   mean, sd (log)     : -0.0135, 0.4226
#>   S = mean(exp(e))   : 1.0785
#>   lognormal check    : 1.0787  (ratio 1.000)
#>   S by prediction    : 1.063 1.074 1.124 1.078 1.054
#> 
#>   A prediction of 1.5 in log space becomes 3.48 (median) or 3.83 (mean).
```
