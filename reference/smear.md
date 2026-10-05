# The conditional MEAN surface, from transform-space predictions.

The conditional MEAN surface, from transform-space predictions.

## Usage

``` r
smear(
  pred_transform,
  cal,
  lower_limit = 0,
  method = c("global", "level", "total")
)
```

## Arguments

- pred_transform:

  Predictions in transform space.

- cal:

  A smearing_cal, or a bare positive number to use as the factor.

- lower_limit:

  Floor, e.g. 0 for a stock. Applied after the correction.

- method:

  Which factor: "global" (the default; Duan's scalar), "level" (S as a
  function of the prediction, from bins of it, interpolated) or "total"
  (the exp(f)-weighted scalar, which unbiases a sum). The last two need
  a smearing_cal.
  [`smearing_check()`](https://moquedace.github.io/soilcnn/reference/smearing_check.md)
  measures them on held-out points.

## Value

Numeric, in native units: an estimate of `E[y | x]` rather than of its
median.

## Examples

``` r
set.seed(1)
f <- runif(500, 2, 5)                    # predictions, in log1p space
z <- f + rnorm(500, 0, 0.4)              # what was observed, in log1p space
cal <- smearing_factor(z, f)
expm1(c(2, 3, 4))                          # the median surface
#> [1]  6.389056 19.085537 53.598150
smear(c(2, 3, 4), cal)                     # the mean surface, by Duan's factor
#> [1]  6.969347 20.662930 57.885950
smear(c(2, 3, 4), cal, method = "level")   # by a factor that follows the level
#> [1]  6.853983 20.722794 58.135546
```
