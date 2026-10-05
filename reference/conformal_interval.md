# Turn predictions into intervals.

Turn predictions into intervals.

## Usage

``` r
conformal_interval(cal, pred, difficulty = NULL, lower_limit = -Inf)
```

## Arguments

- cal:

  A conformal_cal.

- pred:

  Predictions to wrap.

- difficulty:

  Difficulty scores for these predictions. Required when the calibration
  was normalised, and refused when it was not – mixing the two silently
  produces intervals with no guarantee at all.

- lower_limit:

  Floor for the lower bound, e.g. 0 for a stock. Clipping a bound at a
  physical limit can only INCREASE coverage, so the guarantee survives
  it.

## Value

A tibble: pred, lower, upper and width, one row per prediction.

## Examples

``` r
set.seed(1)
obs  <- rlnorm(300, 3, 0.4)
pred <- obs * exp(rnorm(300, 0, 0.25))
cal  <- conformal_calibrate(obs[1:150], pred[1:150], alpha = 0.1)
iv <- conformal_interval(cal, pred[151:300], lower_limit = 0)
head(iv)
#> # A tibble: 6 × 4
#>    pred lower upper width
#>   <dbl> <dbl> <dbl> <dbl>
#> 1 30.6  20.1   41.1  21.0
#> 2 17.1   6.65  27.6  21.0
#> 3 14.7   4.16  25.1  21.0
#> 4  9.39  0     19.9  19.9
#> 5  7.70  0     18.2  18.2
#> 6 13.3   2.76  23.7  21.0
```
