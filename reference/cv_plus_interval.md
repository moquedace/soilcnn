# Turn the fold models' predictions into CV+ intervals.

Turn the fold models' predictions into CV+ intervals.

## Usage

``` r
cv_plus_interval(cal, fold_pred, difficulty = NULL, lower_limit = -Inf)
```

## Arguments

- cal:

  A `cv_plus_cal`, from
  [`cv_plus_calibrate()`](https://moquedace.github.io/soilcnn/reference/cv_plus_calibrate.md).

- fold_pred:

  A matrix, one row per new point and one column per fold: that fold's
  model at the point (for a seed ensemble, the median of the fold's
  seeds). Columns named by fold are matched by name; unnamed ones are
  read in the order of `cal$folds`.

- difficulty:

  The new points' scale, for a normalised calibration; refused for a
  plain one, as in
  [`conformal_interval()`](https://moquedace.github.io/soilcnn/reference/conformal_interval.md).

- lower_limit:

  Floor of the lower bound, e.g. 0 for a stock.

## Value

A tibble: lower, upper and width, one row per new point.

## Examples

``` r
set.seed(1)
fold <- rep(1:4, 25)
oof  <- rnorm(100, 20, 3)              # each point's out-of-fold prediction
obs  <- oof + rnorm(100, 0, 2)
cal  <- cv_plus_calibrate(obs, oof, fold, alpha = 0.1)
# the four fold models at two new points: they disagree a little
cv_plus_interval(cal, rbind(c(19.5, 20.2, 20.0, 20.4), c(30.1, 29.0, 29.8, 30.6)),
                 lower_limit = 0)
#> # A tibble: 2 × 3
#>   lower upper width
#>   <dbl> <dbl> <dbl>
#> 1  16.7  23.3  6.66
#> 2  26.3  33.1  6.82
```
