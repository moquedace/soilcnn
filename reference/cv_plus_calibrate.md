# Calibrate a CV+ interval from the folds of a cross-validation.

Calibrate a CV+ interval from the folds of a cross-validation.

## Usage

``` r
cv_plus_calibrate(obs, pred, fold, alpha = 0.1, difficulty = NULL)
```

## Arguments

- obs:

  Observations of the calibration points: every point validated once, in
  one fold.

- pred:

  Each point's out-of-fold prediction – by the model of the fold that
  held it out (for a seed ensemble, the median of that fold's seeds).

- fold:

  The fold that held each point out.

- alpha:

  Miscoverage rate: 0.1 asks for 90%.

- difficulty:

  Optional positive scale of each point, for a normalised CV+: the
  residuals are divided by it, and
  [`cv_plus_interval()`](https://moquedace.github.io/soilcnn/reference/cv_plus_interval.md)
  multiplies them back by the new point's.

## Value

A `cv_plus_cal`: each fold's sorted scores, the rank, the folds.

## Examples

``` r
set.seed(1)
x <- runif(200)
fold <- rep(1:5, 40)
y <- 10 + 5 * x + rnorm(200)
# each fold's model: a line fitted without that fold
fits <- lapply(1:5, function(k) lm(y ~ x, data = data.frame(x, y)[fold != k, ]))
oof  <- vapply(seq_along(y), function(i)
  unname(predict(fits[[fold[i]]], data.frame(x = x[i]))), numeric(1))
cal <- cv_plus_calibrate(y, oof, fold, alpha = 0.1)
cal
#> 
#> <cv_plus_cal> 90% intervals
#>   calibration points : 200, in 5 fold(s) (40, 40, 40, 40, 40)
#>   rank used          : 181 of 200, both bounds (Barber et al. 2021)
# the five models at three new points, one column per fold
new <- data.frame(x = c(0.1, 0.5, 0.9))
cv_plus_interval(cal, sapply(fits, predict, newdata = new))
#> # A tibble: 3 × 3
#>   lower upper width
#>   <dbl> <dbl> <dbl>
#> 1  8.76  12.2  3.43
#> 2 10.8   14.2  3.36
#> 3 12.8   16.2  3.37
```
