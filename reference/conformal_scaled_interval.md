# Turn predictions into intervals with a fitted scale.

Turn predictions into intervals with a fitted scale.

## Usage

``` r
conformal_scaled_interval(cal, pred, covariates, lower_limit = -Inf)
```

## Arguments

- cal:

  From conformal_scaled_calibrate().

- pred:

  Predictions, native units.

- covariates:

  The same covariates the scale was fitted on, for these predictions –
  the level, and the dissimilarity index computed the same way as the
  calibration points' was.

- lower_limit:

  Floor of the lower bound (0 for a stock).

## Value

A tibble: pred, lower, upper and width, one row per prediction.

## Examples

``` r
set.seed(1)
level <- runif(300, 1, 4)
di    <- runif(300)                      # a dissimilarity index
obs   <- exp(level + rnorm(300, 0, 0.1 + 0.3 * di))
pred  <- exp(level)
covariates <- data.frame(level = pred, di = di)
cal <- conformal_scaled_calibrate(obs[1:200], pred[1:200], covariates[1:200, ],
                                  alpha = 0.1)
iv <- conformal_scaled_interval(cal, pred[201:300], covariates[201:300, ],
                                lower_limit = 0)
picp_report(obs[201:300], iv$lower, iv$upper)
#> 
#> Interval coverage (PICP)
#> ---------------------------------------------------------- 
#>   nominal   : 90%
#>   observed  : 86.0%  (100 point(s))
#>   width     : mean 24.74 | median 14.65
#> 
#>   -> UNDER-COVERAGE by 4.0 points. The map promises more
#>      certainty than it delivers, which is the dangerous direction.
```
