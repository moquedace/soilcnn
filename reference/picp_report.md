# Coverage, and whether it holds where it is needed.

Coverage, and whether it holds where it is needed.

## Usage

``` r
picp_report(obs, lower, upper, group = NULL, alpha = 0.1)
```

## Arguments

- obs, lower, upper:

  As in picp().

- group:

  Optional grouping (a spatial block, a region, a soil class).

- alpha:

  The nominal miscoverage, for the verdict.

## Value

An object of class "picp_report".

## Examples

``` r
set.seed(1)
obs  <- rlnorm(300, 3, 0.4)
pred <- obs * exp(rnorm(300, 0, 0.25))
cal  <- conformal_calibrate(obs[1:150], pred[1:150], alpha = 0.1)
iv <- conformal_interval(cal, pred[151:300], lower_limit = 0)
# one width for every level: right on average, and by level?
picp_report(obs[151:300], iv$lower, iv$upper, alpha = 0.1,
            group = ifelse(pred[151:300] > median(pred), "high", "low"))
#> 
#> Interval coverage (PICP)
#> ---------------------------------------------------------- 
#>   nominal   : 90%
#>   observed  : 90.7%  (150 point(s))
#>   width     : mean 20.81 | median 20.99
#> 
#>   -> Coverage matches the promise. If these are points the
#>      calibration did not see, the uncertainty is calibrated.
#> 
#>   By group (worst coverage first):
#> # A tibble: 2 × 4
#>   group     n  picp mean_width
#>   <chr> <int> <dbl>      <dbl>
#> 1 high     78 0.821       21.0
#> 2 low      72 1           20.6
```
