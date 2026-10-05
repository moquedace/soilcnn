# Compute all regression metrics for one obs/pred pair.

Compute all regression metrics for one obs/pred pair.

## Usage

``` r
calc_metrics(obs, pred)
```

## Arguments

- obs:

  Numeric vector of observed values.

- pred:

  Numeric vector of predicted values (same length as obs).

## Value

A one-row tibble with columns: n, ccc, r2, mae, nse, rmse, rpd, mqi,
bias (signed, native units) and bias_pct (relative to mean(obs)).

## Examples

``` r
obs  <- c(12, 25, 31, 48, 60)
pred <- c(15, 22, 35, 40, 66)
calc_metrics(obs, pred)
#> # A tibble: 1 × 10
#>       n   ccc    r2   mae   nse  rmse   rpd   mqi  bias bias_pct
#>   <int> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>    <dbl>
#> 1     5 0.955 0.914   4.8 0.907  5.18  3.66  6.35   0.4     1.14
```
