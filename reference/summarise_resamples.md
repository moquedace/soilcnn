# Aggregate a unit-level comparison table into one row per config.

Aggregate a unit-level comparison table into one row per config.

## Usage

``` r
summarise_resamples(
  comparison,
  metrics = c("val_ccc", "val_mae", "val_rmse", "val_r2", "val_mqi", "test_ccc",
    "test_mae"),
  by = "config_id"
)
```

## Arguments

- comparison:

  Unit-level table from run_cnn_tuning()/run_cnn_resample().

- metrics:

  Metric column names to aggregate (any that exist).

- by:

  Grouping column; `config_id` unless you have a reason.

## Value

One row per config: n, mean, sd and se of each metric, ranked by the
first metric. Failed units are excluded from the statistics but counted
in `n_failed`, because a config that crashes 2 runs in 3 is not the same
as one that completed all three.

## Examples

``` r
# A tuning table: three configurations, each on three folds under two seeds
set.seed(1)
cfgs <- data.frame(config_id = c("cfg_001", "cfg_002", "cfg_003"),
                   skill = c(0.615, 0.62, 0.55), n_params = c(2e5, 2e6, 5e4))
cmp <- merge(expand.grid(config_id = cfgs$config_id, fold = 1:3, seed = 1:2,
                         stringsAsFactors = FALSE), cfgs)
cmp$status  <- "success"
cmp$val_ccc <- cmp$skill + rnorm(nrow(cmp), 0, 0.02)
summarise_resamples(cmp, metrics = "val_ccc")
#> # A tibble: 3 × 10
#>   config_id n_units n_folds n_seeds val_ccc_mean val_ccc_sd n_params val_ccc_se
#>   <chr>       <int>   <int>   <int>        <dbl>      <dbl>    <dbl>      <dbl>
#> 1 cfg_002         6       3       2        0.631     0.0117  2000000    0.00479
#> 2 cfg_001         6       3       2        0.614     0.0189   200000    0.00770
#> 3 cfg_003         6       3       2        0.547     0.0242    50000    0.00989
#> # ℹ 2 more variables: n_failed <int>, rank <int>
```
