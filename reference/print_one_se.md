# Say what one_se() did, including when it did nothing.

Say what one_se() did, including when it did nothing.

## Usage

``` r
print_one_se(pick, metric = "val_ccc", digits = 4L)
```

## Arguments

- pick:

  From one_se().

- metric:

  The metric it selected on, for the report.

- digits:

  Digits of the threshold.

## Value

`pick`, invisibly.

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
print_one_se(one_se(summarise_resamples(cmp, metrics = "val_ccc"), metric = "val_ccc"))
#>   one_se (val_ccc): 1 config(s) within one standard error of the best (threshold 0.6265)
#>     chosen: cfg_002  -- same as the top-ranked config; the rule changed nothing
```
