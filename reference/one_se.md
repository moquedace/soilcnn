# Pick the simplest config within one standard error of the best.

Pick the simplest config within one standard error of the best.

## Usage

``` r
one_se(by_config, metric = "val_ccc", complexity = "n_params", maximise = NULL)
```

## Arguments

- by_config:

  From summarise_resamples().

- metric:

  Metric column stem, e.g. "val_ccc".

- complexity:

  Column holding the simplicity ordering (lower = simpler), or a numeric
  vector the same length. Defaults to `n_params`, which
  count_model_params() produces.

- maximise:

  TRUE when higher is better (CCC, R2); FALSE for an error.

## Value

One row of `by_config`, with `within_one_se` (how many configs were
tied) and `simpler_than_best` (whether the rule actually moved the
choice) attached – a selection rule that silently returns the same
answer as the default should say so.

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
pick <- one_se(summarise_resamples(cmp, metrics = "val_ccc"), metric = "val_ccc")
pick$config_id
#> [1] "cfg_002"
```
