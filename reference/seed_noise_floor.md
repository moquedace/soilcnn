# How much does a metric move when ONLY the seed changes?

The noise floor of the whole experiment. Computed within (config, fold),
so nothing but the random draw differs between the runs being compared;
then summarised across all configs.

## Usage

``` r
seed_noise_floor(comparison, metric = "val_ccc")
```

## Arguments

- comparison:

  Unit-level comparison table.

- metric:

  Metric column to measure.

## Value

A list: `metric`; `by_config`, one row per (config, fold) trained under
more than one seed, with the mean, sd and range over the seeds;
`n_comparable`, how many such rows; `median_sd`, the floor; and
`max_range`, the widest spread seen.

## Details

Any difference between two configs smaller than this is not evidence.
That sentence is the entire reason the function exists, and it is why
`oneSE`-style selection is worth having: the best mean is frequently the
luckiest draw.

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
nf <- seed_noise_floor(cmp, metric = "val_ccc")
nf$median_sd
#> [1] 0.01246162
```
