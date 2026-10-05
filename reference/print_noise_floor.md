# Print a noise-floor report in the terms it should be read in.

Print a noise-floor report in the terms it should be read in.

## Usage

``` r
print_noise_floor(nf, digits = 4L)
```

## Arguments

- nf:

  From seed_noise_floor().

- digits:

  Digits of the numbers.

## Value

`nf`, invisibly; NULL when no floor could be estimated.

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
print_noise_floor(seed_noise_floor(cmp, metric = "val_ccc"))
#>   Noise floor (val_ccc), with only the seed changing:
#>     median sd between seeds : 0.0125
#>     widest range observed   : 0.0668
#>     estimated over 9 (config x fold) combination(s)
#>     -> a gap between configs smaller than this is NOT evidence
```
