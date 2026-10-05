# Paired comparison between two model families.

WHY PAIRED, AND WHY IT MATTERS HERE.

## Usage

``` r
paired_family_test(
  a,
  b,
  metric = "val_ccc",
  config_a = NULL,
  config_b = NULL,
  label_a = "a",
  label_b = "b",
  conf = 0.95
)
```

## Arguments

- a, b:

  Comparison tibbles (or the `comparison` element of a fit).

- metric:

  Column to compare. Higher-is-better is assumed for the verdict text
  unless the name matches a known error metric.

- config_a, config_b:

  Which config of each family to compare. Defaults to the best by mean
  `metric`, which is the config someone would deploy.

- label_a, label_b:

  Names for the report.

- conf:

  Interval level.

## Value

An object of class "paired_comparison".

## Details

Every family in stage 03b runs on the SAME fold plan with the SAME
seeds, by construction. So each family's units come in matched pairs:
rf_context on fold 2 seed 3 and the CNN on fold 2 seed 3 saw the
identical training rows.

Comparing the two MEANS and their separate spreads throws that away.
Most of the spread across units is the fold – some folds are simply
harder, and both families suffer on them together. A paired comparison
subtracts that shared difficulty out before asking whether anything is
left.

The first 03b run made exactly this mistake in the reader's favour: it
reported +0.0091 CCC against a seed noise floor of 0.0383 and concluded
"smaller than the noise". The conclusion happened to hold, but the
reasoning did not – the noise floor is the spread of ONE family across
seeds, which is not the standard error of a DIFFERENCE, and comparing a
difference to it can hide a real effect as easily as it can invent one.

What is reported is a mean difference with its own standard error and a
paired t interval. Not a p-value on its own: with 9 pairs, "not
significant" is mostly a statement about 9, and the interval says how
large an effect is still compatible with the data – which is the actual
question when the answer is "the convolution buys nothing".

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
# a second family, trained on the same folds under the same seeds
forest <- cmp
forest$config_id <- sub("cfg", "rf", forest$config_id)
forest$val_ccc <- forest$val_ccc - 0.03 + rnorm(nrow(forest), 0, 0.01)
paired_family_test(cmp, forest, metric = "val_ccc", config_a = "cfg_002",
                   config_b = "rf_002", label_a = "cnn", label_b = "forest")
#> 
#> Paired comparison -- val_ccc 
#> -------------------------------------------------------------- 
#>   cnn            cfg_002    mean = 0.6313
#>   forest         rf_002     mean = 0.5995
#>   paired on 6 (fold, seed) unit(s)
#> 
#>   difference   = +0.0319  (SE 0.0030)
#>   95% CI       = [+0.0241, +0.0397]
#>   t(5)        = +10.49   p = 0.000
#>   unpaired SE  = 0.0080  (pairing is worth 2.6x here)
#> 
#>   -> SEPARATED at 95%: cnn is ahead, by 0.0241 to 0.0397.
```
