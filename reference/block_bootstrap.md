# A metric, or two models' difference in it, with an interval from whole blocks.

The points are drawn by blocks: each draw takes as many blocks as there
are, with replacement, and every point of a block drawn comes with it.
With `against`, the statistic is the difference `pred` less `against`,
at the same points. Two weightings answer two questions: "profile",
every point one vote; "block", every block one vote, its points weighing
1/n each. They differ in the estimate as well as in the interval, and
where they disagree the advantage rests on the densely sampled blocks.

## Usage

``` r
block_bootstrap(
  obs,
  pred,
  blocks,
  against = NULL,
  metric = c("mae", "rmse", "ccc", "bias"),
  weights = c("profile", "block"),
  n_boot = 2000L,
  conf = 0.95,
  seed = 42L
)
```

## Arguments

- obs:

  Observations.

- pred:

  Predictions at the same points.

- blocks:

  One block per point: from
  [`equal_area_blocks()`](https://moquedace.github.io/soilcnn/reference/equal_area_blocks.md),
  or the blocks the points were drawn by.

- against:

  NULL, or another model's predictions at the same points.

- metric:

  One or more of "mae", "rmse", "ccc" and "bias" (the mean of pred -
  obs).

- weights:

  "profile" and/or "block".

- n_boot:

  Draws of the blocks.

- conf:

  The interval's level.

- seed:

  Seed of the draws.

## Value

A `block_bootstrap`: a tibble, one row per metric and weighting, with
`estimate` (pred's), `against` (the other model's), `difference`,
`ci_low` and `ci_high` (of the difference when `against` is given, of
the estimate otherwise), `n_points`, `n_blocks`, `n_effective` (for
"profile": the number of independent points that would give the same
interval) and `share_better` (the share of points, or of blocks, where
`pred` errs less).

## Examples

``` r
ex <- example_landscape()
obs <- ex$profiles$soc_stock
set.seed(1)
pred_a <- obs * exp(rnorm(length(obs), 0, 0.20))   # two models' predictions
pred_b <- obs * exp(rnorm(length(obs), 0, 0.25))
blk <- equal_area_blocks(ex$profiles$x, ex$profiles$y, size_km = 2)
block_bootstrap(obs, pred_a, blk, against = pred_b, metric = "mae", n_boot = 500)
#> 
#> <block_bootstrap> 160 point(s) in 60 block(s), 500 draws of the blocks, 95% interval of the difference pred - against
#> # A tibble: 2 × 9
#>   metric weights estimate against difference ci_low ci_high n_effective
#>   <chr>  <chr>      <dbl>   <dbl>      <dbl>  <dbl>   <dbl>       <dbl>
#> 1 mae    profile     6.24    9.20      -2.96  -4.41  -1.70          160
#> 2 mae    block       6.66    9.42      -2.77  -5.03  -0.656          NA
#> # ℹ 1 more variable: share_better <dbl>
#>   profile: every point one vote; the interval from whole blocks drawn with replacement.
#>   block: every block one vote, its points 1/n each -- the estimate moves as well.
#>   n_effective: how many independent points would give the profile interval.
#>   difference: for mae and rmse below zero is pred erring less, for ccc above zero is
#>   pred agreeing more; an interval holding zero is not evidence either way.
```
