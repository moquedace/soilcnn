# The smearing factors on held-out points: which one unbiases the mean surface.

Each factor of
[`smear()`](https://moquedace.github.io/soilcnn/reference/smear.md)
applied to held-out predictions – points none of them was calibrated on,
a test set – against what was observed there: the bias of the total,
which a total taken from the map inherits, and the bias within quintiles
of the prediction, where a factor by level should help. The median
surface, with no factor, is the first row. On the global SOC model,
calibrated on three folds, the global factor won (+2.7% against -3.9% by
level and -5.8% for a total), because those folds saw half the points
the deployed model saw; this measures it again for a run of yours.

## Usage

``` r
smearing_check(
  cal,
  obs_transform,
  pred_transform,
  methods = c("global", "level", "total"),
  lower_limit = 0
)
```

## Arguments

- cal:

  A smearing_cal, from
  [`smearing_factor()`](https://moquedace.github.io/soilcnn/reference/smearing_factor.md)
  or
  [`smearing_from_run()`](https://moquedace.github.io/soilcnn/reference/smearing_from_run.md).

- obs_transform, pred_transform:

  Observed and predicted at held-out points, in transform space.

- methods:

  The factors to compare.

- lower_limit:

  As in
  [`smear()`](https://moquedace.github.io/soilcnn/reference/smear.md).

## Value

A `smearing_check`: a tibble, one row per surface, with `bias_pct` (its
total against the observed total, in %), `mae`, `rmse`, and
`bias_pct_q1` to `bias_pct_q5` (within quintiles of the prediction, low
to high).

## Examples

``` r
set.seed(1)
f <- runif(1000, 2, 5)
z <- f + rnorm(1000, 0, 0.2 + 0.1 * (f - 2))   # the spread grows with the level
cal <- smearing_factor(z[1:500], f[1:500])
smearing_check(cal, z[501:1000], f[501:1000])  # on points it was not calibrated on
#> 
#> <smearing_check> 500 held-out point(s): each surface against what was observed
#> # A tibble: 4 × 9
#>   surface bias_pct   mae  rmse bias_pct_q1 bias_pct_q2 bias_pct_q3 bias_pct_q4
#>   <chr>      <dbl> <dbl> <dbl>       <dbl>       <dbl>       <dbl>       <dbl>
#> 1 median     -6.01  16.6  31.6       -5.31        0.13      -16.4        -0.58
#> 2 global      0.03  17.0  31.4        1.31        6.78      -10.9         5.77
#> 3 level       3.75  17.5  32.0       -7.44        7         -10.5         3.35
#> 4 total       4.08  17.4  31.6        5.75       11.2        -7.29       10.0 
#> # ℹ 1 more variable: bias_pct_q5 <dbl>
#>   bias_pct: the surface's total against the observed total, in % -- what a total
#>   taken from the map inherits. bias_pct_q1..: the same within quintiles of the
#>   prediction, low to high, where a factor by level should help. median: no factor.
```
