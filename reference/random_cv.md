# Random k-fold. Ignores geography by construction.

Right when the rows really are independent, and the cleanest way to
MEASURE what geography is worth: run it against spatial_cv() on the same
points and the gap is the spatial optimism.

## Usage

``` r
random_cv(k = 5L, test_frac = 0.15, group = "auto", seed = 42L)
```

## Arguments

- k:

  Number of folds.

- test_frac:

  Share held out as the test set, drawn first.

- group:

  "auto" keeps the rows of one profile_id together when the table
  repeats profiles; NULL or "row" makes each row its own unit; a column
  name, or a vector with one label per row, names the groups.

- seed:

  Seed for the partition only (see with_local_seed): a fixed plan
  reproduces even when the training seeds change.

## Value

A `resample_spec`, as spatial_cv() returns.

## Examples

``` r
random_cv(k = 10)
#> <resample_spec> random
#>   k               10
#>   test_frac       0.15
#>   group           auto
#>   seed            42
```
