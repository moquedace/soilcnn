# Leave-region-out, on a grouping that already exists (biome, catchment, ...).

Leave-region-out, on a grouping that already exists (biome, catchment,
...).

## Usage

``` r
region_cv(group, k = NULL, test_frac = 0.15, seed = 42L, calibration_frac = 0)
```

## Arguments

- group:

  Group labels, one per row of `meta`.

- k:

  Number of folds; defaults to one per group.

- test_frac:

  Share held out as the test set, drawn first.

- seed:

  Seed for this split only; the training seeds do not move it.

- calibration_frac:

  Share of ALL the points held out as a calibration set for the
  intervals, carved after the test set by the same criterion; 0 for
  none. It trains nothing and chooses nothing: dsm_final() predicts it
  and calibrates its "split" interval there.

## Value

A `resample_spec`, as spatial_cv() returns.

## Examples

``` r
ex <- example_landscape()
region_cv(group = ex$profiles$survey)
#> <resample_spec> region
#>   group           <160 values>
#>   k               NULL
#>   test_frac       0.15
#>   seed            42
#>   calibration_frac 0
```
