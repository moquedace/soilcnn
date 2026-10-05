# A single train/validation/test split.

A single train/validation/test split.

## Usage

``` r
holdout_cv(
  validation_frac = 0.15,
  test_frac = 0.15,
  group = "auto",
  seed = 42L,
  calibration_frac = 0
)
```

## Arguments

- validation_frac:

  Share of the non-test pool that validates.

- test_frac:

  Share held out as the test set, drawn first.

- group:

  "auto" keeps the rows of one profile_id together when the table
  repeats profiles; NULL or "row" makes each row its own unit; a column
  name, or a vector with one label per row, names the groups.

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
holdout_cv(validation_frac = 0.2)
#> <resample_spec> holdout
#>   validation_frac 0.2
#>   test_frac       0.15
#>   group           auto
#>   seed            42
#>   calibration_frac 0
```
