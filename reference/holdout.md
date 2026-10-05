# A single train/validation/test split of a point table.

The fold constructor behind holdout_cv(). Whole groups are drawn, never
single rows (see `group`), so the rows of one profile never straddle the
split.

## Usage

``` r
holdout(
  meta,
  validation_frac = 0.15,
  test_frac = 0.15,
  test_ids = NULL,
  seed = 42L,
  group = "auto",
  calibration_frac = 0,
  calibration_ids = NULL
)
```

## Arguments

- meta:

  Point table (the patch store's meta): needs sample_id, and profile_id
  for `group = "auto"` to have anything to do.

- validation_frac:

  Share of the non-test pool that validates.

- test_frac:

  Share held out as the test set, drawn first.

- test_ids:

  Sample ids that ARE the test set, in place of `test_frac`: a test set
  drawn again on every run is not a test set.

- seed:

  Seed for this split only; the training seeds do not move it.

- group:

  "auto" keeps the rows of one profile_id together when the table
  repeats profiles; NULL or "row" makes each row its own unit; a column
  name, or a vector with one label per row, names the groups.

- calibration_frac:

  Share of ALL the points held out as a calibration set for the
  intervals, carved after the test set by the same criterion; 0 for
  none. It trains nothing and chooses nothing: dsm_final() predicts it
  and calibrates its "split" interval there.

- calibration_ids:

  Sample ids that ARE the calibration set, in place of
  `calibration_frac`, as `test_ids` is for the test set.

## Value

A `fold_plan` with one fold.

## Examples

``` r
ex <- example_landscape()
meta <- data.frame(sample_id = seq_len(nrow(ex$profiles)), x = ex$profiles$x,
                   y = ex$profiles$y)
holdout(meta, validation_frac = 0.2, test_frac = 0.2)
#> <fold_plan> holdout | 1 fold(s) | 160 rows in the store
#>   params: k=1 | validation_frac=0.2 | test_frac=0.2 | n_test=32 | seed=42 | grouping=every row is its own unit (no profile_id column)
#> # A tibble: 1 × 4
#>    fold n_train n_validation n_test
#>   <int>   <int>        <int>  <int>
#> 1     1     103           25     32
# with a calibration set for the intervals beside the test set
holdout(meta, validation_frac = 0.2, test_frac = 0.2, calibration_frac = 0.15)
#> <fold_plan> holdout | 1 fold(s) | 160 rows in the store
#>   params: k=1 | validation_frac=0.2 | test_frac=0.2 | n_test=32 | calibration_frac=0.15 | n_calibration=24 | calibration_frozen=FALSE | seed=42 | grouping=every row is its own unit (no profile_id column)
#> # A tibble: 1 × 5
#>    fold n_train n_validation n_test n_calibration
#>   <int>   <int>        <int>  <int>         <int>
#> 1     1      84           20     32            24
#>   calibration set: 24 point(s) in no fold -- they calibrate the "split" interval and nothing else
```
