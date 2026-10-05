# Spatially blocked folds of a point table.

The fold constructor behind spatial_cv(): the points are binned into
square blocks, whole blocks go to one fold, and the buffer then drops
the training points too close to a validation or test point.

## Usage

``` r
spatial_folds(
  meta,
  k = 5L,
  test_frac = 0,
  block_size = NULL,
  buffer = NULL,
  buffer_metric = c("chebyshev", "euclidean"),
  test_ids = NULL,
  seed = 42L,
  blocks_per_fold = 10L,
  calibration_frac = 0,
  calibration_ids = NULL
)
```

## Arguments

- meta:

  Point table: needs x, y and sample_id.

- k:

  Number of folds.

- test_frac:

  Share held out as the test set, drawn first.

- block_size:

  Side of a block, in the units of x/y. NULL sizes it from the extent,
  for `blocks_per_fold` blocks per fold.

- buffer:

  Distance within which a training point is dropped from a fold, in the
  units of x/y; NULL for none.

- buffer_metric:

  "chebyshev" (the default) or "euclidean". Chebyshev is exact for
  square patches: two patches share a pixel when their centres are
  within the window in both axes.

- test_ids:

  Sample ids that ARE the test set, in place of `test_frac`: a test set
  drawn again on every run is not a test set.

- seed:

  Seed for this split only; the training seeds do not move it.

- blocks_per_fold:

  Blocks per fold, for a `block_size` of NULL.

- calibration_frac:

  Share of ALL the points held out as a calibration set for the
  intervals, carved after the test set by the same criterion; 0 for
  none. It trains nothing and chooses nothing: dsm_final() predicts it
  and calibrates its "split" interval there.

- calibration_ids:

  Sample ids that ARE the calibration set, in place of
  `calibration_frac`, as `test_ids` is for the test set.

## Value

A `fold_plan` with `k` folds.

## Examples

``` r
ex <- example_landscape()
meta <- data.frame(sample_id = seq_len(nrow(ex$profiles)), x = ex$profiles$x,
                   y = ex$profiles$y)
# blocks of 0.05 degrees, and a buffer of the widest window (7 cells of 0.0025)
spatial_folds(meta, k = 3, test_frac = 0.2, block_size = 0.05, buffer = 7 * 0.0025)
#> <fold_plan> spatial_folds | 3 fold(s) | 160 rows in the store
#>   params: block_size=0.05 | block_size_auto=FALSE | n_blocks=15 | k=3 | test_frac=0.2 | n_test=45 | seed=42 | buffer=0.0175 | buffer_metric=chebyshev | buffer_protect=validation+test
#> # A tibble: 3 × 4
#>    fold n_train n_validation n_test
#>   <int>   <int>        <int>  <int>
#> 1     1      58           37     45
#> 2     2      53           37     45
#> 3     3      43           38     45
#>   buffer: 76 training point(s) dropped (33.1% per fold on average)
#>     near validation: 70 | near test: 6 (a point can be both)
#>     validation points dropped for being near test: 3
```
