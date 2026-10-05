# Turn a spec into a fold plan against real points.

Exported because a plan is worth looking at before spending a night on
it: `plan <- resolve_resampling(spatial_cv(k = 5), data); print(plan)`.

## Usage

``` r
resolve_resampling(
  spec,
  data,
  test_ids = NULL,
  windows = NULL,
  verbose = TRUE,
  calibration_ids = NULL
)
```

## Arguments

- spec:

  From spatial_cv() and friends, or an existing fold_plan, which is
  returned unchanged.

- data:

  From dsm_load().

- test_ids:

  Sample ids to force into the test set, so a frozen test set survives a
  change of method.

- windows:

  Windows the grid will use, for `buffer = "auto"`.

- verbose:

  Print the block size that `block_size = "auto"` chose.

- calibration_ids:

  Sample ids to force into the calibration set, in place of the
  `calibration_frac` of `spec`: several designs then calibrate their
  "split" intervals on the same points.

## Value

A `fold_plan`, already checked with check_fold_plan().

## Examples

``` r
ex <- example_landscape()
store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
                     windows = c(3, 7), out_dir = file.path(tempdir(), "landscape"),
                     percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
                     overwrite = TRUE, verbose = FALSE)
data <- dsm_load(store, windows = integer(0), verbose = FALSE)
resolve_resampling(spatial_cv(k = 3, block_size = 0.05, test_frac = 0.2), data,
                   windows = c(3, 7))
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
