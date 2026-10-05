# Fold plan from k-fold Nearest Neighbour Distance Matching.

Fold plan from k-fold Nearest Neighbour Distance Matching.

## Usage

``` r
knndm_folds(
  meta,
  k = 5L,
  predpoints = NULL,
  test_ids = NULL,
  hold_out_test = FALSE,
  crs = 4326,
  project_to = "+proj=moll +lon_0=0 +datum=WGS84 +units=m",
  seed = 42L,
  ...
)
```

## Arguments

- meta:

  Store metadata: needs x, y and sample_id.

- k:

  Number of folds.

- predpoints:

  Where the map will be predicted: a data frame with x and y, or an sf
  object. REQUIRED unless `modeldomain` is passed through `...` – the
  method has no meaning without it, so it is not defaulted.

- test_ids:

  Frozen test sample_ids. Preferred over `hold_out_test` whenever
  several plans are to be compared on the same held-out data.

- hold_out_test:

  Carve a test set by running kNNDM with k + 1 folds and holding one
  out. FALSE by default: a test set that changes with k is not a frozen
  test set.

- crs:

  CRS of `meta$x`/`meta$y`. 4326 by default.

- project_to:

  Projection used for the distance comparison, or NULL to use the
  coordinates as they are (correct only if already projected).

- seed:

  Seed, for the sampling kNNDM does internally.

- ...:

  Passed to CAST::knndm() – `maxp`, `clustering`, `samplesize`,
  `modeldomain`, `space`.

## Value

A fold_plan.

## Examples

``` r
ex <- example_landscape()
meta <- data.frame(sample_id = seq_len(nrow(ex$profiles)), x = ex$profiles$x,
                   y = ex$profiles$y)
elevation <- terra::rast(file.path(ex$raster_dir, "elevation.tif"))
knndm_folds(meta, k = 3, predpoints = prediction_sample(elevation, size = 500))
#> <fold_plan> knndm_folds | 3 fold(s) | 160 rows in the store
#>   params: k=3 | test_frac=0 | n_test=0 | seed=42 | W=138.6 | W_test_split=NA | projection=+proj=moll +lon_0=0 +datum=WGS84 +units=m
#> # A tibble: 3 × 4
#>    fold n_train n_validation n_test
#>   <int>   <int>        <int>  <int>
#> 1     1      95           65      0
#> 2     2     105           55      0
#> 3     3     120           40      0
```
