# Folds matched to where the map will be predicted (kNNDM).

The better-founded alternative to spatial_cv(). Blocks need a block size
and a buffer width, and nothing in the data says whether the chosen ones
were right. kNNDM instead shapes the folds so the distance from a
validation point to its nearest training point is distributed like the
distance from a PREDICTION pixel to its nearest training point.

## Usage

``` r
knndm_cv(
  k = 5L,
  predpoints = NULL,
  hold_out_test = FALSE,
  crs = 4326,
  project_to = "+proj=moll +lon_0=0 +datum=WGS84 +units=m",
  seed = 42L,
  hold_out_calibration = FALSE,
  ...
)
```

## Arguments

- k:

  Number of folds.

- predpoints:

  Where the map will be predicted: a data frame with x and y, or an sf
  object. REQUIRED unless `modeldomain` is passed through `...` – the
  method has no meaning without it, so it is not defaulted.

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

- hold_out_calibration:

  Carve a calibration set for the intervals as the test set is carved:
  kNNDM in k + 1 folds over what the test left, one held out. FALSE by
  default.

- ...:

  Passed to CAST::knndm() – `maxp`, `clustering`, `samplesize`,
  `modeldomain`, `space`.

## Value

A `resample_spec`, as spatial_cv() returns.

## Details

It therefore needs `predpoints`: a sample of where the map will be
drawn. There is no default, because a default would turn the method into
an expensive random split. prediction_sample(raster) produces one.

When the samples are well spread over the prediction area it converges
by itself to ordinary random k-fold – it does not impose separation that
prediction will not face. That is the property blocks cannot have.

Needs the CAST and sf packages. See R/knndm.R for the projection
question, which is not optional on lon/lat data.

## Examples

``` r
ex <- example_landscape()
elevation <- terra::rast(file.path(ex$raster_dir, "elevation.tif"))
knndm_cv(k = 5, predpoints = prediction_sample(elevation, size = 500))
#> <resample_spec> knndm
#>   k               5
#>   predpoints      <528 rows of x, y>
#>   hold_out_test   FALSE
#>   crs             4326
#>   project_to      +proj=moll +lon_0=0 +datum=WGS84 +units=m
#>   seed            42
#>   hold_out_calibration FALSE
#>   extra           
```
