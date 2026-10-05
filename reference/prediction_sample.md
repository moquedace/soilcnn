# A regular sample of the prediction area, as kNNDM's `predpoints`.

kNNDM needs the places the map will cover, not the places it was trained
on. A few thousand regularly spaced pixels describe that well enough –
the method compares DISTRIBUTIONS of distances, and a regular sample
estimates one cheaply. A random sample would work too and is noisier for
the same size.

## Usage

``` r
prediction_sample(raster, size = 5000L)
```

## Arguments

- raster:

  A SpatRaster (terra) covering the prediction area.

- size:

  About how many points to draw. Cells where the raster is NA – the sea
  around a continent – do not count: when too few land on data, the
  sample is drawn again, denser.

## Value

A data frame with x and y, in the raster's own CRS.

## Examples

``` r
ex <- example_landscape()
elevation <- terra::rast(file.path(ex$raster_dir, "elevation.tif"))
head(prediction_sample(elevation, size = 200))
#> # A tibble: 6 × 2
#>       x     y
#>   <dbl> <dbl>
#> 1 -49.6 -20.2
#> 2 -49.6 -20.2
#> 3 -49.6 -20.2
#> 4 -49.6 -20.2
#> 5 -49.5 -20.2
#> 6 -49.5 -20.2
```
