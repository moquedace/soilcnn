# The mean surface of a median map already made.

The conditional mean is `exp(z) S(z) - 1` of the median `z` in transform
space, so a mean map by any of
[`smear()`](https://moquedace.github.io/soilcnn/reference/smear.md)'s
factors comes from the median map alone, cell by cell, without running
the network again.

## Usage

``` r
smear_map(
  median,
  cal,
  method = c("global", "level", "total"),
  lower_limit = 0,
  filename = ""
)
```

## Arguments

- median:

  The ensemble median, in native units: a SpatRaster, or the path of one
  –
  [`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md)'s
  `ensemble_median` band.

- cal:

  A smearing_cal, from
  [`smearing_factor()`](https://moquedace.github.io/soilcnn/reference/smearing_factor.md)
  or
  [`smearing_from_run()`](https://moquedace.github.io/soilcnn/reference/smearing_from_run.md).

- method:

  As in
  [`smear()`](https://moquedace.github.io/soilcnn/reference/smear.md).

- lower_limit:

  As in
  [`smear()`](https://moquedace.github.io/soilcnn/reference/smear.md).

- filename:

  Where to write the map; "" keeps it in memory or in terra's temporary
  files.

## Value

A SpatRaster: the conditional mean, cell by cell.

## Examples

``` r
set.seed(1)
f <- runif(500, 2, 5)                    # predictions, in log1p space
z <- f + rnorm(500, 0, 0.4)              # what was observed, in log1p space
cal <- smearing_factor(z, f)
median_map <- terra::rast(nrows = 10, ncols = 10, vals = expm1(runif(100, 2, 5)))
smear_map(median_map, cal)
#> class       : SpatRaster
#> size        : 10, 10, 1  (nrow, ncol, nlyr)
#> resolution  : 36, 18  (x, y)
#> extent      : -180, 180, -90, 90  (xmin, xmax, ymin, ymax)
#> coord. ref. : lon/lat WGS 84 (CRS84) (OGC:CRS84)
#> source(s)   : memory
#> name        :      lyr.1
#> min value   :   7.048745
#> max value   : 151.717409
```
