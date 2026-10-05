# Maps of SHAP values, from an importance computed at points of the map.

Lays the values of
[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md)`(at = )`
back on a grid: each cell the mean of the points in it. With the points'
own grid (from
[`importance_points()`](https://moquedace.github.io/soilcnn/reference/importance_points.md))
every point is its own cell; with a coarser `resolution`, each cell
averages the points it holds.

## Usage

``` r
importance_map(x, resolution = NULL, output_dir = NULL)
```

## Arguments

- x:

  A `dsm_importance` from
  [`shap_importance()`](https://moquedace.github.io/soilcnn/reference/shap_importance.md)
  at map points.

- resolution:

  Cell size of the maps, in the rasters' units; NULL for the points' own
  grid.

- output_dir:

  Where to write the GeoTIFFs (shap.tif, one layer per variable;
  shap_dominant.tif and its legend, shap_dominant.csv; prediction.tif);
  NULL to only return them.

## Value

An `importance_map`, a list: `shap` (one layer per variable: the mean
SHAP value of the points in each cell, in the network's units),
`dominant` (in each cell, the variable with the largest mean \|SHAP\|,
as the number in `legend`), `legend`, `prediction` (the mean prediction,
native units) and `files`. The three rasters are packed with
[`terra::wrap()`](https://rspatial.github.io/terra/reference/wrap.html),
so the map survives
[`base::saveRDS()`](https://rdrr.io/r/base/readRDS.html);
[`terra::unwrap()`](https://rspatial.github.io/terra/reference/wrap.html)
gives each as a `SpatRaster`.
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) draws it.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
pts <- importance_points(run$final, run$data, every = 4)
imp <- dsm_importance(run$final, run$data, shap_importance(samples = 10, background = 20),
                      at = pts, seeds = 42, verbose = FALSE)
m <- importance_map(imp)
plot(m)
# }
}
```
