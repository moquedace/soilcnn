# Points of the map to explain: a regular grid of the model's raster cells.

The cells of the rasters the model was fitted on, every `every` rows and
columns, inside `extent`, where the first channel has a value. With
`every = 1` every cell – the map at full resolution, for a tile; with 40
at 250 m, one point every 10 km, for a region. The points carry their
grid, so
[`importance_map()`](https://moquedace.github.io/soilcnn/reference/importance_map.md)
can lay the values back on it.

## Usage

``` r
importance_points(final, data, extent = NULL, every = 1L, rasters = NULL)
```

## Arguments

- final:

  A `dsm_final`, or the directory of a final run.

- data:

  The `dsm_data` it was fitted on.

- extent:

  c(xmin, xmax, ymin, ymax) in the rasters' coordinates; NULL for the
  whole raster.

- every:

  Take every `every`-th cell, in rows and in columns.

- rasters:

  Where the rasters are, as in
  [`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md);
  NULL for the store's own raster table.

## Value

A tibble of `x`, `y` (cell centres), with the grid as an attribute.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
pts <- importance_points(run$final, run$data, every = 4)   # every 4th cell
nrow(pts)
# }
}
```
