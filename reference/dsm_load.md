# Load everything a run needs, and check that it fits together.

Replaces the six-step preamble every pipeline script used to repeat:
open the store, read the points, read the predictor types, align the
points to the store, read the raster resolution, and verify the store
can serve this configuration.

## Usage

``` r
dsm_load(
  patch_dir,
  points = NULL,
  type_table = NULL,
  windows = NULL,
  cell_size = NULL,
  raster_table = NULL,
  target_col = NULL,
  verbose = TRUE
)
```

## Arguments

- patch_dir:

  Directory written by the extraction step – or the `dsm_store`
  dsm_prepare() returned, in which case nothing else is needed.

- points:

  Point table, or a path to the CSV written by stage 01. NULL for a
  store written by dsm_prepare(), which carries its own copy.

- type_table:

  Predictor types, or a path to its CSV. NULL as above.

- windows:

  Windows to load. NULL loads every window the store has, which is the
  wrong default when the grid needs two of five – pass the windows the
  grid actually uses and the store reads only those. integer(0) loads
  none: the store's table alone, from which a fold cache reads each
  window it needs from its file (dsm_final()'s workers do).

- cell_size:

  Raster resolution in x/y units. NULL reads it from `raster_table`,
  which is the only source that cannot drift.

- raster_table:

  Path to raster_table_used.csv, for `cell_size`.

- target_col:

  Expected target column, for the store lock.

- verbose:

  Print the loaded data.

## Value

A `dsm_data` object. `$transform` is the target transform the store was
built under – name, forward and inverse – or NULL when the store did not
record one.

## Examples

``` r
ex <- example_landscape()
store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
                     windows = c(3, 7), out_dir = file.path(tempdir(), "landscape"),
                     percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
                     overwrite = TRUE, verbose = FALSE)
dsm_load(store, windows = integer(0))   # the table alone: loading windows needs torch
#> Loading patch store: 160 points x 8 channels | windows none (the table only)
#> <dsm_data>
#>   points     : 160
#>   channels   : 8
#>   windows    : none loaded (the table only)
#>   target     : soc_stock
#>   transform  : log1p
#>   cell size  : 0.0025
#> <dsm_data>
#>   points     : 160
#>   channels   : 8
#>   windows    : none loaded (the table only)
#>   target     : soc_stock
#>   transform  : log1p
#>   cell size  : 0.0025
```
