# Build a patch store from a point table and a folder of aligned rasters.

Build a patch store from a point table and a folder of aligned rasters.

## Usage

``` r
dsm_prepare(
  points,
  target,
  raster_dir,
  windows,
  out_dir = NULL,
  store_dir = NULL,
  metadata_dir = NULL,
  points_file = NULL,
  profile_id = "profile_id",
  coords = c("x", "y"),
  crs = NULL,
  percentage = NULL,
  dummy = "auto",
  drop = NULL,
  na_below = NULL,
  percentage_limits = c(0, 100),
  transform = c("none", "log1p"),
  target_min = NULL,
  subsample = NULL,
  target_label = NULL,
  target_unit = NULL,
  raster_pattern = "\\.tif$",
  chunk_nrows = 1000L,
  n_cores = NULL,
  max_ram_gb = NULL,
  read_gap = 256L,
  read_max_cols = 4096L,
  overwrite = FALSE,
  verbose = TRUE
)
```

## Arguments

- points:

  An sf object, a data.frame with coordinate columns, or a path to a
  spatial file sf can read (GPKG, shapefile, GeoJSON).

- target:

  Name of the target column in `points`.

- raster_dir:

  Folder of aligned single-band rasters, one predictor per file. Only
  files at the top level matching `raster_pattern` are read.

- windows:

  Patch sizes in pixels, odd. REQUIRED: a window's ground extent is
  window x resolution, so there is no default that is right at every
  resolution. 3/9/15 were chosen for 250 m.

- out_dir:

  Where to write, as out_dir/patches (the store), out_dir/metadata
  (tables for a human) and out_dir/points.csv. Or give the three
  separately with store_dir, metadata_dir and points_file.

- store_dir, metadata_dir, points_file:

  Where each part goes when it is not under `out_dir`: the store, the
  tables for a human, the point table.

- profile_id:

  Column that identifies an observation. Rows sharing it are
  de-duplicated (the first is kept) and later kept in the same fold.
  Absent, every row is its own profile.

- coords, crs:

  For a data.frame: the coordinate columns and their CRS. NULL crs means
  the coordinates are already in the rasters' CRS.

- percentage:

  Regular expressions over the CLEANED raster names for channels in
  0-100. Anchor them (^pnv\_, ^peatland_extent\$) to avoid matching more
  than intended; every match is reported.

- dummy:

  "auto" to detect 0/1 channels, or the channel names.

- drop:

  Channel names (raw or cleaned) to leave out.

- na_below:

  Named numeric: names are regexes over cleaned names, values are
  thresholds. A value \<= its threshold becomes NA (a nodata sentinel
  such as -9999, or stage 01's temperature floor of -100).

- percentage_limits:

  Percentages are clamped into this range; NULL to leave them as read.
  The clamp keeps the slight overshoot of an interpolated surface
  instead of turning it into NA.

- transform:

  "none" or "log1p" – the space the model trains in.

- target_min:

  Rows with target \<= target_min are dropped. NULL keeps any finite
  target the transform accepts.

- subsample:

  NULL, or list(frac, block_size, seed, strata): keep a fraction of the
  points in whole spatial blocks, for a run that finishes sooner.
  `strata`, optional, is the side of square strata, a whole multiple of
  `block_size`: each keeps its points up to one common quota, so the cut
  falls on the densest regions and no region is lost. Recorded, so a
  subsampled result is never mistaken for a full one.

- target_label:

  Name of the target in reports. NULL uses `target`.

- target_unit:

  Unit of the target, for reports. NULL leaves it unrecorded.

- raster_pattern:

  Regular expression the raster file names must match.

- chunk_nrows:

  Raster rows read per chunk. A performance knob: the store does not
  depend on it.

- n_cores:

  Cores for the extraction, one band per core. NULL uses the physical
  cores minus one, fewer if `max_ram_gb` cannot hold them. The result
  does not depend on it.

- max_ram_gb:

  RAM the extraction may use in total. NULL for 70% of what is available
  when it starts (read with the ps package; without it, no cap). Decides
  how many of the `n_cores` workers actually run.

- read_gap, read_max_cols:

  How the points of a row chunk are grouped into reads: windows closer
  than `read_gap` columns share a read, and no read is wider than
  `read_max_cols`. Performance knobs only – the store and the point
  table are identical whatever they are, and the test suite proves it.

- overwrite:

  A store_dir that already holds a store is refused unless TRUE, in
  which case that store's files are removed first.

- verbose:

  Report progress.

## Value

A `dsm_store`, which dsm_load() accepts directly.

## Examples

``` r
ex <- example_landscape()
store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
                     windows = c(3, 7), out_dir = file.path(tempdir(), "landscape"),
                     percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
                     overwrite = TRUE, verbose = FALSE)
store
#> <dsm_store> /tmp/Rtmpedyk32/landscape/patches
#>   points     : 160 valid of 160
#>   channels   : 8  (3 dummy, 1 percentage, 4 continuous)
#>   windows    : 3, 7
#>   target     : soc_stock  (trained as log1p)
#>   cell size  : 0.0025
```
