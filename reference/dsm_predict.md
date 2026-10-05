# Predict a map from a final model.

Predict a map from a final model.

## Usage

``` r
dsm_predict(
  final,
  data,
  rasters = NULL,
  qc_table = NULL,
  extent = NULL,
  config = NULL,
  calibration = NULL,
  aoa_weights = NULL,
  alpha = 0.1,
  clamp = NULL,
  bands = NULL,
  engine = c("auto", "patch"),
  output_dir = NULL,
  run_id = NULL,
  resume = TRUE,
  n_cores = NULL,
  threads_per_worker = 5L,
  max_ram_gb = NULL,
  unit_rows = NULL,
  step_rows = NULL,
  chunk_cols = 2048L,
  probe = TRUE,
  verbose = TRUE
)
```

## Arguments

- final:

  A `dsm_final`, or the directory of a final run (dsm_final() or stage
  04).

- data:

  From dsm_load(), or the directory of a store written by dsm_prepare()
  – only its tables are read, no patches.

- rasters:

  NULL for the rasters the store was extracted from; a directory holding
  the same file names (a coarser grid, for a cheap pass end to end); or
  a table (data frame or CSV path) with columns predictor and
  raster_file.

- qc_table:

  NULL for the store's own QC rules; a data frame or CSV path for a
  store written before dsm_prepare().

- extent:

  NULL for the whole grid; a SpatExtent, c(xmin, xmax, ymin, ymax),
  anything terra::ext() reads, or list(rows = c(a, b), cols = c(c, d)).
  Pixels at the extent's edge are predicted with their real neighbours,
  so a part of the map is the same numbers as that part of the whole.

- config:

  NULL for the final run's selected configuration.

- calibration:

  Where the intervals, the smearing factor and the AOA come from: a
  named vector of tuning-run directories whose cross-validated residuals
  calibrate them, e.g.
  `c(block = "<spatial CV run>", knndm = "<kNNDM run>")`. Each source
  gets its own bands. NULL for the final run's own tuning run;
  character(0) for none (then no interval, DI or AOA).

- aoa_weights:

  NULL: every channel weighs alike in the dissimilarity index. Or a
  `dsm_importance` (see
  [`importance_weights()`](https://moquedace.github.io/soilcnn/reference/importance_weights.md)),
  or one non-negative weight per channel, named or in the model's order:
  the DI, the AOA and the level+DI interval then measure a pixel's
  distance from the training data in what the model uses (Meyer &
  Pebesma 2021). A map resumed with other weights is refused.

- alpha:

  Miscoverage of the intervals: 0.1 is 90%.

- clamp:

  c(lower, upper) of a prediction in native units. NULL for the refit's
  own (its evaluation clamps to c(0, Inf) by default).

- bands:

  NULL for all, or some of: ensemble_median, ensemble_mean, ensemble_sd,
  ensemble_mad, ensemble_min, ensemble_max, smeared_mean,
  interval_constant, interval_level_di, di, aoa, valid_mask.

- engine:

  "auto": fully convolutional where that gives exactly the
  patch-by-patch numbers for the network, patch by patch elsewhere.
  "patch": patch by patch everywhere – slow, for checks.

- output_dir:

  Where maps go. NULL for `<final run>/maps`.

- run_id:

  NULL for `map_<timestamp>`. An existing one resumes.

- resume:

  Keep the units already finished.

- n_cores:

  Cores for the whole map. NULL for the physical cores minus one.

- threads_per_worker:

  Torch threads each worker runs with.

- max_ram_gb:

  RAM all workers may use together. NULL for 70% of what is free when
  the map starts (read with the ps package).

- unit_rows, step_rows, chunk_cols:

  The geometry of the work: a unit is unit_rows rows over the whole
  extent width, read step_rows rows at a time (a multiple of 16), with
  the network run over chunk_cols columns at a time. NULL sizes the
  first two from the RAM.

- probe:

  Predict the profiles' own pixels first and compare with the final
  run's stored predictions (see above). A failure stops the map.

- verbose:

  Report progress, and print the result.

## Value

A `dsm_prediction`, printed.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
map <- dsm_predict(run$final, run$data, output_dir = tempdir(), run_id = "map_example",
                   n_cores = 1, threads_per_worker = 1, verbose = FALSE)
map
# }
}
```
