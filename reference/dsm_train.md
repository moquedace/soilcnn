# Fit and tune a model over a resampling plan.

Fit and tune a model over a resampling plan.

## Usage

``` r
dsm_train(
  data,
  model = "cnn",
  resampling = spatial_cv(),
  tune_grid = NULL,
  tune_length = 20L,
  n_seeds = 3L,
  transform = NULL,
  clamp = c(0, Inf),
  features = c("centre", "window_mean"),
  output_dir,
  run_id = format(Sys.time(), "%Y%m%d_%H%M%S"),
  base_seed = 42L,
  device = NULL,
  n_cores = NULL,
  in_session = !is.null(device),
  threads_per_unit = 5L,
  max_ram_gb = NULL,
  test_ids = NULL,
  calibration_ids = NULL,
  resume = TRUE,
  evaluate_test = FALSE,
  verbose = TRUE,
  ...
)
```

## Arguments

- data:

  From dsm_load().

- model:

  A registered model name ("cnn", "rf", "mlp", or anything registered
  with register_model()), or a model_spec.

- resampling:

  A resample_spec, or a fold_plan to use as given.

- tune_grid:

  An explicit grid. NULL asks the model for one – for the CNN, drawn
  over the windows the loaded store holds: every window alone and every
  pair.

- tune_length:

  How many configurations to try when `tune_grid` is NULL. The same
  meaning as caret's: a budget, not a lattice.

- n_seeds:

  Repetitions per (config, fold). One gives a ranking with no error bar,
  which is a ranking of luck as often as of skill.

- transform:

  NULL (the default) uses the inverse of the transform the store was
  built under, which dsm_load() read (`data$transform`). A function is
  the inverse to use instead, and is refused if it disagrees with the
  store's.

- clamp:

  c(lower, upper), the plausible range of the target in native units:
  every prediction is clipped into it before it is scored. c(0, Inf)
  suits a stock or a concentration. A target that can be negative – a
  temperature, a log-ratio – needs c(-Inf, Inf): clipped at zero it
  loses half its predictions, and the metrics still look plausible.
  Refused if the data itself falls outside it. The run keeps it, and
  dsm_final() refits with the same.

- features:

  For tabular models: "centre", "window_mean", or both.

- output_dir:

  Where runs go; each run is a directory under it. Required: nothing is
  written where nobody said – a folder of your project, or
  [`tempdir()`](https://rdrr.io/r/base/tempfile.html) for a try.

- run_id:

  The run's directory name. Reusing one resumes that run (see `resume`);
  the default, a timestamp, always starts a fresh one.

- base_seed:

  Repetition s of every configuration trains under the seed base_seed +
  s - 1, the same for every configuration, so that two configurations
  differ by their hyperparameters and not by their luck.

- device:

  A torch device, to train the CNN on it in this session (see
  `in_session`). NULL builds one with setup_torch_device() when the
  units train in this session.

- n_cores:

  Cores for training. Side by side, the threads of all the workers
  together: n_cores / threads_per_unit workers, rounded down, and fewer
  if the RAM holds fewer. In this session, torch's threads for the CNN;
  the MLP's threads and ranger's for the forest too. NULL is the
  physical cores minus one (see resolve_cores()) – except that a
  `device` passed in keeps the threads it was set up with unless n_cores
  is given too.

- in_session:

  FALSE – the default, unless a `device` is given – trains the CNN's
  units side by side, each in an R process of its own with
  `threads_per_unit` threads. T2 measured three units of 5 threads 1.52x
  faster than one of 15, each giving exactly the numbers it gave alone.
  TRUE trains them one at a time in this session, on `device`, with
  `n_cores` threads; a CUDA device trains that way. The table models
  always train in this session.

- threads_per_unit:

  Threads each unit trains with side by side. It is part of a unit's
  result (T1), so the run records it and a resume with another count is
  refused. 5 was measured best here (T1, T2).

- max_ram_gb:

  RAM the side-by-side workers may use together. NULL for 70% of what is
  available when they start.

- test_ids:

  Sample ids forced into the test set.

- calibration_ids:

  Sample ids forced into the calibration set (see `calibration_frac` in
  [`spatial_cv()`](https://moquedace.github.io/soilcnn/reference/spatial_cv.md)):
  in no fold, kept for the "split" intervals
  [`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)
  calibrates.

- resume:

  Keep the units of `run_id` that already finished, matched by their
  hyperparameters rather than their label. A fold plan or a clamp other
  than the run's is refused.

- evaluate_test:

  Score the test set on every unit? FALSE, and deliberately so: a frozen
  test set stops being frozen once its score sits in the tuning table.
  It is scored once, after the choice, by freeze_selection() and
  score_test_grid().

- verbose:

  Report progress.

- ...:

  Passed to the underlying runner (n_epochs, patience, ...).

## Value

The runner's result, plus the plan and the data it used.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
ex <- example_landscape()
store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
                     windows = c(3, 7), out_dir = file.path(tempdir(), "landscape"),
                     percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
                     overwrite = TRUE, verbose = FALSE)
data <- dsm_load(store, verbose = FALSE)
fit <- dsm_train(data, resampling = spatial_cv(k = 2, block_size = 0.05, test_frac = 0.2),
                 tune_length = 2, n_seeds = 1, output_dir = tempdir(),
                 in_session = TRUE, n_cores = 2, n_epochs = 5, verbose = FALSE)
fit
# }
}
```
