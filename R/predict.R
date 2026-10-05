# ── Prediction over a raster: the network run once per strip, not once per pixel
#
# WHY THIS EXISTS.
#
# Stage 05 predicts a pixel by cutting the w x w patch around it and passing
# the patch through the network. The pixel beside it has a patch that shares
# all but one of its columns, and the network redoes every sum of it again. On
# the deployed SOC model (one 15 x 15 branch, 181 channels, 10 seeds) the
# dev run measured 164 valid pixels per second; a global map at 250 m has
# ~2.3 billion valid pixels. That is four months, and the disk was not the
# limit -- 23 s of reading against 2,184 s of network (2026-09-27).
#
# FULLY CONVOLUTIONAL, AND EXACT WHERE IT CAN BE.
#
# A branch whose convolutions use NO padding ("valid") is exactly
# translation-equivariant: a 3 x 3 valid convolution, an eval-mode BatchNorm
# (an affine map per channel), the activation, and the residual shortcut
# (a 1 x 1 convolution, centre-cropped by the same pixel per side) all compute
# at every position what they compute inside a patch. So the blocks are run
# ONCE over a whole strip of the raster, and what is left per pixel is cheap:
#
#   "gap"      the patch's mean feature is a moving average of the strip's
#              feature map -- avg_pool2d with stride 1
#   "flatten"  the patch's C x o x o features are a window of the strip's
#              feature map, gathered at the pixels that are predicted and
#              passed through the branch's own linear layer
#   SE + gap   the SE weight is constant over the patch, so it commutes with
#              the mean: pooled = s(e) * e, e the moving average -- exact
#
# and then the embedding's BatchNorm, the gate and the head, per pixel. About
# 200k multiply-adds per pixel per seed instead of 27M: ~100x.
#
# The flatten case is gathered rather than convolved on purpose. A convolution
# with the linear weight reshaped to (K, C, o, o) computes the same numbers,
# but at EVERY position of the strip -- and the linear layer is where a
# flatten branch spends most of its arithmetic, so on a coastline that would
# pay it for the sea as well.
#
# WHERE IT IS NOT EXACT, AND WHAT HAPPENS THERE. A "same" branch pads EACH
# PATCH with zeros at its own border; over a strip those positions see real
# neighbours instead, so the numbers would differ. SE with "flatten" does not
# commute with the linear layer. Those branches are run patch by patch --
# which for the usual case, the small 3 x 3 branch of a valid_large model,
# costs little. fcn_supported() says which branch takes which path.
#
# NON-FINITE INPUT. The full-window rule already discards a pixel whose window
# holds any non-finite value. Over a strip, those values are replaced by 0
# before the convolutions -- a NaN would otherwise spread through whatever
# algorithm torch picks for the convolution -- and only the discarded pixels'
# windows ever contain one.

#' Which branches of a network can be run fully convolutionally, exactly.
#'
#' @param model A dual_branch_cnn.
#' @return A logical per branch.
#' @noRd
fcn_supported <- function(model) {
  br <- if (model$n_branches == 1L) list(model$branch1) else list(model$branch1, model$branch2)
  vapply(br, function(b) {
    identical(b$conv_padding, "valid") &&
      (!isTRUE(b$use_se_block) || identical(b$embed_pool, "gap"))
  }, logical(1))
}

# The window a branch reads: its patch size, recovered from the module.
.fcn_branch_window <- function(br) {
  as.integer(if (identical(br$conv_padding, "valid")) br$out_size + 2L * length(br$blocks)
             else br$out_size)
}

# Centres per gather, so one gathered batch stays near `gather_mb`: a 15 x 15
# patch of 181 channels is 160 KB, and 4,096 of them would be 670 MB.
.fcn_gather_batch <- function(n_channels, w, batch, gather_mb) {
  as.integer(max(1L, min(batch, floor(gather_mb * 2^20 / (4 * n_channels * w * w)))))
}

# The patches of `centres` from a strip tensor, (n, C, w, w). centres is a
# 2-column integer matrix of (row, col) in STRIP coordinates, 1-based; every
# centre's window must lie inside the strip. The strip must be contiguous.
.fcn_gather_patches <- function(x_strip, centres, w) {
  C <- x_strip$size(2L); H <- x_strip$size(3L); W <- x_strip$size(4L)
  h <- (w - 1L) %/% 2L
  n <- nrow(centres)
  offs <- expand.grid(dc = (-h):h, dr = (-h):h)          # row-major within a patch
  base <- (centres[, 1] - 1L) * W + centres[, 2]
  cell <- as.vector(t(outer(base, offs$dr * W + offs$dc, "+")))
  x_flat <- x_strip$view(c(C, H * W))
  x_flat[, cell]$view(c(C, n, w, w))$permute(c(2L, 1L, 3L, 4L))$contiguous()
}

# One branch's embedding for every centre, patch by patch. gc_hook, when
# given, runs after every batch: a dense chunk is dozens of batches of
# hundreds of MB each, all of it garbage R does not know the size of.
.fcn_branch_patchwise <- function(branch, x_strip, centres, w, batch, gc_hook = NULL) {
  n <- nrow(centres)
  out <- vector("list", ceiling(n / batch))
  k <- 0L
  for (s in seq(1L, n, by = batch)) {
    e <- min(n, s + batch - 1L)
    k <- k + 1L
    out[[k]] <- branch(.fcn_gather_patches(x_strip, centres[s:e, , drop = FALSE], w))
    if (!is.null(gc_hook)) gc_hook()
  }
  torch::torch_cat(out, dim = 1L)
}

# One branch's embedding for every centre, fully convolutionally. The map
# position (a, b) of the pooled features is the patch whose top-left corner is
# strip pixel (a, b), i.e. the centre (a + h, b + h).
.fcn_branch_convolutional <- function(branch, x_strip, centres, w, batch = 4096L,
                                      gather_mb = 256, gc_hook = NULL) {
  h  <- (w - 1L) %/% 2L
  nb <- length(branch$blocks)
  f  <- x_strip
  for (i in seq_len(nb)) f <- branch$blocks[[i]](f)
  o <- branch$out_size
  C <- f$size(2L)
  if (identical(branch$embed_pool, "gap")) {
    m <- torch::nnf_avg_pool2d(f, kernel_size = c(o, o), stride = c(1L, 1L))
    B <- m$size(4L)
    e <- m$permute(c(1L, 3L, 4L, 2L))$reshape(c(-1L, C))
    e <- e[(centres[, 1] - h - 1L) * B + (centres[, 2] - h), , drop = FALSE]
    if (isTRUE(branch$use_se_block)) {
      s <- torch::torch_sigmoid(branch$se$fc2(branch$se$act(branch$se$fc1(e))))
      e <- e * s
    }
    z <- branch$linear(e)
  } else {
    # The patch's C x o x o features are the window of the feature map centred
    # nb pixels up and left of the patch centre: each valid block trims one
    # pixel per side, so feature row a holds what strip row a + nb held.
    pb <- .fcn_gather_batch(C, o, batch, gather_mb)
    n <- nrow(centres)
    parts <- vector("list", ceiling(n / pb))
    k <- 0L
    for (s in seq(1L, n, by = pb)) {
      e <- min(n, s + pb - 1L)
      k <- k + 1L
      g <- .fcn_gather_patches(f, centres[s:e, , drop = FALSE] - nb, o)
      parts[[k]] <- branch$linear(branch$flatten(g))
      if (!is.null(gc_hook)) gc_hook()
    }
    z <- torch::torch_cat(parts, dim = 1L)
  }
  branch$drop_emb(branch$act_emb(branch$bn_emb(z)))
}

# The gate and the head, on embeddings -- dual_branch_cnn$forward(), after the
# branches. Kept in step with it by the test that compares the two paths.
.fcn_head <- function(model, f1, f2 = NULL) {
  if (model$n_branches == 1L) return(model$head(f1))
  abs_dif <- torch::torch_abs(f1 - f2)
  head_in <- if (identical(model$gate_type, "no_gate_concat")) {
    torch::torch_cat(list(f1, f2), dim = 2L)
  } else {
    gate  <- model$gate_net(torch::torch_cat(list(f1, f2, abs_dif, f1 * f2), dim = 2L))
    fused <- gate * f1 + (1 - gate) * f2
    torch::torch_cat(list(fused, abs_dif), dim = 2L)
  }
  model$head(head_in)
}

#' Predict the centres of a strip, in the transformed space.
#'
#' @param model   A dual_branch_cnn, in eval mode.
#' @param x_strip Tensor (1, C, H, W): the strip, QC'd and scaled, with every
#'   non-finite value replaced by 0.
#' @param centres Integer matrix (n x 2) of (row, col) in strip coordinates,
#'   1-based; every window of every branch must lie inside the strip.
#' @param engine  "fcn" runs each supported branch fully convolutionally and
#'   the rest patch by patch; "patch" runs every branch patch by patch.
#' @param batch   Centres per batch on the patch-by-patch path and the head.
#' @param gather_mb Memory one gathered batch of patches may take.
#' @param gc_hook NULL, or a function called after every batch and branch --
#'   the map passes one that collects R's garbage (see .predict_gc_hook()).
#' @return A numeric vector, one prediction per centre.
#' @noRd
fcn_predict_strip <- function(model, x_strip, centres, engine = c("fcn", "patch"),
                              batch = 4096L, gather_mb = 256, gc_hook = NULL) {
  engine <- match.arg(engine)
  if (nrow(centres) == 0L) return(numeric(0))
  x_strip <- x_strip$contiguous()
  brs <- if (model$n_branches == 1L) list(model$branch1) else list(model$branch1, model$branch2)
  windows <- vapply(brs, .fcn_branch_window, integer(1))
  supported <- fcn_supported(model)
  n_ch <- x_strip$size(2L)
  torch::with_no_grad({
    emb <- lapply(seq_along(brs), function(b) {
      e <- if (identical(engine, "fcn") && supported[b]) {
        .fcn_branch_convolutional(brs[[b]], x_strip, centres, windows[b], batch, gather_mb,
                                  gc_hook)
      } else {
        .fcn_branch_patchwise(brs[[b]], x_strip, centres, windows[b],
                              .fcn_gather_batch(n_ch, windows[b], batch, gather_mb), gc_hook)
      }
      if (!is.null(gc_hook)) gc_hook()
      e
    })
    n <- nrow(centres)
    out <- numeric(n)
    for (s in seq(1L, n, by = batch)) {
      e <- min(n, s + batch - 1L)
      f1 <- emb[[1]][s:e, , drop = FALSE]
      f2 <- if (length(emb) == 2L) emb[[2]][s:e, , drop = FALSE] else NULL
      out[s:e] <- as.numeric(.fcn_head(model, f1, f2)$squeeze(2L)$to(device = "cpu"))
    }
  })
  out
}

# ══════════════════════════════════════════════════════════════════════════════
# dsm_predict(): the map
# ══════════════════════════════════════════════════════════════════════════════
#
# WHAT THIS REPLACED. Stage 05 of the SOC project -- 05_predict_spatial.R and
# the scripts that ran it in parallel, merged its tiles and estimated its ETA,
# all removed on 2026-09-28 and kept in git's history -- predicted one
# dataset's map. dsm_predict() is that stage with the dataset taken out, and
# with what a 250 m global grid needs that stage 05 did not have:
#
#   1. THE NETWORK RUN ONCE PER STRIP (above): ~100x less arithmetic.
#
#   2. EVERY ROW DECOMPRESSED ONCE. The SOC rasters -- 181 files, 1,052 GB on
#      one SATA disk -- are TIFFs in strips of ONE ROW, LZW: reading any
#      column of a row makes GDAL decompress all 160,298 of them. Stage 05
#      cut the grid into 2-D tiles, so every row was decompressed once per
#      column tile, and its margin rows once more per block. Here a unit of
#      work is a band of rows over the whole width of the extent, read
#      step_rows rows at a time, and the 2h rows the next step's windows
#      share with this one are KEPT, not re-read: inside a unit each row of
#      each band is decompressed exactly once, and between units only the 2h
#      halo rows are read twice (1-3% at 250 m).
#
#   3. RESUMABLE, SIDE BY SIDE. Units are claimed by worker processes (callr,
#      a fixed thread count each, as in dsm_final()), and a unit is done when
#      its record is written -- after its files. A run killed on day 2
#      resumes where it stopped, and a resumed run is refused if anything
#      that changes the numbers changed.
#
# WHAT IT WRITES, in native units -- each seed's prediction back-transformed
# and clamped as the refit's own evaluation clamped it, then aggregated:
#
#   ensemble_median, _mean, _sd, _mad, _min, _max   over the seeds
#   smeared_mean_<source>       the conditional MEAN (Duan), log1p only;
#                               smeared_mean_split from the calibration set
#   piNN_<method>_<width>_lower/upper[_<source>]   the intervals:
#       method  cv       calibrated on the source's cross-validated residuals
#               split    split conformal, on the final run's calibration set
#               cv_plus  CV+, the source's fold models at every pixel
#       width   constant, or level_di: fitted on the level and the DI
#   di                          dissimilarity index (Meyer & Pebesma 2021)
#   aoa_<source>                1 inside the area of applicability
#   valid_mask                  1 where the model predicted
#
# One GeoTIFF per band per unit, and one VRT per band over the units. Every
# band's meaning, and every number that calibrated it, is written beside it
# (bands.csv, calibration.csv).
#
# WHY THE INTERVAL HAS A SOURCE. Its residuals decide what it is honest for:
# block CV validates ~16 km from the training data, kNNDM at the distances the
# map predicts at (U1/U2, 2026-09-27). The bands are functions of the median
# and the DI only, so every source asked for is written in the same pass,
# with no second pass of the network.
#
# HOW IT IS CHECKED BEFORE THE EXPENSIVE PART. When the rasters are the grid
# the store was extracted from, the profiles themselves are pixels of the
# map. So before the map, a small unit around the densest group of profiles
# is predicted through the whole chain -- the reader, QC, scaling, channel
# order, the fully convolutional network, the inverse -- and every seed's
# prediction at every one of those profiles must equal the one the final run
# stored for it. Misaligned channels, a stale scaling or the wrong inverse
# cannot pass that, and it costs one unit.

#' Predict a map from a final model.
#'
#' @param final   A `dsm_final`, or the directory of a final run (dsm_final()
#'   or stage 04).
#' @param data    From dsm_load(), or the directory of a store written by
#'   dsm_prepare() -- only its tables are read, no patches.
#' @param rasters NULL for the rasters the store was extracted from; a
#'   directory holding the same file names (a coarser grid, for a cheap pass
#'   end to end); or a table (data frame or CSV path) with columns predictor
#'   and raster_file.
#' @param qc_table NULL for the store's own QC rules; a data frame or CSV path
#'   for a store written before dsm_prepare().
#' @param extent  NULL for the whole grid; a SpatExtent, c(xmin, xmax, ymin,
#'   ymax), anything terra::ext() reads, or list(rows = c(a, b), cols = c(c, d)).
#'   Pixels at the extent's edge are predicted with their real neighbours, so
#'   a part of the map is the same numbers as that part of the whole.
#' @param config  NULL for the final run's selected configuration.
#' @param calibration Where the intervals, the smearing factor and the AOA come
#'   from: a named vector of tuning-run directories whose cross-validated
#'   residuals calibrate them, e.g.
#'   `c(block = "<spatial CV run>", knndm = "<kNNDM run>")`. Each source gets
#'   its own bands. NULL for the final run's own tuning run; character(0) for
#'   none (then no interval, DI or AOA).
#' @param aoa_weights NULL: every channel weighs alike in the dissimilarity
#'   index. Or a `dsm_importance` (see [importance_weights()]), or one
#'   non-negative weight per channel, named or in the model's order: the DI,
#'   the AOA and the level+DI interval then measure a pixel's distance from
#'   the training data in what the model uses (Meyer & Pebesma 2021). A map
#'   resumed with other weights is refused.
#' @param alpha   Miscoverage of the intervals: 0.1 is 90%.
#' @param intervals Which intervals the map carries (see [dsm_final()], which
#'   checks all of them on the test set): "cv", calibrated on each source's
#'   cross-validated residuals; "split", split conformal on the final run's
#'   calibration set; "cv_plus", CV+ with each source's fold models -- which
#'   predicts every pixel once more per fold model, the map's dearest band by
#'   far. NULL: "cv", and "split" when the final run has a calibration set.
#' @param weighting "point" (every calibration point weighs the same) or
#'   "group" (every group of the plan does -- block, region, profile; see
#'   [conformal_calibrate()]), for the "cv" and "split" intervals. CV+ is by
#'   point.
#' @param clamp   c(lower, upper) of a prediction in native units. NULL for the
#'   refit's own (its evaluation clamps to c(0, Inf) by default).
#' @param bands   NULL for all, or some of: ensemble_median, ensemble_mean,
#'   ensemble_sd, ensemble_mad, ensemble_min, ensemble_max, smeared_mean,
#'   interval_constant, interval_level_di, di, aoa, valid_mask.
#' @param engine  "auto": fully convolutional where that gives exactly the
#'   patch-by-patch numbers for the network, patch by patch elsewhere.
#'   "patch": patch by patch everywhere -- slow, for checks.
#' @param output_dir Where maps go. NULL for `<final run>/maps`.
#' @param run_id  NULL for `map_<timestamp>`. An existing one resumes.
#' @param resume  Keep the units already finished.
#' @param n_cores Cores for the whole map. NULL for the physical cores minus one.
#' @param threads_per_worker Torch threads each worker runs with.
#' @param max_ram_gb RAM all workers may use together. NULL for 70% of what is
#'   free when the map starts (read with the ps package).
#' @param unit_rows,step_rows,chunk_cols The geometry of the work: a unit is
#'   unit_rows rows over the whole extent width, read step_rows rows at a time
#'   (a multiple of 16), with the network run over chunk_cols columns at a
#'   time. NULL sizes the first two from the RAM.
#' @param probe   Predict the profiles' own pixels first and compare with the
#'   final run's stored predictions (see above). A failure stops the map.
#' @param verbose Report progress, and print the result.
#' @return A `dsm_prediction`, printed.
#' @examplesIf torch::torch_is_installed()
#' \donttest{
#' run <- example_run()    # a small fitted run, made once a session
#' map <- dsm_predict(run$final, run$data, output_dir = tempdir(), run_id = "map_example",
#'                    n_cores = 1, threads_per_worker = 1, verbose = FALSE)
#' map
#' }
#' @export
dsm_predict <- function(final, data, rasters = NULL, qc_table = NULL, extent = NULL,
                        config = NULL, calibration = NULL, aoa_weights = NULL,
                        alpha = 0.1, intervals = NULL, weighting = c("point", "group"),
                        clamp = NULL,
                        bands = NULL, engine = c("auto", "patch"), output_dir = NULL,
                        run_id = NULL, resume = TRUE, n_cores = NULL,
                        threads_per_worker = 5L, max_ram_gb = NULL, unit_rows = NULL,
                        step_rows = NULL, chunk_cols = 2048L, probe = TRUE,
                        verbose = TRUE) {

  t_start <- Sys.time()
  say <- function(...) if (verbose) message(...)
  engine <- match.arg(engine)
  weighting <- match.arg(weighting)
  if (!is.null(intervals)) intervals <- match.arg(intervals, c("cv", "split", "cv_plus"),
                                                  several.ok = TRUE)

  # ── 0. the arguments, all checked before a raster is opened ───────────────
  fr  <- .predict_final(final, config)
  inp <- .predict_inputs(data, rasters, qc_table, fr$scaling)
  aoa_w <- .predict_aoa_weights(aoa_weights, inp$predictors)
  alpha <- .predict_alpha(alpha)
  clamp <- .predict_clamp(clamp, fr$summ)
  n_cores <- resolve_cores(n_cores, what = "the map")
  tpw <- .predict_whole(threads_per_worker, "threads_per_worker")
  if (tpw > n_cores) {
    say("threads_per_worker = ", tpw, " is more than the ", n_cores, " core(s) the map ",
        "may use; each worker runs with ", n_cores, ".")
    tpw <- n_cores
  }
  chunk_cols <- .predict_whole(chunk_cols, "chunk_cols")
  if (!is.null(step_rows)) step_rows <- .predict_whole(step_rows, "step_rows")
  if (!is.null(unit_rows)) unit_rows <- .predict_whole(unit_rows, "unit_rows")

  # ── 1. the grid, and the part of it to predict ────────────────────────────
  windows <- as.integer(fr$cfg$window_sizes[[1]])
  grid <- .predict_grid(inp, windows, extent)
  say(sprintf("\nGrid: %s x %s cells at %.8g | predicting rows %d-%d, cols %d-%d (%s cells)",
              format(grid$nrow, big.mark = ","), format(grid$ncol, big.mark = ","),
              grid$xres, grid$rows[1], grid$rows[2], grid$cols[1], grid$cols[2],
              format(as.numeric(grid$n_rows_out) * grid$n_cols_out, big.mark = ",")))
  if (!isTRUE(grid$same_res)) {
    say(strrep("!", 78))
    say(sprintf(paste0(
      "THE MODEL WAS TRAINED AT %.8g AND IS PREDICTING AT %.8g (%.1fx).\n",
      "The window is counted in PIXELS, so a %d x %d patch now covers %.1fx the\n",
      "ground it covered in training. This map exercises the pipeline; it does\n",
      "NOT carry the accuracy the validation reported."),
      inp$cell_size, grid$xres, grid$xres / inp$cell_size, grid$window, grid$window,
      grid$xres / inp$cell_size))
    say(strrep("!", 78))
  }

  # ── 2. the calibration: one set of bands per source of residuals ──────────
  sources <- .predict_sources(calibration, fr)
  intervals <- .predict_intervals(intervals, fr, sources)
  cal <- .predict_calibration(sources, fr, inp, alpha, say, weights = aoa_w,
                              intervals = intervals, weighting = weighting)
  if (length(cal$sources) == 0L && is.null(cal$split)) {
    say("\nNo calibration source: the map carries no interval, no DI and no AOA.")
  }

  # ── 3. what is written ────────────────────────────────────────────────────
  band_tbl <- .predict_band_table(bands, cal, inp, length(fr$seeds))

  # ── 4. the run directory; a resumed run must be the same map ──────────────
  output_dir <- output_dir %||% file.path(fr$run_dir, "maps")
  run_id  <- run_id %||% paste0("map_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  run_dir <- normalizePath(file.path(output_dir, run_id), winslash = "/", mustWork = FALSE)
  settings_path <- file.path(run_dir, "settings.rds")
  saved <- NULL
  if (file.exists(settings_path)) {
    if (!isTRUE(resume)) {
      stop("A map already exists at ", run_dir, ". Pass resume = TRUE to finish it, ",
           "or another run_id.", call. = FALSE)
    }
    saved <- readRDS(settings_path)
    # The units are the layout of the files on disk: a resumed run keeps them,
    # and the step they were cut to -- sized from the RAM free on the first
    # call, which is not the RAM free now.
    if (is.null(unit_rows)) unit_rows <- saved$unit_rows
    if (is.null(step_rows)) step_rows <- saved$step_rows_used
  }

  # ── 5. the work: workers, steps, units ────────────────────────────────────
  # CV+ predicts every pixel once more per fold model: they count in the
  # per-pixel memory as the seeds do, and add one strip-sized copy (the
  # fold's scaling of the step's rows).
  n_fold_models <- sum(vapply(cal$sources, function(s) as.integer(s$cv_plus$n_models %||% 0L),
                              integer(1)))
  if (n_fold_models > 0L) {
    say(sprintf("\nCV+: %d fold model(s) predict every pixel beside the %d seed(s) -- the map takes about %.1fx the network time of one without CV+.",
                n_fold_models, length(fr$seeds), (n_fold_models + length(fr$seeds)) / length(fr$seeds)))
  }
  work  <- .predict_work_plan(grid, inp, fr$cfg, band_tbl, n_cores, tpw, max_ram_gb,
                              unit_rows, step_rows, chunk_cols, say,
                              n_seeds = length(fr$seeds) + n_fold_models,
                              fold_strip = n_fold_models > 0L)
  units <- .predict_units(grid$rows, grid$cols, work$unit_rows)
  settings <- .predict_settings(fr, inp, grid, work, band_tbl, alpha, clamp, engine, cal,
                                intervals, weighting)
  # THE WEIGHTS ONLY WHEN GIVEN: a map started before this setting existed has
  # no such field, and an unweighted call must still resume it.
  if (!is.null(aoa_w)) settings$aoa_weights <- unname(aoa_w)
  # THE SETTINGS ARE LOCKED BY THE FIRST FINISHED UNIT, not by the first call:
  # a call stopped before any unit finished -- a probe that failed on a swapped
  # channel -- left nothing its successor could be mixed with.
  if (!is.null(saved) &&
      length(list.files(file.path(run_dir, "units"), pattern = "^done[.]rds$",
                        recursive = TRUE)) == 0L) {
    saved <- NULL
  }
  if (!is.null(saved)) {
    now_id   <- .predict_settings_portable(settings)
    saved_id <- .predict_settings_portable(saved)
    diff_fields <- names(now_id)[!vapply(names(now_id), function(k)
      identical(now_id[[k]], saved_id[[k]]), logical(1))]
    # Checked both ways: weights saved and none given now is a change too,
    # though the loop above walks only the fields this call has.
    if (!identical(now_id$aoa_weights, saved_id$aoa_weights)) {
      diff_fields <- union(diff_fields, "aoa_weights")
    }
    if (length(diff_fields) > 0L) {
      stop("This map was started with different settings (", paste(diff_fields, collapse = ", "),
           "): its finished units would be mixed with units of another map.\n  Use another ",
           "run_id, or resume with the original settings.", call. = FALSE)
    }
  }
  create_output_dirs(c(run_dir, file.path(run_dir, "units"), file.path(run_dir, "logs")))
  if (is.null(saved)) {
    safe_save_rds(c(settings, list(step_rows_used = work$step_rows)), settings_path,
                  compress = FALSE)
  }
  safe_save_rds(cal, file.path(run_dir, "calibration.rds"), compress = FALSE)
  safe_write_csv2(band_tbl, file.path(run_dir, "bands.csv"))
  cal_tbl <- .predict_calibration_table(cal, band_tbl)
  if (nrow(cal_tbl) > 0L) safe_write_csv2(cal_tbl, file.path(run_dir, "calibration.csv"))

  probe_model <- build_cnn_from_config(fr$cfg, length(inp$predictors))
  engine_used <- if (identical(engine, "patch")) rep("patch", probe_model$n_branches) else
    ifelse(fcn_supported(probe_model), "fcn", "patch")
  rm(probe_model)
  say(sprintf("\nWork: %d unit(s) of %d row(s) | %d worker(s) x %d thread(s) | steps of %d row(s), chunks of %d col(s) | ~%.1f GB per worker (estimate)",
              nrow(units), work$unit_rows, work$n_workers, tpw, work$step_rows, chunk_cols,
              work$per_worker_gb))
  say("Engine per branch: ", paste(sprintf("%dx%d %s", windows, windows, engine_used), collapse = ", "),
      " | ", nrow(band_tbl), " band(s) -> ", run_dir)

  job <- .predict_job(fr, inp, grid, work, band_tbl, cal, clamp, engine, tpw, run_dir)

  # ── 6. the probe: the profiles' own pixels, before the map ────────────────
  probe_res <- list(status = "not_run", reason = "probe = FALSE")
  if (isTRUE(probe)) {
    probe_res <- .predict_probe(job, fr, inp, grid, run_dir, say)
    if (identical(probe_res$status, "fail")) {
      stop("The probe failed: the map does not reproduce the final model's own ",
           "predictions at the profiles (", probe_res$reason, ").\n  Nothing was mapped. ",
           "See ", file.path(run_dir, "probe.csv"), call. = FALSE)
    }
  }

  # ── 7. the map ────────────────────────────────────────────────────────────
  done <- file.exists(.predict_done_path(job$units_dir, units$unit_id))
  todo <- units[!done, , drop = FALSE]
  if (any(done)) say("\nResuming: ", sum(done), " of ", nrow(units), " unit(s) already mapped, kept as they are.")
  run_info <- list(n_workers = 0L, restarts = 0L, peak_gb = NA_real_, minutes = NA_real_)
  if (nrow(todo) > 0L) {
    say(sprintf("\nMapping %d unit(s) with %d worker(s)...", nrow(todo),
                min(work$n_workers, nrow(todo))))
    run_info <- .predict_run_units(job, todo, min(work$n_workers, nrow(todo)), run_dir,
                                   "map", say, progress = verbose)
  }
  finished <- file.exists(.predict_done_path(job$units_dir, units$unit_id))
  if (!all(finished)) {
    lost <- units$unit_id[!finished]
    why <- vapply(lost, function(u) {
      fp <- file.path(job$units_dir, u, "failed.rds")
      if (file.exists(fp)) as.character(readRDS(fp)$error) else "never finished"
    }, character(1))
    stop("Unit(s) did not finish: ", paste(sprintf("%s (%s)", lost, why), collapse = "; "),
         ".\n  Worker logs: ", file.path(run_dir, "logs"), "\n  Fix the cause and call ",
         "dsm_predict() again with run_id = \"", run_id, "\" -- the finished units are kept.",
         call. = FALSE)
  }

  # ── 8. the mosaic and the record ──────────────────────────────────────────
  recs <- lapply(units$unit_id, function(u) readRDS(.predict_done_path(job$units_dir, u)))
  vrt <- .predict_write_vrts(run_dir, units, band_tbl)
  unit_tbl <- .predict_unit_table(recs)
  band_sum <- .predict_band_summary(recs, band_tbl)
  safe_write_csv2(unit_tbl, file.path(run_dir, "units.csv"))
  safe_write_csv2(band_sum, file.path(run_dir, "band_summary.csv"))

  minutes <- as.numeric(difftime(Sys.time(), t_start, units = "mins"))
  n_cells <- sum(unit_tbl$n_cells)
  n_valid <- sum(unit_tbl$n_valid)
  manifest <- tibble::tibble(
    run_id = run_id, final_run = fr$run_dir, config_id = fr$config_id,
    seeds = paste(fr$seeds, collapse = ";"), n_seeds = length(fr$seeds),
    transform = inp$transform$name, clamp = paste(clamp, collapse = ";"),
    rasters = paste(unique(dirname(inp$files)), collapse = ";"),
    grid_nrow = grid$nrow, grid_ncol = grid$ncol, cell_size = grid$xres,
    trained_cell_size = inp$cell_size, same_grid_as_store = grid$same_as_store,
    rows = paste(grid$rows, collapse = "-"), cols = paste(grid$cols, collapse = "-"),
    n_cells = n_cells, n_valid = n_valid, n_units = nrow(units), unit_rows = work$unit_rows,
    step_rows = work$step_rows, chunk_cols = chunk_cols, n_workers = run_info$n_workers,
    worker_restarts = run_info$restarts, recycle_gb = job$recycle_gb,
    threads_per_worker = tpw, engine = paste(engine_used, collapse = ";"),
    calibration = paste(sprintf("%s=%s", names(sources), sources), collapse = ";"),
    intervals = paste(intervals, collapse = ";"), weighting = weighting,
    n_fold_models = n_fold_models,
    alpha = paste(alpha, collapse = ";"), n_bands = nrow(band_tbl),
    probe = probe_res$status, probe_max_rel_diff = probe_res$max_rel_diff %||% NA_real_,
    peak_gb_per_worker = run_info$peak_gb, map_minutes = run_info$minutes,
    minutes_this_call = minutes,
    valid_px_per_s = sum(unit_tbl$n_valid) / max(1e-9, sum(unit_tbl$total_s)),
    soilcnn_version = .soilcnn_version(),
    torch_version = as.character(utils::packageVersion("torch")),
    terra_version = as.character(utils::packageVersion("terra")),
    git_commit = .git_commit_at(run_dir), finished_at = as.character(Sys.time()))
  safe_write_csv2(manifest, file.path(run_dir, "prediction_manifest.csv"))

  out <- structure(list(
    run_dir = run_dir, run_id = run_id, final_run = fr$run_dir, config_id = fr$config_id,
    seeds = fr$seeds, grid = grid, units = unit_tbl, bands = band_tbl, vrt = vrt,
    band_summary = band_sum, calibration = cal_tbl, sources = sources, probe = probe_res,
    engine = engine_used, windows = windows, work = work, n_workers = run_info$n_workers,
    manifest = manifest, minutes = minutes, map_minutes = run_info$minutes),
    class = "dsm_prediction")
  if (verbose) print(out)
  invisible(out)
}

#' Print a `dsm_prediction`
#'
#' @param x   A `dsm_prediction`, from [dsm_predict()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.dsm_prediction <- function(x, ...) {
  m <- x$manifest
  cat("\n<dsm_prediction> ", x$run_dir, "\n", sep = "")
  cat("  model      : ", x$config_id, " (", length(x$seeds), " seed(s)) from ", x$final_run, "\n", sep = "")
  cat(sprintf("  grid       : rows %s, cols %s of %s x %s at %.8g%s\n",
              m$rows, m$cols, format(m$grid_nrow, big.mark = ","),
              format(m$grid_ncol, big.mark = ","), m$cell_size,
              if (isTRUE(m$same_grid_as_store)) "" else "  (NOT the training grid)"))
  cat(sprintf("  valid      : %s of %s cell(s) (%.1f%%)\n", format(m$n_valid, big.mark = ","),
              format(m$n_cells, big.mark = ","), 100 * m$n_valid / max(1, m$n_cells)))
  cat("  engine     : ", paste(sprintf("%dx%d %s", x$windows, x$windows, x$engine), collapse = ", "), "\n", sep = "")
  cat(sprintf("  work       : %d unit(s) x %d row(s) | %s valid px/s per worker | %.1f min this call%s\n",
              m$n_units, m$unit_rows, format(round(m$valid_px_per_s), big.mark = ","), x$minutes,
              if (isTRUE(m$worker_restarts > 0)) {
                sprintf(" | %d worker restart(s) over %.1f GB", m$worker_restarts, m$recycle_gb)
              } else ""))
  cat("  probe      : ", x$probe$status,
      if (!is.null(x$probe$max_rel_diff) && is.finite(x$probe$max_rel_diff))
        sprintf(" (%d profile(s), max relative difference %.2e)", x$probe$n, x$probe$max_rel_diff)
      else if (!is.null(x$probe$reason)) paste0(" -- ", x$probe$reason) else "", "\n", sep = "")
  if (nrow(x$calibration) > 0L) {
    cat("  calibration", if (!is.null(m$weighting)) paste0(" (by ", m$weighting, ")") else "", ":\n",
        sep = "")
    for (i in seq_len(nrow(x$calibration))) {
      r <- x$calibration[i, ]
      cat(sprintf("    %-8s %-8s %s  constant %s | level+DI q %.3g x (%.3g %+.3g x level %+.3g x DI)%s | smearing %s\n",
                  r$source, r$method, r$label,
                  if (is.finite(r$q_constant)) sprintf("+/-%.3g", r$q_constant) else "(per pixel)",
                  r$q_level_di, r$scale_intercept, r$scale_level, r$scale_di,
                  if (is.finite(r$aoa_threshold)) sprintf(" | AOA DI <= %.3f", r$aoa_threshold) else "",
                  if (is.finite(r$smearing_s)) sprintf("%.4f", r$smearing_s) else "-"))
    }
  }
  cat("  bands      : ", nrow(x$bands), " -- one VRT each in the run directory; bands.csv says what each is\n", sep = "")
  invisible(x)
}

# ── helpers: the arguments ────────────────────────────────────────────────────

.predict_whole <- function(x, what, min = 1L) {
  v <- suppressWarnings(as.integer(x))
  if (length(x) != 1L || is.na(v) || v < min || v != x) {
    stop(what, " must be a whole number >= ", min, ", got ",
         paste(deparse(x), collapse = ""), ".", call. = FALSE)
  }
  v
}

.predict_alpha <- function(alpha) {
  if (!is.numeric(alpha) || length(alpha) == 0L || anyNA(alpha) ||
      any(alpha <= 0 | alpha >= 1) || anyDuplicated(alpha)) {
    stop("alpha must be distinct numbers in (0, 1), e.g. 0.1 or c(0.1, 0.05).", call. = FALSE)
  }
  as.numeric(alpha)
}

# The interval's name: pi90 for alpha = 0.1, pi87p5 for 0.125.
.predict_level_label <- function(a) {
  lvl <- 100 * (1 - a)
  if (abs(lvl - round(lvl)) < 1e-9) sprintf("pi%02d", as.integer(round(lvl)))
  else paste0("pi", gsub(".", "p", format(lvl), fixed = TRUE))
}

# THE CLAMP IS THE REFIT'S. predict_loader() clamps every native prediction to
# `clamp` before it is scored, and the stored predictions the probe compares
# against went through it: a map clamped differently would describe another
# model at the pixels where the two differ.
.predict_clamp <- function(clamp, summ) {
  clamp <- clamp %||% summ$training$clamp %||% c(0, Inf)
  if (!is.numeric(clamp) || length(clamp) != 2L || anyNA(clamp) || clamp[1] > clamp[2]) {
    stop("clamp must be c(lower, upper) with lower <= upper, no NA -- e.g. c(0, Inf).",
         call. = FALSE)
  }
  as.numeric(clamp)
}

.predict_final <- function(final, config = NULL) {
  tuning_dir <- NULL
  if (inherits(final, "dsm_final")) {
    run_dir <- final$run_dir
    tuning_dir <- final$tuning_dir
  } else if (is.character(final) && length(final) == 1L) {
    run_dir <- final
  } else {
    stop("`final` must be a dsm_final from dsm_final(), or the directory of a final ",
         "run; got a ", class(final)[1], ".", call. = FALSE)
  }
  run_dir <- normalizePath(run_dir, winslash = "/", mustWork = FALSE)
  summ_path <- file.path(run_dir, "comparison", "final_run_summary.rds")
  if (!file.exists(summ_path)) {
    stop("Not a finished final run, missing ", summ_path, call. = FALSE)
  }
  summ <- readRDS(summ_path)
  if (is.null(tuning_dir) && !is.null(summ$tuning_dir)) tuning_dir <- summ$tuning_dir
  # MOVED WITH ITS FINAL RUN. The summary names the tuning run by absolute path
  # and, since 2026-10-02, also relative to the final run: a project moved
  # whole keeps the two in place, and the relative one finds it.
  if (!is.null(tuning_dir) && !dir.exists(tuning_dir) && !is.null(summ$tuning_dir_rel)) {
    moved <- normalizePath(file.path(run_dir, summ$tuning_dir_rel), winslash = "/", mustWork = FALSE)
    if (dir.exists(moved)) tuning_dir <- moved
  }
  cid <- if (is.null(config)) selected_config_id(summ, basename(run_dir)) else as.character(config)
  if (length(cid) != 1L || !cid %in% summ$selected_cfgs$config_id) {
    stop("config '", paste(cid, collapse = ", "), "' is not a configuration this final run ",
         "fitted (", paste(summ$selected_cfgs$config_id, collapse = ", "), ").", call. = FALSE)
  }
  cfg <- summ$selected_cfgs[summ$selected_cfgs$config_id == cid, , drop = FALSE]
  seeds <- summ$seeds
  if (is.null(seeds) || length(seeds) == 0L) {
    stop("The final run's summary lists no seeds: ", summ_path, call. = FALSE)
  }
  if (!is.null(summ$seeds_fitted) && !setequal(summ$seeds_fitted, seeds)) {
    stop("The final run asked for seeds ", paste(seeds, collapse = ", "), " and fitted ",
         paste(summ$seeds_fitted, collapse = ", "), ": a map must use every seed the ",
         "model is said to have.", call. = FALSE)
  }
  model_files <- file.path(run_dir, cid, "models", sprintf("seed%04d_best.pt", seeds))
  miss <- !file.exists(model_files)
  if (any(miss)) {
    stop("Seed checkpoint(s) missing under ", file.path(run_dir, cid, "models"), ": ",
         paste(basename(model_files[miss]), collapse = ", "), ".\n  A map built from the ",
         "seeds that remain would not be the model the run describes.", call. = FALSE)
  }
  sc_path <- file.path(run_dir, cid, "predictor_scaling.csv")
  if (!file.exists(sc_path)) stop("The model's scaling is missing: ", sc_path, call. = FALSE)
  list(run_dir = run_dir, tuning_dir = tuning_dir, summ = summ, config_id = cid, cfg = cfg,
       seeds = as.integer(seeds), model_files = model_files,
       scaling = safe_read_csv2(sc_path))
}

# The store's tables, the rasters and the QC -- everything in the model's
# channel order, which is the contract tying channel i to band i.
.predict_inputs <- function(data, rasters, qc_table, scaling) {
  type_table <- NULL
  if (inherits(data, "dsm_data")) {
    store_dir  <- data$patch_dir
    recipe     <- data$recipe
    points     <- data$points
    meta       <- data$store$meta
    manifest   <- data$store$manifest
    transform  <- data$transform
    predictors <- as.character(data$store$predictors)
    cell_size  <- data$cell_size
    type_table <- data$type_table
  } else {
    if (inherits(data, "dsm_store")) data <- data$store_dir
    if (!is.character(data) || length(data) != 1L || !dir.exists(data)) {
      stop("`data` must come from dsm_load(), or be the directory of a store written by ",
           "dsm_prepare().", call. = FALSE)
    }
    store_dir <- data
    rp <- file.path(store_dir, "recipe.rds")
    if (!file.exists(rp)) {
      stop(store_dir, " holds no recipe.rds, so it was not written by dsm_prepare() and ",
           "does not carry its own tables. Pass dsm_load(<store>, points, type_table) ",
           "instead.", call. = FALSE)
    }
    recipe   <- readRDS(rp)
    manifest <- readRDS(file.path(store_dir, "patch_manifest.rds"))
    meta     <- safe_read_csv2(file.path(store_dir, "patch_meta.csv"))
    points   <- align_points_to_meta(safe_read_csv2(file.path(store_dir, recipe$files$points)), meta)
    tr <- if ("target_transform" %in% names(manifest)) as.character(manifest$target_transform[1]) else recipe$transform
    transform  <- if (is.null(tr) || is.na(tr)) NULL else target_transform_spec(tr)
    predictors <- strsplit(as.character(manifest$predictor_cols_final[1]), ";")[[1]]
    cell_size  <- recipe$cell_size
    tt <- if (!is.null(recipe$files$type_table)) file.path(store_dir, recipe$files$type_table) else ""
    if (file.exists(tt)) type_table <- safe_read_csv2(tt)
  }
  if (is.null(transform)) {
    stop("The store does not record its target transform, so the map could not be ",
         "back-transformed with the model's own inverse.", call. = FALSE)
  }
  if (is.null(cell_size) && "cell_size" %in% names(manifest)) {
    cell_size <- suppressWarnings(as.numeric(manifest$cell_size[1]))
  }
  gone <- setdiff(predictors, names(points))
  if (length(gone) > 0L) {
    stop("The point table lacks ", length(gone), " predictor column(s) (", paste(utils::head(gone, 5), collapse = ", "),
         "): the dissimilarity index is measured against the profiles' own values.",
         call. = FALSE)
  }

  # The scaling travels with the weights, in their order.
  scaling <- scaling[match(predictors, scaling$predictor), , drop = FALSE]
  if (!identical(as.character(scaling$predictor), predictors)) {
    stop("The model's predictor_scaling.csv does not list the store's channels: the ",
         "network would be fed one channel while the map is built from another.",
         call. = FALSE)
  }
  if (any(!is.finite(scaling$center)) || any(!is.finite(scaling$scale)) || any(scaling$scale <= 0)) {
    stop("The model's scaling has a non-finite centre or a scale <= 0 for: ",
         paste(scaling$predictor[!is.finite(scaling$center) | !is.finite(scaling$scale) |
                                   scaling$scale <= 0], collapse = ", "), call. = FALSE)
  }

  read_table <- function(z, what) {
    if (is.data.frame(z)) return(z)
    if (is.character(z) && length(z) == 1L && file.exists(z) && !dir.exists(z)) return(safe_read_csv2(z))
    stop(what, " must be a data frame or the path of a CSV file.", call. = FALSE)
  }
  store_file <- function(key) {
    if (is.null(recipe) || is.null(recipe$files[[key]])) return(NULL)
    p <- file.path(store_dir, recipe$files[[key]])
    if (file.exists(p)) p else NULL
  }

  qc <- if (is.null(qc_table)) {
    p <- store_file("qc_table")
    if (is.null(p)) {
      stop("This store does not carry its QC rules (it predates dsm_prepare()): pass ",
           "qc_table = <qc_table.csv>.", call. = FALSE)
    }
    safe_read_csv2(p)
  } else read_table(qc_table, "qc_table")
  need <- c("predictor", "na_below", "clamp_lower", "clamp_upper")
  if (!all(need %in% names(qc))) {
    stop("qc_table needs columns ", paste(need, collapse = ", "), ".", call. = FALSE)
  }
  qc <- qc[match(predictors, qc$predictor), , drop = FALSE]
  if (!identical(as.character(qc$predictor), predictors)) {
    stop("qc_table does not list every channel of the model.", call. = FALSE)
  }

  store_rt <- store_file("raster_table")
  rt <- if (is.null(rasters)) {
    if (is.null(store_rt)) {
      stop("This store does not carry its raster table (it predates dsm_prepare()): ",
           "pass rasters = <raster_table_used.csv>, or a directory.", call. = FALSE)
    }
    safe_read_csv2(store_rt)
  } else if (is.character(rasters) && length(rasters) == 1L && dir.exists(rasters)) {
    # The same FILE NAMES in another directory -- the channel order is still
    # the model's, never the order a directory listing returns.
    if (is.null(store_rt)) {
      stop("A raster directory is matched to the store's raster table by file name, and ",
           "this store carries none: pass rasters = a table (predictor, raster_file).",
           call. = FALSE)
    }
    base_rt <- safe_read_csv2(store_rt)
    base_rt$raster_file <- file.path(rasters, basename(base_rt$raster_file))
    base_rt
  } else read_table(rasters, "rasters")
  if (!all(c("predictor", "raster_file") %in% names(rt))) {
    stop("The raster table needs columns predictor and raster_file.", call. = FALSE)
  }
  rt <- rt[match(predictors, rt$predictor), , drop = FALSE]
  if (anyNA(rt$predictor)) {
    stop("The raster table lacks ", sum(is.na(rt$predictor)), " of the model's ",
         length(predictors), " channels. Predicting without one is not possible: the ",
         "network has a weight for every one of them.", call. = FALSE)
  }
  files <- normalizePath(as.character(rt$raster_file), winslash = "/", mustWork = FALSE)
  if (any(!file.exists(files))) {
    stop("Raster file(s) not found: ", paste(utils::head(files[!file.exists(files)], 5),
                                             collapse = ", "),
         "\n  If the folder moved, pass rasters = \"<the folder they are in now>\": they are ",
         "matched to the store's channels by file name.", call. = FALSE)
  }
  list(store_dir = store_dir, points = points, meta = meta, predictors = predictors,
       qc_table = qc, scaling = scaling, files = files, transform = transform,
       type_table = type_table,
       cell_size = as.numeric(cell_size %||% NA_real_),
       raster_nrow = if ("raster_nrow" %in% names(manifest)) as.numeric(manifest$raster_nrow[1]) else NULL,
       raster_ncol = if ("raster_ncol" %in% names(manifest)) as.numeric(manifest$raster_ncol[1]) else NULL)
}

.predict_grid <- function(inp, windows, extent) {
  ref <- terra::rast(inp$files[1])
  bad <- character(0)
  for (i in seq_along(inp$files)[-1L]) {
    if (!isTRUE(terra::compareGeom(ref, terra::rast(inp$files[i]), stopOnError = FALSE))) {
      bad <- c(bad, inp$predictors[i])
    }
  }
  if (length(bad) > 0L) {
    stop(length(bad), " raster(s) do not share the first one's geometry: ",
         paste(utils::head(bad, 8), collapse = ", "), call. = FALSE)
  }
  e <- as.vector(terra::ext(ref))
  g <- list(nrow = as.integer(terra::nrow(ref)), ncol = as.integer(terra::ncol(ref)),
            xmin = e[["xmin"]], xmax = e[["xmax"]], ymin = e[["ymin"]], ymax = e[["ymax"]],
            xres = terra::xres(ref), yres = terra::yres(ref), crs = terra::crs(ref),
            window = max(windows), h = (max(windows) - 1L) %/% 2L)
  rc <- .predict_extent_rc(g, extent)
  g$rows <- rc$rows
  g$cols <- rc$cols
  g$n_rows_out <- rc$rows[2] - rc$rows[1] + 1L
  g$n_cols_out <- rc$cols[2] - rc$cols[1] + 1L
  cs <- inp$cell_size
  g$same_res <- is.finite(cs) && abs(g$xres - cs) <= 1e-9 * max(1, abs(cs))
  g$same_as_store <- g$same_res &&
    (is.null(inp$raster_nrow) || isTRUE(g$nrow == inp$raster_nrow)) &&
    (is.null(inp$raster_ncol) || isTRUE(g$ncol == inp$raster_ncol))
  g
}

# An extent in cells: the rows and columns whose cells it touches.
.predict_extent_rc <- function(g, extent) {
  if (is.null(extent)) return(list(rows = c(1L, g$nrow), cols = c(1L, g$ncol)))
  if (is.list(extent) && !is.null(extent$rows) && !is.null(extent$cols)) {
    r <- as.integer(extent$rows); cc <- as.integer(extent$cols)
    if (length(r) != 2L || length(cc) != 2L || anyNA(c(r, cc)) || r[1] > r[2] || cc[1] > cc[2] ||
        r[1] < 1L || r[2] > g$nrow || cc[1] < 1L || cc[2] > g$ncol) {
      stop("extent rows/cols must be c(first, last) inside the grid (", g$nrow, " x ",
           g$ncol, ").", call. = FALSE)
    }
    return(list(rows = r, cols = cc))
  }
  v <- if (inherits(extent, "SpatExtent")) as.vector(extent)
       else if (inherits(extent, c("SpatRaster", "SpatVector"))) as.vector(terra::ext(extent))
       else if (is.numeric(extent) && length(extent) == 4L) stats::setNames(as.numeric(extent), c("xmin", "xmax", "ymin", "ymax"))
       else stop("extent must be NULL, a SpatExtent, c(xmin, xmax, ymin, ymax), a raster or ",
                 "vector, or list(rows = , cols = ).", call. = FALSE)
  tol <- 1e-6
  c0 <- max(1L, as.integer(floor((v[["xmin"]] - g$xmin) / g$xres + tol)) + 1L)
  c1 <- min(g$ncol, as.integer(ceiling((v[["xmax"]] - g$xmin) / g$xres - tol)))
  r0 <- max(1L, as.integer(floor((g$ymax - v[["ymax"]]) / g$yres + tol)) + 1L)
  r1 <- min(g$nrow, as.integer(ceiling((g$ymax - v[["ymin"]]) / g$yres - tol)))
  if (c0 > c1 || r0 > r1) stop("extent does not overlap the rasters.", call. = FALSE)
  list(rows = c(r0, r1), cols = c(c0, c1))
}

.predict_source_name <- function(method) {
  switch(as.character(method %||% ""),
         spatial_folds = "block", knndm_folds = "knndm", random_folds = "random",
         region_folds = "region", holdout = "holdout", "cv")
}

.predict_sources <- function(calibration, fr) {
  if (isFALSE(calibration)) calibration <- character(0)
  if (is.null(calibration)) {
    if (is.null(fr$tuning_dir)) {
      stop("The intervals are calibrated on a tuning run's cross-validated residuals, ",
           "and this final run does not say where its tuning run is (its summary names '",
           fr$summ$tuning_run_id %||% "?", "' without a path).\n  Pass calibration = ",
           "c(block = \"<tuning run directory>\"), or character(0) for a map without ",
           "intervals.", call. = FALSE)
    }
    pp <- file.path(fr$tuning_dir, "fold_plan.rds")
    nm <- if (file.exists(pp)) .predict_source_name(readRDS(pp)$method) else "cv"
    calibration <- stats::setNames(fr$tuning_dir, nm)
  }
  if (length(calibration) == 0L) return(character(0))
  nm <- names(calibration)
  if (!is.character(calibration) || is.null(nm) || any(!grepl("^[a-z][a-z0-9_]*$", nm)) ||
      anyDuplicated(nm)) {
    stop("calibration must be a named character vector of tuning-run directories, the ",
         "names distinct, lowercase, starting with a letter -- e.g. c(block = \"...\", ",
         "knndm = \"...\"). The names become the bands' suffixes.", call. = FALSE)
  }
  if ("split" %in% nm) {
    stop("\"split\" names the final run's own calibration set in the bands; give the ",
         "tuning run another name.", call. = FALSE)
  }
  dirs <- normalizePath(as.character(calibration), winslash = "/", mustWork = FALSE)
  for (i in seq_along(dirs)) {
    for (f in c("fold_plan.rds", "tune_grid.rds")) {
      if (!file.exists(file.path(dirs[i], f))) {
        stop("calibration source '", nm[i], "' is not a finished tuning run, missing ", f,
             ": ", dirs[i], call. = FALSE)
      }
    }
  }
  stats::setNames(dirs, nm)
}

# WHICH INTERVALS. NULL is what costs nothing: "cv" for every source, and
# "split" when the final run predicted a calibration set. "split" asked for
# where there is none, or "cv_plus" without a source, is refused rather than
# skipped: a map said to carry an interval must carry it.
.predict_intervals <- function(intervals, fr, sources) {
  has_split <- .predict_split_rows(fr, check = TRUE)
  if (is.null(intervals)) return(c("cv", if (has_split) "split"))
  if ("split" %in% intervals && !has_split) {
    stop("intervals = \"split\" needs a calibration set, and this final run has none: its ",
         "tuning plan carved no calibration_frac (spatial_cv(calibration_frac = 0.15), ",
         "and the others), or the run predates it.", call. = FALSE)
  }
  if (any(c("cv", "cv_plus") %in% intervals) && length(sources) == 0L) {
    stop("intervals \"cv\" and \"cv_plus\" are calibrated on a tuning run's folds, and ",
         "calibration = character(0) gives none.", call. = FALSE)
  }
  intervals
}

# The final ensemble's calibration rows (ensemble_predictions.csv): sample_id,
# obs, pred. With check = TRUE, only whether there are any.
.predict_split_rows <- function(fr, check = FALSE) {
  p <- file.path(fr$run_dir, fr$config_id, "ensemble_predictions.csv")
  if (!file.exists(p)) return(if (check) FALSE else NULL)
  e <- safe_read_csv2(p)
  e <- e[e$dataset_role == "calibration", , drop = FALSE]
  if (check) return(nrow(e) >= 2L)
  if (nrow(e) < 2L) NULL else e
}

# ONE CALIBRATION PER SOURCE, from that source's residuals and that source's
# folds: a residual's DI is the distance its own fold model faced, and the AOA
# threshold is the CV DI of the plan whose error it delimits.
# The AOA weights a map was given, one per channel in the model's order --
# from an importance, a named vector, or one in that order already.
.predict_aoa_weights <- function(w, predictors) {
  if (is.null(w)) return(NULL)
  if (inherits(w, "dsm_importance")) w <- importance_weights(w)
  if (!is.numeric(w) || length(w) == 0L || any(!is.finite(w)) || any(w < 0)) {
    stop("aoa_weights must be a dsm_importance, or non-negative numbers, one per channel.",
         call. = FALSE)
  }
  if (!is.null(names(w))) {
    miss  <- setdiff(predictors, names(w))
    extra <- setdiff(names(w), predictors)
    if (length(miss) > 0L || length(extra) > 0L) {
      stop("aoa_weights must name every channel of the model and no other",
           if (length(miss)) paste0("; missing: ", paste(utils::head(miss, 5L), collapse = ", ")) else "",
           if (length(extra)) paste0("; not a channel: ", paste(utils::head(extra, 5L), collapse = ", ")) else "",
           ".", call. = FALSE)
    }
    w <- w[predictors]
  } else if (length(w) != length(predictors)) {
    stop("aoa_weights has ", length(w), " value(s) for ", length(predictors), " channels.",
         call. = FALSE)
  }
  if (sum(w) == 0) stop("aoa_weights are all zero: no axis would be measured.", call. = FALSE)
  stats::setNames(as.numeric(w), predictors)
}

.predict_calibration <- function(sources, fr, inp, alpha, say, weights = NULL,
                                 intervals = "cv", weighting = "point") {
  out <- list()
  for (nm in names(sources)) {
    dir <- sources[[nm]]
    res <- cv_residuals_for_config(dir, fr$cfg, required = TRUE)
    plan <- readRDS(file.path(dir, "fold_plan.rds"))
    if (!isTRUE(as.integer(plan$n_rows) == nrow(inp$points))) {
      stop("calibration source '", nm, "': its fold plan was cut over ", plan$n_rows,
           " point(s) and the store holds ", nrow(inp$points), " -- the plan's indices would ",
           "name other profiles.", call. = FALSE)
    }
    if (!is.null(plan$assignment) && "sample_id" %in% names(plan$assignment)) {
      alien <- setdiff(plan$assignment$sample_id, inp$points$sample_id)
      if (length(alien) > 0L) {
        stop("calibration source '", nm, "': its fold plan names ", length(alien),
             " sample_id(s) this store does not hold.", call. = FALSE)
      }
    }
    aref <- aoa_reference(inp$points, inp$predictors, inp$qc_table, inp$scaling, plan,
                          weights = weights)
    tab <- dplyr::inner_join(res, dplyr::select(aref$cv, sample_id, fold, di = cv_di),
                             by = "sample_id")
    pos <- match(as.character(tab$sample_id), as.character(inp$points$sample_id))
    grp <- if (identical(weighting, "group")) .plan_groups(plan)[pos] else NULL
    iv <- list()
    if ("cv" %in% intervals) {
      for (a in alpha) {
        lab <- .predict_level_label(a)
        const <- conformal_calibrate(tab$obs, tab$pred, alpha = a, group = grp)
        ldi <- tryCatch(
          conformal_scaled_calibrate(tab$obs, tab$pred,
                                     data.frame(level = tab$pred, di = tab$di), alpha = a,
                                     group = grp),
          error = function(e) stop("calibration source '", nm, "': ", conditionMessage(e),
                                   call. = FALSE))
        iv[[lab]] <- list(alpha = a, label = lab, constant = const, level_di = ldi)
      }
    }
    cvp <- if ("cv_plus" %in% intervals) {
      .predict_cv_plus(nm, dir, attr(res, "config_id"), plan, tab, inp, alpha)
    } else NULL
    sm <- if (identical(inp$transform$name, "log1p")) {
      smearing_from_run(dir, attr(res, "config_id"))
    } else NULL
    say(sprintf("\nCalibration '%s': %s (%s, as %s) | %d residual(s), %d with a CV DI | AOA DI <= %.4f%s%s",
                nm, basename(dir), plan$method, attr(res, "config_id"), nrow(res), nrow(tab),
                as.numeric(aref$threshold),
                if (!is.null(weights)) " (importance-weighted)" else "",
                if (!is.null(sm)) sprintf(" | smearing S = %.4f", sm$s) else ""))
    for (lab in names(iv)) {
      say(sprintf("  cv %s: constant +/- %.3f | level+DI: q %.3f x (%.3f %+.4f x level %+.3f x DI), floor %.3f",
                  lab, iv[[lab]]$constant$q, iv[[lab]]$level_di$q, iv[[lab]]$level_di$coef[[1]],
                  iv[[lab]]$level_di$coef[["level"]], iv[[lab]]$level_di$coef[["di"]],
                  iv[[lab]]$level_di$floor))
    }
    if (!is.null(cvp)) {
      say(sprintf("  cv_plus: %d fold model(s) in %d fold(s), %d residual(s); the interval is computed at every pixel",
                  cvp$n_models, length(cvp$folds), cvp$n_residuals))
    }
    out[[nm]] <- list(name = nm, dir = dir, config_id = attr(res, "config_id"),
                      method = plan$method, n_residuals = nrow(res), n_calibration = nrow(tab),
                      aref = aref, threshold = as.numeric(aref$threshold), table = tab,
                      intervals = iv, cv_plus = cvp, smearing = sm)
  }
  split <- if ("split" %in% intervals) {
    .predict_split(fr, inp, alpha, out, weights, weighting, say)
  } else NULL

  # Sources whose references are the same points in the same space share one
  # DI -- the split interval's reference among them.
  refs <- list()
  group_of <- function(r) {
    hit <- which(vapply(refs, function(q) identical(q$x, r$x) && identical(q$avg_dist, r$avg_dist) &&
                          identical(q$weights, r$weights), logical(1)))
    if (length(hit) == 0L) {
      refs[[length(refs) + 1L]] <<- r
      hit <- length(refs)
    }
    hit[1]
  }
  for (nm in names(out)) out[[nm]]$di_group <- group_of(out[[nm]]$aref$ref)
  if (!is.null(split)) split$di_group <- group_of(split$aref$ref)
  list(sources = out, split = split, refs = refs, intervals = intervals, weighting = weighting)
}

# Every entry that measures a DI: the sources, and the split interval.
.predict_cal_members <- function(cal) {
  c(cal$sources, if (!is.null(cal$split)) list(split = cal$split))
}

# CV+ FOR ONE SOURCE: the residuals with their folds, each fold's models, and
# how to turn the map's rows -- scaled by the final model's constants -- into
# what each fold model was trained on. Scaling is per channel, x' = (x - c) /
# s, so a fold's input is the final's times s_final / s_fold, plus (c_final -
# c_fold) / s_fold: one multiply-add over the step's rows, which leaves the
# window's out-of-raster cells at other values than zero -- cells no
# predicted pixel's window holds (.predict_valid()). Each fold's scaling is
# rebuilt as the tuning run built it (fit_scaling() on its training rows), and
# the probe holds every fold model to the predictions its run wrote.
.predict_cv_plus <- function(nm, dir, cid_there, plan, tab, inp, alpha) {
  pat <- paste0("^", cid_there, "_f([0-9]+)_s([0-9]+)_best\\.pt$")
  files <- list.files(file.path(dir, "models"), pattern = pat)
  if (length(files) == 0L) {
    stop("calibration source '", nm, "': intervals = \"cv_plus\" needs its fold models, and the ",
         "tuning run kept none of ", cid_there, " (", file.path(dir, "models"), ").", call. = FALSE)
  }
  if (is.null(inp$type_table)) {
    stop("intervals = \"cv_plus\" rebuilds each fold's scaling from the store's type table, ",
         "and this store does not carry one: pass data = dsm_load(<store>).", call. = FALSE)
  }
  fold <- as.integer(sub(pat, "\\1", files))
  seed <- as.integer(sub(pat, "\\2", files))
  t2 <- tab[!is.na(tab$fold), , drop = FALSE]
  folds <- sort(unique(as.integer(t2$fold)))
  if (length(folds) < 2L) {
    stop("calibration source '", nm, "': CV+ needs two folds at least, and its plan holds ",
         "out ", length(folds), " (", plan$method, "). Leave \"cv_plus\" out of intervals ",
         "for it.", call. = FALSE)
  }
  gone <- setdiff(folds, fold)
  if (length(gone) > 0L) {
    stop("calibration source '", nm, "': fold(s) ", paste(gone, collapse = ", "), " hold ",
         "cross-validated residuals and no model -- CV+ needs the model of every fold its ",
         "residuals came from.", call. = FALSE)
  }
  tt <- inp$type_table[match(inp$predictors, inp$type_table$predictor), , drop = FALSE]
  scal <- lapply(folds, function(k) {
    s <- fit_scaling(inp$points, tt, plan$folds[[k]]$train)
    if (any(s$degenerate)) {
      stop("calibration source '", nm, "': degenerate scaling on fold ", k, "'s training rows ",
           "-- this is not the store the tuning run was fitted on.", call. = FALSE)
    }
    s
  })
  # The residuals divided by the level+DI scale, for the level+DI CV+.
  sc <- conformal_scale_fit(t2$obs, t2$pred, data.frame(level = t2$pred, di = t2$di))
  d_cv <- .conformal_scale(sc$coef, sc$floor, data.frame(level = t2$pred, di = t2$di))
  iv <- list()
  for (a in alpha) {
    lab <- .predict_level_label(a)
    iv[[lab]] <- list(alpha = a, label = lab,
                      constant = cv_plus_calibrate(t2$obs, t2$pred, t2$fold, alpha = a),
                      level_di = cv_plus_calibrate(t2$obs, t2$pred, t2$fold, alpha = a,
                                                   difficulty = d_cv))
  }
  seeds <- lapply(folds, function(k) sort(seed[fold == k]))
  list(folds = folds, seeds = seeds, config_id = cid_there, dir = dir,
       models = lapply(seq_along(folds), function(j)
         file.path(dir, "models", sprintf("%s_f%d_s%d_best.pt", cid_there, folds[j], seeds[[j]]))),
       preds = lapply(seq_along(folds), function(j)
         file.path(dir, "predictions", sprintf("%s_f%d_s%d_pred_all.csv", cid_there, folds[j], seeds[[j]]))),
       a = lapply(scal, function(s) as.numeric(inp$scaling$scale / s$scale)),
       b = lapply(scal, function(s) as.numeric((inp$scaling$center - s$center) / s$scale)),
       intervals = iv, scale = sc, n_models = sum(lengths(seeds)), n_residuals = nrow(t2))
}

# THE SPLIT INTERVAL: the final run's calibration set, predicted by its seeds
# and trained on by none (refit_split()). Its DI is measured as a pixel's is,
# against the final model's own tuning plan; the level+DI scale is fitted on
# that run's cross-validated residuals and q taken on the whole calibration
# set (conformal_scaled_calibrate(scale = )).
.predict_split <- function(fr, inp, alpha, sources, weights, weighting, say) {
  e <- .predict_split_rows(fr)
  tdir <- fr$tuning_dir
  if (is.null(tdir) || !dir.exists(tdir)) {
    stop("The split interval's dissimilarity is measured against the final model's own tuning ",
         "plan, and its tuning run is not found", if (!is.null(tdir)) paste0(" (", tdir, ")"),
         ".", call. = FALSE)
  }
  plan <- readRDS(file.path(tdir, "fold_plan.rds"))
  same <- function(a, b) identical(normalizePath(a, winslash = "/", mustWork = FALSE),
                                   normalizePath(b, winslash = "/", mustWork = FALSE))
  own <- Filter(function(s) same(s$dir, tdir), sources)
  aref <- if (length(own)) own[[1]]$aref else
    aoa_reference(inp$points, inp$predictors, inp$qc_table, inp$scaling, plan, weights = weights)
  pos <- match(as.character(e$sample_id), as.character(inp$points$sample_id))
  if (anyNA(pos)) {
    stop("The final run's calibration set names ", sum(is.na(pos)), " sample_id(s) this store ",
         "does not hold.", call. = FALSE)
  }
  di <- aoa_di(aref, inp$points[pos, inp$predictors, drop = FALSE])
  res <- cv_residuals_for_config(tdir, fr$cfg, required = TRUE)
  tab <- dplyr::inner_join(res, dplyr::select(aref$cv, sample_id, di = cv_di), by = "sample_id")
  sc  <- conformal_scale_fit(tab$obs, tab$pred, data.frame(level = tab$pred, di = tab$di))
  grp <- if (identical(weighting, "group")) .plan_groups(plan)[pos] else NULL
  iv <- list()
  for (a in alpha) {
    lab <- .predict_level_label(a)
    iv[[lab]] <- list(alpha = a, label = lab,
                      constant = conformal_calibrate(e$obs, e$pred, alpha = a, group = grp),
                      level_di = conformal_scaled_calibrate(e$obs, e$pred,
                                                            data.frame(level = e$pred, di = di),
                                                            alpha = a, scale = sc, group = grp))
  }
  sm_path <- file.path(fr$run_dir, fr$config_id, "smearing_split.rds")
  sm <- if (identical(inp$transform$name, "log1p") && file.exists(sm_path)) readRDS(sm_path) else NULL
  say(sprintf("\nCalibration 'split': the final run's %d calibration point(s)%s",
              nrow(e), if (!is.null(sm)) sprintf(" | smearing S = %.4f", sm$s) else ""))
  for (lab in names(iv)) {
    say(sprintf("  split %s: constant +/- %.3f | level+DI: q %.3f x the cross-validated scale",
                lab, iv[[lab]]$constant$q, iv[[lab]]$level_di$q))
  }
  list(name = "split", dir = fr$run_dir, config_id = fr$config_id, method = "split",
       n_residuals = nrow(e), n_calibration = sum(is.finite(di)), aref = aref,
       threshold = NA_real_, intervals = iv, smearing = sm, scale = sc)
}

# One row per source, interval method and level. di_band names the DI band the
# level+DI interval and the AOA were computed from: sources whose fold plans
# used different profiles (a buffer drops some) have different references, and
# so a DI band each. plan is the source's fold plan ("split" for the final
# run's calibration set). A CV+ interval has no single q: it is computed at
# every pixel from the fold models, so its q columns are NA and its scale is
# the one its residuals were divided by.
.predict_calibration_table <- function(cal, band_tbl = NULL) {
  rows <- list()
  one <- function(s, method, iv, q_const, q_ldi, sc) {
    di_band <- NA_character_
    if (!is.null(band_tbl)) {
      b <- band_tbl$band[band_tbl$kind == "di" & band_tbl$group %in% s$di_group]
      if (length(b)) di_band <- b[1]
    }
    tibble::tibble(
      source = s$name, method = method, dir = s$dir, config_id_there = s$config_id,
      plan = s$method, n_residuals = s$n_residuals, n_calibration = s$n_calibration,
      weighting = if (identical(method, "cv_plus")) "point" else (cal$weighting %||% "point"),
      label = iv$label, alpha = iv$alpha, q_constant = q_const, q_level_di = q_ldi,
      scale_intercept = sc$coef[[1]], scale_level = sc$coef[["level"]],
      scale_di = sc$coef[["di"]], scale_floor = sc$floor, scale_r2_fit = sc$r2_fit,
      aoa_threshold = s$threshold, di_group = s$di_group, di_band = di_band,
      smearing_s = if (is.null(s$smearing)) NA_real_ else s$smearing$s)
  }
  for (s in cal$sources) {
    for (iv in s$intervals) {
      rows[[length(rows) + 1L]] <- one(s, "cv", iv, iv$constant$q, iv$level_di$q, iv$level_di)
    }
    for (iv in s$cv_plus$intervals) {
      rows[[length(rows) + 1L]] <- one(s, "cv_plus", iv, NA_real_, NA_real_, s$cv_plus$scale)
    }
  }
  for (iv in cal$split$intervals) {
    rows[[length(rows) + 1L]] <- one(cal$split, "split", iv, iv$constant$q, iv$level_di$q,
                                     iv$level_di)
  }
  dplyr::bind_rows(rows)
}

.predict_band_kinds <- c("ensemble_median", "ensemble_mean", "ensemble_sd", "ensemble_mad",
                         "ensemble_min", "ensemble_max", "smeared_mean", "interval_constant",
                         "interval_level_di", "di", "aoa", "valid_mask")

.predict_band_table <- function(bands, cal, inp, n_seeds) {
  asked <- !is.null(bands)
  want <- bands %||% .predict_band_kinds
  bad <- setdiff(want, .predict_band_kinds)
  if (length(bad) > 0L) {
    stop("Unknown band(s): ", paste(bad, collapse = ", "), ". Known: ",
         paste(.predict_band_kinds, collapse = ", "), ".", call. = FALSE)
  }
  per_source <- c("smeared_mean", "interval_constant", "interval_level_di", "di", "aoa")
  if (length(cal$sources) == 0L && is.null(cal$split) && asked && any(want %in% per_source)) {
    stop("Band(s) ", paste(intersect(want, per_source), collapse = ", "), " need a ",
         "calibration source, and none was given.", call. = FALSE)
  }
  rows <- list()
  # method: the interval's calibration (cv, split, cv_plus); width: constant
  # or level_di.
  add <- function(band, kind, meaning, datatype = "FLT4S", stat = NA_character_,
                  source = NA_character_, alpha = NA_real_, label = NA_character_,
                  method = NA_character_, width = NA_character_, side = NA_character_,
                  group = NA_integer_) {
    rows[[length(rows) + 1L]] <<- tibble::tibble(
      band = band, kind = kind, datatype = datatype, stat = stat, source = source,
      alpha = alpha, label = label, method = method, width = width, side = side,
      group = group, meaning = meaning)
  }
  by_group <- if (identical(cal$weighting, "group")) ", every group of the plan weighing the same" else ""
  # The interval bands of one calibration (cv or split): both widths, both sides.
  add_intervals <- function(ivs, method, src, suffix, from) {
    for (iv in ivs) {
      lvl <- 100 * (1 - iv$alpha)
      if ("interval_constant" %in% want) {
        for (side in c("lower", "upper")) {
          add(sprintf("%s_%s_constant_%s%s", iv$label, method, side, suffix), "interval",
              sprintf("%s bound of the %g%% %s interval, constant width (+/- %.4g), calibrated on %s%s",
                      side, lvl, if (method == "split") "split conformal" else "conformal",
                      iv$constant$q, from, by_group),
              source = src, alpha = iv$alpha, label = iv$label, method = method,
              width = "constant", side = side)
        }
      }
      if ("interval_level_di" %in% want) {
        cf <- iv$level_di$coef
        for (side in c("lower", "upper")) {
          add(sprintf("%s_%s_level_di_%s%s", iv$label, method, side, suffix), "interval",
              sprintf("%s bound of the %g%% %s interval of width %.4g x (%.4g %+.4g x level %+.4g x DI, floor %.4g), calibrated on %s%s",
                      side, lvl, if (method == "split") "split conformal" else "conformal",
                      iv$level_di$q, cf[[1]], cf[["level"]], cf[["di"]], iv$level_di$floor,
                      from, by_group),
              source = src, alpha = iv$alpha, label = iv$label, method = method,
              width = "level_di", side = side)
        }
      }
    }
  }
  stat_meaning <- c(
    median = "median over the seeds of each seed's native prediction: the map's central value. A conditional MEDIAN -- do not sum it for a total",
    mean   = "mean over the seeds of the native predictions: still a conditional median of the target, the seeds do not remove the back-transform bias",
    sd     = "standard deviation between the seeds: how much the answer moves with the initialisation, NOT a prediction interval",
    mad    = "median absolute deviation between the seeds (x 1.4826): as sd, robust",
    min    = "smallest of the seeds' predictions",
    max    = "largest of the seeds' predictions")
  for (st in names(stat_meaning)) {
    b <- paste0("ensemble_", st)
    if (b %in% want) add(b, "ensemble", sprintf("%s (%d seed(s))", stat_meaning[[st]], n_seeds), stat = st)
  }
  for (s in cal$sources) {
    if ("smeared_mean" %in% want && !is.null(s$smearing)) {
      add(paste0("smeared_mean_", s$name), "smeared_mean",
          sprintf("conditional MEAN, Duan's smearing with S = %.4f from '%s' residuals: the only band that may be summed for a total", s$smearing$s, s$name),
          source = s$name)
    }
    add_intervals(s$intervals, "cv", s$name, paste0("_", s$name),
                  sprintf("'%s' cross-validated residuals", s$name))
    cp <- s$cv_plus
    for (iv in cp$intervals) {
      lvl <- 100 * (1 - iv$alpha)
      for (w in intersect(c("interval_constant", "interval_level_di"), want)) {
        width <- sub("^interval_", "", w)
        for (side in c("lower", "upper")) {
          add(sprintf("%s_cv_plus_%s_%s_%s", iv$label, width, side, s$name), "interval",
              sprintf("%s bound of the %g%% CV+ interval (Barber et al. 2021): the %d fold model(s) of '%s' at the pixel, with each one's cross-validated residuals%s",
                      side, lvl, cp$n_models, s$name,
                      if (width == "level_di") sprintf(", scaled by (%.4g %+.4g x level %+.4g x DI, floor %.4g)",
                                                       cp$scale$coef[[1]], cp$scale$coef[["level"]],
                                                       cp$scale$coef[["di"]], cp$scale$floor) else ""),
              source = s$name, alpha = iv$alpha, label = iv$label, method = "cv_plus",
              width = width, side = side)
        }
      }
    }
  }
  if (!is.null(cal$split)) {
    sp <- cal$split
    if ("smeared_mean" %in% want && !is.null(sp$smearing)) {
      add("smeared_mean_split", "smeared_mean",
          sprintf("conditional MEAN, Duan's smearing with S = %.4f from the final run's calibration set: the only band that may be summed for a total", sp$smearing$s),
          source = "split")
    }
    add_intervals(sp$intervals, "split", "split", "",
                  sprintf("the final run's %d calibration point(s)", sp$n_residuals))
  }
  members <- .predict_cal_members(cal)
  if ("di" %in% want && length(cal$refs) > 0L) {
    for (g in seq_along(cal$refs)) {
      users <- names(members)[vapply(members, function(s) identical(s$di_group, g), logical(1))]
      nm <- if (length(cal$refs) == 1L) "di" else paste0("di_", users[1])
      add(nm, "di", sprintf("dissimilarity index (Meyer & Pebesma 2021): distance in the model's scaled predictor space to the nearest of %d training profile(s), over their mean pairwise distance%s",
                           cal$refs[[g]]$n, if (length(users) > 1L) paste0("; shared by ", paste(users, collapse = ", ")) else ""),
          group = g)
    }
  }
  if ("aoa" %in% want) {
    for (s in cal$sources) {
      add(paste0("aoa_", s$name), "aoa",
          sprintf("1 inside the area of applicability (DI <= %.4f, the Q75 + 1.5 IQR of '%s' cross-validated DI), 0 outside, NA where there is no prediction", s$threshold, s$name),
          datatype = "INT1U", source = s$name, group = s$di_group)
    }
  }
  if ("valid_mask" %in% want) {
    add("valid_mask", "valid_mask", "1 where the model predicted (the whole window finite in every channel), 0 elsewhere",
        datatype = "INT1U")
  }
  if (length(rows) == 0L) stop("No band left to write.", call. = FALSE)
  dplyr::bind_rows(rows)
}

# ── helpers: the work ─────────────────────────────────────────────────────────
#
# THE RAM MODEL, per worker, for a step of g rows over W columns of C channels:
#
#   the step's window as float32 -- the kept halo and the new rows, made once
#     per worker (.predict_window()) -- and its masks
#   one band in flight: its R doubles, its float32 tensor, its masks
#   the network's strip and feature maps over one chunk
#   the step's per-pixel R vectors -- the seeds' predictions twice (as read and
#     native), every band's values, four columns per interval computed -- as
#     if every pixel were valid, since a band of land can be
#
# then GDAL's cache (0.5 GB), an R + torch + terra session with the seeds'
# models (2 GB), and the torch garbage the collector leaves between two full
# collections plus the freed blocks torch keeps for reuse (3.5 GB: ~2.4 GB/s
# for 1 s -- P4 measured that rate for the deployed network -- and a cache
# capped at the 1 GB light-collection threshold; see .predict_gc_hook()).
#
# And 0.35 GB for every thread above 7, MEASURED RATHER THAN DERIVED. With 7
# threads the model met T3's peak (14.7 GB against 14.6); with 15 it fell
# short, by 1.3 GB at full width (T3) and by 4.5 GB over Brazil, where one
# worker mapped all 69 units and peaked at 12.1 GB against 7.6 (P5). mimalloc
# keeps what a thread frees in that thread's own heap (T6), so what 7 threads
# share 15 hold apart: the likely cause, not a proven one. 0.25 GB a thread
# was first fitted to fresh workers' first units (P5's first run, where
# restarts kept every worker young); 0.35 is fitted to a worker's whole life,
# the peak the plan must hold. It puts every run measured so far within the
# 25% t3_03 and p5_03 allow: Brazil 12.1 of 10.4, T3 15.9 of 17.4 and 14.7 of
# 14.6, P5's small box 8.0 of 9.6, P4 6.6 of 8.5.
#
# The first version counted neither the halo's clone nor the R vectors, and a
# band's copies as three doubles; on full-width rows T3 measured 24-32 GB
# against its 12.4 (with the reader's own garbage, since fixed). The clone
# left the model with the window that replaced it. An estimate, said to be
# one: the real peak of every worker is measured and written beside it.
# With CV+ (fold_strip), the seeds count every fold model too, and a worker
# holds one more strip-sized float32 copy: the step's rows in a fold's scaling
# (.predict_fold_strip()).
.predict_worker_gb <- function(g, h, w_out, n_ch, chunk_cols, conv_sum, n_seeds = 10L,
                               n_bands = 21L, n_iv = 4L, threads = 7L, fold_strip = FALSE) {
  w_buf <- w_out + 2 * h
  bytes <- (g + 2 * h) * w_buf * (4 * n_ch + 1) +
    g * w_buf * 20 +
    2 * (g + 2 * h) * (min(chunk_cols, w_out) + 2 * h) * (n_ch + 4 * conv_sum) * 4 +
    g * w_out * (8 * (2 * n_seeds + n_bands + 4 * n_iv) + 16) +
    if (fold_strip) (g + 2 * h) * (min(chunk_cols, w_out) + 2 * h) * n_ch * 4 else 0
  bytes / 1e9 + 0.5 + 2 + 3.5 + 0.35 * max(0, threads - 7)
}

.predict_work_plan <- function(grid, inp, cfg, band_tbl, n_cores, tpw, max_ram_gb,
                               unit_rows, step_rows, chunk_cols, say, n_seeds = 10L,
                               fold_strip = FALSE) {
  n_ch <- length(inp$predictors)
  conv_sum <- sum(as.integer(cfg$conv_channels[[1]]))
  iv <- band_tbl[band_tbl$kind == "interval", , drop = FALSE]
  n_iv <- nrow(unique(iv[, c("source", "label", "method", "width"), drop = FALSE]))
  gb <- function(g) .predict_worker_gb(g, grid$h, grid$n_cols_out, n_ch, chunk_cols, conv_sum,
                                       n_seeds = n_seeds, n_bands = nrow(band_tbl), n_iv = n_iv,
                                       threads = tpw, fold_strip = fold_strip)
  n_workers <- max(1L, n_cores %/% tpw)
  budget <- max_ram_gb
  if (is.null(budget) && requireNamespace("ps", quietly = TRUE)) {
    avail <- tryCatch(ps::ps_system_memory()$avail / 1e9, error = function(e) NA_real_)
    if (is.finite(avail)) budget <- 0.7 * avail
  }
  cap <- 16L * as.integer(ceiling(grid$n_rows_out / 16))
  n_asked <- n_workers
  if (is.null(step_rows)) {
    cand <- seq(128L, 16L, by = -16L)
    cand <- cand[cand <= max(16L, cap)]
    # As many workers as fit, then the largest step those fit with: the step
    # was sized for the workers asked for and the workers then cut, which left
    # two workers on the step three would have needed.
    step_rows <- if (is.null(budget)) cand[1] else {
      repeat {
        fit <- cand[vapply(cand, gb, numeric(1)) <= budget / n_workers]
        if (length(fit) > 0L || n_workers == 1L) break
        n_workers <- n_workers - 1L
      }
      if (length(fit)) fit[1] else 16L
    }
    if (n_workers < n_asked) {
      say(sprintf("RAM fits %d of the %d workers asked for (%.1f GB budget, ~%.1f GB each at steps of %d rows). The numbers do not change -- only the time.",
                  n_workers, n_asked, budget, gb(step_rows), step_rows))
    }
  } else if (step_rows %% 16L != 0L) {
    step_rows <- 16L * as.integer(ceiling(step_rows / 16))
    say("step_rows rounded up to ", step_rows, ": a step writes whole tile rows, and ",
        "GeoTIFF tiles are multiples of 16.")
  }
  # A step at least as tall as the halo: the next step's halo is then the last
  # 2h rows of this one, moved within the window (.predict_unit()). Only a
  # unit's last step may be shorter, and it keeps no halo.
  if (step_rows < 2L * grid$h) {
    step_rows <- 16L * as.integer(ceiling(2 * grid$h / 16))
    say("step_rows raised to ", step_rows, ": a step keeps the ", 2L * grid$h,
        " rows of its halo.")
  }
  if (!is.null(budget)) {
    fits <- max(1L, as.integer(floor(budget / gb(step_rows))))
    if (fits < n_workers) {
      say(sprintf("RAM caps the workers at %d of %d (%.1f GB budget, ~%.1f GB each). The numbers do not change -- only the time.",
                  fits, n_workers, budget, gb(step_rows)))
      n_workers <- fits
    }
    if (gb(step_rows) > budget) {
      say(sprintf("WARNING: one worker is estimated at %.1f GB against a budget of %.1f GB. It is started anyway.",
                  gb(step_rows), budget))
    }
  }
  idle <- n_cores - n_workers * tpw
  if (n_workers < n_asked && idle >= n_workers) {
    say(sprintf("  %d of the %d cores would sit idle: threads_per_worker = %d would use them.",
                idle, n_cores, n_cores %/% n_workers))
  }
  if (is.null(unit_rows)) {
    k <- max(1L, min(8L, as.integer(ceiling(grid$n_rows_out / (4 * n_workers * step_rows)))))
    unit_rows <- k * step_rows
  } else if (unit_rows %% step_rows != 0L) {
    unit_rows <- step_rows * as.integer(ceiling(unit_rows / step_rows))
    say("unit_rows rounded up to ", unit_rows, ", a multiple of step_rows (", step_rows, ").")
  }
  blocky <- max(Filter(function(b) step_rows %% b == 0L, seq(16L, 256L, by = 16L)))
  n_units <- as.integer(ceiling(grid$n_rows_out / unit_rows))
  list(n_workers = max(1L, min(n_workers, n_units)), threads = tpw, step_rows = step_rows,
       unit_rows = as.integer(unit_rows), chunk_cols = chunk_cols, blocky = blocky,
       per_worker_gb = gb(step_rows), budget_gb = budget %||% NA_real_)
}

# WHEN A WORKER GIVES ITS MEMORY BACK.
#
# mimalloc, under libtorch on Windows, keeps what it frees (see
# .predict_window()), so a worker's working set can only grow, and the global
# map runs a worker through ~250 units. The window takes the large tensors
# out of that, but not every allocation a step makes, and a slow climb over
# two days would end the run where no test could see it coming. So a worker
# whose working set, after the full collection that ends a unit, is above its
# share of the RAM budget -- with no budget, 25% above the RAM model's
# estimate -- exits, and the run starts a fresh one in its place
# (.predict_run_units()). A restart costs seconds; a unit of the global map,
# minutes.
#
# THE SHARE, NOT THE ESTIMATE. The first version took the tighter of the two,
# 1.25 x the estimate, and P5 showed what that does: over Brazil, with 15
# threads, a fresh worker already held ~9.5 GB after its first unit where the
# RAM model said 7.6, and a healthy worker was restarted 27 times in 69 units.
# The estimate is a model, and sizes the plan; the share is what the plan
# promised to stay inside, and the only thing a restart should defend. Never
# below the estimate itself: a worker estimated above its share (the plan
# warned) would otherwise restart after every unit.
.predict_recycle_gb <- function(work) {
  share <- work$budget_gb / max(1L, work$n_workers)
  if (is.finite(share)) max(work$per_worker_gb, share) else 1.25 * work$per_worker_gb
}

.predict_units <- function(rows, cols, unit_rows) {
  starts <- seq(rows[1], rows[2], by = unit_rows)
  tibble::tibble(unit_id = sprintf("u%05d", seq_along(starts)), r0 = as.integer(starts),
                 r1 = as.integer(pmin(starts + unit_rows - 1L, rows[2])),
                 c0 = as.integer(cols[1]), c1 = as.integer(cols[2]))
}

# WHAT MAKES TWO RUNS THE SAME MAP. Anything that changes a number or the
# layout of the files: the model, the rasters, the part, the units, the bands
# and every number that calibrated them. Not the workers, threads, steps or
# chunks -- those change the time.
# A MAP IS THE SAME MAP WHEREVER ITS FOLDERS ARE. The settings name the final
# run, the rasters and the calibration runs by absolute path, and a resume
# compared them as written: when the project and the rasters moved
# (2026-10-01), every map would have been refused, the finished units with it.
# Compared here by what identifies them -- the final run's and the calibration
# runs' folder names, the rasters' file names, and, unchanged, the model's
# bytes, the seeds, the grid and every number of the calibration. The
# settings on disk keep the full paths, for the record.
.predict_settings_portable <- function(x) {
  x$final_run <- basename(as.character(x$final_run))
  x$rasters   <- basename(as.character(x$rasters))
  if (!is.null(x$calibration)) {
    x$calibration <- lapply(x$calibration, function(v) {
      v[1] <- basename(as.character(v[1]))
      v
    })
  }
  x[setdiff(names(x), "step_rows_used")]
}

.predict_settings <- function(fr, inp, grid, work, band_tbl, alpha, clamp, engine, cal,
                              intervals = "cv", weighting = "point") {
  list(final_run = fr$run_dir, config_id = fr$config_id, seeds = fr$seeds,
       model_bytes = as.numeric(file.size(fr$model_files)), rasters = inp$files,
       predictors = inp$predictors,
       grid = c(grid$nrow, grid$ncol, grid$xmin, grid$ymax, grid$xres, grid$yres),
       rows = grid$rows, cols = grid$cols, unit_rows = work$unit_rows,
       bands = band_tbl$band, alpha = alpha, clamp = clamp, engine = engine,
       intervals = intervals, weighting = weighting,
       calibration = lapply(.predict_cal_members(cal), function(s) c(
         s$dir, s$config_id, s$threshold,
         unlist(lapply(s$intervals, function(iv) c(iv$constant$q, iv$level_di$q, iv$level_di$coef))),
         # CV+ is computed at every pixel: what makes two maps the same is the
         # fold models, their residuals and the scale.
         if (!is.null(s$cv_plus)) c(s$cv_plus$n_models, s$cv_plus$n_residuals,
                                    unlist(s$cv_plus$seeds), s$cv_plus$scale$coef,
                                    unlist(lapply(s$cv_plus$intervals, function(iv)
                                      c(iv$constant$r, sum(iv$constant$residuals))))),
         if (is.null(s$smearing)) NA_real_ else s$smearing$s)))
}

.predict_job <- function(fr, inp, grid, work, band_tbl, cal, clamp, engine, tpw, run_dir) {
  list(cfg = fr$cfg, n_channels = length(inp$predictors), model_files = fr$model_files,
       seeds = fr$seeds, files = inp$files, rules = as.data.frame(inp$qc_table),
       center = as.numeric(inp$scaling$center), scale = as.numeric(inp$scaling$scale),
       transform = inp$transform$name, clamp = clamp,
       grid = grid[c("nrow", "ncol", "xmin", "ymax", "xres", "yres", "crs")],
       h = grid$h, step_rows = work$step_rows, chunk_cols = work$chunk_cols,
       batch = 4096L, gather_mb = 256, col_quantum = min(512L, work$chunk_cols),
       engine = engine, bands = band_tbl,
       di = .predict_di_plan(cal),
       # The split interval is a source of its own here, named "split" (no
       # tuning source may take the name, .predict_sources()).
       sources = lapply(.predict_cal_members(cal), function(s) list(
         name = s$name, di_group = s$di_group, threshold = s$threshold,
         smearing = s$smearing, intervals = s$intervals,
         cv_plus = if (is.null(s$cv_plus)) NULL else
           s$cv_plus[c("folds", "seeds", "models", "preds", "a", "b", "intervals", "scale",
                       "n_models", "config_id")])),
       threads = tpw, gdal_cache_mb = 512L, blocky = work$blocky,
       gc_threshold_mb = 1000L, gc_every_s = 1,
       units_dir = file.path(run_dir, "units"), probe_cells = NULL,
       # A worker whose working set is above this after a unit exits, and the
       # run starts a fresh one in its place (see .predict_recycle_gb()).
       recycle_gb = getOption("dsm.predict.recycle_gb", .predict_recycle_gb(work)),
       # options(dsm.predict.trace_mem = TRUE): every unit records the worker's
       # CURRENT working set after each phase of each step (see .predict_unit).
       trace_mem = isTRUE(getOption("dsm.predict.trace_mem", FALSE)))
}

.predict_done_path <- function(units_dir, unit_id) file.path(units_dir, unit_id, "done.rds")

# The workers: as dsm_final()'s, each its own R process with its thread count
# set before torch loads, claiming units by dir.create() (atomic: made or
# found made). A unit's numbers depend on the unit and not on the worker --
# which is what lets a worker that gave its memory back be replaced by a
# fresh one mid-run (see .predict_recycle_gb()).
.predict_run_units <- function(job, units, n_workers, run_dir, tag, say, progress = TRUE) {
  if (!requireNamespace("callr", quietly = TRUE)) {
    stop("dsm_predict() maps in worker processes and needs the callr package. ",
         "install.packages(\"callr\").", call. = FALSE)
  }
  loader <- .pkg_loader()          # the framework this session runs, for each worker
  claims_dir <- file.path(run_dir, paste0(".claims_", tag))
  logs_dir   <- file.path(run_dir, "logs")
  unlink(claims_dir, recursive = TRUE)          # stale claims of a run that died
  create_output_dirs(c(claims_dir, logs_dir, job$units_dir))
  job$claims_dir <- claims_dir
  job$units <- units
  .pkg_check_portable(job, "dsm_predict()")
  done_paths <- .predict_done_path(job$units_dir, units$unit_id)
  cells <- as.numeric(units$r1 - units$r0 + 1L) * (units$c1 - units$c0 + 1L)

  t0 <- Sys.time()
  # A slot's workers in turn: its first, then each fresh one that took the
  # place of one that gave its memory back, logged apart.
  log_of <- function(w, gen) {
    file.path(logs_dir, sprintf("%s_worker_%02d%s.log", tag, w,
                                if (gen > 1L) sprintf("_%02d", gen) else ""))
  }
  start <- function(w, gen) {
    callr::r_bg(
      .predict_worker_entry,
      args = list(job = c(job, list(worker = w)), loader = loader),
      # The primitive caches of ideep (LRU_CACHE_CAPACITY) and oneDNN
      # (ONEDNN_PRIMITIVE_CACHE_CAPACITY) default to 1,024 entries each, and
      # an entry holds buffers sized to its input: bounded here, beside the
      # few strip shapes the unit loop keeps to (see "THE STRIP'S SHAPE").
      env = c(callr::rcmd_safe_env(),
              OMP_NUM_THREADS = as.character(job$threads),
              MKL_NUM_THREADS = as.character(job$threads),
              LRU_CACHE_CAPACITY = "64",
              ONEDNN_PRIMITIVE_CACHE_CAPACITY = "64",
              # A diagnosis can add to the workers' environment (T4 does).
              getOption("dsm.predict.worker_env", character(0))),
      stdout = log_of(w, gen), stderr = "2>&1", supervise = TRUE)
  }
  gen <- rep(1L, n_workers)
  procs <- lapply(seq_len(n_workers), start, gen = 1L)
  # An interrupted map must not leave workers writing into its units: the next
  # call deletes the claims and would hand the same units out again.
  on.exit(for (p in procs) if (p$is_alive()) p$kill(), add = TRUE)

  seen <- rep(FALSE, nrow(units))
  valid_done <- 0
  res <- vector("list", n_workers)
  over <- rep(FALSE, n_workers)
  peaks <- numeric(0)
  restarts <- 0L
  repeat {
    alive <- vapply(procs, function(p) p$is_alive(), logical(1))
    now <- file.exists(done_paths)
    fresh <- which(now & !seen)
    if (length(fresh) > 0L) {
      for (i in fresh) {
        r <- tryCatch(readRDS(done_paths[i]), error = function(e) NULL)
        if (!is.null(r)) valid_done <- valid_done + r$n_valid
      }
      seen[fresh] <- TRUE
      if (progress) {
        el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
        rate <- sum(cells[seen]) / max(el, 1e-9)
        eta <- (sum(cells) - sum(cells[seen])) / max(rate, 1e-9) / 60
        say(sprintf("  %7.1f min | %d of %d unit(s) | %s valid px | %s valid px/s | ETA %.0f min",
                    el / 60, sum(seen), nrow(units), format(valid_done, big.mark = ","),
                    format(round(valid_done / max(el, 1e-9)), big.mark = ","), eta))
      }
    }
    for (w in which(!alive & !over)) {
      r <- tryCatch(procs[[w]]$get_result(), error = function(e) e)
      if (!inherits(r, "error")) peaks <- c(peaks, as.numeric(r$peak_gb %||% NA_real_))
      # Units no worker has claimed: a worker that stopped on an error is not
      # restarted, and one that gave its memory back only while any are left.
      unclaimed <- !file.exists(done_paths) & !dir.exists(file.path(claims_dir, units$unit_id))
      if (!inherits(r, "error") && isTRUE(r$recycled) && any(unclaimed)) {
        say(sprintf("  worker %d gave its memory back (%.1f GB after a unit, above %.1f GB): a fresh one takes its place.",
                    w, r$rss_gb, job$recycle_gb))
        gen[w] <- gen[w] + 1L
        restarts <- restarts + 1L
        procs[[w]] <- start(w, gen[w])
      } else {
        res[[w]] <- r
        over[w] <- TRUE
      }
    }
    if (all(over)) break
    Sys.sleep(if (progress) 5 else 1)
  }
  errs <- vapply(res, function(r) inherits(r, "error"), logical(1))
  for (w in which(errs)) {
    say("  worker ", w, " stopped with an error: ", conditionMessage(res[[w]]),
        "\n    log: ", log_of(w, gen[w]))
  }
  peaks <- peaks[is.finite(peaks)]
  unlink(claims_dir, recursive = TRUE)
  list(n_workers = n_workers, restarts = restarts,
       peak_gb = if (length(peaks)) max(peaks) else NA_real_,
       minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
}

# At the top level on purpose, as .final_worker_entry(): nothing a worker
# does may depend on a frame serialised along with it.
.predict_worker_entry <- function(job, loader) {
  # BEFORE torch loads -- the threshold is read once, when torch starts, and
  # loading the framework loads torch, which it imports. torch then runs a
  # LIGHT collection every job$gc_threshold_mb of new allocations (lantern's
  # CPU allocator, src/lantern/src/Allocator.cpp), and caches freed blocks up
  # to the same amount. See .predict_gc_hook() for the full collections.
  options(torch.threshold_call_gc = job$gc_threshold_mb)
  ns <- loader$open(loader)
  get(".predict_worker", envir = ns)(job)
}

# THE COLLECTOR THE MAP RUNS, and why it has to run a FULL collection.
#
# R frees a tensor only when its own collector finds it unreachable, and R
# cannot see a tensor's size. torch compensates with a light collection every
# N MB it allocates -- but a light collection only reaches the youngest
# generation, and every tensor still in use when one runs (the seed being
# computed) is promoted past it, to die later where only a full collection
# looks. P4's first run showed the result: 8-13 GB per worker for the fully
# convolutional path, 17 GB for the patch-by-patch one, against ~6 estimated.
#
# So: a light collection at every call (milliseconds), and a full one every
# `every_s` seconds. Garbage is allocated at a rate the computation sets, so a
# clock bounds it as well as a byte count would, without a model of every
# layer. The rate is not small: every BatchNorm and SiLU of the deployed
# network allocates a new map, ~1.9 GB per seed over a 20 km strip, ~2.4 GB/s
# per worker -- and at 2 s the 20 km map still peaked at 9.3 GB (P4, second
# run). 1 s halves what is left; a full collection of a worker's session costs
# tens of ms, a few percent of that second.
.predict_gc_hook <- function(every_s) {
  last <- Sys.time()
  function() {
    if (as.numeric(difftime(Sys.time(), last, units = "secs")) >= every_s) {
      invisible(gc(verbose = FALSE, full = TRUE))
      last <<- Sys.time()
    } else {
      invisible(gc(verbose = FALSE, full = FALSE))
    }
    invisible(NULL)
  }
}

# The worker's current working set, in GB: what it holds now, not its peak.
.predict_rss_gb <- function() {
  if (!requireNamespace("ps", quietly = TRUE)) return(NA_real_)
  mi <- tryCatch(ps::ps_memory_info(ps::ps_handle()), error = function(e) NULL)
  if (is.null(mi)) NA_real_ else as.numeric(mi[["rss"]]) / 1e9
}

# Its private memory, in GB: what it has committed, in RAM or not. Windows
# moves a process's pages out of its working set when another program needs
# the RAM -- over Brazil one worker's working set fell 2.5 GB between two
# units while the PC was in use (P5) -- so a leak can hide from the working
# set, but not from the private memory. NA where ps does not report it (it
# does on Windows, as mem_private -- psutil's name is private, and the first
# version looked for that one, finding nothing).
.predict_private_gb <- function() {
  if (!requireNamespace("ps", quietly = TRUE)) return(NA_real_)
  mi <- tryCatch(ps::ps_memory_info(ps::ps_handle()), error = function(e) NULL)
  nm <- intersect(c("mem_private", "private"), names(mi))
  if (length(nm) == 0L) NA_real_ else as.numeric(mi[[nm[1]]]) / 1e9
}

.predict_worker <- function(job) {
  env <- .predict_worker_setup(job)
  on.exit(for (s in env$srcs) try(terra::readStop(s), silent = TRUE), add = TRUE)
  for (u in seq_len(nrow(job$units))) {
    uid  <- job$units$unit_id[u]
    done <- .predict_done_path(job$units_dir, uid)
    if (file.exists(done)) next
    if (!dir.create(file.path(job$claims_dir, uid), showWarnings = FALSE)) next
    udir <- file.path(job$units_dir, uid)
    create_output_dirs(udir)
    unlink(file.path(udir, "failed.rds"))
    message(sprintf("\n-- unit %s: rows %d-%d, cols %d-%d (worker %d, %d thread(s)) --",
                    uid, job$units$r0[u], job$units$r1[u], job$units$c0[u], job$units$c1[u],
                    job$worker, job$threads))
    rec <- tryCatch(.predict_unit(job$units[u, , drop = FALSE], env, job), error = function(e) e)
    # A full collection, then the working set it leaves: what the worker holds
    # between units, which the record keeps (rss_gb) and a restart is judged
    # on. A peak says only how high a unit once went, with whatever garbage
    # sat between two collections -- over Brazil, fresh workers' first-unit
    # peaks spread from 8.9 to 11.1 GB on units alike (P5).
    invisible(gc(verbose = FALSE))
    rss <- .predict_rss_gb()
    if (inherits(rec, "error")) {
      message("  ERROR in ", uid, ": ", conditionMessage(rec))
      safe_save_rds(list(unit_id = uid, status = "failed", error = conditionMessage(rec),
                         worker = job$worker, at = Sys.time()),
                    file.path(udir, "failed.rds"), compress = FALSE)
    } else {
      rec$worker  <- job$worker
      rec$threads <- job$threads
      rec$peak_gb <- .final_peak_gb()
      # The process too: a slot's worker is replaced after a restart, and a
      # resumed map starts new ones, so memory is followed by process.
      rec$pid     <- Sys.getpid()
      rec$rss_gb  <- rss
      rec$private_gb <- .predict_private_gb()
      # The record LAST: its existence is what says the unit is finished.
      safe_save_rds(rec, done, compress = FALSE)
      s <- rec$seconds
      message(sprintf("  %s valid px of %s in %.1f s (read %.1f | network %.1f | DI %.1f | bands %.1f | write %.1f)",
                      format(rec$n_valid, big.mark = ","), format(rec$n_cells, big.mark = ","),
                      rec$total_s, s[["read"]], s[["net"]], s[["di"]], s[["bands"]], s[["write"]]))
    }
    if (is.finite(rss) && is.finite(job$recycle_gb %||% NA_real_) && rss > job$recycle_gb) {
      message(sprintf("  working set %.1f GB after %s, above %.1f GB: this worker exits, and a fresh one takes its place.",
                      rss, uid, job$recycle_gb))
      # A note beside the unit it followed, so a restart is known after the
      # run -- a resumed map's manifest counts only its own call's.
      safe_save_rds(list(unit_id = uid, rss_gb = rss, limit_gb = job$recycle_gb,
                         pid = Sys.getpid(), at = Sys.time()),
                    file.path(udir, "recycled.rds"), compress = FALSE)
      return(list(worker = job$worker, peak_gb = .final_peak_gb(), recycled = TRUE, rss_gb = rss))
    }
  }
  list(worker = job$worker, peak_gb = .final_peak_gb(), recycled = FALSE)
}

.predict_worker_setup <- function(job) {
  set_torch_threads(job$threads)
  tryCatch(terra::gdalCache(job$gdal_cache_mb), error = function(e) NULL)
  # GDAL decodes the strips of one read on several threads (GDAL >= 3.6); the
  # torch threads are idle while a band is read, so this costs nothing. A
  # GDAL_NUM_THREADS already in the worker's environment wins (T4 sets it).
  if (!nzchar(Sys.getenv("GDAL_NUM_THREADS"))) {
    tryCatch(terra::setGDALconfig("GDAL_NUM_THREADS", as.character(job$threads)),
             error = function(e) NULL)
  }
  models <- lapply(job$model_files, function(f) {
    m <- build_cnn_from_config(job$cfg, job$n_channels)
    m$load_state_dict(torch::torch_load(f))
    m$eval()
    m
  })
  srcs <- lapply(job$files, terra::rast)
  for (s in srcs) terra::readStart(s)
  rules <- job$rules
  has_rule <- !is.na(rules$na_below) | !is.na(rules$clamp_lower) | !is.na(rules$clamp_upper)
  di <- lapply(job$di, function(u) {
    x32 <- torch::torch_tensor(u$x, dtype = torch::torch_float32())
    list(xt_t = x32$t()$contiguous(), r2 = (x32 * x32)$sum(dim = 2L)$unsqueeze(1L),
         x64 = torch::torch_tensor(u$x, dtype = torch::torch_float64()),
         sqrt_w = torch::torch_tensor(sqrt(u$weights), dtype = torch::torch_float32())$unsqueeze(1L),
         groups = u$groups)
  })
  n_di <- sum(vapply(job$di, function(u) length(u$groups), integer(1)))
  # Whether torch hands back 0- or 1-based indices, asked rather than assumed:
  # an off-by-one here would pair every pixel with the wrong profile.
  index_base <- 2L - as.integer(torch::torch_argmin(torch::torch_tensor(c(3, 1, 2))))
  # CV+: every source's fold models, each fold with the multiply-add that
  # takes the step's rows from the final model's scaling to its own.
  fold_models <- list()
  for (src in job$sources) {
    cp <- src$cv_plus
    if (is.null(cp)) next
    fold_models[[src$name]] <- lapply(seq_along(cp$folds), function(j) list(
      fold = cp$folds[j],
      a = torch::torch_tensor(cp$a[[j]], dtype = torch::torch_float32())$view(c(1L, -1L, 1L, 1L)),
      b = torch::torch_tensor(cp$b[[j]], dtype = torch::torch_float32())$view(c(1L, -1L, 1L, 1L)),
      models = lapply(cp$models[[j]], function(f) {
        m <- build_cnn_from_config(job$cfg, job$n_channels)
        m$load_state_dict(torch::torch_load(f))
        m$eval()
        m
      })))
  }
  list(models = models, srcs = srcs, n_ch = length(job$files), n_seeds = length(models),
       rules = rules, has_rule = has_rule, inverse = target_transform_spec(job$transform)$inverse,
       di = di, n_di = n_di, index_base = index_base, gc_hook = .predict_gc_hook(job$gc_every_s),
       fcn_engine = if (identical(job$engine, "patch")) "patch" else "fcn",
       fold_models = fold_models,
       # The worker's step window, kept across its units (.predict_window()):
       # an environment, so one unit's allocation is the next unit's to reuse.
       bufs = new.env(parent = emptyenv()))
}

# THE FOLD'S COPY OF THE STRIP, made once per strip shape and kept by the
# worker, as the step's window is (.predict_window()): every fold of every
# chunk writes into it -- the strip, times the fold's a, plus its b -- and no
# strip-sized tensor is allocated per fold. The strips come in a handful of
# shapes (the columns are rounded to quanta), so a new one is rare.
.predict_fold_strip <- function(env, x4) {
  shp <- as.integer(x4$shape)
  key <- paste(shp, collapse = "x")
  b <- env$bufs
  if (!identical(b$fold_key, key)) {
    b$fold_strip <- NULL
    invisible(gc(verbose = FALSE, full = TRUE))
    b$fold_strip <- torch::torch_empty(shp, dtype = torch::torch_float32())
    b$fold_key <- key
  }
  b$fold_strip
}

# THE STEP'S WINDOW LIVES IN THE WORKER, MADE ONCE.
#
# libtorch on Windows allocates CPU memory with mimalloc (2.2.3 here, built
# into c10.dll), and T6 found that it neither gives back nor reuses the memory
# of a freed block of a few hundred MB: a 1 GB tensor made and dropped three
# times left 3 GB behind, 0.5 GB ones 1.5 GB, while blocks of ~20 MB were
# reused. R let go of every tensor -- T6 counted their finalizers -- emptying
# a tensor's storage by hand gave nothing back, and neither one thread nor
# mimalloc's own options (purge at once, no arenas) changed it. Its purge
# never runs in 2.2.3: mi_arenas_try_purge() returns when the delay HAS
# expired (arenas_expire < now; upstream's dev branch has > now). A full-width
# step allocated its rows (3.7 GB), its first halo (1.6 GB) and the next
# halo's clone (1.6 GB) afresh, and each worker grew by about that a step.
# Holding the rows in blocks of channels under 2^31 bytes, the guess T5's
# first run suggested, changed nothing: its second run kept a 1.0 GB tensor
# like a 3.7 GB one.
#
# So the halo and the step's rows are one tensor, [halo; rows], made once in
# the worker and kept for all its units: every read writes into it, the
# network's strips are copies of its columns, and a step's last 2h rows move
# to its top for the next step. Nothing that size is allocated again; a unit
# of another shape (a part narrower than the map) makes its own, once.
.predict_window <- function(env, n_ch, rows, w_buf) {
  key <- paste(n_ch, rows, w_buf, sep = "x")
  b <- env$bufs
  if (!identical(b$key, key)) {
    b$W <- NULL
    invisible(gc(verbose = FALSE, full = TRUE))
    b$W <- torch::torch_zeros(c(n_ch, rows, w_buf))
    b$key <- key
  }
  b$W
}

# Rows of every band, QC'd and scaled, as float32 over the unit's buffer
# columns, with the mask of cells finite in every channel: written into
# `into` from its row at + 1 (the window), or into a tensor of their own.
# Rows and columns outside the raster are 0 and masked: the full-window rule
# then discards any centre whose window leaves the raster, which is the rule
# stage 05 applied.
.predict_read_rows <- function(rows, rc0, rc1, pad_l, pad_r, w_buf, env, job, into = NULL,
                               at = 0L) {
  nr <- length(rows)
  x <- into %||% torch::torch_empty(c(env$n_ch, nr, w_buf))
  if (is.null(into)) at <- 0L
  inside <- rows >= 1L & rows <= job$grid$nrow
  if (!any(inside)) {
    x[, (at + 1L):(at + nr), ] <- 0
    return(list(x = x, fin = torch::torch_zeros(c(nr, w_buf), dtype = torch::torch_bool())))
  }
  ra <- min(rows[inside]); rn <- sum(inside)
  pads <- c(pad_l, pad_r, ra - rows[1], nr - (ra - rows[1]) - rn)
  wn <- rc1 - rc0 + 1L
  fok <- NULL
  for (k in seq_len(env$n_ch)) {
    v <- terra::readValues(env$srcs[[k]], row = ra, nrows = rn, col = rc0, ncols = wn, mat = FALSE)
    if (env$has_rule[k]) v <- qc_band_values(v, env$rules[k, , drop = FALSE])
    # FLOAT32 FROM HERE, AND IN PLACE. A band of a full-width step is ~5
    # million cells, and the double tensor and the four copies this used to
    # make of it (scaled, masked, cast, padded) were ~300 MB per band, 181
    # times per step -- with torch's light collections promoting what was
    # alive, T3 saw workers reach 24-32 GB. Now one float32 tensor, scaled and
    # masked in place, and a collection after every band. float32 is also how
    # training scaled its patches (scale_patches(), in the tensor's dtype);
    # the raster's float32 values convert exactly.
    t <- torch::torch_tensor(v, dtype = torch::torch_float32())$view(c(rn, wn))
    rm(v)
    t$sub_(job$center[k])$div_(job$scale[k])
    f <- torch::torch_isfinite(t)
    fok <- if (is.null(fok)) f else torch::torch_logical_and(fok, f)
    t$masked_fill_(f$logical_not(), 0)
    # `:` inside torch's brackets is a slice: this writes into x's own rows.
    x[k, (at + 1L):(at + nr), ] <- if (any(pads > 0L)) torch::nnf_pad(t, pads) else t
    rm(t, f)
    env$gc_hook()
  }
  fin <- if (any(pads > 0L)) {
    torch::nnf_pad(fok$to(dtype = torch::torch_uint8()), pads)$to(dtype = torch::torch_bool())
  } else fok
  list(x = x, fin = fin)
}

# Which of a step's pixels are predicted: every cell of the largest window
# finite in every channel -- a max-pool of the "not finite" mask.
.predict_valid <- function(fin_full, h, gs, w_out) {
  nf <- fin_full$logical_not()$to(dtype = torch::torch_float32())$unsqueeze(1L)$unsqueeze(1L)
  if (h > 0L) {
    nf <- torch::nnf_max_pool2d(nf, kernel_size = c(2L * h + 1L, 2L * h + 1L), stride = c(1L, 1L))
  }
  v <- as.array(nf$reshape(c(gs, w_out)) == 0)
  matrix(as.logical(v), gs, w_out)
}

# ONE DISTANCE MATRIX FOR EVERY REFERENCE.
#
# Sources whose fold plans kept different profiles have different references
# -- but the same profiles in the same scaled space: the SOC block plan's
# 3,092 are all among the kNNDM plan's 3,137. A matrix product per reference
# computed the same distances twice, and the DI took 25% of T3's full-width
# map. So the distances go to the UNION of the references' profiles, once,
# and each reference takes its nearest profile among its own columns -- the
# same neighbour, the same distance, the same DI. References the union cannot
# hold (another weighting, or rows of one profile that differ) keep a matrix
# of their own.
#
# A list of unions, each list(x = the union's weighted rows, weights, groups =
# list(list(g = the DI group, cols = its rows in x, all = every row?,
# avg_dist))).
.predict_di_plan <- function(cal) {
  n_g <- length(cal$refs)
  if (n_g == 0L) return(list())
  alone <- function(g) {
    r <- cal$refs[[g]]
    list(x = r$x, weights = r$weights,
         groups = list(list(g = g, cols = seq_len(nrow(r$x)), all = TRUE, avg_dist = r$avg_dist)))
  }
  w <- cal$refs[[1]]$weights
  if (n_g == 1L || !all(vapply(cal$refs, function(r) identical(r$weights, w), logical(1)))) {
    return(lapply(seq_len(n_g), alone))
  }
  # Each reference's profiles, in its rows' order (aoa_reference() builds the
  # rows and cv$sample_id from the same index).
  members <- .predict_cal_members(cal)
  ids <- lapply(seq_len(n_g), function(g) {
    s <- members[[which(vapply(members, function(z) identical(z$di_group, g), logical(1)))[1]]]
    s$aref$cv$sample_id
  })
  u_ids <- unique(unlist(ids))
  u_x <- matrix(NA_real_, length(u_ids), ncol(cal$refs[[1]]$x))
  for (g in seq_len(n_g)) {
    pos  <- match(ids[[g]], u_ids)
    seen <- !is.na(u_x[pos, 1L])
    if (any(seen) && !identical(unname(u_x[pos[seen], , drop = FALSE]),
                                unname(cal$refs[[g]]$x[seen, , drop = FALSE]))) {
      return(lapply(seq_len(n_g), alone))
    }
    u_x[pos, ] <- cal$refs[[g]]$x
  }
  list(list(x = u_x, weights = w, groups = lapply(seq_len(n_g), function(g) {
    cols <- match(ids[[g]], u_ids)
    list(g = g, cols = cols, all = identical(cols, seq_along(u_ids)),
         avg_dist = cal$refs[[g]]$avg_dist)
  })))
}

# The DI of pixels against every reference: pixels x groups. The nearest
# profile by float32 distances, then that distance recomputed in double --
# the same number aoa_di() gives, to the rounding of the scaled values.
.predict_di_all <- function(di, x, index_base, n_groups, batch = 8192L) {
  n <- x$size(1L)
  out <- matrix(NA_real_, n, n_groups)
  for (u in di) {
    xw <- x * u$sqrt_w
    for (s in seq(1L, n, by = batch)) {
      e <- min(n, s + batch - 1L)
      xb <- xw[s:e, , drop = FALSE]
      # |r|^2 - 2 x.r in ONE fused product: |x|^2 is the same for every
      # profile a pixel is compared with, so it cannot change which is
      # nearest -- and leaving it out takes a pass over the pixels x profiles
      # matrix away, and the cancellation that came with adding it. The
      # distance itself is recomputed in double below.
      d2 <- torch::torch_addmm(u$r2, xb, u$xt_t, beta = 1, alpha = -2)
      x64 <- xb$to(dtype = torch::torch_float64())
      for (grp in u$groups) {
        sub <- if (isTRUE(grp$all)) d2 else d2[, grp$cols, drop = FALSE]
        nn  <- grp$cols[as.integer(torch::torch_argmin(sub, dim = 2L)) + index_base]
        out[s:e, grp$g] <- as.numeric(((x64 - u$x64[nn, , drop = FALSE])^2)$sum(dim = 2L)$sqrt()) /
          grp$avg_dist
      }
    }
  }
  out
}

.predict_gdal_opts <- function(datatype, blocky) {
  c("COMPRESS=DEFLATE", paste0("PREDICTOR=", if (grepl("^INT|^UINT", datatype)) 2L else 3L),
    "TILED=YES", "BLOCKXSIZE=512", paste0("BLOCKYSIZE=", blocky), "BIGTIFF=IF_SAFER")
}

.predict_open_writers <- function(u, job, unit_dir) {
  g <- job$grid
  nr <- u$r1 - u$r0 + 1L; nc <- u$c1 - u$c0 + 1L
  xmin <- g$xmin + (u$c0 - 1L) * g$xres; xmax <- g$xmin + u$c1 * g$xres
  ymax <- g$ymax - (u$r0 - 1L) * g$yres; ymin <- g$ymax - u$r1 * g$yres
  lapply(seq_len(nrow(job$bands)), function(b) {
    r <- terra::rast(nrows = nr, ncols = nc, xmin = xmin, xmax = xmax, ymin = ymin,
                     ymax = ymax, crs = g$crs)
    names(r) <- job$bands$band[b]
    f <- file.path(unit_dir, paste0(job$bands$band[b], ".tif"))
    dt <- job$bands$datatype[b]
    terra::writeStart(r, f, overwrite = TRUE, datatype = dt, gdal = .predict_gdal_opts(dt, job$blocky))
    list(rast = r, file = f)
  })
}

# The bands of a step, from the seeds' transformed predictions P (pixels x
# seeds), the DI (pixels x reference groups) and, for CV+, each source's fold
# predictions FP (pixels x folds, native). The interval and smearing
# arithmetic is the library's own -- conformal_interval(),
# conformal_scaled_interval(), cv_plus_interval(), smear() -- so the map
# applies exactly what was calibrated.
.predict_step_values <- function(P, D, env, job, FP = NULL) {
  cl <- job$clamp
  nat <- env$inverse(P)
  if (is.finite(cl[1])) nat[nat < cl[1]] <- cl[1]
  if (is.finite(cl[2])) nat[nat > cl[2]] <- cl[2]
  nv <- nrow(nat)
  st <- if (ncol(nat) >= 2L) {
    list(median = matrixStats::rowMedians(nat), mean = rowMeans(nat),
         sd = matrixStats::rowSds(nat), mad = matrixStats::rowMads(nat),
         min = matrixStats::rowMins(nat), max = matrixStats::rowMaxs(nat))
  } else {
    v <- nat[, 1]
    list(median = v, mean = v, sd = rep(0, nv), mad = rep(0, nv), min = v, max = v)
  }
  cache <- list()
  out <- vector("list", nrow(job$bands))
  for (b in seq_len(nrow(job$bands))) {
    r <- job$bands[b, ]
    val <- switch(r$kind,
      ensemble = st[[r$stat]],
      smeared_mean = smear(log1p(st$median), job$sources[[r$source]]$smearing, lower_limit = cl[1]),
      interval = {
        key <- paste(r$source, r$label, r$method, r$width)
        if (is.null(cache[[key]])) {
          src <- job$sources[[r$source]]
          cache[[key]] <- if (identical(r$method, "cv_plus")) {
            ivs <- src$cv_plus$intervals[[r$label]]
            if (identical(r$width, "constant")) {
              cv_plus_interval(ivs$constant, FP[[r$source]], lower_limit = cl[1])
            } else {
              sc <- src$cv_plus$scale
              cv_plus_interval(ivs$level_di, FP[[r$source]],
                               difficulty = .conformal_scale(sc$coef, sc$floor,
                                                             data.frame(level = st$median,
                                                                        di = D[, src$di_group])),
                               lower_limit = cl[1])
            }
          } else {
            ivs <- src$intervals[[r$label]]
            if (identical(r$width, "constant")) {
              conformal_interval(ivs$constant, st$median, lower_limit = cl[1])
            } else {
              conformal_scaled_interval(ivs$level_di, st$median,
                                        data.frame(level = st$median, di = D[, src$di_group]),
                                        lower_limit = cl[1])
            }
          }
        }
        cache[[key]][[r$side]]
      },
      di = D[, r$group],
      aoa = as.integer(inside_aoa(D[, job$sources[[r$source]]$di_group],
                                  job$sources[[r$source]]$threshold)),
      valid_mask = NULL)
    out[b] <- list(val)
  }
  out
}

# ONE UNIT: rows r0..r1 over columns c0..c1, in steps of step_rows rows.
#
# The step's rows are the 2h rows kept from the step before (the halo) and
# the step_rows rows read now. The network sees them chunk by chunk, each
# chunk cropped to the box of its valid pixels plus their windows -- over the
# sea there is nothing to crop, and nothing runs.
.predict_unit <- function(u, env, job) {
  t_unit <- Sys.time()
  secs <- function(t) as.numeric(difftime(Sys.time(), t, units = "secs"))
  h <- job$h; C <- env$n_ch; S <- env$n_seeds
  w_out <- u$c1 - u$c0 + 1L
  w_buf <- w_out + 2L * h
  bc0 <- u$c0 - h                               # raster column of buffer column 1
  rc0 <- max(1L, bc0); rc1 <- min(job$grid$ncol, u$c1 + h)
  pad_l <- rc0 - bc0; pad_r <- (u$c1 + h) - rc1
  # [halo; rows], the worker's own and reused (see .predict_window()).
  W <- .predict_window(env, C, 2L * h + job$step_rows, w_buf)

  unit_dir <- file.path(job$units_dir, u$unit_id)
  create_output_dirs(unit_dir)
  writers <- .predict_open_writers(u, job, unit_dir)
  open <- TRUE
  on.exit(if (open) for (w in writers) try(terra::writeStop(w$rast), silent = TRUE), add = TRUE)

  tm <- c(read = 0, net = 0, di = 0, bands = 0, write = 0)
  nb <- nrow(job$bands)
  bstat <- data.frame(band = job$bands$band, n = 0, sum = 0, min = Inf, max = -Inf)
  probe <- job$probe_cells
  probe_pred <- if (!is.null(probe)) matrix(NA_real_, nrow(probe), S) else NULL
  # CV+'s fold models, counted in the order the probe reads them: source, fold,
  # seed.
  n_fold_models <- sum(vapply(env$fold_models, function(fm)
    sum(vapply(fm, function(z) length(z$models), integer(1))), integer(1)))
  probe_fold <- if (!is.null(probe) && n_fold_models > 0L) {
    matrix(NA_real_, nrow(probe), n_fold_models)
  } else NULL
  n_valid <- 0
  fin_halo <- NULL
  first <- TRUE
  # THE WORKER'S CURRENT WORKING SET after each phase of each step, when the
  # map is asked for it (options(dsm.predict.trace_mem = TRUE); T4 does). A
  # peak says only how high memory once went; a phase after which the level
  # left by a full collection climbs step after step is a leak, and names
  # itself.
  trace <- list()
  si <- 0L
  mark <- function(phase) {
    if (isTRUE(job$trace_mem)) {
      mi <- tryCatch(ps::ps_memory_info(ps::ps_handle()), error = function(e) NULL)
      trace[[length(trace) + 1L]] <<- data.frame(
        step = si, phase = phase,
        rss_gb = if (is.null(mi)) NA_real_ else as.numeric(mi[["rss"]]) / 1e9)
    }
    invisible(NULL)
  }

  for (o0 in seq(u$r0, u$r1, by = job$step_rows)) {
    o1 <- min(o0 + job$step_rows - 1L, u$r1)
    gs <- o1 - o0 + 1L
    si <- si + 1L
    mark("start")

    # ── the rows: read once, the halo kept ──────────────────────────────────
    tr <- Sys.time()
    # The first step reads its halo and its own rows as two reads -- disjoint
    # rows, so no row is decompressed twice -- instead of one block cut in
    # two afterwards: the block, its halo's clone and its rows' copy alive
    # together were ~11 GB of a full-width worker's peak (T3).
    if (first && h > 0L) {
      fin_halo <- .predict_read_rows((o0 - h):(o0 + h - 1L), rc0, rc1, pad_l, pad_r, w_buf, env,
                                     job, into = W, at = 0L)$fin
    }
    first <- FALSE
    fin_new <- .predict_read_rows((o0 + h):(o1 + h), rc0, rc1, pad_l, pad_r, w_buf, env, job,
                                  into = W, at = 2L * h)$fin
    tm[["read"]] <- tm[["read"]] + secs(tr)
    mark("read")

    fin_full <- if (h > 0L) torch::torch_cat(list(fin_halo, fin_new), dim = 1L) else fin_new
    valid <- .predict_valid(fin_full, h, gs, w_out)
    rm(fin_full)

    # ── the network and the DI, chunk by chunk ──────────────────────────────
    I_all <- list(); P_all <- list(); D_all <- list(); FP_all <- list(); FR_all <- list()
    k <- 0L
    for (j0 in seq(1L, w_out, by = job$chunk_cols)) {
      j1 <- min(w_out, j0 + job$chunk_cols - 1L)
      vm <- valid[, j0:j1, drop = FALSE]
      if (!any(vm)) next
      ij <- which(vm, arr.ind = TRUE)
      i <- as.integer(ij[, 1]); j <- as.integer(ij[, 2]) + j0 - 1L
      # THE STRIP'S SHAPE COMES FROM A SHORT LIST. oneDNN keeps a compiled
      # primitive, with its buffers, for every input shape it has seen, and a
      # strip cropped to the box of its valid pixels had a new shape at
      # nearly every coastal chunk: T3 saw a worker's peak climb ~5 GB a
      # unit, unit after unit (15.9 -> 21.1 GB; 16.9 -> 21.9 -> 25.9 with 15
      # threads) -- on the globe, until the machine gave out. So every row of
      # the step, and the columns rounded out to whole quanta from the
      # chunk's own start: a handful of shapes, reused all run. The ocean in
      # a coastal chunk's quantum is computed and discarded; ~5% of a chunk
      # at most, on the chunks that have any.
      q <- job$col_quantum
      a0 <- j0 + ((min(j) - j0) %/% q) * q
      a1 <- min(j1, a0 + q * as.integer(ceiling((max(j) - a0 + 1) / q)) - 1L)
      # The window's halo and rows over the strip's columns: `:` inside the
      # brackets makes slices, so this is a view of W and contiguous() its one
      # copy.
      strip <- W[, 1:(2L * h + gs), a0:(a1 + 2L * h), drop = FALSE]$contiguous()
      cs <- cbind(i + h, j - a0 + 1L + h)
      tn <- Sys.time()
      x4 <- strip$unsqueeze(1L)
      P <- matrix(NA_real_, length(i), S)
      for (s in seq_len(S)) {
        P[, s] <- fcn_predict_strip(env$models[[s]], x4, cs, engine = env$fcn_engine,
                                    batch = job$batch, gather_mb = job$gather_mb,
                                    gc_hook = env$gc_hook)
      }
      # CV+: each fold's models on the strip in that fold's scaling (see
      # .predict_cv_plus()), the median of a fold's seeds in native units --
      # the ensemble each residual came from. The raw predictions are kept
      # only for the probe.
      FP <- NULL; FR <- NULL
      if (length(env$fold_models) > 0L) {
        xb <- .predict_fold_strip(env, x4)
        FP <- list(); FR <- list()
        for (src in names(env$fold_models)) {
          fm <- env$fold_models[[src]]
          Fm <- matrix(NA_real_, length(i), length(fm),
                       dimnames = list(NULL, vapply(fm, function(z) as.character(z$fold), character(1))))
          for (jf in seq_along(fm)) {
            xb$copy_(x4)
            xb$mul_(fm[[jf]]$a)$add_(fm[[jf]]$b)
            Pk <- matrix(NA_real_, length(i), length(fm[[jf]]$models))
            for (s2 in seq_along(fm[[jf]]$models)) {
              Pk[, s2] <- fcn_predict_strip(fm[[jf]]$models[[s2]], xb, cs, engine = env$fcn_engine,
                                            batch = job$batch, gather_mb = job$gather_mb,
                                            gc_hook = env$gc_hook)
            }
            if (!is.null(probe)) FR[[length(FR) + 1L]] <- Pk
            nat <- env$inverse(Pk)
            if (is.finite(job$clamp[1])) nat[nat < job$clamp[1]] <- job$clamp[1]
            if (is.finite(job$clamp[2])) nat[nat > job$clamp[2]] <- job$clamp[2]
            Fm[, jf] <- if (ncol(nat) > 1L) matrixStats::rowMedians(nat) else nat[, 1]
          }
          FP[[src]] <- Fm
        }
      }
      tm[["net"]] <- tm[["net"]] + secs(tn)
      td <- Sys.time()
      D <- NULL
      if (length(env$di) > 0L) {
        ws <- strip$size(3L)
        xc <- strip$view(c(C, -1L))[, (cs[, 1] - 1L) * ws + cs[, 2], drop = FALSE]$t()
        D <- .predict_di_all(env$di, xc, env$index_base, env$n_di)
        env$gc_hook()
      }
      tm[["di"]] <- tm[["di"]] + secs(td)
      k <- k + 1L
      I_all[[k]] <- (i - 1L) * w_out + j
      P_all[[k]] <- P
      D_all[[k]] <- D
      FP_all[[k]] <- FP
      FR_all[[k]] <- if (length(FR)) do.call(cbind, FR) else NULL
      rm(strip, x4)
    }
    mark("network_di")
    idx <- unlist(I_all)
    nv <- length(idx)
    n_valid <- n_valid + nv

    # ── the bands ───────────────────────────────────────────────────────────
    tb <- Sys.time()
    vals <- NULL
    if (nv > 0L) {
      P <- do.call(rbind, P_all)
      D <- if (length(env$di) > 0L) do.call(rbind, D_all) else NULL
      FPm <- if (length(env$fold_models) > 0L) {
        lapply(stats::setNames(nm = names(env$fold_models)), function(src)
          do.call(rbind, lapply(FP_all, function(z) z[[src]])))
      } else NULL
      vals <- .predict_step_values(P, D, env, job, FPm)
      if (!is.null(probe)) {
        here <- probe$row >= o0 & probe$row <= o1 & probe$col >= u$c0 & probe$col <= u$c1
        if (any(here)) {
          pos <- match((probe$row[here] - o0) * w_out + (probe$col[here] - u$c0 + 1L), idx)
          probe_pred[here, ] <- P[pos, , drop = FALSE]
          if (!is.null(probe_fold)) {
            probe_fold[here, ] <- do.call(rbind, FR_all)[pos, , drop = FALSE]
          }
        }
      }
    }
    tm[["bands"]] <- tm[["bands"]] + secs(tb)
    mark("bands")

    tw <- Sys.time()
    for (b in seq_len(nb)) {
      kind <- job$bands$kind[b]
      if (identical(kind, "valid_mask")) {
        v <- as.integer(t(valid))
        w <- v
      } else {
        v <- if (identical(job$bands$datatype[b], "INT1U")) rep(NA_integer_, gs * w_out) else rep(NA_real_, gs * w_out)
        w <- if (nv > 0L) vals[[b]] else numeric(0)
        if (nv > 0L) v[idx] <- w
      }
      terra::writeValues(writers[[b]]$rast, v, o0 - u$r0 + 1L, gs)
      w <- w[is.finite(w)]
      if (length(w) > 0L) {
        bstat$n[b]   <- bstat$n[b] + length(w)
        bstat$sum[b] <- bstat$sum[b] + sum(as.numeric(w))
        bstat$min[b] <- min(bstat$min[b], min(w))
        bstat$max[b] <- max(bstat$max[b], max(w))
      }
    }
    tm[["write"]] <- tm[["write"]] + secs(tw)
    mark("write")

    # ── the halo the next step keeps ────────────────────────────────────────
    # The step's last 2h rows move to the window's top, in place. A step is at
    # least 2h rows (.predict_work_plan()) but a unit's last, and the last
    # keeps no halo: the next unit reads its own.
    if (h > 0L && o1 < u$r1) {
      if (gs < 2L * h) {
        stop("A step of ", gs, " rows cannot keep a halo of ", 2L * h, " rows.", call. = FALSE)
      }
      W[, 1:(2L * h), ]$copy_(W[, (gs + 1L):(gs + 2L * h), ])
      fin_halo <- fin_new[(gs - 2L * h + 1L):gs, , drop = FALSE]$clone()
    }
    rm(fin_new, vals, P_all, D_all, I_all, FP_all, FR_all)
    invisible(gc(verbose = FALSE))
    mark("gc")
  }

  for (w in writers) terra::writeStop(w$rast)
  open <- FALSE
  mark("unit_end")
  list(unit_id = u$unit_id, r0 = u$r0, r1 = u$r1, c0 = u$c0, c1 = u$c1,
       n_cells = as.numeric(u$r1 - u$r0 + 1L) * w_out, n_valid = n_valid,
       seconds = tm, total_s = secs(t_unit), band_stats = bstat,
       probe_pred = probe_pred, probe_fold = probe_fold,
       mem_trace = if (length(trace)) do.call(rbind, trace) else NULL,
       step_rows = job$step_rows, chunk_cols = job$chunk_cols, engine = job$engine,
       status = "success", finished_at = Sys.time())
}

# ── helpers: the probe ────────────────────────────────────────────────────────

# The densest group of profiles in one step's rows and chunk_cols columns: the
# cheapest unit that still puts dozens of them through the whole chain.
.predict_probe_points <- function(inp, grid, step_rows, keep = NULL, max_points = 48L,
                                  max_cols = 2048L) {
  meta <- inp$meta
  if (!isTRUE(grid$same_as_store) || !all(c("x", "y", "sample_id") %in% names(meta))) return(NULL)
  ref  <- terra::rast(inp$files[1])
  cell <- terra::cellFromXY(ref, cbind(meta$x, meta$y))
  row  <- as.integer(terra::rowFromCell(ref, cell))
  col  <- as.integer(terra::colFromCell(ref, cell))
  h <- grid$h
  ok <- !is.na(cell) & row > h & row <= grid$nrow - h & col > h & col <= grid$ncol - h
  if (!is.null(keep)) ok <- ok & meta$sample_id %in% keep
  if (!any(ok)) return(NULL)
  d <- data.frame(sample_id = meta$sample_id[ok], row = row[ok], col = col[ok])
  d <- d[order(d$row, d$col), , drop = FALSE]
  n_in <- findInterval(d$row + step_rows - 1L, d$row) - seq_len(nrow(d)) + 1L
  i0 <- which.max(n_in)
  band <- d[d$row >= d$row[i0] & d$row <= d$row[i0] + step_rows - 1L, , drop = FALSE]
  band <- band[order(band$col), , drop = FALSE]
  n_in <- findInterval(band$col + max_cols - 1L, band$col) - seq_len(nrow(band)) + 1L
  j0 <- which.max(n_in)
  sel <- band[band$col >= band$col[j0] & band$col <= band$col[j0] + max_cols - 1L, , drop = FALSE]
  if (nrow(sel) > max_points) {
    sel <- sel[unique(round(seq(1, nrow(sel), length.out = max_points))), , drop = FALSE]
  }
  rownames(sel) <- NULL
  sel
}

# Every seed's transformed prediction for every profile the final run stored
# one for (predictions/seedNNNN_pred_all.csv): list(ids, pred), pred a matrix
# ids x seeds; NULL when a seed stored none.
#
# NOT EVERY PROFILE IS THERE. The refit's split drops the profiles its buffer
# puts too near the test and validation sets -- 129 of the SOC store's 3,728
# -- and those have no prediction to compare with. The probe draws from the
# profiles that have one; the first P4 run drew from all of them, met a gap
# and gave up.
#
# CV+'S FOLD MODELS ARE PROBED TOO, against the predictions their tuning run
# wrote -- the one check that the fold's multiply-add, its scaling and the
# engine give the fold model's own numbers. A fold model predicted its own
# training and validation rows, so the profiles the probe can use are those
# EVERY model -- seed and fold model alike -- has a prediction for. fold_pred
# is in the order the map's worker keeps them: source, fold, seed.
.predict_stored_all <- function(fr, job = NULL) {
  read_one <- function(p) {
    if (!file.exists(p)) return(NULL)
    d <- safe_read_csv2(p)
    if (!all(c("sample_id", "pred_transform") %in% names(d))) return(NULL)
    # A row predicted under two roles is one prediction: the first.
    d <- d[is.finite(d$pred_transform), c("sample_id", "pred_transform"), drop = FALSE]
    d[!duplicated(d$sample_id), , drop = FALSE]
  }
  tabs <- lapply(fr$seeds, function(s) {
    read_one(file.path(fr$run_dir, fr$config_id, "predictions", sprintf("seed%04d_pred_all.csv", s)))
  })
  if (any(vapply(tabs, is.null, logical(1)))) return(NULL)
  fold_files <- unlist(lapply(job$sources, function(s) unlist(s$cv_plus$preds)), use.names = FALSE)
  ftabs <- lapply(fold_files, read_one)
  if (any(vapply(ftabs, is.null, logical(1)))) {
    stop("CV+: a fold model's predictions are missing or unreadable (",
         paste(fold_files[vapply(ftabs, is.null, logical(1))][1], collapse = ""),
         "), and the probe holds every fold model to them.", call. = FALSE)
  }
  ids <- Reduce(intersect, lapply(c(tabs, ftabs), function(d) d$sample_id))
  if (length(ids) == 0L) return(NULL)
  as_mat <- function(tt) {
    m <- vapply(tt, function(d) as.numeric(d$pred_transform[match(ids, d$sample_id)]),
                numeric(length(ids)))
    matrix(m, nrow = length(ids))
  }
  list(ids = ids, pred = as_mat(tabs),
       fold_pred = if (length(ftabs)) as_mat(ftabs) else NULL,
       fold_files = fold_files)
}

.predict_probe <- function(job, fr, inp, grid, run_dir, say, tol = 1e-3) {
  st <- .predict_stored_all(fr, job)
  has_folds <- any(vapply(job$sources, function(s) !is.null(s$cv_plus), logical(1)))
  if (is.null(st)) {
    reason <- if (has_folds) {
      "no profile has a stored prediction from every seed and every fold model"
    } else "the final run stored no transform-space prediction (pred_transform) for its seeds"
    # CV+ rests on the fold models giving their own numbers on the map, and
    # nothing else shows it: a probe that cannot run stops the map.
    if (has_folds) {
      return(list(status = "fail", reason = paste0(reason, " -- the CV+ bands cannot be checked")))
    }
    say("\nProbe: not applicable -- ", reason, ".")
    return(list(status = "not_applicable", reason = reason))
  }
  pts <- .predict_probe_points(inp, grid, job$step_rows, keep = st$ids)
  if (is.null(pts)) {
    reason <- if (!isTRUE(grid$same_as_store)) {
      "these rasters are not the grid the store was extracted from, so no profile is a pixel of this map"
    } else "no profile with a stored prediction is far enough from the grid's edge"
    say("\nProbe: not applicable -- ", reason, ".")
    return(list(status = "not_applicable", reason = reason))
  }
  stored <- st$pred[match(pts$sample_id, st$ids), , drop = FALSE]
  say(sprintf("\nProbe: %d profile(s) in rows %d-%d, cols %d-%d, through the whole chain (%d of %d profiles have a stored prediction)...",
              nrow(pts), min(pts$row), max(pts$row), min(pts$col), max(pts$col),
              length(st$ids), nrow(inp$meta)))
  pdir <- file.path(run_dir, "probe")
  unlink(pdir, recursive = TRUE)
  pj <- job
  pj$units_dir <- pdir
  pj$probe_cells <- pts
  unit <- tibble::tibble(unit_id = "probe", r0 = min(pts$row), r1 = max(pts$row),
                         c0 = min(pts$col), c1 = max(pts$col))
  t0 <- Sys.time()
  .predict_run_units(pj, unit, 1L, run_dir, "probe", say, progress = FALSE)
  rp <- .predict_done_path(pdir, "probe")
  if (!file.exists(rp)) {
    fp <- file.path(pdir, "probe", "failed.rds")
    stop("The probe unit did not finish: ",
         if (file.exists(fp)) readRDS(fp)$error else "see logs/probe_worker_01.log",
         call. = FALSE)
  }
  rec <- readRDS(rp)
  # The seeds and, for CV+, every fold model, side by side: the same check.
  mp <- rec$probe_pred
  if (!is.null(st$fold_pred)) {
    if (is.null(rec$probe_fold) || ncol(rec$probe_fold) != ncol(st$fold_pred)) {
      stop("The probe unit returned ", if (is.null(rec$probe_fold)) 0L else ncol(rec$probe_fold),
           " fold model prediction(s) per profile for the ", ncol(st$fold_pred), " fold model(s) ",
           "stored.", call. = FALSE)
    }
    mp <- cbind(mp, rec$probe_fold)
    stored <- cbind(stored, st$fold_pred[match(pts$sample_id, st$ids), , drop = FALSE])
  }
  valid <- stats::complete.cases(mp)
  rel <- abs(mp - stored) / (1 + abs(stored))
  worst <- if (any(valid)) max(rel[valid, , drop = FALSE]) else NA_real_
  n_seed <- ncol(rec$probe_pred)
  tab <- tibble::tibble(sample_id = pts$sample_id, row = pts$row, col = pts$col,
                        valid_on_map = valid,
                        max_abs_diff = apply(abs(mp - stored), 1L, max),
                        max_rel_diff = apply(rel, 1L, max),
                        max_rel_diff_seeds = apply(rel[, seq_len(n_seed), drop = FALSE], 1L, max),
                        max_rel_diff_fold_models = if (ncol(rel) > n_seed) {
                          apply(rel[, -seq_len(n_seed), drop = FALSE], 1L, max)
                        } else NA_real_)
  safe_write_csv2(tab, file.path(run_dir, "probe.csv"))
  pass <- all(valid) && is.finite(worst) && worst <= tol
  reason <- if (!all(valid)) {
    sprintf("%d of %d profile pixel(s) were not predicted on the map", sum(!valid), length(valid))
  } else sprintf("max relative difference %.2e against a tolerance of %.0e", worst, tol)
  say(sprintf("Probe: %s -- %d profile(s) x %d seed(s)%s, %s (%.0f s).",
              if (pass) "PASS" else "FAIL", nrow(pts), n_seed,
              if (ncol(mp) > n_seed) sprintf(" and %d fold model(s)", ncol(mp) - n_seed) else "",
              reason, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  list(status = if (pass) "pass" else "fail", reason = reason, n = nrow(pts),
       max_rel_diff = worst, tol = tol, table = tab)
}

# ── helpers: the record ───────────────────────────────────────────────────────

.predict_write_vrts <- function(run_dir, units, band_tbl) {
  out <- character(nrow(band_tbl))
  for (b in seq_len(nrow(band_tbl))) {
    files <- file.path(run_dir, "units", units$unit_id, paste0(band_tbl$band[b], ".tif"))
    out[b] <- file.path(run_dir, paste0(band_tbl$band[b], ".vrt"))
    terra::vrt(files, out[b], overwrite = TRUE)
  }
  stats::setNames(out, band_tbl$band)
}

.predict_unit_table <- function(recs) {
  dplyr::bind_rows(lapply(recs, function(r) tibble::tibble(
    unit_id = r$unit_id, r0 = r$r0, r1 = r$r1, c0 = r$c0, c1 = r$c1,
    n_cells = r$n_cells, n_valid = r$n_valid, read_s = r$seconds[["read"]],
    net_s = r$seconds[["net"]], di_s = r$seconds[["di"]], bands_s = r$seconds[["bands"]],
    write_s = r$seconds[["write"]], total_s = r$total_s,
    valid_px_per_s = r$n_valid / max(1e-9, r$total_s), worker = r$worker %||% NA_integer_,
    threads = r$threads %||% NA_integer_, step_rows = r$step_rows, chunk_cols = r$chunk_cols,
    peak_gb = r$peak_gb %||% NA_real_, rss_gb = r$rss_gb %||% NA_real_,
    private_gb = r$private_gb %||% NA_real_,
    pid = r$pid %||% NA_integer_,
    finished_at = as.character(r$finished_at))))
}

.predict_band_summary <- function(recs, band_tbl) {
  st <- dplyr::bind_rows(lapply(recs, function(r) r$band_stats))
  st <- dplyr::summarise(dplyr::group_by(st, .data$band), n = sum(.data$n), sum = sum(.data$sum),
                         min = min(.data$min), max = max(.data$max), .groups = "drop")
  st$mean <- ifelse(st$n > 0, st$sum / st$n, NA_real_)
  st$min[!is.finite(st$min)] <- NA_real_
  st$max[!is.finite(st$max)] <- NA_real_
  st <- st[match(band_tbl$band, st$band), , drop = FALSE]
  tibble::tibble(band = st$band, n = st$n, mean = st$mean, min = st$min, max = st$max)
}
