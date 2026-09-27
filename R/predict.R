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
# limit -- 23 s of reading against 2,184 s of network (docs/project_log.md,
# 2026-09-27).
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
# WHAT THIS REPLACES. Stage 05 of the SOC example -- 05_predict_spatial.R and
# the scripts that ran it in parallel, merged its tiles and estimated its ETA
# -- predicted one dataset's map. dsm_predict() is that stage with the dataset
# taken out, and with what a 250 m global grid needs that stage 05 did not
# have:
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
#   smeared_mean_<source>       the conditional MEAN (Duan), log1p only
#   piNN_constant_lower/upper_<source>   split conformal, constant width
#   piNN_level_di_lower/upper_<source>   width fitted on the level and the DI
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
#'   residuals calibrate them, e.g. c(block = "<spatial CV run>", knndm =
#'   "<kNNDM run>"). Each source gets its own bands. NULL for the final run's
#'   own tuning run; character(0) for none (then no interval, DI or AOA).
#' @param alpha   Miscoverage of the intervals: 0.1 is 90%.
#' @param clamp   c(lower, upper) of a prediction in native units. NULL for the
#'   refit's own (its evaluation clamps to c(0, Inf) by default).
#' @param bands   NULL for all, or some of: ensemble_median, ensemble_mean,
#'   ensemble_sd, ensemble_mad, ensemble_min, ensemble_max, smeared_mean,
#'   interval_constant, interval_level_di, di, aoa, valid_mask.
#' @param engine  "auto": fully convolutional where exact (fcn_supported()),
#'   patch by patch elsewhere. "patch": patch by patch everywhere -- slow, for
#'   checks.
#' @param output_dir Where maps go. NULL for <final run>/maps.
#' @param run_id  NULL for map_<timestamp>. An existing one resumes.
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
#' @return A `dsm_prediction`, printed.
dsm_predict <- function(final, data, rasters = NULL, qc_table = NULL, extent = NULL,
                        config = NULL, calibration = NULL, alpha = 0.1, clamp = NULL,
                        bands = NULL, engine = c("auto", "patch"), output_dir = NULL,
                        run_id = NULL, resume = TRUE, n_cores = NULL,
                        threads_per_worker = 5L, max_ram_gb = NULL, unit_rows = NULL,
                        step_rows = NULL, chunk_cols = 2048L, probe = TRUE,
                        verbose = TRUE) {

  t_start <- Sys.time()
  say <- function(...) if (verbose) message(...)
  engine <- match.arg(engine)

  # ── 0. the arguments, all checked before a raster is opened ───────────────
  fr  <- .predict_final(final, config)
  inp <- .predict_inputs(data, rasters, qc_table, fr$scaling)
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
  cal <- .predict_calibration(sources, fr, inp, alpha, say)
  if (length(cal$sources) == 0L) {
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
  work  <- .predict_work_plan(grid, inp, fr$cfg, band_tbl, n_cores, tpw, max_ram_gb,
                              unit_rows, step_rows, chunk_cols, say, n_seeds = length(fr$seeds))
  units <- .predict_units(grid$rows, grid$cols, work$unit_rows)
  settings <- .predict_settings(fr, inp, grid, work, band_tbl, alpha, clamp, engine, cal)
  # THE SETTINGS ARE LOCKED BY THE FIRST FINISHED UNIT, not by the first call:
  # a call stopped before any unit finished -- a probe that failed on a swapped
  # channel -- left nothing its successor could be mixed with.
  if (!is.null(saved) &&
      length(list.files(file.path(run_dir, "units"), pattern = "^done[.]rds$",
                        recursive = TRUE)) == 0L) {
    saved <- NULL
  }
  if (!is.null(saved)) {
    diff_fields <- names(settings)[!vapply(names(settings), function(k)
      identical(settings[[k]], saved[[k]]), logical(1))]
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
  run_info <- list(n_workers = 0L, peak_gb = NA_real_, minutes = NA_real_)
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
    threads_per_worker = tpw, engine = paste(engine_used, collapse = ";"),
    calibration = paste(sprintf("%s=%s", names(sources), sources), collapse = ";"),
    alpha = paste(alpha, collapse = ";"), n_bands = nrow(band_tbl),
    probe = probe_res$status, probe_max_rel_diff = probe_res$max_rel_diff %||% NA_real_,
    peak_gb_per_worker = run_info$peak_gb, map_minutes = run_info$minutes,
    minutes_this_call = minutes,
    valid_px_per_s = sum(unit_tbl$n_valid) / max(1e-9, sum(unit_tbl$total_s)),
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
  cat(sprintf("  work       : %d unit(s) x %d row(s) | %s valid px/s per worker | %.1f min this call\n",
              m$n_units, m$unit_rows, format(round(m$valid_px_per_s), big.mark = ","), x$minutes))
  cat("  probe      : ", x$probe$status,
      if (!is.null(x$probe$max_rel_diff) && is.finite(x$probe$max_rel_diff))
        sprintf(" (%d profile(s), max relative difference %.2e)", x$probe$n, x$probe$max_rel_diff)
      else if (!is.null(x$probe$reason)) paste0(" -- ", x$probe$reason) else "", "\n", sep = "")
  if (nrow(x$calibration) > 0L) {
    cat("  calibration:\n")
    for (i in seq_len(nrow(x$calibration))) {
      r <- x$calibration[i, ]
      cat(sprintf("    %-8s %s  constant +/-%.3g | level+DI q %.3g x (%.3g %+.3g x level %+.3g x DI) | AOA DI <= %.3f | smearing %s\n",
                  r$source, r$label, r$q_constant, r$q_level_di, r$scale_intercept,
                  r$scale_level, r$scale_di, r$aoa_threshold,
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
  if (inherits(data, "dsm_data")) {
    store_dir  <- data$patch_dir
    recipe     <- data$recipe
    points     <- data$points
    meta       <- data$store$meta
    manifest   <- data$store$manifest
    transform  <- data$transform
    predictors <- as.character(data$store$predictors)
    cell_size  <- data$cell_size
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
                                             collapse = ", "), call. = FALSE)
  }
  list(store_dir = store_dir, points = points, meta = meta, predictors = predictors,
       qc_table = qc, scaling = scaling, files = files, transform = transform,
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

# ONE CALIBRATION PER SOURCE, from that source's residuals and that source's
# folds: a residual's DI is the distance its own fold model faced, and the AOA
# threshold is the CV DI of the plan whose error it delimits.
.predict_calibration <- function(sources, fr, inp, alpha, say) {
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
    aref <- aoa_reference(inp$points, inp$predictors, inp$qc_table, inp$scaling, plan)
    tab <- dplyr::inner_join(res, dplyr::select(aref$cv, sample_id, di = cv_di), by = "sample_id")
    iv <- list()
    for (a in alpha) {
      lab <- .predict_level_label(a)
      const <- conformal_calibrate(tab$obs, tab$pred, alpha = a)
      ldi <- tryCatch(
        conformal_scaled_calibrate(tab$obs, tab$pred,
                                   data.frame(level = tab$pred, di = tab$di), alpha = a),
        error = function(e) stop("calibration source '", nm, "': ", conditionMessage(e),
                                 call. = FALSE))
      iv[[lab]] <- list(alpha = a, label = lab, constant = const, level_di = ldi)
    }
    sm <- if (identical(inp$transform$name, "log1p")) {
      smearing_from_run(dir, attr(res, "config_id"))
    } else NULL
    say(sprintf("\nCalibration '%s': %s (%s, as %s) | %d residual(s), %d with a CV DI | AOA DI <= %.4f%s",
                nm, basename(dir), plan$method, attr(res, "config_id"), nrow(res), nrow(tab),
                as.numeric(aref$threshold),
                if (!is.null(sm)) sprintf(" | smearing S = %.4f", sm$s) else ""))
    for (lab in names(iv)) {
      say(sprintf("  %s: constant +/- %.3f | level+DI: q %.3f x (%.3f %+.4f x level %+.3f x DI), floor %.3f",
                  lab, iv[[lab]]$constant$q, iv[[lab]]$level_di$q, iv[[lab]]$level_di$coef[[1]],
                  iv[[lab]]$level_di$coef[["level"]], iv[[lab]]$level_di$coef[["di"]],
                  iv[[lab]]$level_di$floor))
    }
    out[[nm]] <- list(name = nm, dir = dir, config_id = attr(res, "config_id"),
                      method = plan$method, n_residuals = nrow(res), n_calibration = nrow(tab),
                      aref = aref, threshold = as.numeric(aref$threshold), table = tab,
                      intervals = iv, smearing = sm)
  }
  # Sources whose references are the same points in the same space share one DI.
  refs <- list()
  for (nm in names(out)) {
    r <- out[[nm]]$aref$ref
    hit <- which(vapply(refs, function(q) identical(q$x, r$x) && identical(q$avg_dist, r$avg_dist) &&
                          identical(q$weights, r$weights), logical(1)))
    if (length(hit) == 0L) {
      refs[[length(refs) + 1L]] <- r
      hit <- length(refs)
    }
    out[[nm]]$di_group <- hit[1]
  }
  list(sources = out, refs = refs)
}

# One row per source and level. di_band names the DI band the source's
# level+DI interval and AOA were computed from: sources whose fold plans used
# different profiles (a buffer drops some) have different references, and so
# a DI band each.
.predict_calibration_table <- function(cal, band_tbl = NULL) {
  rows <- list()
  for (s in cal$sources) {
    di_band <- NA_character_
    if (!is.null(band_tbl)) {
      b <- band_tbl$band[band_tbl$kind == "di" & band_tbl$group %in% s$di_group]
      if (length(b)) di_band <- b[1]
    }
    for (iv in s$intervals) {
      rows[[length(rows) + 1L]] <- tibble::tibble(
        source = s$name, dir = s$dir, config_id_there = s$config_id, method = s$method,
        n_residuals = s$n_residuals, n_calibration = s$n_calibration,
        label = iv$label, alpha = iv$alpha, q_constant = iv$constant$q,
        q_level_di = iv$level_di$q, scale_intercept = iv$level_di$coef[[1]],
        scale_level = iv$level_di$coef[["level"]], scale_di = iv$level_di$coef[["di"]],
        scale_floor = iv$level_di$floor, scale_r2_fit = iv$level_di$r2_fit,
        aoa_threshold = s$threshold, di_group = s$di_group, di_band = di_band,
        smearing_s = if (is.null(s$smearing)) NA_real_ else s$smearing$s)
    }
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
  if (length(cal$sources) == 0L && asked && any(want %in% per_source)) {
    stop("Band(s) ", paste(intersect(want, per_source), collapse = ", "), " need a ",
         "calibration source, and none was given.", call. = FALSE)
  }
  rows <- list()
  add <- function(band, kind, meaning, datatype = "FLT4S", stat = NA_character_,
                  source = NA_character_, alpha = NA_real_, label = NA_character_,
                  method = NA_character_, side = NA_character_, group = NA_integer_) {
    rows[[length(rows) + 1L]] <<- tibble::tibble(
      band = band, kind = kind, datatype = datatype, stat = stat, source = source,
      alpha = alpha, label = label, method = method, side = side, group = group,
      meaning = meaning)
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
    for (iv in s$intervals) {
      lvl <- 100 * (1 - iv$alpha)
      if ("interval_constant" %in% want) {
        for (side in c("lower", "upper")) {
          add(sprintf("%s_constant_%s_%s", iv$label, side, s$name), "interval",
              sprintf("%s bound of the %g%% split-conformal interval, constant width (+/- %.4g), calibrated on '%s' cross-validated residuals", side, lvl, iv$constant$q, s$name),
              source = s$name, alpha = iv$alpha, label = iv$label, method = "constant", side = side)
        }
      }
      if ("interval_level_di" %in% want) {
        cf <- iv$level_di$coef
        for (side in c("lower", "upper")) {
          add(sprintf("%s_level_di_%s_%s", iv$label, side, s$name), "interval",
              sprintf("%s bound of the %g%% conformal interval of width %.4g x (%.4g %+.4g x level %+.4g x DI, floor %.4g), calibrated on '%s' residuals", side, lvl, iv$level_di$q, cf[[1]], cf[["level"]], cf[["di"]], iv$level_di$floor, s$name),
              source = s$name, alpha = iv$alpha, label = iv$label, method = "level_di", side = side)
        }
      }
    }
  }
  if ("di" %in% want && length(cal$refs) > 0L) {
    for (g in seq_along(cal$refs)) {
      users <- names(cal$sources)[vapply(cal$sources, function(s) s$di_group == g, logical(1))]
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
#   the step's rows as float32, the kept halo and the new rows, and the mask
#   the next halo, cloned while the step's rows are still alive
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
# The first version counted neither the halo's clone nor the R vectors, and a
# band's copies as three doubles; on full-width rows T3 measured 24-32 GB
# against its 12.4 (with the reader's own garbage, since fixed). An estimate,
# said to be one: the real peak of every worker is measured and written
# beside it.
.predict_worker_gb <- function(g, h, w_out, n_ch, chunk_cols, conv_sum, n_seeds = 10L,
                               n_bands = 21L, n_iv = 4L) {
  w_buf <- w_out + 2 * h
  bytes <- (g + 2 * h) * w_buf * (4 * n_ch + 1) +
    2 * h * w_buf * 4 * n_ch +
    g * w_buf * 20 +
    2 * (g + 2 * h) * (min(chunk_cols, w_out) + 2 * h) * (n_ch + 4 * conv_sum) * 4 +
    g * w_out * (8 * (2 * n_seeds + n_bands + 4 * n_iv) + 16)
  bytes / 1e9 + 0.5 + 2 + 3.5
}

.predict_work_plan <- function(grid, inp, cfg, band_tbl, n_cores, tpw, max_ram_gb,
                               unit_rows, step_rows, chunk_cols, say, n_seeds = 10L) {
  n_ch <- length(inp$predictors)
  conv_sum <- sum(as.integer(cfg$conv_channels[[1]]))
  iv <- band_tbl[band_tbl$kind == "interval", , drop = FALSE]
  n_iv <- nrow(unique(iv[, c("source", "label", "method"), drop = FALSE]))
  gb <- function(g) .predict_worker_gb(g, grid$h, grid$n_cols_out, n_ch, chunk_cols, conv_sum,
                                       n_seeds = n_seeds, n_bands = nrow(band_tbl), n_iv = n_iv)
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
.predict_settings <- function(fr, inp, grid, work, band_tbl, alpha, clamp, engine, cal) {
  list(final_run = fr$run_dir, config_id = fr$config_id, seeds = fr$seeds,
       model_bytes = as.numeric(file.size(fr$model_files)), rasters = inp$files,
       predictors = inp$predictors,
       grid = c(grid$nrow, grid$ncol, grid$xmin, grid$ymax, grid$xres, grid$yres),
       rows = grid$rows, cols = grid$cols, unit_rows = work$unit_rows,
       bands = band_tbl$band, alpha = alpha, clamp = clamp, engine = engine,
       calibration = lapply(cal$sources, function(s) c(
         s$dir, s$config_id, s$threshold,
         unlist(lapply(s$intervals, function(iv) c(iv$constant$q, iv$level_di$q, iv$level_di$coef))),
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
       sources = lapply(cal$sources, function(s) list(
         name = s$name, di_group = s$di_group, threshold = s$threshold,
         smearing = s$smearing, intervals = s$intervals)),
       threads = tpw, gdal_cache_mb = 512L, blocky = work$blocky,
       gc_threshold_mb = 1000L, gc_every_s = 1,
       units_dir = file.path(run_dir, "units"), probe_cells = NULL)
}

.predict_done_path <- function(units_dir, unit_id) file.path(units_dir, unit_id, "done.rds")

# The workers: as dsm_final()'s, each its own R process with its thread count
# set before torch loads, claiming units by dir.create() (atomic: made or
# found made). A unit's numbers depend on the unit and not on the worker.
.predict_run_units <- function(job, units, n_workers, run_dir, tag, say, progress = TRUE) {
  if (!requireNamespace("callr", quietly = TRUE)) {
    stop("dsm_predict() maps in worker processes and needs the callr package. ",
         "install.packages(\"callr\").", call. = FALSE)
  }
  root <- get0(".dlc_root", envir = globalenv(), inherits = FALSE)
  if (is.null(root)) {
    stop("dsm_predict() starts its workers from R/load_all.R and cannot find it: ",
         "source R/load_all.R first.", call. = FALSE)
  }
  claims_dir <- file.path(run_dir, paste0(".claims_", tag))
  logs_dir   <- file.path(run_dir, "logs")
  unlink(claims_dir, recursive = TRUE)          # stale claims of a run that died
  create_output_dirs(c(claims_dir, logs_dir, job$units_dir))
  job$claims_dir <- claims_dir
  job$units <- units
  done_paths <- .predict_done_path(job$units_dir, units$unit_id)
  cells <- as.numeric(units$r1 - units$r0 + 1L) * (units$c1 - units$c0 + 1L)

  t0 <- Sys.time()
  procs <- lapply(seq_len(n_workers), function(w) {
    callr::r_bg(
      .predict_worker_entry,
      args = list(job = c(job, list(worker = w)), root = root),
      # The primitive caches of ideep (LRU_CACHE_CAPACITY) and oneDNN
      # (ONEDNN_PRIMITIVE_CACHE_CAPACITY) default to 1,024 entries each, and
      # an entry holds buffers sized to its input: bounded here, beside the
      # few strip shapes the unit loop keeps to (see "THE STRIP'S SHAPE").
      env = c(callr::rcmd_safe_env(),
              OMP_NUM_THREADS = as.character(job$threads),
              MKL_NUM_THREADS = as.character(job$threads),
              LRU_CACHE_CAPACITY = "64",
              ONEDNN_PRIMITIVE_CACHE_CAPACITY = "64"),
      stdout = file.path(logs_dir, sprintf("%s_worker_%02d.log", tag, w)), stderr = "2>&1",
      supervise = TRUE)
  })
  # An interrupted map must not leave workers writing into its units: the next
  # call deletes the claims and would hand the same units out again.
  on.exit(for (p in procs) if (p$is_alive()) p$kill(), add = TRUE)

  seen <- rep(FALSE, nrow(units))
  valid_done <- 0
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
    if (!any(alive)) break
    Sys.sleep(if (progress) 5 else 1)
  }
  res  <- lapply(procs, function(p) tryCatch(p$get_result(), error = function(e) e))
  errs <- vapply(res, function(r) inherits(r, "error"), logical(1))
  for (w in which(errs)) {
    say("  worker ", w, " stopped with an error: ", conditionMessage(res[[w]]),
        "\n    log: ", file.path(logs_dir, sprintf("%s_worker_%02d.log", tag, w)))
  }
  peaks <- vapply(res[!errs], function(r) as.numeric(r$peak_gb %||% NA_real_), numeric(1))
  unlink(claims_dir, recursive = TRUE)
  list(n_workers = n_workers,
       peak_gb = if (any(is.finite(peaks))) max(peaks[is.finite(peaks)]) else NA_real_,
       minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
}

# At the top level on purpose, as .final_worker_entry(): nothing a worker
# does may depend on a frame serialised along with it.
.predict_worker_entry <- function(job, root) {
  # BEFORE torch loads -- the threshold is read once, when torch starts, and
  # R/load_all.R builds torch modules as it sources. torch then runs a LIGHT
  # collection every job$gc_threshold_mb of new allocations (lantern's CPU
  # allocator, src/lantern/src/Allocator.cpp), and caches freed blocks up to
  # the same amount. See .predict_gc_hook() for the full collections.
  options(torch.threshold_call_gc = job$gc_threshold_mb)
  suppressMessages(source(file.path(root, "R", "load_all.R")))
  .predict_worker(job)
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
    if (inherits(rec, "error")) {
      message("  ERROR in ", uid, ": ", conditionMessage(rec))
      safe_save_rds(list(unit_id = uid, status = "failed", error = conditionMessage(rec),
                         worker = job$worker, at = Sys.time()),
                    file.path(udir, "failed.rds"), compress = FALSE)
    } else {
      rec$worker  <- job$worker
      rec$threads <- job$threads
      rec$peak_gb <- .final_peak_gb()
      # The record LAST: its existence is what says the unit is finished.
      safe_save_rds(rec, done, compress = FALSE)
      s <- rec$seconds
      message(sprintf("  %s valid px of %s in %.1f s (read %.1f | network %.1f | DI %.1f | bands %.1f | write %.1f)",
                      format(rec$n_valid, big.mark = ","), format(rec$n_cells, big.mark = ","),
                      rec$total_s, s[["read"]], s[["net"]], s[["di"]], s[["bands"]], s[["write"]]))
    }
    invisible(gc(verbose = FALSE))
  }
  list(worker = job$worker, peak_gb = .final_peak_gb())
}

.predict_worker_setup <- function(job) {
  set_torch_threads(job$threads)
  tryCatch(terra::gdalCache(job$gdal_cache_mb), error = function(e) NULL)
  # GDAL decodes the strips of one read on several threads (GDAL >= 3.6); the
  # torch threads are idle while a band is read, so this costs nothing.
  tryCatch(terra::setGDALconfig("GDAL_NUM_THREADS", as.character(job$threads)),
           error = function(e) NULL)
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
  list(models = models, srcs = srcs, n_ch = length(job$files), n_seeds = length(models),
       rules = rules, has_rule = has_rule, inverse = target_transform_spec(job$transform)$inverse,
       di = di, n_di = n_di, index_base = index_base, gc_hook = .predict_gc_hook(job$gc_every_s),
       fcn_engine = if (identical(job$engine, "patch")) "patch" else "fcn")
}

# Rows of every band, QC'd and scaled, as float32 over the unit's buffer
# columns, with the mask of cells finite in every channel. Rows and columns
# outside the raster are 0 and masked: the full-window rule then discards any
# centre whose window leaves the raster, which is the rule stage 05 applied.
.predict_read_rows <- function(rows, rc0, rc1, pad_l, pad_r, w_buf, env, job) {
  nr <- length(rows)
  inside <- rows >= 1L & rows <= job$grid$nrow
  if (!any(inside)) {
    return(list(x = torch::torch_zeros(c(env$n_ch, nr, w_buf)),
                fin = torch::torch_zeros(c(nr, w_buf), dtype = torch::torch_bool())))
  }
  ra <- min(rows[inside]); rn <- sum(inside)
  pads <- c(pad_l, pad_r, ra - rows[1], nr - (ra - rows[1]) - rn)
  wn <- rc1 - rc0 + 1L
  x <- torch::torch_empty(c(env$n_ch, nr, w_buf))
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
    x[k, , ] <- if (any(pads > 0L)) torch::nnf_pad(t, pads) else t
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
  ids <- lapply(seq_len(n_g), function(g) {
    s <- cal$sources[[which(vapply(cal$sources, function(z) z$di_group == g, logical(1)))[1]]]
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
# seeds) and the DI (pixels x reference groups). The interval and smearing
# arithmetic is the library's own -- conformal_interval(),
# conformal_scaled_interval(), smear() -- so the map applies exactly what was
# calibrated.
.predict_step_values <- function(P, D, env, job) {
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
        key <- paste(r$source, r$label, r$method)
        if (is.null(cache[[key]])) {
          src <- job$sources[[r$source]]
          ivs <- src$intervals[[r$label]]
          cache[[key]] <- if (identical(r$method, "constant")) {
            conformal_interval(ivs$constant, st$median, lower_limit = cl[1])
          } else {
            conformal_scaled_interval(ivs$level_di, st$median,
                                      data.frame(level = st$median, di = D[, src$di_group]),
                                      lower_limit = cl[1])
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
  n_valid <- 0
  halo <- NULL; fin_halo <- NULL
  first <- TRUE

  for (o0 in seq(u$r0, u$r1, by = job$step_rows)) {
    o1 <- min(o0 + job$step_rows - 1L, u$r1)
    gs <- o1 - o0 + 1L

    # ── the rows: read once, the halo kept ──────────────────────────────────
    tr <- Sys.time()
    # The first step reads its halo and its own rows as two reads -- disjoint
    # rows, so no row is decompressed twice -- instead of one block cut in
    # two afterwards: the block, its halo's clone and its rows' copy alive
    # together were ~11 GB of a full-width worker's peak (T3).
    if (first && h > 0L) {
      hb <- .predict_read_rows((o0 - h):(o0 + h - 1L), rc0, rc1, pad_l, pad_r, w_buf, env, job)
      halo <- hb$x; fin_halo <- hb$fin
      rm(hb)
    }
    first <- FALSE
    blk <- .predict_read_rows((o0 + h):(o1 + h), rc0, rc1, pad_l, pad_r, w_buf, env, job)
    new <- blk$x; fin_new <- blk$fin
    rm(blk)
    tm[["read"]] <- tm[["read"]] + secs(tr)

    fin_full <- if (h > 0L) torch::torch_cat(list(fin_halo, fin_new), dim = 1L) else fin_new
    valid <- .predict_valid(fin_full, h, gs, w_out)
    rm(fin_full)

    # ── the network and the DI, chunk by chunk ──────────────────────────────
    I_all <- list(); P_all <- list(); D_all <- list(); k <- 0L
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
      bcols <- a0:(a1 + 2L * h)
      strip <- if (h > 0L) {
        torch::torch_cat(list(halo[, , bcols, drop = FALSE], new[, , bcols, drop = FALSE]), dim = 2L)
      } else new[, , bcols, drop = FALSE]
      strip <- strip$contiguous()
      cs <- cbind(i + h, j - a0 + 1L + h)
      tn <- Sys.time()
      x4 <- strip$unsqueeze(1L)
      P <- matrix(NA_real_, length(i), S)
      for (s in seq_len(S)) {
        P[, s] <- fcn_predict_strip(env$models[[s]], x4, cs, engine = env$fcn_engine,
                                    batch = job$batch, gather_mb = job$gather_mb,
                                    gc_hook = env$gc_hook)
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
      rm(strip, x4)
    }
    idx <- unlist(I_all)
    nv <- length(idx)
    n_valid <- n_valid + nv

    # ── the bands ───────────────────────────────────────────────────────────
    tb <- Sys.time()
    vals <- NULL
    if (nv > 0L) {
      P <- do.call(rbind, P_all)
      D <- if (length(env$di) > 0L) do.call(rbind, D_all) else NULL
      vals <- .predict_step_values(P, D, env, job)
      if (!is.null(probe)) {
        here <- probe$row >= o0 & probe$row <= o1 & probe$col >= u$c0 & probe$col <= u$c1
        if (any(here)) {
          pos <- match((probe$row[here] - o0) * w_out + (probe$col[here] - u$c0 + 1L), idx)
          probe_pred[here, ] <- P[pos, , drop = FALSE]
        }
      }
    }
    tm[["bands"]] <- tm[["bands"]] + secs(tb)

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

    # ── the halo the next step keeps ────────────────────────────────────────
    if (h > 0L) {
      if (gs >= 2L * h) {
        halo <- new[, (gs - 2L * h + 1L):gs, , drop = FALSE]$clone()
        fin_halo <- fin_new[(gs - 2L * h + 1L):gs, , drop = FALSE]$clone()
      } else {
        halo <- torch::torch_cat(list(halo[, (gs + 1L):(2L * h), , drop = FALSE], new), dim = 2L)
        fin_halo <- torch::torch_cat(list(fin_halo[(gs + 1L):(2L * h), , drop = FALSE], fin_new), dim = 1L)
      }
    }
    rm(new, fin_new, vals, P_all, D_all, I_all)
    invisible(gc(verbose = FALSE))
  }

  for (w in writers) terra::writeStop(w$rast)
  open <- FALSE
  list(unit_id = u$unit_id, r0 = u$r0, r1 = u$r1, c0 = u$c0, c1 = u$c1,
       n_cells = as.numeric(u$r1 - u$r0 + 1L) * w_out, n_valid = n_valid,
       seconds = tm, total_s = secs(t_unit), band_stats = bstat,
       probe_pred = probe_pred,
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
.predict_stored_all <- function(fr) {
  tabs <- lapply(fr$seeds, function(s) {
    p <- file.path(fr$run_dir, fr$config_id, "predictions", sprintf("seed%04d_pred_all.csv", s))
    if (!file.exists(p)) return(NULL)
    d <- safe_read_csv2(p)
    if (!all(c("sample_id", "pred_transform") %in% names(d))) return(NULL)
    d[is.finite(d$pred_transform), c("sample_id", "pred_transform"), drop = FALSE]
  })
  if (any(vapply(tabs, is.null, logical(1)))) return(NULL)
  ids <- Reduce(intersect, lapply(tabs, function(d) d$sample_id))
  if (length(ids) == 0L) return(NULL)
  pred <- vapply(tabs, function(d) as.numeric(d$pred_transform[match(ids, d$sample_id)]),
                 numeric(length(ids)))
  list(ids = ids, pred = matrix(pred, nrow = length(ids)))
}

.predict_probe <- function(job, fr, inp, grid, run_dir, say, tol = 1e-3) {
  st <- .predict_stored_all(fr)
  if (is.null(st)) {
    reason <- "the final run stored no transform-space prediction (pred_transform) for its seeds"
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
  mp <- rec$probe_pred
  valid <- stats::complete.cases(mp)
  rel <- abs(mp - stored) / (1 + abs(stored))
  worst <- if (any(valid)) max(rel[valid, , drop = FALSE]) else NA_real_
  tab <- tibble::tibble(sample_id = pts$sample_id, row = pts$row, col = pts$col,
                        valid_on_map = valid,
                        max_abs_diff = apply(abs(mp - stored), 1L, max),
                        max_rel_diff = apply(rel, 1L, max))
  safe_write_csv2(tab, file.path(run_dir, "probe.csv"))
  pass <- all(valid) && is.finite(worst) && worst <= tol
  reason <- if (!all(valid)) {
    sprintf("%d of %d profile pixel(s) were not predicted on the map", sum(!valid), length(valid))
  } else sprintf("max relative difference %.2e against a tolerance of %.0e", worst, tol)
  say(sprintf("Probe: %s -- %d profile(s) x %d seed(s), %s (%.0f s).",
              if (pass) "PASS" else "FAIL", nrow(pts), ncol(mp), reason,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
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
    peak_gb = r$peak_gb %||% NA_real_, finished_at = as.character(r$finished_at))))
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
