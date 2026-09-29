# ── The tuning units side by side ─────────────────────────────────────────────
#
# WHY THIS EXISTS.
#
# dsm_train() trained its units one at a time, in the user's session, with
# every core on each: n_cores, the physical cores minus one, 15 here. Two
# measurements say that wastes the machine (2026-09-27):
#
#   T1  one unit uses this CPU poorly. Tripling its threads from 5 to 15 made
#       the heaviest configuration only 1.5x faster (5.58 against 3.72 s an
#       epoch), and the lightest not at all (0.67 against 0.70 s).
#   T2  three units of 5 threads side by side trained 1.52x faster than one of
#       15 in sequence, and each gave, to the bit, the numbers it gave alone.
#       A unit's result depends on its seed and its thread count, and on
#       nothing else.
#
# dsm_final() has trained its seeds that way since. Its tuning did not, and
# what brought it here is the SOC 0-30 cm trial's full run (2026-09-29): 3,690
# units, about three months in sequence and about two side by side.
#
# HOW.
#
# The mechanism is dsm_final()'s (R/final.R), and its pieces are called here,
# not copied:
#   - each worker is an R process of its own, started with OMP_NUM_THREADS
#     already set;
#   - it claims a unit with dir.create(), which makes a directory or finds it
#     made, atomically, so no two workers train the same unit;
#   - it reads the store in turn with the others (.final_one_reader());
#   - how many run at once follows from n_cores and the RAM
#     (.final_worker_gb(), .final_workers_that_fit()).
# What tuning adds:
#
#   FOLDS. A unit trains from its fold's cache: the fold's rows of every
#   window, scaled with the fold's own training rows. Building one reads the
#   whole store. So the units are claimed in fold order. A worker takes the
#   next free unit of the fold it is in, builds that fold's cache once, when it
#   claims its first unit there, and moves to the next fold only when no unit
#   of this one is left to claim -- never back. Each worker keeps its caches in
#   one fold buffer (new_fold_buffer()), as the fold loop does, for the same
#   reason (T8: mimalloc keeps the blocks a fold releases).
#
#   THE TABLE. Three processes appending to one comparison_all.rds would lose
#   rows. So each unit writes its row to a record of its own, units/<unit>.rds,
#   and writes it LAST: a record means the unit is finished. The session
#   gathers the records into comparison_all as they arrive, and ranks the run
#   when the workers are done. A resume reads the table and the records, the
#   record winning, and keeps what finished -- matched by the hyperparameters,
#   as always (.resumable_units()).
#
#   THE SCALING of every fold is fitted here, once, from the fold's training
#   rows -- the scaling build_fold_cache() would fit in each worker. It is
#   written as the fold loop writes it (scaling_foldNN.csv) and handed to
#   every worker.
#
# WHAT IT DOES NOT CHANGE. The units, their names, their seeds, the files each
# writes and the rows of the table. A unit trained in a worker runs the code the
# fold loop runs for it (.train_unit(), R/train_cnn.R).
#
# WHAT IT CHANGES. The thread count of each unit, 5 instead of n_cores, and so
# the numbers: in T1, another thread count moved the heavy configuration's
# val_ccc by 0.067 in 12 epochs. That is why the run records the count
# (.train_threads_record(), R/api.R) and refuses a resume with another.

.train_unit_record_path <- function(run_dir, unit_id) {
  file.path(run_dir, "units", paste0(unit_id, ".rds"))
}

# The run's table as the disk holds it: comparison_all (the RDS, or the CSV
# for a run from before the RDS existed), with each unit's record in place of
# its row there. with_table = FALSE reads the records alone: a run started over
# (resume = FALSE) must not inherit the rows of the last one.
.train_comparison_so_far <- function(run_dir, tune_grid, with_table = TRUE) {
  cmp <- tibble::tibble()
  if (isTRUE(with_table)) {
    rds <- file.path(run_dir, "comparison", "comparison_all.rds")
    csv <- file.path(run_dir, "comparison", "comparison_all.csv")
    cmp <- if (file.exists(rds)) {
      readRDS(rds)
    } else if (file.exists(csv)) {
      .comparison_from_csv(csv, tune_grid)
    } else {
      tibble::tibble()
    }
    if (nrow(cmp) > 0L && !"unit_id" %in% names(cmp)) cmp$unit_id <- cmp$config_id
  }
  files <- list.files(file.path(run_dir, "units"), pattern = "[.]rds$", full.names = TRUE)
  rows <- lapply(files, function(f) tryCatch(readRDS(f)$row, error = function(e) NULL))
  rows <- rows[!vapply(rows, is.null, logical(1))]
  if (length(rows) > 0L) {
    new <- dplyr::bind_rows(rows)
    if (nrow(cmp) > 0L) cmp <- cmp[!cmp$unit_id %in% new$unit_id, , drop = FALSE]
    cmp <- dplyr::bind_rows(cmp, new)
  }
  cmp
}

# A duration as a person reads it: minutes, then hours, then days.
.train_duration <- function(minutes) {
  if (length(minutes) != 1L || !is.finite(minutes)) return("?")
  if (minutes < 120) return(sprintf("%.0f min", minutes))
  if (minutes < 48 * 60) return(sprintf("%.1f h", minutes / 60))
  sprintf("%.1f d", minutes / 1440)
}

#' Train a tuning run's units side by side, each worker an R process of its own.
#'
#' Called by run_cnn_resample() once the plan is checked and written; it
#' returns the run's ranked table. See the header of R/train_workers.R.
#'
#' @param training The arguments for train_one_cnn() (n_epochs, patience,
#'   clamp, ...): what the fold loop passes through `...`.
#' @param windows  The windows the grid needs: every worker's fold cache holds
#'   them all.
#' @param n_cores  The cores of the whole run; n_cores %/% threads_per_unit
#'   workers, fewer if the RAM holds fewer.
#' @param say      How to report: message(), or nothing.
#' @return list(comparison, n_workers, peak_gb).
#' @noRd
.train_side_by_side <- function(tune_grid, store, points, type_table, plan, transform,
                                run_dir, base_seed, n_seeds, resume, evaluate_test,
                                training, windows, n_cores, threads_per_unit,
                                max_ram_gb, say = message) {
  if (!requireNamespace("callr", quietly = TRUE)) {
    stop("dsm_train() trains its units side by side, each in an R process of its ",
         "own, and needs the callr package: install.packages(\"callr\"). Or ",
         "in_session = TRUE, one unit at a time in this session.", call. = FALSE)
  }
  if (is.null(store$patch_dir)) {
    stop("The workers read the store from its files, and this store does not say ",
         "where they are. Load it with dsm_load().", call. = FALSE)
  }
  n_cores <- n_cores %||% resolve_cores(NULL, what = "training")
  n_seeds <- as.integer(n_seeds)
  create_output_dirs(file.path(run_dir, c("models", "history", "predictions", "metrics",
                                          "gates", "comparison", "units", "logs")))
  .write_tune_grid(run_dir, tune_grid, resume)

  # A run started over keeps none of the last one's records.
  if (!isTRUE(resume)) {
    unlink(file.path(run_dir, "units"), recursive = TRUE)
    create_output_dirs(file.path(run_dir, "units"))
  }

  # THE SCALING OF EVERY FOLD, fitted here once from the fold's training rows.
  # It is what build_fold_cache() fits in the fold loop, written as the fold
  # loop writes it, and a channel constant within a fold stops the run here,
  # before any worker has started, and not inside each of them.
  scalings <- lapply(seq_along(plan$folds), function(j) {
    sc <- fit_scaling(points, type_table, plan$folds[[j]]$train)
    if (any(sc$degenerate)) {
      stop("Degenerate scaling (zero or non-finite sd) on fold ", j, "'s training ",
           "rows for: ", paste(sc$predictor[sc$degenerate], collapse = ", "),
           ".\n  A channel constant within a fold cannot be z-scored.", call. = FALSE)
    }
    safe_write_csv2(sc, file.path(run_dir, sprintf("scaling_fold%02d.csv", j)))
    sc
  })

  # THE UNITS, IN FOLD ORDER, then by configuration, the seeds innermost -- the
  # fold loop's order, so that the units of a fold are claimed together and a
  # configuration's seeds finish together. Their names and seeds are the fold
  # loop's.
  n_cfg <- nrow(tune_grid)
  g <- expand.grid(seed_i = seq_len(n_seeds), i = seq_len(n_cfg),
                   fold = seq_along(plan$folds))
  g <- g[order(g$fold, g$i, g$seed_i), , drop = FALSE]
  units <- tibble::tibble(fold = as.integer(g$fold), i = as.integer(g$i),
                          seed_i = as.integer(g$seed_i))
  units$unit_id <- sprintf("%s_f%d_s%d", tune_grid$config_id[units$i], units$fold,
                           units$seed_i)
  units$seed <- as.integer(base_seed) + units$seed_i - 1L

  # What is done: a success row, its checkpoint on disk, and the hyperparameters
  # its name claims -- the three conditions of the fold loop's resume.
  comparison <- .train_comparison_so_far(run_dir, tune_grid, with_table = resume)
  done <- character(0)
  if (nrow(comparison) > 0L) {
    done <- comparison$unit_id[comparison$status == "success"]
    done <- done[file.exists(file.path(run_dir, "models", paste0(done, "_best.pt")))]
    done <- .resumable_units(done, comparison, tune_grid)
  }
  kept <- units$unit_id %in% done
  todo <- units[!kept, , drop = FALSE]
  if (any(kept)) {
    say(sprintf("Resume: %d of %d unit(s) already trained, kept as they are.",
                sum(kept), nrow(units)))
  }

  run_info <- list(n_workers = 0L, peak_gb = NA_real_)
  if (nrow(todo) > 0L) {
    run_info <- .train_run_workers(
      todo = todo, comparison = comparison, tune_grid = tune_grid, store = store,
      points = points, type_table = type_table, plan = plan, scalings = scalings,
      transform = transform, run_dir = run_dir, evaluate_test = evaluate_test,
      training = training, windows = windows, n_cores = n_cores,
      threads_per_unit = threads_per_unit, max_ram_gb = max_ram_gb, say = say)
  }

  comparison <- .train_comparison_so_far(run_dir, tune_grid, with_table = resume)
  comparison_path <- file.path(run_dir, "comparison", "comparison_all.csv")
  write_comparison(comparison, comparison_path,
                   file.path(run_dir, "comparison", "comparison_all.rds"))

  # EVERY UNIT, OR STOP. A unit this call set out to train and left without a
  # record never finished -- its worker died under it. Its row, if the table
  # has one, is the last call's (a failure, or other hyperparameters), and a
  # table ranked with it would say nothing about it. Said here, while the run is
  # still in front of someone; the next call trains only what is missing.
  missing <- todo$unit_id[!file.exists(.train_unit_record_path(run_dir, todo$unit_id))]
  if (length(missing) > 0L) {
    stop(length(missing), " unit(s) did not finish: ",
         paste(utils::head(missing, 6), collapse = ", "),
         if (length(missing) > 6L) ", ..." else "",
         ".\n  Worker logs: ", file.path(run_dir, "logs"),
         "\n  Fix the cause and call dsm_train() again with the same run_id -- the ",
         "finished units are kept.", call. = FALSE)
  }
  comparison <- .rank_comparison(comparison, run_dir, "the run", comparison_path)
  list(comparison = comparison, n_workers = run_info$n_workers,
       peak_gb = run_info$peak_gb)
}

# The workers, started and watched. The table is gathered as their records
# arrive, so comparison_all on disk is never more than a unit behind: the
# run can be read while it trains, and a session that dies loses nothing that
# a worker finished.
.train_run_workers <- function(todo, comparison, tune_grid, store, points, type_table,
                               plan, scalings, transform, run_dir, evaluate_test,
                               training, windows, n_cores, threads_per_unit,
                               max_ram_gb, say) {
  loader <- .pkg_loader()          # the framework this session runs, for each worker

  est <- .final_worker_gb(store, windows)
  n_workers <- max(1L, min(n_cores %/% threads_per_unit, nrow(todo)))
  budget <- max_ram_gb
  if (is.null(budget) && requireNamespace("ps", quietly = TRUE)) {
    avail <- tryCatch(ps::ps_system_memory()$avail / 1e9, error = function(e) NA_real_)
    if (is.finite(avail)) budget <- 0.7 * avail
  }
  if (!is.null(budget)) {
    fits <- .final_workers_that_fit(est, budget)
    if (fits < n_workers) {
      say("  RAM caps the workers at ", fits, " of ", n_workers, " (", sprintf("%.1f", budget),
          " GB budget, ~", sprintf("%.1f", est[["steady"]]), " GB each, one at a time up to ",
          sprintf("%.1f", est[["peak"]]), " while it reads). The numbers do not ",
          "change -- only the time.")
      n_workers <- fits
    }
    if (est[["peak"]] > budget) {
      say("  WARNING: one worker is estimated at ", sprintf("%.1f", est[["peak"]]),
          " GB at its peak, against a budget of ", sprintf("%.1f", budget),
          " GB. It is started anyway.")
    }
  }
  say(sprintf(paste0("\nTraining %d unit(s): %d worker(s) side by side x %d thread(s), ",
                     "~%.1f GB each, up to %.1f while one reads the store -- they read one ",
                     "at a time (estimate). Each worker's log: %s"),
              nrow(todo), n_workers, threads_per_unit, est[["steady"]], est[["peak"]],
              file.path(run_dir, "logs")))

  # The workers are given absolute paths: they start in this session's working
  # directory today, and nothing here should depend on that.
  run_abs    <- normalizePath(run_dir, winslash = "/", mustWork = FALSE)
  claims_dir <- file.path(run_abs, ".claims")
  logs_dir   <- file.path(run_abs, "logs")
  unlink(claims_dir, recursive = TRUE)       # stale claims of a run that died
  create_output_dirs(c(claims_dir, logs_dir))
  # A unit trained again -- it failed, or its checkpoint is gone -- starts with
  # no record: an old one would read as this call's.
  record_paths <- .train_unit_record_path(run_dir, todo$unit_id)
  unlink(record_paths)

  job <- list(
    patch_dir = normalizePath(store$patch_dir, winslash = "/", mustWork = FALSE),
    points = points, type_table = type_table, windows = windows,
    folds = plan$folds, scalings = scalings, grid = tune_grid, units = todo,
    training = training, transform = transform, evaluate_test = evaluate_test,
    run_dir = run_abs, claims_dir = claims_dir, threads = threads_per_unit,
    fold_buffer = .use_fold_buffer(),
    # options(dsm.train.trace_mem = TRUE): each worker records its memory at
    # every phase and every unit, as the fold loop does in the session.
    trace_mem = isTRUE(getOption("dsm.train.trace_mem", FALSE)))
  .pkg_check_portable(job, "dsm_train()")

  t0 <- Sys.time()
  procs <- lapply(seq_len(n_workers), function(w) {
    callr::r_bg(
      .train_worker_entry,
      args = list(job = c(job, list(worker = w)), loader = loader),
      env = c(callr::rcmd_safe_env(),
              OMP_NUM_THREADS = as.character(threads_per_unit),
              MKL_NUM_THREADS = as.character(threads_per_unit),
              # A diagnosis can add to the workers' environment, as dsm_final()'s can.
              getOption("dsm.train.worker_env", character(0))),
      stdout = file.path(logs_dir, sprintf("worker_%02d.log", w)), stderr = "2>&1",
      supervise = TRUE)
  })
  # An interrupted run must not leave workers training into it: the next call
  # clears the claims and would hand their units out again.
  on.exit(for (p in procs) if (p$is_alive()) p$kill(), add = TRUE)

  comparison_path <- file.path(run_dir, "comparison", "comparison_all.csv")
  comparison_rds  <- file.path(run_dir, "comparison", "comparison_all.rds")
  seen <- rep(FALSE, nrow(todo))
  repeat {
    # Alive first, records after: a worker that has exited wrote its last
    # record before it did, so the pass that finds no worker alive reads them all.
    alive <- vapply(procs, function(p) p$is_alive(), logical(1))
    fresh <- which(!seen & file.exists(record_paths))
    rows  <- lapply(record_paths[fresh], function(f) tryCatch(readRDS(f)$row, error = function(e) NULL))
    got   <- !vapply(rows, is.null, logical(1))
    if (any(got)) {
      new <- dplyr::bind_rows(rows[got])
      if (nrow(comparison) > 0L) {
        comparison <- comparison[!comparison$unit_id %in% new$unit_id, , drop = FALSE]
      }
      comparison <- dplyr::bind_rows(comparison, new)
      write_comparison(comparison, comparison_path, comparison_rds)
      seen[fresh[got]] <- TRUE
      el   <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
      n    <- sum(seen)
      eta  <- el / n * (nrow(todo) - n)
      last <- new[nrow(new), , drop = FALSE]
      say(sprintf("  %8s | %d of %d unit(s) | %d worker(s) running | ETA %s | %s: %s",
                  .train_duration(el), n, nrow(todo), sum(alive), .train_duration(eta),
                  last$unit_id,
                  if (identical(last$status, "success")) {
                    sprintf("val_ccc %.3f, %.0f min", last$val_ccc, last$runtime_min)
                  } else "FAILED -- see its error_message"))
    }
    if (!any(alive)) break
    Sys.sleep(5)
  }

  res  <- lapply(procs, function(p) tryCatch(p$get_result(), error = function(e) e))
  errs <- vapply(res, function(r) inherits(r, "error"), logical(1))
  for (w in which(errs)) {
    say("  worker ", w, " stopped with an error: ", conditionMessage(res[[w]]),
        "\n    log: ", file.path(logs_dir, sprintf("worker_%02d.log", w)))
  }
  peaks <- vapply(res[!errs], function(r) as.numeric(r$peak_gb %||% NA_real_), numeric(1))
  peak  <- if (any(is.finite(peaks))) max(peaks[is.finite(peaks)]) else NA_real_
  say(sprintf("  workers done in %s | peak RAM per worker %.1f GB (estimated %.1f)",
              .train_duration(as.numeric(difftime(Sys.time(), t0, units = "mins"))),
              peak, est[["peak"]]))
  unlink(claims_dir, recursive = TRUE)
  list(n_workers = n_workers, peak_gb = peak)
}

# What each worker process runs. At the top level on purpose, as
# .final_worker_entry() is: nothing a worker does may depend on a frame
# serialised along with it. The framework is loaded the way the session loaded
# it, and the worker taken from its namespace, where it is not exported.
.train_worker_entry <- function(job, loader) {
  ns <- loader$open(loader)
  get(".train_worker", envir = ns)(job)
}

# ONE WORKER: the store's table, then units claimed one at a time, in fold
# order, until none is left. A fold's cache is built when the worker claims
# its first unit there, and replaced in the same buffer when it moves on.
.train_worker <- function(job) {
  set_torch_threads(job$threads)

  # THE WORKER'S MEMORY, when the run is asked for it (options(dsm.train.
  # trace_mem = TRUE)): the fold loop's marks, per worker, written after every
  # one, so that a worker that dies leaves its trace up to its last mark.
  t0 <- Sys.time()
  trace <- list()
  mark <- function(phase, fold = NA_integer_, unit_id = NA_character_) {
    if (!isTRUE(job$trace_mem)) return(invisible(NULL))
    g <- gc(verbose = FALSE)
    trace[[length(trace) + 1L]] <<- data.frame(
      worker = job$worker, fold = as.integer(fold), unit_id = unit_id, phase = phase,
      seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")),
      rss_gb = .predict_rss_gb(), private_gb = .predict_private_gb(),
      peak_gb = .final_peak_gb(), r_heap_gb = sum(g[, 2]) / 1024,
      stringsAsFactors = FALSE)
    safe_save_rds(do.call(rbind, trace),
                  file.path(job$run_dir, "logs", sprintf("worker_%02d_mem_trace.rds", job$worker)),
                  compress = FALSE)
    invisible(NULL)
  }

  mark("start")
  # THE TABLE ONLY: each fold's cache reads its windows from the store's files
  # (build_fold_cache()), as dsm_final()'s workers do. The points come aligned
  # from the session, and are held to this store's rows again here.
  store  <- load_patch_store(job$patch_dir, integer(0), verbose = FALSE)
  points <- align_points_to_meta(job$points, store$meta)
  mark("store_loaded")
  buffer <- if (isTRUE(job$fold_buffer)) new_fold_buffer() else NULL
  device <- torch::torch_device("cpu")
  n_cfg  <- nrow(job$grid)

  cache   <- NULL
  pv      <- NULL
  at_fold <- NA_integer_
  for (u in seq_len(nrow(job$units))) {
    uid <- job$units$unit_id[u]
    if (!dir.create(file.path(job$claims_dir, uid), showWarnings = FALSE)) next
    j <- job$units$fold[u]
    if (!identical(at_fold, j)) {
      # This fold's roles are written into the buffer the last fold's were in:
      # the last fold's cache goes first, and with it the views of the buffer.
      cache <- NULL
      invisible(gc(verbose = FALSE))
      idx <- job$folds[[j]]
      message("\n", strrep("=", 78), "\nFOLD ", j, "/", length(job$folds), " -- ",
              paste(sprintf("%s=%d", names(idx), lengths(idx)), collapse = " | "),
              " (worker ", job$worker, ")\n", strrep("=", 78))
      fold <- .final_one_reader(job$claims_dir, build_fold_cache(
        store, points, job$type_table, idx, job$windows,
        scaling = job$scalings[[j]], verbose = FALSE, buffer = buffer))
      cache <- fold$cache
      rm(fold)
      pv <- fold_points_valid(store, idx)
      at_fold <- j
      mark("fold_cache", j)
    }
    i <- job$units$i[u]
    row <- .train_unit(
      cfg = job$grid[i, , drop = FALSE], unit_id = uid, fold = j,
      this_seed = job$units$seed[u],
      header = sprintf("%s  (config %d/%d, fold %d, seed %d | worker %d, %d thread(s))",
                       uid, i, n_cfg, j, job$units$seed[u], job$worker, job$threads),
      cache = cache, points_valid = pv, n_channels = store$n_channels,
      transform = job$transform, device = device, run_dir = job$run_dir,
      evaluate_test = job$evaluate_test, training = job$training)
    mark("unit_trained", j, uid)

    # The record LAST, and whole: written beside its place and renamed into it,
    # so the session never reads half of one.
    rec <- list(row = row, unit_id = uid, status = row$status[1], threads = job$threads,
                worker = job$worker, finished_at = Sys.time())
    path <- .train_unit_record_path(job$run_dir, uid)
    safe_save_rds(rec, paste0(path, ".part"), compress = FALSE)
    if (!file.rename(paste0(path, ".part"), path)) {
      stop("Could not move the record of ", uid, " into place: ", path, call. = FALSE)
    }
  }
  list(worker = job$worker, peak_gb = .final_peak_gb())
}
