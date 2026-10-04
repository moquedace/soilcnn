# ── Refit importance: the variable left out, the network trained again ───────
#
# WHAT IT ANSWERS. A permutation asks how much the fitted network relies on a
# variable; SHAP and SAGE, how it uses it. None asks whether the variable is
# NEEDED: a network trained without it may find the same signal in another.
# The refit does -- leave one covariate out (LOCO; Lei et al. 2018, Williamson
# et al. 2023): the variable's channels are set to their training mean,
# everywhere, and the final run's seeds are trained again. The test skill they
# lose is the variable's importance.
#
# HOW A VARIABLE IS LEFT OUT. Its channels are not taken out of the network:
# that would change the first layer, and with it every random draw after it,
# and the refit would differ from the run's seed in its initial weights as
# well as in the variable. They are set to their training mean -- 0 for a
# z-scored channel, the class frequency for a dummy -- in every pixel, window
# and row, so they carry nothing, and the network, its initial weights, its
# batches and its augmentations are the seed's own.
#
# THE SAME TRAINING AS THE RUN'S, AND CHECKED. The units are trained by
# dsm_final()'s own workers (R/final.R), with what the final run recorded in
# run_spec.rds: the split, the scaling -- the exact one, not the CSV beside
# the weights, whose last digits a training would amplify -- the schedule and
# the thread count. A seed's numbers depend on its seed and its thread count
# and on nothing else (T2), so a refit with nothing left out gives the run's
# predictions to the digit. It is trained first, for `check_seeds` seeds, and
# must; if it does not, another torch, store or point table is computing, and
# what a variable's absence costs could not be told from what changed.
#
# THE NOISE. A refit is another training: one without a variable the network
# does not need still lands elsewhere than the seed did, by about the spread
# between the run's seeds. A seed's drop within sqrt(2) times that sd is not
# evidence, nor a mean within sqrt(2 / seeds) times it; the print says how
# large that is.
#
# WHERE IT GOES. <final run>/refit_importance/<run_id>/: v000/ the check, v001/
# ... one directory per variable, each laid out as a final run's configuration
# is (models/, predictions/, units/ ...); variables.csv says which is which, and
# refit_spec.rds what the refit is of. A refit stopped midway resumes with its
# run_id, the units it finished kept.

#' Refit importance: what the model loses when trained without a variable.
#'
#' Leave one covariate out (LOCO; Lei et al. 2018, Williamson et al. 2023):
#' each variable's channels are set to their training mean -- every pixel,
#' every window, every row -- and the final run's seeds are trained again,
#' everything else as the run trained them: its split, scaling, schedule,
#' thread count, and the seed. The test skill lost is the variable's
#' importance. A permutation asks how much the fitted network relies on a
#' variable; this asks whether the variable is needed at all: variables that
#' carry the same information each cost little, because a network trained
#' without one finds it in the other.
#'
#' Before any variable is left out, the first `check_seeds` seeds are trained
#' again with nothing left out, and must give the run's own predictions -- or
#' the refit is not the run's training, and it stops. It trains (variables x
#' seeds + check_seeds) networks, side by side as [dsm_final()] does: give
#' [dsm_importance()] themes as `groups`, and a few of the run's `seeds`.
#'
#' @param metric What the table is ranked by: "ccc" (the drop in Lin's
#'   concordance, native units), "rmse" (the rise in RMSE, native units) or
#'   "rmse_transform" (in the space the model was trained in). All three are
#'   kept.
#' @param check_seeds How many of the seeds are trained again with nothing left
#'   out first, which must reproduce the run. 1 or more: a seed's numbers depend
#'   only on its seed and its thread count, so one checks the path.
#' @param run_id NULL for `refit_<timestamp>`. An existing one, with resume =
#'   TRUE, finishes a refit that stopped: its finished units are kept.
#' @param resume Keep the units an existing run_id already trained.
#' @param output_dir Where refits go. NULL: refit_importance/ in the final
#'   run's directory.
#' @param n_cores Cores for the refit; NULL is the physical cores minus one.
#'   Each unit trains with the final run's threads per unit -- part of its
#'   numbers -- so n_cores must be at least that.
#' @param max_ram_gb RAM the workers may use in total, as in [dsm_final()].
#' @return An `importance_spec`, for [dsm_importance()].
#' @export
refit_importance <- function(metric = c("ccc", "rmse", "rmse_transform"), check_seeds = 1L,
                             run_id = NULL, resume = TRUE, output_dir = NULL,
                             n_cores = NULL, max_ram_gb = NULL) {
  metric <- match.arg(metric)
  if (!is.numeric(check_seeds) || length(check_seeds) != 1L || !is.finite(check_seeds) ||
      check_seeds < 1 || check_seeds != round(check_seeds)) {
    stop("check_seeds must be one whole number of 1 or more: the check that a refit is the ",
         "run's training is not skipped.", call. = FALSE)
  }
  if (!is.null(run_id) && (!is.character(run_id) || length(run_id) != 1L || !nzchar(run_id))) {
    stop("run_id must be NULL or one name.", call. = FALSE)
  }
  if (!is.null(output_dir) && (!is.character(output_dir) || length(output_dir) != 1L)) {
    stop("output_dir must be NULL or one path.", call. = FALSE)
  }
  # draws, within and fill are what the frame reads of every method; a refit
  # fills what it leaves out with the training mean, once per unit.
  structure(list(kind = "refit", metric = metric, check_seeds = as.integer(check_seeds),
                 run_id = run_id, resume = isTRUE(resume), output_dir = output_dir,
                 n_cores = n_cores, max_ram_gb = max_ram_gb,
                 draws = 1L, within = NULL, fill = "mean"),
            class = "importance_spec")
}

# LEFT OUT, IN PLACE: each of `left_out`'s channels takes its training mean in
# every window and role of a fold cache, and what it held is kept, channel by
# channel -- blocks of a few MB, never one the size of a theme of the cache,
# which mimalloc would keep (T6). .final_worker() calls this before a unit
# trains and .importance_put_back() after it. Nothing to leave out, nothing
# done: a dsm_final() unit passes here untouched.
.importance_leave_out <- function(cache, left_out, fill) {
  if (length(left_out) == 0L) return(NULL)
  if (is.null(fill) || length(fill) < max(left_out) || any(!is.finite(fill[left_out]))) {
    stop("A unit leaves channel(s) out with no finite training mean to give them.", call. = FALSE)
  }
  kept <- list()
  for (role in names(cache)) {
    for (k in grep("^w[0-9]+$", names(cache[[role]]), value = TRUE)) {
      t <- cache[[role]][[k]]
      vals <- vector("list", length(left_out))
      for (j in seq_along(left_out)) {
        ch <- left_out[j]
        vals[[j]] <- t[, ch, , ]$clone()
        t[, ch, , ]$fill_(fill[[ch]])
      }
      kept[[length(kept) + 1L]] <- list(role = role, key = k, values = vals)
    }
  }
  list(channels = left_out, kept = kept)
}

# What .importance_leave_out() took, put back where it was.
.importance_put_back <- function(cache, kept) {
  if (is.null(kept)) return(invisible(NULL))
  for (e in kept$kept) {
    t <- cache[[e$role]][[e$key]]
    for (j in seq_along(kept$channels)) {
      ch <- kept$channels[j]
      t[, ch, , ]$copy_(e$values[[j]])
    }
  }
  invisible(NULL)
}

# WHAT A REFIT IS OF, written when it starts: the final run (by name and by
# when its spec was written, which a project moved whole keeps), the
# configuration, the variables and their channels, the fill they take, and
# the thread count. A resume that would leave out other channels, or refit
# another run, is refused: its units and the first ones would not be one
# refit. The seeds are not in it -- adding seeds is what a resume is for.
.importance_refit_spec <- function(run_dir, run_id, spec, resume) {
  f <- file.path(run_dir, "refit_spec.rds")
  if (!file.exists(f)) {
    held <- if (dir.exists(run_dir)) {
      list.files(run_dir, recursive = TRUE, all.files = TRUE, no.. = TRUE)
    } else character(0)
    if (length(held) > 0L) {
      stop("run_id \"", run_id, "\" names a directory refit_importance() did not start: ",
           run_dir, " holds ", length(held), " file(s) and no refit_spec.rds. Give another ",
           "run_id.", call. = FALSE)
    }
    create_output_dirs(run_dir)
    safe_save_rds(c(spec, list(written_at = Sys.time())), f, compress = FALSE)
    safe_write_csv2(spec$variables, file.path(run_dir, "variables.csv"))
    return(invisible(spec))
  }
  if (!isTRUE(resume)) {
    stop("A refit already exists at ", run_dir, ". Pass refit_importance(resume = TRUE) to ",
         "finish it, or another run_id.", call. = FALSE)
  }
  old <- readRDS(f)
  what <- c(final_run = "another final run", final_spec_written = "another final run",
            config_id = "another configuration", variables = "other variables or channels",
            fill = "other training means", threads_per_unit = "another thread count")
  diffs <- unique(unname(what[names(what)[!vapply(names(what), function(k)
    isTRUE(all.equal(old[[k]], spec[[k]])), logical(1))]]))
  if (length(diffs) > 0L) {
    stop("run_id \"", run_id, "\" was started for ", paste(diffs, collapse = ", "),
         " than this call gives: its units and this call's would not be one refit. Resume ",
         "it with the call that started it, or give another run_id.", call. = FALSE)
  }
  invisible(old)
}

# Every unit of `us` trained, or stop -- naming each that did not, and why.
.importance_refit_all_done <- function(run_dir, us) {
  done <- vapply(seq_len(nrow(us)), function(i) .final_unit_done(run_dir, us$dir[i], us$seed[i]),
                 logical(1))
  if (all(done)) return(invisible(TRUE))
  lost <- which(!done)
  msgs <- vapply(lost, function(i) {
    r <- .final_unit_record(run_dir, us$dir[i], us$seed[i])
    if (is.null(r)) "never trained" else as.character(r$error %||% r$status)
  }, character(1))
  stop("Refit unit(s) did not finish: ",
       paste(sprintf("%s (%s)", us$unit_id[lost], msgs), collapse = "; "),
       ".\n  Worker logs: ", file.path(run_dir, "logs"),
       "\n  Fix the cause and call again with refit_importance(run_id = \"", basename(run_dir),
       "\") -- the finished units are kept.", call. = FALSE)
}

# THE CHECK THAT STOPS: each check seed, trained again with nothing left out,
# against the predictions the final run wrote for it -- every row, every role.
.importance_refit_check <- function(fr, run_dir, seeds) {
  chk <- dplyr::bind_rows(lapply(seeds, function(s) {
    sl <- sprintf("seed%04d_pred_all.csv", s)
    a <- safe_read_csv2(file.path(fr$run_dir, fr$config_id, "predictions", sl))
    b <- safe_read_csv2(file.path(run_dir, "v000", "predictions", sl))
    at <- match(paste(a$dataset_role, a$sample_id), paste(b$dataset_role, b$sample_id))
    dev <- if (nrow(a) != nrow(b) || anyNA(at)) Inf else
      max(abs(a$pred_transform - b$pred_transform[at]) / pmax(1, abs(a$pred_transform)))
    tibble::tibble(seed = as.integer(s), n_rows = nrow(a), max_difference = dev)
  }))
  bad <- !is.finite(chk$max_difference) | chk$max_difference > 1e-4
  if (any(bad)) {
    stop(sprintf(paste0("Trained again with nothing left out, seed %d does not give the final ",
                        "run's predictions (they differ by %.2g, relative, at most). The refit ",
                        "here is not the run's training -- another torch, thread count, store or ",
                        "point table -- and what a variable's absence costs could not be told ",
                        "from what changed. Nothing was left out; the check's unit is in %s."),
                 chk$seed[bad][1], chk$max_difference[bad][1], file.path(run_dir, "v000")),
         call. = FALSE)
  }
  chk
}

# A unit's three scores, from the test rows of the predictions it wrote,
# computed as every importance computes them (.importance_scores()).
.importance_refit_scores <- function(pred_file, meta, transform, clamp) {
  pa <- safe_read_csv2(pred_file)
  pa <- pa[pa$dataset_role == "test", , drop = FALSE]
  at <- match(as.character(meta$sample_id), as.character(pa$sample_id))
  if (anyNA(at)) {
    stop(pred_file, " lacks ", sum(is.na(at)), " of the ", length(at), " test row(s).",
         call. = FALSE)
  }
  tibble::as_tibble(.importance_scores(pa$pred_transform[at], as.numeric(meta$target_native),
                                       as.numeric(meta$target_transform), transform, clamp))
}

# refit_importance() through the final run's seeds: the cheap check, the
# refit's record, the check seeds trained and compared, the variables left out,
# then their scores against the run's.
.importance_refit_run <- function(fr, data, units, grp, vars, method, transform, clamp,
                                  batch_size, say) {
  spec <- readRDS(file.path(fr$run_dir, "run_spec.rds"))
  need <- c("grid", "split", "scaling", "threads_per_unit", "training", "transform_probe")
  if (!all(need %in% names(spec))) {
    stop("The final run's run_spec.rds records no ", paste(setdiff(need, names(spec)), collapse = ", "),
         ": a refit could not train as its seeds trained.", call. = FALSE)
  }
  ch <- as.character(data$store$predictors)
  if (!identical(as.character(spec$scaling$predictor), ch)) {
    stop("The final run's scaling is not over the store's channels: this is not the store it ",
         "was fitted on.", call. = FALSE)
  }
  if (!isTRUE(all.equal(as.numeric(transform(c(0, 0.5, 1, 2.5, 5))),
                        as.numeric(spec$transform_probe)))) {
    stop("The store's inverse transform is not the one the final run trained with: a refit ",
         "would stop on, and be scored in, another space.", call. = FALSE)
  }
  cfg <- spec$grid[spec$grid$config_id == fr$config_id, , drop = FALSE]
  if (nrow(cfg) != 1L) {
    stop("Configuration ", fr$config_id, " is not in the final run's grid.", call. = FALSE)
  }
  windows <- sort(unique(as.integer(unlist(cfg$window_sizes))))
  tpu <- as.integer(spec$threads_per_unit)
  n_cores <- resolve_cores(method$n_cores, what = "the refit")
  if (tpu > n_cores) {
    stop(sprintf(paste0("The final run's seeds trained with %d thread(s) each, and the refit may ",
                        "use %d core(s). A seed's numbers depend on its thread count (T1): with ",
                        "fewer, the refit could not reproduce them. Give n_cores = %d or more."),
                 tpu, n_cores, tpu), call. = FALSE)
  }
  seeds   <- as.integer(units$seed)
  n_check <- min(method$check_seeds, length(seeds))
  fill    <- .importance_fill_values(data, spec$scaling, as.integer(spec$split$train))
  vtab <- tibble::tibble(dir = sprintf("v%03d", seq_along(vars)), variable = names(vars),
                         n_channels = as.integer(lengths(vars)),
                         channels = vapply(vars, function(v) paste(ch[v], collapse = ", "),
                                           character(1)))
  if (any(!is.finite(fill[unlist(vars)]))) {
    stop("A channel has no finite training mean to be left out with.", call. = FALSE)
  }
  say(sprintf("Importance -- %s | test set of final run '%s' (%s)", .importance_label(method),
              basename(fr$run_dir), fr$config_id))
  say("  ", .importance_groups_line(grp))

  # 1. THE CHEAP CHECK, in seconds where a refit takes minutes to hours: each
  # seed's checkpoint gives the predictions its run wrote, so the store, the
  # rows and the scaling are the run's before anything trains.
  rows_t <- units$rows[[1]]
  meta_t <- data$store$meta[rows_t, , drop = FALSE]
  cache  <- build_fold_cache(data$store, data$points, data$type_table, list(test = rows_t),
                             windows, scaling = units$scaling[[1]], verbose = FALSE)$cache$test
  inputs <- lapply(patch_window_key(as.integer(cfg$window_sizes[[1]])), function(k) cache[[k]])
  device <- torch::torch_device("cpu")
  for (u in seq_len(nrow(units))) {
    un <- units[u, , drop = FALSE]
    model <- build_cnn_from_config(un$cfg[[1]], data$store$n_channels)
    model$load_state_dict(torch::torch_load(un$model_file))
    model$to(device = device)
    f <- .importance_forward(model, inputs, batch_size)
    base <- tibble::as_tibble(.importance_scores(f, as.numeric(meta_t$target_native),
                                                 as.numeric(meta_t$target_transform),
                                                 transform, clamp))
    .importance_check_baseline(base, un, pred_t = f, sample_id = meta_t$sample_id,
                               obs = as.numeric(meta_t$target_native))
    rm(model)
  }
  rm(cache, inputs); invisible(gc(verbose = FALSE))
  say("  each seed's checkpoint gives the predictions its run wrote")

  # 2. THE REFIT'S RECORD, and its units: the check seeds with nothing left
  # out, then every variable under every seed.
  out_dir <- method$output_dir %||% file.path(fr$run_dir, "refit_importance")
  run_id  <- method$run_id %||% paste0("refit_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  run_dir <- normalizePath(file.path(out_dir, run_id), winslash = "/", mustWork = FALSE)
  .importance_refit_spec(run_dir, run_id, list(
    final_run = basename(fr$run_dir), final_spec_written = spec$written_at,
    config_id = fr$config_id, variables = vtab, fill = unname(fill), threads_per_unit = tpu),
    method$resume)
  check <- tibble::tibble(config_id = fr$config_id, seed = seeds[seq_len(n_check)], dir = "v000",
                          left_out = rep(list(integer(0)), n_check))
  g <- expand.grid(s = seq_along(seeds), v = seq_along(vars))
  left <- tibble::tibble(config_id = fr$config_id, seed = seeds[g$s], dir = vtab$dir[g$v],
                         left_out = unname(vars[g$v]))
  check$unit_id <- sprintf("%s_seed%04d", check$dir, check$seed)
  left$unit_id  <- sprintf("%s_seed%04d", left$dir, left$seed)
  for (d in c("v000", vtab$dir)) {
    create_output_dirs(file.path(run_dir, d, c("models", "history", "predictions", "metrics",
                                               "gates", "units")))
  }
  done <- function(us) vapply(seq_len(nrow(us)), function(i)
    .final_unit_done(run_dir, us$dir[i], us$seed[i]), logical(1))
  n_todo <- sum(!done(check)) + sum(!done(left))
  runtime <- vapply(seeds, function(s)
    as.numeric(.final_unit_record(fr$run_dir, fr$config_id, s)$runtime_min %||% NA_real_),
    numeric(1))
  n_work <- max(1L, n_cores %/% tpu)
  say(sprintf(paste0("  %d unit(s) to train of %d: %d seed(s) with nothing left out first, the ",
                     "check, then %d variable(s) x %d seed(s), %d thread(s) each%s"),
              n_todo, nrow(check) + nrow(left), n_check, length(vars), length(seeds), tpu,
              if (n_todo > 0L && any(is.finite(runtime))) {
                sprintf(" -- the run's seeds took %s each, so about %s with %d worker(s)",
                        .train_duration(mean(runtime, na.rm = TRUE)),
                        .train_duration(mean(runtime, na.rm = TRUE) * n_todo / n_work), n_work)
              } else ""))
  say("  refit run: ", run_dir)
  train <- function(us) {
    todo <- us[!done(us), , drop = FALSE]
    if (nrow(todo) > 0L) {
      .final_train_units(todo = todo, selected = cfg, data = data, index = spec$split,
                         scaling = spec$scaling, windows = windows, training = spec$training,
                         transform = transform, run_dir = run_dir, n_cores = n_cores,
                         threads_per_unit = tpu, max_ram_gb = method$max_ram_gb, say = say,
                         fill = fill)
    }
    .importance_refit_all_done(run_dir, us)
  }

  # 3. THE CHECK THAT STOPS, before any variable is left out.
  train(check)
  chk <- .importance_refit_check(fr, run_dir, check$seed)
  say(sprintf("  checked: %d seed(s) trained again with nothing left out give the run's own predictions (largest difference %.2g)",
              nrow(chk), max(chk$max_difference)))

  # 4. THE VARIABLES LEFT OUT, and what each costs against the seed's own.
  train(left)
  baseline <- dplyr::bind_rows(lapply(seeds, function(s) {
    dplyr::mutate(.importance_refit_scores(
      file.path(fr$run_dir, fr$config_id, "predictions", sprintf("seed%04d_pred_all.csv", s)),
      meta_t, transform, clamp), unit = sprintf("seed%04d", s), .before = 1)
  }))
  raw <- dplyr::bind_rows(lapply(seq_len(nrow(left)), function(i) {
    rec <- .final_unit_record(run_dir, left$dir[i], left$seed[i])
    dplyr::mutate(.importance_refit_scores(
      file.path(run_dir, left$dir[i], "predictions", sprintf("seed%04d_pred_all.csv", left$seed[i])),
      meta_t, transform, clamp),
      unit = sprintf("seed%04d", left$seed[i]), variable = vtab$variable[match(left$dir[i], vtab$dir)],
      dir = left$dir[i], best_epoch = as.integer(rec$best_epoch %||% NA_integer_),
      runtime_min = as.numeric(rec$runtime_min %||% NA_real_), .before = 1)
  }))
  agg <- .importance_aggregate(raw, baseline, method$metric)
  tab <- agg$table
  tab$target <- tab$variable
  tab$n_channels <- vtab$n_channels[match(tab$variable, vtab$variable)]
  tab <- dplyr::relocate(tab, "rank", "target", "variable", "n_channels")
  sd_seeds <- if (nrow(baseline) > 1L) stats::sd(baseline[[method$metric]]) else NA_real_

  out <- structure(list(
    table = tab, by_model = agg$by_model, baseline = baseline, raw = raw, groups = grp,
    variables = vtab, check = chk,
    noise = list(sd_seeds = sd_seeds, per_seed = sqrt(2) * sd_seeds,
                 mean = sqrt(2 / length(seeds)) * sd_seeds),
    method = method, rows = "test",
    units = units[, c("unit", "kind", "fold", "seed", "role", "n_rows", "model_file")],
    run_dir = fr$run_dir, refit_dir = run_dir, config_id = fr$config_id,
    window_sizes = as.integer(cfg$window_sizes[[1]]),
    transform_name = data$transform$name %||% "none", rows_alone_share = 0, label = NULL),
    class = "dsm_importance")
  out$label <- .importance_object_label(out)
  out
}

# The body of a refit's print, after the frame's lines.
.importance_print_refit <- function(x, n) {
  chk <- x$check
  cat(sprintf("  checked: %s, trained again with nothing left out, the run's own predictions (largest difference %.2g)\n",
              if (nrow(chk) == 1L) sprintf("seed %d gives", chk$seed) else
                sprintf("seeds %s give", paste(chk$seed, collapse = ", ")),
              max(chk$max_difference)))
  nz <- x$noise
  if (is.finite(nz$sd_seeds)) {
    cat(sprintf("  noise: a variable the network does not need moves a seed's score by about +/- %.2g\n",
                nz$per_seed))
    cat(sprintf("  (sqrt 2 x the sd between the run's seeds), the mean of %d seed(s) by +/- %.2g\n",
                nrow(x$units), nz$mean))
  } else {
    cat("  noise: not estimable from one seed -- give dsm_importance(seeds =) two or more.\n")
  }
  cat("  ", .importance_groups_line(x$groups), "\n", sep = "")
  cat(strrep("-", 72), "\n")
  show <- utils::head(x$table, n)
  shown <- tibble::tibble(rank = show$rank, variable = show$variable, channels = show$n_channels,
                          importance = signif(show$importance, 4),
                          sd_models = signif(show$sd_models, 3),
                          in_models = sprintf("%d/%d", show$n_models_positive, show$n_models))
  if (is.finite(nz$mean)) shown$beyond_noise <- ifelse(show$importance > nz$mean, "yes", "")
  print(shown, n = Inf)
  if (nrow(x$table) > n) cat(sprintf("  ... %d more in $table\n", nrow(x$table) - n))
  # WHAT THE NUMBER IS, the converse of the permutation's warning.
  cat("  Necessity of each variable to this training, not reliance: a variable whose\n")
  cat("  information another also carries costs little, the refit finding it there.\n")
  cat("  refit run: ", x$refit_dir, "\n", sep = "")
  invisible(NULL)
}
