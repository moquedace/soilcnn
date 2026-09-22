# ══════════════════════════════════════════════════════════════════════════════
# CAPABILITY SWEEP -- tier A of docs/test_plan.md
#
# WHAT THIS IS FOR.
#
# On 2026-09-17 the chain 01 -> 07 ran end to end. That proved ONE path:
# spatial_cv, one CNN config of three, the rf and mlp baselines, a
# constant-width interval, a single-tile map. It did not prove the framework.
# An inventory of what the code offers against what has ever executed:
#
#   126 public functions | 37 with no unit test | 60 never called by any script
#
# The capabilities below are the ones a reader of the README would expect to
# work and that have never touched real data.
#
# THE RISK IS NOT THAT THEY ERROR OUT. Code that errors out is cheap -- it
# announces itself. The risk is a capability that runs, returns plausible
# numbers, and is wrong. The three costliest defects in this project all did
# exactly that:
#
#   rf_grid() drew the same config four times and reported four results
#   the buffer protected the validation set and left the test set exposed
#   the conformal interval was calibrated on one spatial block and applied
#     to another
#
# None raised an error. So every capability here reports THREE things, and the
# interesting row is `ran = TRUE, asserted = FALSE`.
#
# WHY RANDOM FOREST ALMOST EVERYWHERE. Nearly every capability is a property of
# the PLUMBING -- the fold plan, the table view, the registry, the comparison
# table -- not of the CNN. A forest unit costs seconds where a CNN unit costs
# two minutes and exercises the same wiring. CNN time is spent only where the
# network itself is the subject (A7).
#
# ORDER. The three inference-only capabilities run first: they are minutes
# rather than tens of minutes, and two of them (A7 occlusion, A8 selection
# optimism) produce numbers for the paper rather than only a PASS.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_capability_sweep.R")
# ══════════════════════════════════════════════════════════════════════════════

# ── Packages ──────────────────────────────────────────────────────────────────
#
# dplyr IS A PREREQUISITE OF THE SWEEP, not a convenience of one capability.
# source(R/load_all.R) attaches nothing -- there is no library() call anywhere
# under R/ and no `%>%` <- assignment -- while R/train_table.R:289 uses %>%
# INSIDE the per-unit fitting loop and R/resample.R uses it in
# summarise_resamples(), which is on every dsm_train() return path. Without
# dplyr attached, every tabular capability here fits its first forest and dies.

suppressPackageStartupMessages({
  library(torch); library(coro)
  library(dplyr); library(readr); library(tibble); library(purrr)
  library(randomForest)
  library(caret); library(xgboost); library(plyr)
})

options(width = 200)

# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R and in R/load_all.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/load_all.R.
project_root <- (function() {
  cand <- character(0)
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) cand <- c(cand, dirname(normalizePath(f[1], mustWork = FALSE)))
  for (i in seq_len(sys.nframe())) {
    of <- sys.frame(i)$ofile
    if (!is.null(of) && is.character(of)) {
      cand <- c(cand, dirname(normalizePath(of, mustWork = FALSE)))
    }
  }
  cand <- c(cand, getwd())
  for (d in cand) for (up in c(".", "..", "../..", "../../..")) {
    r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
    if (file.exists(file.path(r, "R", "load_all.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
setwd(project_root)
source(file.path(project_root, "R", "load_all.R"))

target_label <- "soc_stock_0_5cm"
data_dir     <- file.path(project_root, "data", "processed",
                          "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)
patch_dir    <- file.path(project_root, "outputs", "patches",
                          "soc_stock_modeling", target_label)
tuning_base  <- file.path(project_root, "outputs", "tuning",
                          "soc_stock_modeling", target_label)

# The stage 03 run every capability measures itself against. PINNED BY NAME, not
# resolved as "latest": every capability below is only readable beside the
# others, and beside the 03 numbers, if all of them are defined against ONE run.
# "latest" would silently re-point the whole sweep at the next 03 run somebody
# starts, and every capability would still report PASS -- a PASS carries no
# record of which run produced it. Point this line at a newer run on purpose.
#
# NOT because "latest" would land on a baselines run: those keep their
# fold_plan.rds one level down, one per family, so 03b_run_baselines.R:92, which
# scans for a TOP-LEVEL fold_plan.rds, never sees them, and sort(decreasing =
# TRUE) puts "soc_" ahead of "baselines_" anyway.
cnn_run_dir <- file.path(tuning_base, "soc_0_5cm_20260916_232318")

# ── The harness ───────────────────────────────────────────────────────────────
#
# WHY A HARNESS AND NOT A SCRIPT OF CALLS. A sweep that prints as it goes
# produces a scroll, and a scroll is read by looking for the word "Error". That
# finds what CRASHED and misses what ran and was wrong -- the entire class of
# defect this sweep exists to catch.

.cap_results <- list()

#' Run one capability, record what happened, never stop the sweep.
#'
#' @param id    Short id, e.g. "A1".
#' @param title One line.
#' @param expr  The capability. Its value is passed to `check`.
#' @param check function(value) -> list(ok = logical(1), measured = character(1)).
#'   It must test a property that is FALSE when the capability is silently
#'   wrong. "It returned something" is not such a property.
#' @param cost  Expected cost, printed so a surprise is visible.
capability <- function(id, title, expr, check, cost = "") {
  message("\n", strrep("-", 78))
  message("[", id, "] ", title, if (nzchar(cost)) paste0("   (", cost, ")") else "")
  message(strrep("-", 78))
  t0  <- Sys.time()
  val <- tryCatch(force(expr), error = function(e) e)
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  if (inherits(val, "error")) {
    row <- tibble::tibble(id = id, title = title, ran = FALSE, asserted = NA,
                          measured = conditionMessage(val),
                          minutes = round(mins, 2))
  } else {
    # THE CHECK IS ALSO ALLOWED TO FAIL. A check that errors was written against
    # a shape the capability does not return -- itself a finding, and a common
    # one here. It must not be mistaken for a pass.
    ck <- tryCatch(check(val), error = function(e)
      list(ok = NA, measured = paste("check errored:", conditionMessage(e))))
    row <- tibble::tibble(id = id, title = title, ran = TRUE,
                          asserted = isTRUE(ck$ok),
                          measured = as.character(ck$measured)[1],
                          minutes = round(mins, 2))
  }

  .cap_results[[id]] <<- row
  flag <- if (!row$ran) "CRASH " else if (isTRUE(row$asserted)) "PASS  " else
          if (is.na(row$asserted)) "CHECK?" else "WRONG "
  message(sprintf("\n  >> %s %s  %s", flag, id, substr(row$measured, 1, 90)))
  invisible(val)
}

capability_report <- function(path = NULL) {
  out <- dplyr::bind_rows(.cap_results)
  if (nrow(out) == 0L) { message("No capability ran."); return(invisible(out)) }
  message("\n", strrep("=", 78))
  message("CAPABILITY SWEEP -- tier A")
  message(strrep("=", 78))
  print_wide(dplyr::select(out, id, ran, asserted, minutes, measured), n = Inf)

  n_pass  <- sum(out$ran & out$asserted %in% TRUE)
  n_crash <- sum(!out$ran)
  n_wrong <- sum(out$ran & out$asserted %in% FALSE)
  n_unk   <- sum(out$ran & is.na(out$asserted))
  message(sprintf(
    "\n  %d capability(ies): %d passed | %d crashed | %d RAN AND FAILED ITS CHECK | %d check unusable",
    nrow(out), n_pass, n_crash, n_wrong, n_unk))
  if (n_wrong > 0L) {
    message("\n  The rows that matter are the ones that RAN AND FAILED. A capability")
    message("  that crashes announces itself; one that runs and returns a plausible")
    message("  wrong answer is what this sweep exists to find.")
  }
  if (!is.null(path)) { safe_write_csv2(out, path); message("\n  ", path) }
  invisible(out)
}

# ── One dsm_data, one device, one frozen test set ─────────────────────────────
#
# ONE STORE SERVES EVERYTHING, and the reason is that every runner scopes its
# own fold cache from its own argument rather than from the store:
#
#   tabular runner   windows_needed <- if (is.null(windows)) store$window_sizes
#                                      else windows          R/train_table.R:161
#   occlusion        ws_needed <- unlist(cfg$window_sizes)   R/occlusion.R:324
#   score_test_grid  store$window_sizes                      R/test_optimism.R:148
#
# So A7 does NOT need a window-15-only store -- occlusion_report() reads the
# window off the CONFIG and slices w15 out of a 3/9/15 store. score_test_grid()
# is the only capability that reads store$window_sizes, and the 03 grid spans
# {3, 9, 15}, so that fixes the answer from both sides: not fewer (the loader
# aborts mid-run at R/train_cnn.R:1064 on a missing window) and not more.
#
# Cost: 0.85 GB of tensors, ~2.3 GB peak while a fold cache is built. On 68 GB
# that is not a consideration; comparability is.

windows_needed <- sort(unique(unlist(
  readRDS(file.path(cnn_run_dir, "tune_grid.rds"))$window_sizes)))
stopifnot(identical(as.integer(windows_needed), c(3L, 9L, 15L)))

data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = windows_needed,
  target_col   = readr::read_csv2(file.path(metadata_dir, "target_config.csv"),
                                  show_col_types = FALSE)$target_col[1]
)

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# The frozen test set, read once. A1, A3 and B1 all pass it; A2 deliberately
# does NOT, and draws its own ceiling(0.15 * 3728) = 560 rows. That 560 vs 591
# is the only thing distinguishing the two plans, so it is asserted, not assumed.
ds <- readr::read_csv2(file.path(metadata_dir, "data_split.csv"),
                       show_col_types = FALSE)
frozen_test <- ds$sample_id[ds$role == "test"]
stopifnot(!anyNA(frozen_test), length(frozen_test) == 591L)

# The 03 plan, for the capabilities that must run on the SAME folds.
plan_cnn <- readRDS(file.path(cnn_run_dir, "fold_plan.rds"))

# EVERY RUN ID IS A FIXED STRING UNDER ONE PARENT.
#
# Fixed, because a timestamped run_id defeats resume: .resumable_units() needs
# comparison_all.rds in the SAME run_dir (R/train_table.R:151), so a session
# limit mid-capability would cost the whole capability rather than the unit in
# flight -- which is exactly how the first attempt at this sweep's contract
# analysis died.
#
# Under one parent, because 03b resolves "latest" over directories holding a
# top-level fold_plan.rds and 04 over names matching ^soc_. A nested
# capability_sweep/<id> is invisible to both by construction rather than by the
# lexical luck of "c" sorting before "s".
sweep_dir <- function(id) file.path("capability_sweep", id)

# ══════════════════════════════════════════════════════════════════════════════
# A7 -- does the trained network use the neighbourhood at all?
#
# Inference only, nothing retrained. This is the question 03b answers from the
# outside (CNN against a forest fed the same neighbourhood as window means);
# occlusion answers it from inside the trained model. If the two disagree, one
# of the measurements is wrong and that is worth more than either answer.
#
# The hidden region is PERMUTED from another sample, never zeroed: after scaling
# zero is the training mean, and a patch whose rim is the mean everywhere is a
# landscape that does not exist -- the drop would then mix "this region
# mattered" with "this input is impossible", and the second grows with the area
# hidden, which is precisely the comparison being made.
# ══════════════════════════════════════════════════════════════════════════════

capability(
  "A7", "occlusion_report() on the real cfg_003", cost = "~5-10 min, inference",
  expr = {
    occ <- occlusion_report(
      run_dir   = cnn_run_dir,
      data      = data,
      config_id = "cfg_003",
      fold      = 1L,
      seed_i    = 1L,
      role      = "validation",
      transform = expm1,
      device    = device,
      method    = "permute",
      clamp     = c(0, Inf)
    )
    print(occ)
    safe_write_csv2(occ$table, file.path(cnn_run_dir, "occlusion",
                    "occlusion_cfg_003_f1_s1_validation_permute.csv"))
    occ
  },
  check = function(occ) {
    tb  <- occ$table
    ctx <- tb$delta_ccc[tb$scope == "context_all"][1]
    ctr <- tb$delta_ccc[tb$scope == "centre_only_hidden"][1]
    rng <- tb$n_pixels_hidden[grepl("^ring_", tb$scope)]
    # The MASKS are what a silent failure would corrupt: a 15x15 patch has one
    # centre, 224 non-centre pixels, and rings of 8d that must sum to 224.
    ok <- nrow(tb) == 10L &&
      setequal(tb$scope, c("baseline", "context_all", "centre_only_hidden",
                           sprintf("ring_%02d", 1:7))) &&
      all(tb$n == 1037L) &&
      tb$n_pixels_hidden[tb$scope == "baseline"] == 0L &&
      tb$n_pixels_hidden[tb$scope == "context_all"] == 224L &&
      tb$n_pixels_hidden[tb$scope == "centre_only_hidden"] == 1L &&
      identical(as.integer(rng), 8L * 1:7) && sum(rng) == 224L &&
      tb$delta_ccc[tb$scope == "baseline"] == 0 &&
      # the baseline must reproduce what stage 03 measured for this unit
      abs(occ$baseline_ccc - 0.4974) < 0.02 &&
      # and permuting SOMETHING must move SOMETHING: a report where every delta
      # is zero means the mask never reached the tensor
      is.finite(ctx) && is.finite(ctr) && max(abs(tb$delta_ccc)) > 1e-6
    list(ok = ok, measured = sprintf(
      "baseline %.4f | context %+.4f | centre %+.4f | %s carries more",
      occ$baseline_ccc, ctx, ctr,
      if (abs(ctx) < abs(ctr)) "THE CENTRE PIXEL" else "the neighbourhood"))
  })

# ══════════════════════════════════════════════════════════════════════════════
# A8 -- how much would choosing on the test set have overstated the result?
#
# Inference only: the 27 checkpoints are reloaded and re-scored. The selection
# is already frozen (comparison/selection.rds, cfg_003, 2026-09-17 10:31:26), so
# score_test_grid() will run; it refuses when it is not.
#
# READ THIS BEFORE QUOTING THE NUMBER. The test set was ALSO scored during the
# tuning run itself and written to metrics/*_perf.csv for all 27 units, eleven
# hours before the selection was frozen -- evaluate_test = FALSE blanked the
# comparison table and, until 2026-09-17, nothing else. The leak is closed now
# (.drop_test_rows() in R/utils.R) but it was open then. The number below is
# therefore an audit of what the checkpoints say, not evidence that nobody could
# have looked. Report it that way.
# ══════════════════════════════════════════════════════════════════════════════

capability(
  "A8", "score_test_grid() on the real 03 run", cost = "~10-15 min, inference",
  expr = {
    opt <- score_test_grid(
      run_dir        = cnn_run_dir,
      data           = data,
      transform      = expm1,
      device         = device,
      config_ids     = NULL,
      allow_unfrozen = FALSE
    )
    print(opt)
    safe_save_rds(opt, file.path(cnn_run_dir, "comparison", "test_optimism.rds"))
    safe_write_csv2(opt$by_config, file.path(cnn_run_dir, "comparison",
                    "test_optimism_by_config.csv"))
    opt
  },
  check = function(opt) {
    bc <- opt$by_config
    ok <- inherits(opt, "test_optimism") &&
      nrow(bc) == 3L &&
      setequal(bc$config_id, c("cfg_001", "cfg_002", "cfg_003")) &&
      all(bc$n_units == 9L) &&
      nrow(opt$units) == 27L && anyDuplicated(opt$units$unit_id) == 0L &&
      setequal(opt$units$fold, 1:3) && setequal(opt$units$seed_i, 1:3) &&
      all(opt$units$test_n == 591L) &&
      all(is.finite(opt$units$test_ccc)) &&
      identical(opt$chosen, "cfg_003") && !isTRUE(opt$unfrozen) &&
      identical(sort(bc$test_rank), 1:3)
    r <- bc[bc$config_id == opt$chosen, ][1, ]
    list(ok = ok, measured = sprintf(
      "chosen cfg_003 test CCC %.4f (rank %d of 3) | best %.4f | optimism %+.4f",
      r$test_ccc_mean, r$test_rank, bc$test_ccc_mean[1],
      bc$test_ccc_mean[1] - r$test_ccc_mean))
  })

# ══════════════════════════════════════════════════════════════════════════════
# A6 -- does the interval cover what it promises, calibrated across folds?
#
# Seconds: it reads the 03 run's out-of-fold predictions and takes quantiles.
#
# conformal_cv() has only ever run on simulated data. The table it needs has
# columns fold / obs / pred, and the 03 run wrote ONE FILE PER UNIT with the
# fold in the FILENAME and no fold column -- so the parsing below is the part
# most likely to be silently wrong, and it is cross-checked against
# fold_assignment.csv rather than trusted.
# ══════════════════════════════════════════════════════════════════════════════

capability(
  "A6", "conformal_cv() on the real out-of-fold predictions", cost = "seconds",
  expr = {
    pat <- "^cfg_003_f([0-9]+)_s([0-9]+)_pred_all[.]csv$"
    pred_files <- list.files(file.path(cnn_run_dir, "predictions"),
                             pattern = pat, full.names = TRUE)
    stopifnot(length(pred_files) == 9L)

    long <- purrr::map_dfr(pred_files, function(f) {
      m <- regmatches(basename(f), regexec(pat, basename(f)))[[1]]
      suppressMessages(readr::read_csv2(f, show_col_types = FALSE)) %>%
        dplyr::filter(.data$dataset_role == "validation") %>%
        dplyr::transmute(fold      = as.integer(m[2]),
                         seed      = as.integer(m[3]),
                         sample_id = as.integer(.data$sample_id),
                         obs       = as.numeric(.data$obs),
                         pred      = as.numeric(.data$pred))
    })

    pred_obs <- long %>%
      dplyr::group_by(.data$fold, .data$sample_id) %>%
      dplyr::summarise(obs     = dplyr::first(.data$obs),
                       spread  = stats::sd(.data$pred),
                       pred    = stats::median(.data$pred),
                       n_seeds = dplyr::n(), .groups = "drop")

    # THE FOLD IN THE FILENAME MUST BE THE FOLD IN THE PLAN. Nothing else in
    # this block would notice if a file were mis-parsed; the metrics would come
    # back plausible and calibrated on the wrong residuals.
    fold_map <- suppressMessages(readr::read_csv2(
        file.path(cnn_run_dir, "fold_assignment.csv"), show_col_types = FALSE)) %>%
      dplyr::transmute(sample_id = as.integer(.data$sample_id),
                       fold_ref  = as.integer(.data$fold))
    stopifnot(!any(duplicated(fold_map$sample_id)))
    pred_obs <- dplyr::left_join(pred_obs, fold_map, by = "sample_id")
    stopifnot(
      all(pred_obs$n_seeds == 3L),
      !anyNA(pred_obs$fold_ref),
      all(pred_obs$fold == pred_obs$fold_ref),
      !any(duplicated(pred_obs$sample_id)),
      nrow(pred_obs) == 3092L,
      all(is.finite(pred_obs$obs)), all(is.finite(pred_obs$pred)),
      all(is.finite(pred_obs$spread) & pred_obs$spread > 0)
    )

    cp      <- conformal_cv(pred_obs, alpha = 0.1, group = "fold")
    cp_norm <- conformal_cv(pred_obs, alpha = 0.1, difficulty = "spread",
                            group = "fold")
    print(cp)
    message("\n-- normalised by the seed spread --")
    print(cp_norm)
    list(cp = cp, cp_norm = cp_norm, pred_obs = pred_obs)
  },
  check = function(r) {
    cp <- r$cp; cn <- r$cp_norm
    # NOT a golden value: the property. With 3092 points a correct 90% interval
    # lands within a couple of points of nominal, and the per-fold sizes are the
    # thing a mis-parse would break.
    ok <- inherits(cp, "picp_report") && cp$overall$n == 3092L &&
      abs(cp$overall$picp - 0.9) < 0.03 &&
      !is.null(cp$by_group) && nrow(cp$by_group) == 3L &&
      setequal(cp$by_group$group, 1:3) &&
      setequal(cp$by_group$n, c(1037L, 1038L, 1017L)) &&
      cp$overall$mean_width > 0 &&
      inherits(cn, "picp_report") && cn$overall$n == 3092L &&
      abs(cn$overall$picp - 0.9) < 0.04
    list(ok = ok, measured = sprintf(
      "PICP %.4f (width %.1f) | per fold %s | normalised %.4f (width %.1f)",
      cp$overall$picp, cp$overall$mean_width,
      paste(sprintf("%.3f", cp$by_group$picp[order(cp$by_group$group)]),
            collapse = "/"),
      cn$overall$picp, cn$overall$mean_width))
  })

# ══════════════════════════════════════════════════════════════════════════════
# A1 -- random_cv() on real points
#
# The property under test is GROUPING. random_cv() carries group = "auto", and
# in 3-D soil mapping one profile yields several rows at identical coordinates.
# Split those across roles and the model is scored on a depth of a profile it
# already learned -- silently, with a plausible CCC.
#
# The frozen test set is passed so this plan holds out exactly the 591 points
# the spatial plan held out; otherwise the two val_ccc values answer different
# questions. .carve_test() gives test_ids precedence over test_frac.
# ══════════════════════════════════════════════════════════════════════════════

capability(
  "A1", "random_cv() with profile grouping", cost = "~15 min, 36 forests",
  expr = dsm_train(
    data          = data,
    model         = "rf",
    resampling    = random_cv(k = 3L, test_frac = 0.15, group = "auto",
                              seed = 42L),
    features      = "centre",
    windows       = 3L,
    test_ids      = frozen_test,
    transform     = expm1,
    output_dir    = tuning_base,
    run_id        = sweep_dir("A1_random_cv"),
    base_seed     = 42L, n_seeds = 3L, tune_length = 4L,
    evaluate_test = FALSE, resume = TRUE
  ),
  check = function(fit) {
    fl   <- fit$plan$folds
    meta <- fit$data$store$meta
    pid  <- as.character(meta$profile_id)
    cmp  <- fit$comparison
    ok <- identical(fit$plan$method, "random_folds") &&
      grepl("one row per profile", fit$plan$params$grouping, fixed = TRUE) &&
      nrow(cmp) == 36L && all(cmp$status == "success") &&
      all(cmp$n_features == 181L) &&
      identical(sort(unique(cmp$config_id)), sprintf("rf_%03d", 1:4)) &&
      all(is.na(cmp$test_ccc)) &&
      # THE TEST SET'S IDENTITY, not its size. A fresh draw would be
      # ceiling(0.15 * 3728) = 560, and 560 != 591 catches a total failure by
      # arithmetic luck -- that is a coincidence, not a check.
      identical(sort(as.numeric(meta$sample_id[fl[[1]]$test])),
                sort(as.numeric(frozen_test))) &&
      # every non-test row validated exactly once
      identical(sort(unlist(lapply(fl, "[[", "validation"), use.names = FALSE)),
                sort(setdiff(seq_len(nrow(meta)), fl[[1]]$test))) &&
      # and no profile appears in two roles of the same fold
      all(vapply(fl, function(f) {
        roles <- Filter(length, list(f$train, f$validation, f$test))
        !anyDuplicated(unlist(lapply(roles, function(ix) unique(pid[ix]))))
      }, logical(1)))
    list(ok = ok, measured = sprintf(
      "%d units | val_ccc %.4f | test n=%d identical to frozen: %s",
      nrow(cmp), mean(cmp$val_ccc, na.rm = TRUE), length(fl[[1]]$test),
      identical(sort(as.numeric(meta$sample_id[fl[[1]]$test])),
                sort(as.numeric(frozen_test)))))
  })

# ══════════════════════════════════════════════════════════════════════════════
# A2 -- holdout_cv(), the single split
#
# Deliberately NOT given the frozen test set: this draws its own, and the
# difference (560 against 591) is the only thing separating this plan from A1's.
# Asserting it is how a silent fall-through to the frozen ids would be caught.
#
# With one fold and three seeds the noise floor IS estimable -- it is measured
# within (config, fold) across seeds -- and that is asserted, because "not
# estimable" would be the quiet symptom of the seeds not varying.
# ══════════════════════════════════════════════════════════════════════════════

capability(
  "A2", "holdout_cv() drawing its own test set", cost = "~8 min, 12 forests",
  expr = dsm_train(
    data          = data,
    model         = "rf",
    resampling    = holdout_cv(validation_frac = 0.15, test_frac = 0.15,
                               group = "auto", seed = 42L),
    features      = "centre",
    windows       = 3L,
    test_ids      = NULL,
    transform     = expm1,
    output_dir    = tuning_base,
    run_id        = sweep_dir("A2_holdout"),
    base_seed     = 42L, n_seeds = 3L, tune_length = 4L,
    evaluate_test = FALSE, resume = TRUE
  ),
  check = function(fit) {
    meta <- fit$data$store$meta
    n    <- nrow(meta)
    nt   <- ceiling(0.15 * n)          # .draw_groups_for_frac: ceiling over points
    nv   <- floor(0.15 * (n - nt))     # holdout(): floor over groups of the pool
    f    <- fit$plan$folds[[1]]
    cmp  <- fit$comparison
    nf   <- seed_noise_floor(cmp, metric = "val_ccc")

    u1 <- cmp$unit_id[1]
    pa <- suppressMessages(readr::read_csv2(
      file.path(fit$run_dir, "predictions", paste0(u1, "_pred_all.csv")),
      show_col_types = FALSE))
    role_ids <- function(r)  sort(as.numeric(pa$sample_id[pa$dataset_role == r]))
    plan_ids <- function(ix) sort(as.numeric(meta$sample_id[ix]))

    ok <- identical(fit$plan$method, "holdout") && fit$plan$n_folds == 1L &&
      fit$plan$params$n_test == nt &&
      length(f$test) == nt && length(f$validation) == nv &&
      length(f$train) == n - nt - nv &&
      setequal(c(f$train, f$validation, f$test), seq_len(n)) &&
      # THE PLAN IS WHAT WAS ACTUALLY FITTED, by membership rather than by count
      identical(role_ids("train"), plan_ids(f$train)) &&
      identical(role_ids("validation"), plan_ids(f$validation)) &&
      # NO TEST ROW REACHES DISK while evaluate_test is FALSE. Before
      # 2026-09-17 both predictions/ and metrics/ carried them; the rule now
      # lives in .drop_test_rows() and this is where it is checked on real data.
      !("test" %in% pa$dataset_role) &&
      # this test set is NOT the frozen one
      !setequal(plan_ids(f$test), sort(as.numeric(frozen_test))) &&
      nrow(cmp) == 12L && all(cmp$status == "success") &&
      all(cmp$val_n == nv) && all(is.na(cmp$test_ccc)) &&
      dplyr::n_distinct(cmp$fold) == 1L && setequal(cmp$seed, 42:44) &&
      # one fold, three seeds: the floor is estimable, and non-zero
      nf$n_comparable == 4L && is.finite(nf$median_sd) && nf$median_sd > 0
    list(ok = ok, measured = sprintf(
      "train/val/test %d/%d/%d (frozen would be 591) | noise floor %.4f over %d",
      length(f$train), length(f$validation), length(f$test),
      nf$median_sd, nf$n_comparable))
  })

# ══════════════════════════════════════════════════════════════════════════════
# A3 -- region_cv() on a group derived from the FAO soil-class dummies
#
# There is no region column in the point table, so the group is derived. THAT
# DERIVATION IS THE RISK, not region_folds(): a group vector in CSV order
# against a store in store order would produce a plan that looks perfect and
# groups the wrong points.
#
# So the label is derived TWICE, from independent sources -- from the aligned
# point table, and from the soil-class channels at the centre pixel of the patch
# tensors -- and the two must agree. A permuted point table cannot survive that.
# ══════════════════════════════════════════════════════════════════════════════

capability(
  "A3", "region_cv() on derived FAO soil classes", cost = "~6 min, 10 forests",
  expr = {
    stopifnot(nrow(data$points) == nrow(data$store$meta),
              isTRUE(all.equal(as.numeric(data$points$sample_id),
                               as.numeric(data$store$meta$sample_id))))

    fao_cols <- grep("^soil_class_fao_", names(data$points), value = TRUE)
    fao_mat  <- as.matrix(data$points[, fao_cols])
    storage.mode(fao_mat) <- "double"
    # max.col() on an all-zero row still returns a column, so an unclassified
    # point would silently join class 1. One-hot is proven, never assumed.
    stopifnot(length(fao_cols) > 1L, !anyNA(fao_mat),
              all(abs(rowSums(fao_mat) - 1) < 1e-8))
    soil_group <- as.character(sub("^soil_class_fao_", "",
                    fao_cols[max.col(fao_mat, ties.method = "first")]))

    # THE SECOND, INDEPENDENT DERIVATION. Patch tensor row i IS store$meta row i
    # by construction, so reading the same channels at the centre pixel never
    # touches the point table. Indexing is fold_table_view()'s own idiom.
    w_min <- min(data$store$window_sizes)
    c_pix <- (as.integer(w_min) %/% 2L) + 1L
    m_ctr <- as.matrix(data$store$windows[[patch_window_key(w_min)]][
      , , c_pix, c_pix]$to(device = "cpu"))
    colnames(m_ctr) <- data$store$predictors
    store_group <- as.character(sub("^soil_class_fao_", "",
                     fao_cols[max.col(m_ctr[, fao_cols, drop = FALSE],
                                      ties.method = "first")]))
    stopifnot(identical(store_group, soil_group))
    message("Derived group: ", dplyr::n_distinct(soil_group), " FAO classes")
    print(sort(table(soil_group), decreasing = TRUE))

    # k = 5, NOT NULL. k = NULL is leave-one-class-out over ~30 classes, and the
    # smallest classes would validate on one point, where ccc() returns NA.
    plan_region <- resolve_resampling(
      region_cv(group = soil_group, k = 5L, test_frac = 0, seed = 42L),
      data, test_ids = frozen_test)
    print(plan_region)

    # region_folds() applies NO buffer -- it has no buffer argument. This says
    # how much that costs, in stage 03's own metric, instead of leaving it
    # unstated.
    leak <- fold_leakage_report(plan_region, data$store$meta,
                                cell_size = data$cell_size, windows = w_min)
    print(leak[leak$matters, ], n = Inf)

    fit <- dsm_train(
      data = data, model = "rf", resampling = plan_region,
      features = "centre", windows = w_min, transform = expm1,
      output_dir = tuning_base, run_id = sweep_dir("A3_region_cv"),
      base_seed = 42L, n_seeds = 1L, tune_length = 2L,
      evaluate_test = FALSE, resume = TRUE)
    list(fit = fit, group = soil_group, leak = leak)
  },
  check = function(r) {
    fit <- r$fit; g <- r$group; fl <- fit$plan$folds
    meta <- fit$data$store$meta; cmp <- fit$comparison
    ok <- identical(fit$plan$method, "region_folds") &&
      fit$plan$n_folds == 5L &&
      # the frozen test survived: without it, region_folds would draw WHOLE
      # CLASSES into the test set and still look like a valid region plan
      fit$plan$params$n_test == 591L &&
      all(vapply(fl, function(f)
        identical(sort(as.numeric(meta$sample_id[f$test])),
                  sort(as.numeric(frozen_test))), logical(1))) &&
      # NO CLASS STRADDLES A FOLD -- the property region_cv exists for
      all(vapply(fl, function(f)
        length(intersect(g[f$train], g[f$validation])) == 0L, logical(1))) &&
      # every non-test row validated exactly once
      identical(sort(unlist(lapply(fl, "[[", "validation"), use.names = FALSE)),
                sort(setdiff(seq_len(nrow(meta)), fl[[1]]$test))) &&
      nrow(cmp) == 10L && all(cmp$status == "success") &&
      all(is.finite(cmp$val_ccc)) && all(cmp$n_features == 181L) &&
      all(is.na(cmp$test_ccc))
    list(ok = ok, measured = sprintf(
      "%d classes | %d folds | no class split: %s | val_ccc %.4f",
      dplyr::n_distinct(g), fit$plan$n_folds,
      all(vapply(fl, function(f)
        length(intersect(g[f$train], g[f$validation])) == 0L, logical(1))),
      mean(cmp$val_ccc, na.rm = TRUE)))
  })

# ══════════════════════════════════════════════════════════════════════════════
# A4 -- features = "window_mean" alone
#
# Every run so far used "centre" or c("centre", "window_mean"). The path where
# only the window means are requested has never executed.
#
# THE COLUMN NAMES CANNOT CHECK THEMSELVES. They are pasted from the window key,
# never read from the data, so they are right whatever the tensor held. The
# means are therefore recomputed in base R on 20 rows of every window and
# compared -- on the fold cache, while it is still alive, before 36 forests.
# ══════════════════════════════════════════════════════════════════════════════

capability(
  "A4", "features = \"window_mean\" alone", cost = "~35 min, 36 forests",
  expr = {
    fold1 <- build_fold_cache(data$store, data$points, data$type_table,
                              plan_cnn$folds[[1]], windows_needed)
    tv <- fold_table_view(fold1$cache, data$store$predictors,
                          windows = windows_needed,
                          features = "window_mean")$train$x

    mean_ok <- vapply(patch_window_key(windows_needed), function(k) {
      sub <- 1:20
      arr <- as.array(fold1$cache$train[[k]][sub, , , , drop = FALSE]$to(
        device = "cpu"))
      man <- apply(arr, c(1L, 2L), mean)
      blk <- tv[sub, grep(paste0("_mean_", k, "$"), colnames(tv)), drop = FALSE]
      isTRUE(all.equal(unname(man), unname(blk), tolerance = 1e-4))
    }, logical(1))
    print(mean_ok)
    rm(fold1); invisible(gc(verbose = FALSE))

    message("A4 table: ", nrow(tv), " x ", ncol(tv))
    # STOP BEFORE THE 35 MINUTES if the table is wrong.
    stopifnot(ncol(tv) == 543L, all(mean_ok),
              !any(data$store$predictors %in% colnames(tv)))

    fit <- dsm_train(
      data = data, model = "rf", resampling = plan_cnn,
      features = "window_mean", windows = windows_needed, transform = expm1,
      output_dir = tuning_base, run_id = sweep_dir("A4_window_mean"),
      base_seed = 42L, n_seeds = 3L, tune_length = 4L,
      evaluate_test = FALSE, resume = TRUE)
    list(fit = fit, cols = colnames(tv), mean_ok = mean_ok)
  },
  check = function(r) {
    p   <- data$store$predictors
    cmp <- r$fit$comparison
    expect <- unlist(lapply(patch_window_key(windows_needed),
                            function(k) paste0(p, "_mean_", k)))
    ok <- identical(r$cols, expect) &&
      length(r$cols) == 3L * length(p) &&
      !any(p %in% r$cols) &&
      all(r$mean_ok) &&
      nrow(cmp) == 36L && all(cmp$status == "success") &&
      all(cmp$n_features == 543L) &&
      setequal(cmp$fold, 1:3) && setequal(cmp$seed, 42:44) &&
      all(is.finite(cmp$val_ccc)) &&
      # rf_centre and rf_context span 0.436-0.537 on these folds; 0.20 separates
      # "learned something" from "the table was garbage but well shaped"
      min(cmp$val_ccc) > 0.20
    list(ok = ok, measured = sprintf(
      "%d columns, means verified in base R: %s | val_ccc %.4f",
      length(r$cols), all(r$mean_ok), mean(cmp$val_ccc, na.rm = TRUE)))
  })

# ══════════════════════════════════════════════════════════════════════════════
# A5 -- caret_spec(): borrowing a model library without borrowing a resampler
#
# xgboost and caret are installed; ranger and glmnet are not.
#
# THE FAILURE MODE THAT MATTERS is caret's own resampling leaking in. caret is
# borrowed for its model LIBRARY -- parameter names, sensible grids, fit and
# predict closures -- never for its cross-validation, because the fold plan here
# carries blocks and a buffer that caret knows nothing about. Two objects must
# never both believe they own the split.
#
# The fitted objects are destroyed after each unit, so the adapter is probed
# directly on a tiny synthetic fit: trainControl(method = "none"), no resample
# table, no retained training data.
# ══════════════════════════════════════════════════════════════════════════════

# THE METHOD IS caret's "rf", NOT xgbTree, AND THE REASON IS NOT TIMIDITY.
#
# xgbTree was tried first and died instantly:
#
#   ALTLIST classes must provide a Set_elt method [class: XGBAltrepPointerClass]
#
# That is xgboost's own R binding failing against this R build -- it fires
# inside xgboost::xgb.train before any of this framework's code is reached, and
# reproducing it needs neither caret nor the adapter. Testing the adapter
# through a broken backend measures the backend.
#
# caret's "rf" is the better probe anyway, and not a weaker one: it wraps
# randomForest, which is also what our NATIVE rf spec wraps. So the adapter's
# answer can be compared against a number this project already measured on the
# same folds -- 03b's rf_context at 0.487 CCC. An adapter that quietly
# resamples, or that drops the tuning grid, cannot land on that number by
# accident. A method with no native counterpart could only be checked for "it
# ran".
capability(
  "A5", "caret_spec(\"rf\") through dsm_train()", cost = "~25-35 min",
  expr = {
    register_model(caret_spec("rf", name = "caret_rf"), overwrite = TRUE)

    probe <- get_model("caret_rf")$fit(
      x = matrix(stats::rnorm(600L), 200L, 3L,
                 dimnames = list(NULL, c("a", "b", "c"))),
      y = stats::rnorm(200L),
      cfg = tibble::tibble(config_id = "probe", mtry = 2L))

    fit <- dsm_train(
      data = data, model = "caret_rf", resampling = plan_cnn,
      tune_grid = NULL, features = c("centre", "window_mean"),
      windows = windows_needed, transform = expm1,
      output_dir = tuning_base, run_id = sweep_dir("A5_caret_rf"),
      base_seed = 42L, n_seeds = 1L, tune_length = 3L,
      evaluate_test = FALSE, resume = TRUE)
    list(fit = fit, probe = probe)
  },
  check = function(r) {
    ctl <- r$probe$fit$control
    cmp <- r$fit$comparison
    g   <- readRDS(file.path(r$fit$run_dir, "tune_grid.rds"))
    # THE ADAPTER: caret must be a fit/predict shim and nothing else.
    adapter_ok <- identical(ctl$method, "none") &&
      identical(ctl$returnData, FALSE) &&
      is.null(r$probe$fit[["resample"]]) &&
      is.null(r$probe$fit[["trainingData"]]) &&
      NROW(r$probe$fit$results) == 0L
    # THE FOLDS ARE THE CNN'S, by row index rather than by fold count.
    folds_ok <- identical(
      lapply(r$fit$plan$folds, function(f) sort(f$validation)),
      lapply(plan_cnn$folds,   function(f) sort(f$validation)))
    ok <- adapter_ok && folds_ok &&
      nrow(g) == 3L && !anyDuplicated(g$config_id) &&
      nrow(cmp) == 3L * 3L * 1L && all(cmp$status == "success") &&
      all(cmp$n_features == 724L) &&
      all(is.finite(cmp$val_ccc)) &&
      all(is.na(cmp$test_ccc)) &&
      # THE NUMBER, not merely a number. caret's rf wraps randomForest, which
      # is what the native rf spec wraps, on the same folds and the same 724
      # columns -- so 03b's rf_context (0.487) is the target. An adapter that
      # quietly resamples, or that ignores the grid, does not land here.
      #
      # THE BEST CONFIG'S MEAN, not the best UNIT. The first version compared
      # max(cmp$val_ccc) -- a per-unit maximum over 9 units -- against 0.487,
      # which is a per-config mean over 9. Those are different quantities and
      # the looser one is systematically higher, so the tolerance was doing the
      # work the comparison should have done. It passed at 0.523 against 0.487;
      # the right comparison is 0.483 against 0.487.
      abs(r$fit$by_config$val_ccc_mean[1] - 0.487) < 0.02
    list(ok = ok, measured = sprintf(
      "method='%s' no-resample=%s | folds match CNN: %s | best config mean %.4f vs native rf_context 0.487",
      ctl$method, is.null(r$probe$fit[["resample"]]), folds_ok,
      r$fit$by_config$val_ccc_mean[1]))
  })

# ══════════════════════════════════════════════════════════════════════════════

capability_report(file.path(tuning_base, "capability_sweep",
                            "capability_sweep_tierA.csv"))
