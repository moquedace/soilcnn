# ══════════════════════════════════════════════════════════════════════════════
# B6 -- resume after an interruption, and the refusal that protects it
#
# WHAT IS UNDER TEST.
#
# 03 can be killed half way and started again, and it is supposed to pick up
# where it stopped instead of retraining everything. Three pieces of code make
# that claim, and each of them has a comment in R/utils.R describing the run
# where it came apart:
#
#   run_cnn_tuning()        a unit is done when its comparison row says success
#                           AND models/{unit}_best.pt exists
#   .resumable_units()      ...AND the cached unit still describes the
#                           hyperparameters the grid now asks for under that
#                           name. A config_id is a label, not an identity.
#   check_plan_unchanged()  ...AND the fold plan on disk is the plan being asked
#                           for. Cached units fitted on a training set that no
#                           longer exists are not comparable with new ones.
#
# WHY THIS IS A SCRIPT AND NOT A UNIT TEST.
#
# A genuine interruption cannot be scripted from inside the thing being
# interrupted: the process that would trigger it is the process that dies. A
# unit test can only SIMULATE the half-finished state -- delete some rows, drop
# some checkpoints -- and a simulation tests the assumption the test author had
# about what an interrupt leaves behind, which is precisely the thing worth
# measuring. So the interrupt here is real, performed by hand, and the script's
# whole job is to make the human half foolproof.
#
# Hence TWO PHASES, each one source() call:
#
#   phase 1  prepare  builds the plan and the grid, runs an UNINTERRUPTED
#                     control run to have something to compare against and to
#                     measure what a unit costs, records a fingerprint of all
#                     of it, and then starts a second, identical run for the
#                     user to kill by hand.
#   phase 2  verify   after the restart: confirms the finished units were
#                     reused and not retrained, that the unfinished ones ran,
#                     that check_plan_unchanged() stayed quiet, and that the
#                     final table is complete and stands up against the control
#                     run -- identical where the runner COPIED a value, and no
#                     further from it than one seed is from another where the
#                     runner TRAINED one. Then it proves the guards REFUSE a
#                     changed plan.
#
# THE NEGATIVE CASE IS HALF THE POINT. A resume check that only shows resume
# working is the check this project's own suite already had: "resume skips
# finished units" passed for weeks while resume could happily continue onto a
# DIFFERENT split. What protects the user is the refusal, so the refusal is
# tested as hard as the success -- at the function level and again through
# dsm_train(), because a correct guard that nothing calls is not a guard.
#
# WHAT IT COSTS
#
# Two passes over the same small grid: 3 configs x 2 folds x 2 seeds = 12 units,
# window 3 only, 30 epochs, no early stopping. One pass is the control, the
# other is the interrupted run finished off. Phase 1 measures the per-unit cost
# from the control and prints it, so the second number is never a guess. The
# control survives a retry: if the interrupt lands too early or too late, only
# the run to be interrupted is deleted and redone.
#
# HOW TO RUN IT
#
#   Sys.setenv(soc_b6_phase = "prepare")
#   source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_b6_resume_check.R")
#     ... phase 1 tells you, in a banner, exactly when to press Esc ...
#   Sys.setenv(soc_b6_phase = "verify")
#   source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_b6_resume_check.R")
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Packages ──────────────────────────────────────────────────────────────────
#
# dplyr is a prerequisite of the runner, not a convenience of this script:
# R/load_all.R attaches nothing, and R/resample.R uses %>% inside
# summarise_resamples(), which is on every dsm_train() return path.
suppressPackageStartupMessages({
  library(torch); library(coro)
  library(dplyr); library(readr); library(tibble); library(purrr)
})

# WIPING THE WORKSPACE IS NOT COSMETIC HERE, IT IS PART OF THE TEST.
#
# The natural way to run phase 2 is in the same console that just had phase 1
# interrupted -- where every object phase 1 built is still lying around. If this
# script ever read one of them, phase 2 would pass in that console and fail in a
# fresh R session, which is the session a real crash leaves you with. Clearing
# the workspace forces the two phases to communicate ONLY through the state file
# on disk, which is the channel a crash cannot take away.
#
# Sys.setenv() survives this; a workspace object does not. That is exactly why
# the phase is carried by an environment variable.
rm(list = ls())
gc()

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

# The path this script prints in its own instructions, so the two phases
# can never be told to source a different file from the one running.
b6_script <- file.path(project_root, "examples", "soc_stock_0_5cm",
                       "_b6_resume_check.R")

# ── Which phase ───────────────────────────────────────────────────────────────
#
# EXPLICIT, NOT INFERRED. The obvious alternative is to infer the phase from
# whether the state file exists -- and it is wrong. After phase 1 the state file
# always exists, so the same source() line would silently become "verify",
# including when the user interrupted too early and meant to start over. He
# would then be verifying a run with nothing in it, and the message would be
# about the run rather than about what he actually did.
#
# An explicit phase costs one line of typing and makes every wrong order
# produce a message that names the mistake.
b6_phase <- tolower(trimws(Sys.getenv("soc_b6_phase", "")))

# "worker" is not for a human to type. Phase 1 launches this same file in a
# subprocess under that phase, and kills it partway: that IS the interruption.
# It is a phase rather than a separate script so that the interrupted run is
# built by exactly the code that builds the control run -- a second file would
# be a second chance for them to drift apart.
if (!b6_phase %in% c("prepare", "verify", "worker")) {
  stop(
    "B6 is a TWO-PHASE check and the phase has to be chosen explicitly.\n",
    "soc_b6_phase is ", if (nzchar(b6_phase)) paste0("'", b6_phase, "'")
                        else "not set", ".\n\n",
    "  PHASE 1 -- prepare (run this first):\n",
    "    Sys.setenv(soc_b6_phase = \"prepare\")\n",
    "    source(\"", b6_script, "\")\n\n",
    "  PHASE 2 -- verify (only after you have interrupted the run that\n",
    "                     phase 1 starts, and it will refuse if you have not):\n",
    "    Sys.setenv(soc_b6_phase = \"verify\")\n",
    "    source(\"", b6_script, "\")\n\n",
    "A genuine interruption cannot be scripted from INSIDE the thing being\n",
    "interrupted, which is why this is not one command. Phase 1 does script it\n",
    "from outside, by killing a subprocess -- but the two phases stay separate,\n",
    "because phase 2 has to reconstruct the experiment in a fresh session.",
    call. = FALSE)
}

setwd(project_root)
source(file.path(project_root, "R", "load_all.R"))

# ── Paths ─────────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"

patch_dir    <- file.path(project_root, "outputs", "patches",
                          "soc_stock_modeling", target_label)
data_dir     <- file.path(project_root, "data", "processed",
                          "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)
tuning_base  <- file.path(project_root, "outputs", "tuning",
                          "soc_stock_modeling", target_label)

# EVERY RUN ID IS A FIXED STRING UNDER ONE PARENT, exactly as
# _capability_sweep.R does it, and for the same two reasons.
#
# Fixed, because a timestamped run_id defeats resume -- and resume is the
# subject here, so a run_id that changes between the two phases would make the
# check impossible rather than merely inconvenient.
#
# Nested under capability_sweep/, because 03b resolves "latest" over the
# directories holding a top-level fold_plan.rds and 04 over names matching
# ^soc_. A nested directory is invisible to both by construction rather than by
# the lexical luck of which name sorts first.
b6_control_run_id  <- file.path("capability_sweep", "b6_control")
b6_resume_run_id   <- file.path("capability_sweep", "b6_resume")
b6_negative_run_id <- file.path("capability_sweep", "b6_negative_plan")

control_run_dir  <- file.path(tuning_base, b6_control_run_id)
resume_run_dir   <- file.path(tuning_base, b6_resume_run_id)
negative_run_dir <- file.path(tuning_base, b6_negative_run_id)

# ── Where phase 2 learns what phase 1 saw ─────────────────────────────────────
#
# A SMALL RDS INSIDE THE RUN DIRECTORY BEING VERIFIED.
#
# What was rejected, and why:
#
#   an environment variable   dies with the session, and the session dying is
#                             the scenario under test. It carries the phase,
#                             which is one word the user retypes; it cannot
#                             carry a fold plan.
#   a workspace object        every script here opens with rm(list = ls()), and
#                             a genuine crash takes the workspace anyway.
#   one file under outputs/   this is the failure the brief names: a second
#                             experiment would verify against the first one's
#                             fingerprint and pass while measuring nothing.
#   nothing -- derive the     the run directory is the thing under test.
#   expectation from the      Deriving the expectation from it makes the check
#   run directory itself      a tautology: it cannot tell a run that was
#                             interrupted from one that never started, and it
#                             would agree with any plan it happened to find.
#
# Inside the run directory it is per-run by construction, it disappears when the
# run disappears, and it cannot describe some other directory. The remaining
# hazard -- a state file left by an EARLIER experiment under the same fixed
# run_id -- is closed by checking the fingerprints rather than merely reading
# them: phase 2 rebuilds the plan and the grid from this script and refuses if
# they are not the ones phase 1 recorded.
b6_state_version <- 1L
state_path <- file.path(resume_run_dir, "b6_state.rds")

# ── Settings ──────────────────────────────────────────────────────────────────
#
# SMALL ON PURPOSE, AND IN THE TWO DIMENSIONS THAT MATTER.
#
# The subject is the plumbing -- the fold plan, the unit ledger, the comparison
# table -- not the network. So the network is made as cheap as it can be while
# still being the CNN runner that stage 03 uses: window 3 only (the smallest
# patches in the store, and the only one this script loads), the smallest
# channel pair, gap pooling, a single branch.
#
# Why the CNN at all, when the sweep's rule is "test the framework with the
# cheap model": because a forest unit finishes in seconds, and an interruption
# performed by a human needs units long enough to land INSIDE one. The runner
# under test is also the one B6 names.
#
# n_epochs is fixed below the patience so early stopping never fires. Every unit
# then costs the same, which is what lets the control run predict when to press
# Esc -- and it removes one source of difference between the control and the
# resumed run before it can be mistaken for a resume defect.
b6_k_folds     <- 2L
b6_n_seeds     <- 2L
b6_base_seed   <- 42L
b6_windows     <- 3L
b6_tune_length <- 3L
b6_test_frac   <- 0.15

training_args <- list(
  n_epochs            = 30L,
  patience            = 40L,   # > n_epochs: early stopping cannot fire
  es_min_delta        = 0.0005,
  warmup_start_lr     = 1e-5,
  lr_plateau_factor   = 0.5,
  lr_plateau_patience = 25L,
  lr_plateau_min_delta = 0.0005,
  min_lr              = 1e-6,
  gradient_clip       = 1.0,
  print_every         = 10L,
  # `augment` is NOT a tune_grid dimension -- it reaches train_one_cnn()
  # through `...`, and 03_run_tuning.R sets it here, in the config list. TRUE,
  # because the run being imitated is 03 and 03 has it on; it consumes R RNG
  # per batch, which is reseeded per unit and therefore reproducible.
  augment             = TRUE
)

# EVERY NAME IN training_args MUST BE A FORMAL OF train_one_cnn().
#
# The list travels dsm_train(...) -> run_cnn_resample(...) -> run_cnn_tuning(...)
# -> train_one_cnn(), which has no `...` of its own. A misspelt name therefore
# dies with "unused argument" -- but only once the first unit starts training,
# minutes in and after the store is loaded. Checked here, in the first second.
#
# This matters more here than anywhere else in the project, because two of these
# names are load-bearing for the comparison rather than for the model:
# patience (40) is deliberately greater than n_epochs (30) so early stopping
# cannot fire, which is what makes the resumed run and the control run
# comparable unit for unit. A silently dropped `patience` would let stopping
# fire at different epochs in the two runs and the difference would read as a
# resume defect.
#
# _b3_augmentation.R carries the same guard for the same reason.
unknown_args <- setdiff(names(training_args), names(formals(train_one_cnn)))
if (length(unknown_args) > 0L) {
  stop("training_args names that train_one_cnn() does not accept: ",
       paste(unknown_args, collapse = ", "),
       "\n  They would travel through `...` and abort the first unit, ",
       "after the store is loaded.", call. = FALSE)
}
stopifnot(training_args$patience > training_args$n_epochs)

# ── Small helpers ─────────────────────────────────────────────────────────────

# The fold plan reduced to what training actually consumed: MEMBERSHIP, not the
# plan object. This is the same comparison check_plan_unchanged() makes, and it
# is made the same way on purpose -- params differ for irrelevant reasons (a new
# field, a rounded buffer) while the split is identical, and the split is what
# the cached units were fitted to.
#
# Stored whole rather than hashed: 3,728 integers are nothing, digest is not a
# dependency of this project, and an exact object compares exactly.
b6_fold_membership <- function(plan) {
  lapply(plan$folds, function(f) list(
    train      = sort(as.integer(f$train)),
    validation = sort(as.integer(f$validation)),
    test       = sort(as.integer(f$test))
  ))
}

# The grid reduced to its VALUES, used only to detect that the grid moved
# between the two phases. The framework's own staleness rule lives in
# .resumable_units() and is tested below, never reimplemented here.
#
# Values rather than a printed signature, and that is deliberate. The obvious
# version pastes format() over every cell -- and format() reads options(scipen),
# which belongs to the R session, not to the experiment. Phase 1 and phase 2 are
# two sessions. A user who had scipen set in one of them would get 1e-04 against
# 0.0001, a fingerprint mismatch, and a refusal to verify a run that was
# perfectly fine. Stripping the tibble to plain vectors compares types and
# values with nothing in between.
b6_grid_fingerprint <- function(g) {
  lapply(stats::setNames(names(g), names(g)), function(p) {
    v <- g[[p]]
    if (is.list(v)) lapply(v, as.vector) else as.vector(v)
  })
}

# THE UNIT NAMES THE RUN WILL USE, DERIVED FROM THE GRID AND THE PLAN.
#
# Never a literal count: the grid may come back smaller than tune_length when
# the restricted space is exhausted, and a hardcoded 12 would then verify a
# 12-unit expectation against an 8-unit run and blame the resume.
#
# The shape is run_cnn_tuning()'s: "{config_id}_f{fold}_s{seed_index}", where
# the suffix is the REPETITION INDEX (1..n_seeds), not the seed value.
b6_expected_units <- function(grid, n_folds, n_seeds) {
  unlist(lapply(grid$config_id, function(cid) {
    unlist(lapply(seq_len(n_folds), function(j) {
      sprintf("%s_f%d_s%d", cid, j, seq_len(n_seeds))
    }))
  }), use.names = FALSE)
}

b6_read_comparison <- function(run_dir) {
  p <- file.path(run_dir, "comparison", "comparison_all.rds")
  if (!file.exists(p)) return(tibble::tibble())
  readRDS(p)
}

# Every checkpoint in the run directory, with the identity a file name carries.
#
# The pattern is deliberately wider than "_best.pt": safe_torch_save() falls
# back to a TIMESTAMPED name when the target file cannot be removed (a Windows
# lock), and such a file is a unit that looks trained and that nothing will ever
# read. Those come back with an unparsed unit_id and are reported, not ignored.
b6_checkpoints <- function(run_dir) {
  d <- file.path(run_dir, "models")
  f <- if (dir.exists(d)) list.files(d, pattern = "[.]pt$", full.names = TRUE)
       else character(0)
  if (length(f) == 0L) {
    return(tibble::tibble(file = character(0), unit_id = character(0),
                          mtime = numeric(0), size = numeric(0)))
  }
  info <- file.info(f)
  tibble::tibble(
    file    = basename(f),
    unit_id = sub("_best[.]pt$", "", basename(f)),
    mtime   = as.numeric(info$mtime),
    size    = as.numeric(info$size)
  )
}

# What the runner will treat as already done, worked out the way the runner
# works it out: a success row AND a checkpoint on disk.
#
# This is a deliberate COPY of run_cnn_tuning()'s rule rather than a call into
# it, because the rule is inlined there. The copy is not trusted on its own --
# the runner is made to announce every unit it skips, and those announcements
# are checked against this list. If the two ever disagree, that disagreement is
# the finding.
b6_done_units <- function(run_dir, comparison = NULL) {
  cmp <- if (is.null(comparison)) b6_read_comparison(run_dir) else comparison
  if (nrow(cmp) == 0L || !"unit_id" %in% names(cmp)) return(character(0))
  ok <- cmp$unit_id[cmp$status %in% "success"]
  ok[file.exists(file.path(run_dir, "models", paste0(ok, "_best.pt")))]
}

# ── The check ledger ──────────────────────────────────────────────────────────
#
# WHY A LEDGER AND NOT A SCROLL OF stopifnot(). A script that stops at the first
# failed assertion reports one defect per run, and this one costs the user a
# real interruption to reach. Everything that can still be measured after a
# failure is measured, and the answer is a table.
#
# stopifnot() is still used for PRECONDITIONS -- the things that make the rest
# of the script meaningless -- and those stop, loudly, with what to do about it.
#
# The check expression is forced INSIDE tryCatch: a check written against a
# shape the framework does not return is itself a finding and must never be
# mistaken for a pass.
.b6_checks <- list()

b6_check <- function(id, what, expr) {
  r <- tryCatch(force(expr), error = function(e)
    list(ok = NA, measured = paste("check errored:", conditionMessage(e))))
  if (!is.list(r) || is.null(r$ok)) {
    r <- list(ok = NA, measured = "the check returned no ok/measured")
  }
  # as.logical()[1] rather than isTRUE() alone: a check that returns a VECTOR is
  # a check that forgot an all(), and `if` on a vector is an error in R >= 4.2 --
  # which would kill the script instead of recording the mistake.
  flag_ok <- suppressWarnings(as.logical(r$ok)[1])
  row <- tibble::tibble(
    id = id, what = what,
    ok = if (is.na(flag_ok)) NA else isTRUE(flag_ok),
    measured = as.character(r$measured)[1])
  .b6_checks[[id]] <<- row
  flag <- if (isTRUE(row$ok)) "PASS  " else
          if (is.na(row$ok))  "CHECK?" else "FAIL  "
  message(sprintf("  %s %-4s %s", flag, id, substr(row$measured, 1, 110)))
  invisible(row)
}

b6_report <- function(path = NULL) {
  out <- dplyr::bind_rows(.b6_checks)
  if (nrow(out) == 0L) stop("No check ran at all.", call. = FALSE)
  message("\n", strrep("=", 78))
  message("B6 -- resume after an interruption")
  message(strrep("=", 78))
  print_wide(dplyr::select(out, id, ok, what, measured), n = Inf)

  n_pass <- sum(out$ok %in% TRUE)
  n_fail <- sum(out$ok %in% FALSE)
  n_unk  <- sum(is.na(out$ok))
  message(sprintf("\n  %d check(s): %d passed | %d FAILED | %d unusable",
                  nrow(out), n_pass, n_fail, n_unk))
  if (!is.null(path)) {
    safe_write_csv2(out, path)
    message("  ", path)
  }
  message("\n", strrep("=", 78))
  message(sprintf("B6: %s", if (n_fail == 0L && n_unk == 0L) "PASS" else "FAIL"))
  message(strrep("=", 78))
  if (n_fail > 0L || n_unk > 0L) {
    stop("B6 FAILED -- read the table above. A resume defect does not announce ",
         "itself at run time; it produces a comparison table whose rows were ",
         "fitted on different things.", call. = FALSE)
  }
  invisible(out)
}

# ══════════════════════════════════════════════════════════════════════════════
# THE PRELUDE BOTH PHASES SHARE
#
# Built identically in both phases, from this file alone, so that phase 2
# reconstructs the experiment rather than inheriting it. Everything here is
# deterministic given the data on disk: the grid is drawn under a fixed seed,
# the plan under another, and the test set is the frozen one.
# ══════════════════════════════════════════════════════════════════════════════

split_file <- file.path(metadata_dir, "data_split.csv")
if (!file.exists(split_file)) {
  stop("The frozen test set is missing:\n  ", split_file, "\n\n",
       "It is written by the first run of 03_run_tuning.R. B6 uses it so that ",
       "its fold plan is fixed rather than redrawn, which is what lets the two ",
       "phases build the same plan in two different R sessions. Run 03 first.",
       call. = FALSE)
}
ds <- readr::read_csv2(split_file, show_col_types = FALSE)
frozen_test <- ds$sample_id[ds$role == "test"]
if (length(frozen_test) == 0L || anyNA(frozen_test)) {
  stop("data_split.csv holds no usable test ids: ", split_file, call. = FALSE)
}

tune_grid <- make_tune_grid(
  tune_length = b6_tune_length,
  seed        = 666L,
  fixed = list(
    window_sizes  = list(c(3L)),          # the store's smallest patches
    conv_channels = list(c(32L, 64L)),
    use_residual  = TRUE,
    use_se_block  = FALSE,
    se_reduction  = 16L,
    embedding_dim = 128L,
    embed_pool    = "gap",
    conv_padding  = "same",
    gate_type     = "no_gate_concat",     # single branch: no gate to choose
    dropout       = 0.1,
    weight_decay  = 0.0,
    batch_size    = 512L,
    loss_fn       = "smooth_l1",
    # THE ONLY AXIS LEFT. Three learning rates give three configs that are
    # genuinely different objects to train while everything else is pinned, so
    # the grid is exhaustive and the draw cannot come back short by luck.
    base_lr       = c(3e-4, 1e-3, 2e-3)
  )
)
if (nrow(tune_grid) != b6_tune_length) {
  stop("The grid came back with ", nrow(tune_grid), " config(s) instead of ",
       b6_tune_length, ". The restricted space above should hold exactly ",
       b6_tune_length, "; something in .cnn_param_space or in `fixed` moved. ",
       "Fix the grid before using it to measure resume.", call. = FALSE)
}

message("\n-- B6 grid --")
print_wide(dplyr::mutate(
  tune_grid,
  window_sizes  = purrr::map_chr(window_sizes,  paste, collapse = "x"),
  conv_channels = purrr::map_chr(conv_channels, paste, collapse = "_")))

data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = b6_windows,
  target_col   = readr::read_csv2(file.path(metadata_dir, "target_config.csv"),
                                  show_col_types = FALSE)$target_col[1]
)

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# The plan is resolved HERE, once, and passed to every dsm_train() call below --
# the control, the interruptible run, the resumed run and the negative test all
# have to be talking about the same split or none of the comparisons mean
# anything. resolve_resampling() returns a fold_plan unchanged, so passing the
# resolved object is also what makes the negative test constructible.
cv <- spatial_cv(k = b6_k_folds, block_size = "auto", buffer = "auto",
                 test_frac = b6_test_frac, max_share = 0.10, seed = 42L)
plan <- resolve_resampling(cv, data, test_ids = frozen_test,
                           windows = b6_windows)
message("\n-- B6 fold plan --")
print(plan)

expected_units <- b6_expected_units(tune_grid, b6_k_folds, b6_n_seeds)
n_expected     <- length(expected_units)
message("\nUnits this experiment will train: ", n_expected,
        "  (", nrow(tune_grid), " configs x ", b6_k_folds, " folds x ",
        b6_n_seeds, " seeds)")

# One argument list, used by every run below. The only thing that ever changes
# is run_id and the plan -- which is the point: a difference between the control
# and the resumed run cannot come from a setting that drifted between two calls.
b6_train_args <- c(
  list(
    data          = data,
    model         = "cnn",
    tune_grid     = tune_grid,
    transform     = expm1,
    output_dir    = tuning_base,
    device        = device,
    base_seed     = b6_base_seed,
    n_seeds       = b6_n_seeds,
    resume        = TRUE,
    evaluate_test = FALSE,
    verbose       = FALSE
  ),
  training_args
)

b6_train <- function(run_id, resampling = plan) {
  do.call(dsm_train, c(b6_train_args,
                       list(run_id = run_id, resampling = resampling)))
}

# ══════════════════════════════════════════════════════════════════════════════
# THE WORKER -- phase 1 launches this file under this phase, and kills it
#
# Nothing else runs here: no control run, no banner, no ledger. It exists so
# that the interrupted run is produced by the same b6_train() that produced the
# control run, in a process the parent is free to kill at a moment of its
# choosing. A second script would be a second chance for the two to drift.
# ══════════════════════════════════════════════════════════════════════════════

if (identical(b6_phase, "worker")) {
  message("B6 worker: training into ", b6_resume_run_id,
          " -- the parent kills this process partway, on purpose.")
  invisible(b6_train(b6_resume_run_id))
  message("B6 worker: reached the end WITHOUT being killed.")
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 -- prepare
# ══════════════════════════════════════════════════════════════════════════════

if (identical(b6_phase, "prepare")) {

  # ── What may already be on disk ────────────────────────────────────────────
  #
  # TWO DIRECTORIES, TWO DIFFERENT RULES, because they are two different things.
  #
  # The run to be interrupted must start from ZERO units. Otherwise "the
  # finished units were reused" is measured against units finished by some
  # earlier session, and the interrupt this script is built around is not what
  # produced the state being verified.
  #
  # The control may be reused, but only if it is COMPLETE and was fitted on this
  # experiment's plan. A half-finished control cannot be finished off, because
  # then the reference for "identical in shape to an uninterrupted run" would
  # itself be a resumed run -- circular in exactly the direction that hides a
  # defect. A complete one is a legitimate reference, and reusing it is what
  # makes a second attempt at the interrupt cost minutes instead of twice that:
  # phase 2 sends the user back here whenever he interrupts too early or too
  # late, and charging him for the control every time would make the retry
  # expensive enough to be worth avoiding, which is how a check stops being run.
  #
  # Deleting either directory automatically was rejected. This is the one action
  # in the script that destroys work, and an unattended delete is how a check
  # quietly starts measuring nothing. The exact lines to paste are printed.
  ctrl_disk  <- b6_read_comparison(control_run_dir)
  ctrl_plan_file <- file.path(control_run_dir, "fold_plan.rds")
  control_complete <-
    nrow(ctrl_disk) == n_expected &&
    all(ctrl_disk$status %in% "success") &&
    setequal(b6_done_units(control_run_dir, ctrl_disk), expected_units) &&
    file.exists(ctrl_plan_file) &&
    identical(b6_fold_membership(readRDS(ctrl_plan_file)),
              b6_fold_membership(plan))
  control_dirty <- !control_complete &&
    (nrow(ctrl_disk) > 0L || nrow(b6_checkpoints(control_run_dir)) > 0L)
  resume_dirty <- nrow(b6_read_comparison(resume_run_dir)) > 0L ||
    nrow(b6_checkpoints(resume_run_dir)) > 0L

  if (control_dirty || resume_dirty) {
    stop(
      "B6 phase 1 cannot start on what is already in its run directories.\n\n",
      if (control_dirty) paste0(
        "  The control run is present but not complete, or it was fitted on a ",
        "different\n  plan. A control that had to be resumed is not a control.",
        "\n\n  unlink(\"", control_run_dir, "\", recursive = TRUE)\n\n") else "",
      if (resume_dirty) paste0(
        "  The run to be interrupted already holds units. It has to start from ",
        "zero, or\n  'the finished units were reused' is about somebody else's ",
        "units.\n\n  unlink(\"", resume_run_dir, "\", recursive = TRUE)\n\n")
      else "",
      "Then run phase 1 again.", call. = FALSE)
  }

  # ── Step 1: the control run ────────────────────────────────────────────────
  #
  # It buys two things. The reference table, which is the only honest answer to
  # "identical in shape to an uninterrupted run" -- the 03 run on disk is NOT
  # that answer, because it was written before val_bias/val_bias_pct existed and
  # a correct run today has two columns it does not. And the COST of a unit,
  # measured rather than guessed, so the instruction that follows can say when
  # to press Esc in minutes rather than in hope.
  #
  # dsm_train() is called even when the control is complete: every unit is then
  # skipped in seconds, and it returns the RANKED table, which exists nowhere
  # else (see the state file below). Reading it back off disk would mean reading
  # a CSV whose types are guessed -- the hazard .comparison_from_csv() documents.
  message("\n", strrep("=", 78))
  if (control_complete) {
    message("PHASE 1, STEP 1 OF 2 -- REUSING THE COMPLETED CONTROL RUN.")
    message("  ", n_expected, " units already trained in:")
  } else {
    message("PHASE 1, STEP 1 OF 2 -- THE CONTROL RUN. DO NOT INTERRUPT THIS ONE.")
    message("  ", n_expected, " units, uninterrupted, into:")
  }
  message("  ", control_run_dir)
  message("  The banner that asks you to interrupt comes AFTER this finishes.")
  message(strrep("=", 78))

  t0 <- Sys.time()
  control <- b6_train(b6_control_run_id)
  control_minutes <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  ctrl_cmp <- control$comparison
  if (!setequal(ctrl_cmp$unit_id, expected_units) ||
      !all(ctrl_cmp$status %in% "success")) {
    stop("The control run did not complete: ",
         sum(ctrl_cmp$status %in% "success"), "/", n_expected,
         " unit(s) succeeded.\n",
         "Everything phase 2 compares against comes from this run, so there is ",
         "no point\ncontinuing. Read ",
         file.path(control_run_dir, "comparison", "comparison_ranked.csv"),
         "\nfor the error_message column, fix it, delete both B6 directories ",
         "and start again.", call. = FALSE)
  }

  # PER-UNIT COST FROM WHAT THE RUN WROTE, not from this session's stopwatch.
  # The stopwatch reads ~0 whenever the control was reused, and the advice below
  # would then tell the user to interrupt after no time at all. runtime_min is
  # in every row, so the estimate survives the reuse. It counts training only,
  # so it slightly understates the wall clock -- the per-fold cache build sits
  # outside it -- and the instruction is given in unit headers as well for that
  # reason.
  per_unit_min <- mean(ctrl_cmp$runtime_min, na.rm = TRUE)
  message(sprintf(
    "\nControl run: %d units, %.2f min per unit (median %.2f) | this session %.1f min.",
    n_expected, per_unit_min, stats::median(ctrl_cmp$runtime_min, na.rm = TRUE),
    control_minutes))

  # ── Step 2: record what phase 2 will be verifying ──────────────────────────
  #
  # Written BEFORE the interruptible run starts, because after the interrupt
  # this script is dead and cannot write anything. Everything phase 2 needs
  # about phase 1 therefore has to be knowable in advance -- which it is: the
  # plan, the grid, the unit names, the control table and the moment the
  # interruptible run began. What phase 1 cannot know (how far it got) is
  # exactly what phase 2 reads off the run directory itself.
  #
  # The control's comparison table is carried IN the state, as the object the
  # runner returned. The copy on disk is not equivalent: comparison_all.rds is
  # written inside the unit loop and so has no `rank` column, and
  # comparison_ranked.csv is a CSV whose types are guessed on re-read -- the
  # hazard .comparison_from_csv() exists to describe. The returned object is the
  # one thing that is neither truncated nor re-typed.
  state <- list(
    state_version      = b6_state_version,
    written_at         = Sys.time(),
    script             = b6_script,
    project_root       = project_root,
    target_label       = target_label,
    control_run_id     = b6_control_run_id,
    resume_run_id      = b6_resume_run_id,
    control_run_dir    = control_run_dir,
    resume_run_dir     = resume_run_dir,
    k_folds            = b6_k_folds,
    n_seeds            = b6_n_seeds,
    base_seed          = b6_base_seed,
    windows            = b6_windows,
    test_frac          = b6_test_frac,
    training_args      = training_args,
    plan_method        = plan$method,
    fold_membership    = b6_fold_membership(plan),
    grid_config_ids    = tune_grid$config_id,
    grid_fingerprint   = b6_grid_fingerprint(tune_grid),
    expected_units     = expected_units,
    control_units      = sort(ctrl_cmp$unit_id),
    control_columns    = names(ctrl_cmp),
    control_comparison = ctrl_cmp,
    control_by_config  = control$by_config,
    control_minutes    = control_minutes,
    per_unit_min       = per_unit_min,
    started_run_at     = Sys.time()
  )
  safe_save_rds(state, state_path)
  message("State written: ", state_path)

  # ── Step 3: the run to interrupt ───────────────────────────────────────────
  #
  # The banner is the last thing printed before the call, so it is the last
  # thing on screen while the user decides.
  #
  # A third of the units is the target: enough finished for "these were reused"
  # to be a real claim, and enough left for "these ran" to be one too. It is a
  # target, not a requirement -- phase 2 measures where the interrupt actually
  # landed and refuses only the two useless extremes, none finished and all
  # finished.
  suggest_units <- max(1L, as.integer(floor(n_expected / 3)))
  suggest_min   <- per_unit_min * suggest_units

  # WHICH KIND OF INTERRUPTION. Automatic by default: this file is launched in
  # a subprocess and killed once `suggest_units` units are on disk. "manual"
  # keeps the original path, where the user presses Esc -- a different failure
  # mode worth being able to reach, because Esc unwinds the stack and torch can
  # surface it as an ordinary error that the runner records as a failed unit,
  # while a killed process cannot record anything. Phase 2 accepts both.
  b6_interrupt <- tolower(env_chr("soc_b6_interrupt", "auto"))
  if (!b6_interrupt %in% c("auto", "manual")) {
    stop("soc_b6_interrupt must be \"auto\" or \"manual\", got '", b6_interrupt,
         "'.", call. = FALSE)
  }
  if (identical(b6_interrupt, "auto") && !requireNamespace("processx", quietly = TRUE)) {
    message("\nprocessx is not installed, so the interruption cannot be ",
            "scripted. Falling back\nto the manual path.")
    b6_interrupt <- "manual"
  }

  if (identical(b6_interrupt, "manual")) {
    message("\n", strrep("=", 78))
    message("PHASE 1, STEP 2 OF 2 -- THIS IS THE RUN YOU INTERRUPT.")
    message(strrep("-", 78))
    message(sprintf("  Let it run about %.1f minute(s) -- until roughly %d of the %d",
                    suggest_min, suggest_units, n_expected))
    message("  unit headers have gone by -- and then INTERRUPT IT:")
    message("")
    message("      RStudio : press Esc")
    message("      Rterm   : press Ctrl-C")
    message("")
    message("  Press it again if the run does not stop within a few seconds:")
    message("  torch can surface an interrupt as an error, which the runner ")
    message("  records as a failed unit and carries on. Either way is fine -- ")
    message("  phase 2 handles a failed unit and a missing one the same way.")
    message("")
    message("  Do NOT delete anything. Then run, here or in a fresh R session:")
    message("")
    message("      Sys.setenv(soc_b6_phase = \"verify\")")
    message("      source(\"", b6_script, "\")")
    message(strrep("=", 78))

    invisible(b6_train(b6_resume_run_id))

  } else {
    # ── The interruption, performed from outside ───────────────────────────
    #
    # THE MOMENT IS WATCHED, NOT TIMED. Sleeping for `suggest_min` minutes and
    # then killing would be a bet on the subprocess loading packages, the store
    # and the fold cache at the speed this machine did it last time. The run
    # directory says what actually happened, so the watcher polls it and kills
    # the instant the target is reached -- mid-unit, which is the point.
    rscript <- file.path(R.home("bin"), "Rscript.exe")
    if (!file.exists(rscript)) rscript <- file.path(R.home("bin"), "Rscript")
    if (!file.exists(rscript)) {
      stop("Rscript not found under ", R.home("bin"),
           "; re-run with Sys.setenv(soc_b6_interrupt = \"manual\").",
           call. = FALSE)
    }
    # Beside the run, not inside it: an unlink() of the run to start over must
    # not take the log that explains why starting over was needed.
    worker_log <- file.path(dirname(resume_run_dir), "b6_worker.log")
    dir.create(dirname(worker_log), recursive = TRUE, showWarnings = FALSE)

    message("\n", strrep("=", 78))
    message("PHASE 1, STEP 2 OF 2 -- THE RUN THAT GETS INTERRUPTED.")
    message(strrep("-", 78))
    message(sprintf("  Launched in a subprocess. It will be KILLED as soon as %d of the",
                    suggest_units))
    message(sprintf("  %d units are on disk -- around %.1f minute(s), but measured, not timed.",
                    n_expected, suggest_min))
    message("  Nothing to press. Killing the process is a harder interruption than")
    message("  Esc: it is what a power cut does.")
    message("")
    message("  Worker log: ", worker_log)
    message(strrep("=", 78))

    p <- processx::process$new(
      rscript, args = b6_script,
      env = c("current", soc_b6_phase = "worker"),
      stdout = worker_log, stderr = worker_log, cleanup = TRUE)
    message("  worker PID ", p$get_pid(), " -- watching ", basename(resume_run_dir))

    t0_watch  <- Sys.time()
    # Generous: the worker has to load packages, the store and the first fold
    # cache before any unit finishes. Ten times the control's own per-unit cost
    # for the units wanted, floored at 5 minutes.
    watch_cap <- max(5, 10 * suggest_min)
    n_seen    <- -1L
    killed    <- FALSE
    repeat {
      if (!p$is_alive()) break
      n_now <- length(b6_done_units(resume_run_dir))
      if (n_now != n_seen) {
        message(sprintf("    %5.1f min | %d of %d unit(s) finished",
                        as.numeric(difftime(Sys.time(), t0_watch, units = "mins")),
                        n_now, n_expected))
        n_seen <- n_now
      }
      if (n_now >= suggest_units) {
        message("  target reached -- killing the worker MID-UNIT now.")
        p$kill()
        killed <- TRUE
        break
      }
      if (as.numeric(difftime(Sys.time(), t0_watch, units = "mins")) > watch_cap) {
        p$kill()
        stop("The worker ran ", round(watch_cap, 1), " minute(s) without ",
             "finishing ", suggest_units, " unit(s), and was killed.\n  Read ",
             worker_log, " -- it failed before training, or this machine is ",
             "far slower than the control run suggested.", call. = FALSE)
      }
      Sys.sleep(2)
    }
    if (!killed) {
      message("\n  The worker exited on its own (status ", p$get_exit_status(),
              ") before the target was reached.")
      message("  Its log is at ", worker_log)
    }
    # The kill is asynchronous; give the OS a moment before counting files.
    p$wait(timeout = 10000)
    Sys.sleep(1)
  }

  # ── Reaching this line means the call RETURNED rather than being killed ─────
  #
  # There are two ways that happens and they need opposite advice, so the ledger
  # on disk decides rather than the assumption that "returned" means "finished".
  #
  # Esc is not guaranteed to propagate: torch can surface an interrupt as an
  # ordinary error, run_cnn_tuning() records that unit as failed and carries on,
  # and the loop then ends normally with units missing. That is a perfectly good
  # half-finished run -- phase 2 treats a failed unit exactly like a missing one
  # -- and telling the user to start over would throw away the interrupt he just
  # performed.
  n_done_now <- length(b6_done_units(resume_run_dir))
  message("\n", strrep("=", 78))
  if (n_done_now == 0L) {
    message("THE RUN WAS INTERRUPTED BEFORE ITS FIRST UNIT FINISHED.")
    message(strrep("=", 78))
    message("Phase 2 needs at least one finished unit to have anything to reuse,")
    message("and it will refuse. Delete this run and do phase 1 again:")
    message("")
    message("  unlink(\"", resume_run_dir, "\", recursive = TRUE)")
    message("  Sys.setenv(soc_b6_phase = \"prepare\")")
    message("  source(\"", b6_script, "\")")
    message("")
    message("The control run is untouched and does NOT need to be redone.")
  } else if (n_done_now >= n_expected) {
    message("THIS RUN FINISHED WITHOUT BEING INTERRUPTED.")
    message(strrep("=", 78))
    message("There is nothing left for phase 2 to resume, and it will refuse.")
    message("Delete the run to be interrupted and do phase 1 again -- and this")
    message("time let it interrupt itself, which is the default:")
    message("")
    message("  unlink(\"", resume_run_dir, "\", recursive = TRUE)")
    message("  Sys.unsetenv(\"soc_b6_interrupt\")")
    message("  Sys.setenv(soc_b6_phase = \"prepare\")")
    message("  source(\"", b6_script, "\")")
    message("")
    message("The control run is untouched and does NOT need to be redone.")
  } else {
    message(sprintf("THE RUN STOPPED EARLY: %d of %d units finished.",
                    n_done_now, n_expected))
    message(strrep("=", 78))
    message("That is exactly the half-finished run phase 2 needs: some units on ")
    message("disk, some never started. Go on:")
    message("")
    message("      Sys.setenv(soc_b6_phase = \"verify\")")
    message("      source(\"", b6_script, "\")")
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 -- verify
# ══════════════════════════════════════════════════════════════════════════════

if (identical(b6_phase, "verify")) {

  # ── Preconditions ──────────────────────────────────────────────────────────
  #
  # These stop rather than joining the ledger, because each of them makes every
  # check below meaningless. None of them is allowed to print "skipping".

  if (!file.exists(state_path)) {
    stop(
      "PHASE 1 HAS NOT RUN -- there is no state file at:\n  ", state_path,
      "\n\nPhase 2 verifies a run that phase 1 started and you interrupted. ",
      "Without\nphase 1 there is no plan fingerprint, no control run and no ",
      "half-finished run\nto resume, so there is nothing here that could be ",
      "checked.\n\nRun phase 1 first:\n\n",
      "  Sys.setenv(soc_b6_phase = \"prepare\")\n",
      "  source(\"", b6_script, "\")\n", call. = FALSE)
  }
  state <- readRDS(state_path)

  if (!identical(state$state_version, b6_state_version)) {
    stop("The state file was written by a different version of this script ",
         "(state_version ", state$state_version %||% "missing", " against ",
         b6_state_version, ").\nDelete both B6 run directories and redo phase ",
         "1 with the current script:\n\n",
         "  unlink(\"", control_run_dir, "\", recursive = TRUE)\n",
         "  unlink(\"", resume_run_dir,  "\", recursive = TRUE)\n",
         call. = FALSE)
  }

  # THE FINGERPRINT IS CHECKED, NOT MERELY READ.
  #
  # This is what stops a state file from an EARLIER experiment being verified
  # against a plan or a grid that has since moved. Reading the fingerprint and
  # trusting it would leave the exact hole the brief names: a second experiment
  # silently verifying against the first one's expectations, and passing.
  drift <- c(
    if (!identical(state$fold_membership, b6_fold_membership(plan)))
      "the fold plan",
    if (!identical(state$grid_fingerprint, b6_grid_fingerprint(tune_grid)))
      "the tune grid",
    if (!identical(state$expected_units, expected_units))
      "the list of units",
    if (!identical(state$training_args, training_args))
      "the training arguments",
    if (!identical(state$n_seeds, b6_n_seeds) ||
        !identical(state$k_folds, b6_k_folds) ||
        !identical(state$base_seed, b6_base_seed))
      "the resampling settings"
  )
  if (length(drift) > 0L) {
    stop(
      "THE EXPERIMENT MOVED BETWEEN PHASE 1 AND PHASE 2.\n\n",
      "  changed: ", paste(drift, collapse = ", "), "\n\n",
      "The state file describes a different experiment from the one this ",
      "script\nbuilds now, so verifying one against the other would measure ",
      "nothing.\nEither this script was edited between the phases, or the ",
      "point table or\nthe frozen test set changed under it.\n\n",
      "Delete both B6 run directories and start from phase 1:\n\n",
      "  unlink(\"", control_run_dir, "\", recursive = TRUE)\n",
      "  unlink(\"", resume_run_dir,  "\", recursive = TRUE)\n",
      call. = FALSE)
  }

  ctrl_cmp     <- state$control_comparison
  ctrl_on_disk <- b6_read_comparison(control_run_dir)
  if (nrow(ctrl_on_disk) == 0L ||
      !setequal(ctrl_on_disk$unit_id, state$control_units)) {
    stop("The control run recorded by phase 1 is not in ", control_run_dir,
         " any more.\nIt is the reference every comparison below is made ",
         "against. Delete both B6\nrun directories and redo phase 1.",
         call. = FALSE)
  }

  # ── The state phase 1 left behind, read BEFORE anything is restarted ───────
  #
  # This snapshot is the only record of what the interrupt produced. Once the
  # resumed run starts, the run directory stops being evidence.
  snap_cmp   <- b6_read_comparison(resume_run_dir)
  snap_ckpt  <- b6_checkpoints(resume_run_dir)
  done_before <- b6_done_units(resume_run_dir, snap_cmp)
  todo_before <- setdiff(expected_units, done_before)

  failed_before <- if (nrow(snap_cmp) > 0L)
    snap_cmp$unit_id[!snap_cmp$status %in% "success"] else character(0)
  orphan_ckpt <- setdiff(snap_ckpt$unit_id, done_before)

  message("\n", strrep("=", 78))
  message("WHAT THE INTERRUPT LEFT BEHIND")
  message(strrep("=", 78))
  message(sprintf("  finished units      : %d of %d", length(done_before),
                  n_expected))
  message(sprintf("  left to do          : %d", length(todo_before)))
  message(sprintf("  rows marked failed  : %d%s", length(failed_before),
                  if (length(failed_before) > 0L)
                    "  (an interrupt surfaced as an error -- expected)" else ""))
  message(sprintf("  checkpoints with no success row : %d%s",
                  length(orphan_ckpt),
                  if (length(orphan_ckpt) > 0L)
                    "  (killed between the save and the row -- expected)"
                  else ""))

  # THE TWO USELESS EXTREMES. Neither is a failure of the framework, and neither
  # can be quietly tolerated: verifying them would produce a table of passes
  # that measured nothing at all.
  if (length(done_before) == 0L) {
    stop(
      "NOTHING FINISHED BEFORE THE INTERRUPT, so there is nothing that could ",
      "have been\nreused, and 'resume skipped the finished units' is not a ",
      "claim this run can\ntest.\n\n",
      "Redo phase 1 and let it run longer before pressing Esc -- phase 1 ",
      "prints the\ntarget in minutes, measured from the control run:\n\n",
      "  unlink(\"", resume_run_dir, "\", recursive = TRUE)\n",
      "  Sys.setenv(soc_b6_phase = \"prepare\")\n",
      "  source(\"", b6_script, "\")\n\n",
      "The control run is untouched and does not need to be redone.",
      call. = FALSE)
  }
  if (length(todo_before) == 0L) {
    stop(
      "EVERY UNIT WAS ALREADY FINISHED, so there is nothing to resume ONTO and ",
      "'the\nunfinished units ran' is not a claim this run can test. Either the ",
      "interrupt\ncame too late, or phase 2 has already been run once against ",
      "this directory --\nthe second time round, the run it verified is ",
      "complete.\n\n",
      "Redo phase 1 and press Esc earlier:\n\n",
      "  unlink(\"", resume_run_dir, "\", recursive = TRUE)\n",
      "  Sys.setenv(soc_b6_phase = \"prepare\")\n",
      "  source(\"", b6_script, "\")\n\n",
      "The control run is untouched and does not need to be redone.",
      call. = FALSE)
  }

  # ── The resumed run ────────────────────────────────────────────────────────
  #
  # Messages and warnings are RECORDED rather than muffled: the runner's own
  # announcements are evidence here ("already trained, skipping" names every
  # unit it reused), and the user still needs to watch the run.
  #
  # restart_at is the line every new checkpoint must fall after. The five
  # seconds of slack absorb clock granularity on a mounted filesystem; it is far
  # below the cost of a unit, so it cannot let a stale checkpoint through.
  b6_messages <- character(0)
  b6_warnings <- character(0)
  restart_at  <- as.numeric(Sys.time())

  message("\n", strrep("=", 78))
  message("RESTARTING THE INTERRUPTED RUN (same run_id, resume = TRUE)")
  message("  ", resume_run_dir)
  message(strrep("=", 78))

  resumed <- withCallingHandlers(
    b6_train(b6_resume_run_id),
    message = function(m) b6_messages <<- c(b6_messages, conditionMessage(m)),
    warning = function(w) b6_warnings <<- c(b6_warnings, conditionMessage(w))
  )

  final_cmp  <- resumed$comparison
  final_ckpt <- b6_checkpoints(resume_run_dir)
  said <- function(pattern) {
    any(grepl(pattern, b6_messages, fixed = TRUE)) ||
    any(grepl(pattern, b6_warnings, fixed = TRUE))
  }

  message("\n", strrep("=", 78))
  message("CHECKS")
  message(strrep("=", 78))

  # ── (a) the finished units were reused, not retrained ──────────────────────

  b6_check("B6-1", "every finished unit was announced as skipped", {
    missing <- done_before[!vapply(done_before, function(u)
      said(paste0(u, " -- already trained, skipping")), logical(1))]
    list(ok = length(missing) == 0L, measured = sprintf(
      "%d/%d finished unit(s) announced as skipped by the runner%s",
      length(done_before) - length(missing), length(done_before),
      if (length(missing) > 0L)
        paste0(" | silent: ", paste(utils::head(missing, 4), collapse = ", "))
      else ""))
  })

  b6_check("B6-2", "their checkpoints were not touched", {
    # THE PHYSICAL EVIDENCE. A retrained unit under a fixed seed produces the
    # same numbers, so a table comparison alone cannot tell "reused" from
    # "retrained and happened to agree". The file's modification time can.
    a <- snap_ckpt[match(done_before, snap_ckpt$unit_id), ]
    b <- final_ckpt[match(done_before, final_ckpt$unit_id), ]
    drift <- abs(a$mtime - b$mtime)
    drift[is.na(drift)] <- Inf          # the file went away: not "unchanged"
    same <- drift < 1e-3 & !is.na(b$size) & a$size == b$size
    list(ok = all(same), measured = sprintf(
      "%d/%d checkpoint(s) unchanged in mtime and size | worst drift %s s",
      sum(same), length(done_before), format(max(drift), digits = 3)))
  })

  b6_check("B6-3", "their comparison rows are byte-identical", {
    cols <- intersect(names(snap_cmp), names(final_cmp))
    a <- snap_cmp[match(done_before, snap_cmp$unit_id), cols, drop = FALSE]
    b <- final_cmp[match(done_before, final_cmp$unit_id), cols, drop = FALSE]
    differ <- cols[!vapply(cols, function(cc) identical(a[[cc]], b[[cc]]),
                           logical(1))]
    list(ok = length(differ) == 0L, measured = sprintf(
      "%d row(s) x %d column(s) identical%s", length(done_before), length(cols),
      if (length(differ) > 0L)
        paste0(" | differ: ", paste(utils::head(differ, 6), collapse = ", "))
      else ""))
  })

  # ── (b) the unfinished units ran ───────────────────────────────────────────

  b6_check("B6-4", "every unfinished unit trained on the restart", {
    b <- final_ckpt[match(todo_before, final_ckpt$unit_id), ]
    fresh <- !is.na(b$mtime) & b$mtime >= restart_at - 5
    wrongly_skipped <- todo_before[vapply(todo_before, function(u)
      said(paste0(u, " -- already trained, skipping")), logical(1))]
    ok_rows <- final_cmp$status[match(todo_before, final_cmp$unit_id)] %in% "success"
    list(ok = all(fresh) && length(wrongly_skipped) == 0L && all(ok_rows),
         measured = sprintf(
      "%d/%d wrote a NEW checkpoint after the restart | %d succeeded | %d wrongly skipped",
      sum(fresh), length(todo_before), sum(ok_rows), length(wrongly_skipped)))
  })

  b6_check("B6-5", "no checkpoint under a name nothing reads", {
    # safe_torch_save() falls back to a timestamped file when it cannot remove
    # the old one. Such a file is a unit that trained and that resume will never
    # find -- it would be retrained forever, silently.
    stray <- setdiff(final_ckpt$unit_id, expected_units)
    list(ok = length(stray) == 0L, measured = sprintf(
      "%d checkpoint file(s), %d stray%s", nrow(final_ckpt), length(stray),
      if (length(stray) > 0L)
        paste0(": ", paste(utils::head(stray, 3), collapse = ", ")) else ""))
  })

  # ── (c) the guards stayed quiet ────────────────────────────────────────────

  b6_check("B6-6", "check_plan_unchanged() stayed quiet", {
    # Three ways of asking, because "quiet" is the absence of something and an
    # absence is easy to get for the wrong reason. The run finished (it would
    # have stopped), it said nothing about the plan, and a direct call returns
    # TRUE without emitting a condition.
    spoke <- said("THE FOLD PLAN IN THIS RUN DIRECTORY")
    # The cached plan must EXIST, or the direct call returns TRUE from its first
    # line and this check passes by finding nothing to compare -- which is the
    # exact shape of the quiet pass this project keeps being bitten by.
    cached <- file.exists(file.path(resume_run_dir, "fold_plan.rds"))
    heard <- character(0)
    direct <- withCallingHandlers(
      tryCatch(check_plan_unchanged(plan, resume_run_dir, resume = TRUE),
               error = function(e) e),
      message = function(m) heard <<- c(heard, conditionMessage(m)),
      warning = function(w) heard <<- c(heard, conditionMessage(w)))
    list(ok = !spoke && cached && isTRUE(direct) && length(heard) == 0L,
         measured = sprintf(
      "runner complained: %s | fold_plan.rds on disk: %s | direct call returned %s | conditions: %d",
      spoke, cached,
      if (inherits(direct, "error")) "an ERROR" else format(direct),
      length(heard)))
  })

  b6_check("B6-7", ".resumable_units() kept every cached unit", {
    # The other half of quiet. A resume that refits everything is not a crash --
    # it is hours of work and a table that looks fine, which is why the noisy
    # "REFITTED" message exists and why its ABSENCE is asserted here.
    kept <- .resumable_units(done_before, snap_cmp, tune_grid, verbose = FALSE)
    list(ok = setequal(kept, done_before) && !said("REFITTED"),
         measured = sprintf("%d/%d cached unit(s) still resumable | runner said REFITTED: %s",
                            length(kept), length(done_before), said("REFITTED")))
  })

  # ── (d) the final table ────────────────────────────────────────────────────

  b6_check("B6-8", "the comparison table is complete", {
    dup <- sum(duplicated(final_cmp$unit_id))
    ok_status <- sum(final_cmp$status %in% "success")
    ranks <- sort(final_cmp$rank[!is.na(final_cmp$rank)])
    list(ok = nrow(final_cmp) == n_expected && dup == 0L &&
              setequal(final_cmp$unit_id, expected_units) &&
              ok_status == n_expected && !anyNA(final_cmp$val_ccc) &&
              identical(as.integer(ranks), seq_len(n_expected)),
         measured = sprintf(
      "%d row(s) for %d expected units | %d success | %d duplicate | ranks 1..%d: %s",
      nrow(final_cmp), n_expected, ok_status, dup, n_expected,
      identical(as.integer(ranks), seq_len(n_expected))))
  })

  b6_check("B6-9", "same shape as the uninterrupted control run", {
    list(ok = identical(names(final_cmp), names(ctrl_cmp)) &&
              identical(vapply(final_cmp, function(z) class(z)[1], character(1)),
                        vapply(ctrl_cmp,  function(z) class(z)[1], character(1))) &&
              nrow(final_cmp) == nrow(ctrl_cmp) &&
              setequal(final_cmp$unit_id, ctrl_cmp$unit_id) &&
              identical(names(resumed$by_config), names(state$control_by_config)) &&
              nrow(resumed$by_config) == nrow(state$control_by_config),
         measured = sprintf(
      "comparison %d x %d against %d x %d | by_config %d x %d against %d x %d",
      nrow(final_cmp), ncol(final_cmp), nrow(ctrl_cmp), ncol(ctrl_cmp),
      nrow(resumed$by_config), ncol(resumed$by_config),
      nrow(state$control_by_config), ncol(state$control_by_config)))
  })

  # ── (e) the contents, against the uninterrupted control run ────────────────
  #
  # A table can have the right columns and the wrong contents, so the shape
  # check above needs a companion that reads the cells. TWO checks do it, and
  # they are deliberately different in KIND, because the two claims are
  # different in kind.
  #
  # B6-10 asserts the IDENTITY columns EXACTLY. Every one of them is copied from
  # the config, the plan or the loop counter; nothing about them is computed
  # from a training. A difference of any size is a defect.
  #
  # B6-11 asserts the OUTCOME columns, and it cannot ask for exactness. EVERY
  # unit of the resumed run is an INDEPENDENT training of the same
  # (config, fold, seed) as its twin in the control -- the carried-over ones
  # included, because they were trained by phase 1's SECOND run, which is not
  # the control run either. A CNN trained on a GPU is not promised to the last
  # bit: cuDNN picks convolution algorithms non-deterministically, its backward
  # kernels accumulate through atomics, and nothing under R/ asks torch for a
  # deterministic backend -- R/train_cnn.R sets set.seed() and
  # torch_manual_seed() and stops there. Thirty epochs amplify whatever that
  # leaves behind. _b3_augmentation.R states the same fact and refuses, for the
  # same reason, to read anything into two runs of one config disagreeing.
  #
  # WHAT THE YARDSTICK IS, AND WHY IT IS NOT A NUMBER TYPED HERE.
  #
  # The first version of B6-11 borrowed _b4_shard_merge_check.R's
  # |x - y| <= 1e-6 + 1e-3 * |y|, and that was wrong twice over. _b4 compares
  # two TILINGS OF ONE TRAINED MODEL -- deterministic inference, not two
  # trainings -- and it anchors its relative term to the band's global maximum,
  # never to the value in hand, so the allowance cannot collapse at the bottom
  # of the range. Here it collapses: val_bias is a signed mean residual that
  # sits near zero, 1e-3 * |y| is then worth about nothing, and the allowance
  # falls back to 1e-6 absolute -- which no GPU meets. The whole ledger would
  # have printed FAIL, b6_report() would have stopped, and the diagnosis handed
  # to the user would have been "resume defect" on a framework that resumed
  # perfectly, at the end of a two-phase procedure that costs a hand-performed
  # interrupt and a control run to reach.
  #
  # The yardstick used instead is one THE CONTROL RUN MEASURED ITSELF:
  # seed_noise_floor() (R/resample.R) reports how far the same config on the
  # same fold moves when only the seed changes. That turns B6-11 into a claim
  # that is both falsifiable and true -- a SAME-seed pair must not differ by
  # more than a DIFFERENT-seed pair does. It is far too coarse to notice GPU
  # jitter, which is the point: jitter is a fact about the machine and B6 is not
  # the script that measures it. It is far too tight to let through the defect
  # this check exists for, a unit retrained after the restart against a fold
  # cache or a training set that is not the one the control used -- that moves a
  # metric by much more than a seed does, which is the whole reason the seed
  # floor is worth reporting in the first place.
  #
  # Alternatives considered and discarded:
  #
  #   assert bit-identity        contradicted by the repo. B3 says so in as many
  #                              words and prints its own cross-run table as
  #                              context rather than asserting it.
  #   drop the comparison        B6-3 already carries the resume claim exactly
  #   entirely                   (the carried rows ARE the rows that were on
  #                              disk) and B6-2 carries the physical evidence.
  #                              But neither looks at the units that trained
  #                              AFTER the restart, and those are the ones a
  #                              rebuilt fold cache would spoil.
  #   a fixed multiplier below   nobody has measured this machine yet. The
  #   1.0, e.g. 0.05             realised ratio is printed on every run; that is
  #                              the number a future tightening should be argued
  #                              from, not a guess that fails the first time the
  #                              check is honestly run.
  #
  # best_epoch is left OUT of both checks and REPORTED below instead. It is an
  # argmin over the epoch axis, decided at R/train_cnn.R:291 by a
  # `monitor < best_metric - es_min_delta` comparison with es_min_delta = 5e-4,
  # between validation losses that sit within ~1e-3 of each other for the whole
  # second half of a run (any history CSV under outputs/final_model/ shows it:
  # 0.163776 at epoch 20 against 0.164601 at 28 and 0.160137 at 29). A drift far
  # below any metric's noise floor flips it by an epoch, and an epoch index has
  # no scale on which "close" means anything.

  # The two column sets, worked out once so that the checks and the diagnosis
  # below cannot drift apart, and derived from the tables rather than listed.
  b6_identity_cols <- unique(c("unit_id", "config_id", "fold", "seed", "status",
                               "error_message", "window_sizes", "conv_channels",
                               names(tune_grid), "n_params", "val_n", "test_n"))

  # Everything else the runner COMPUTED, minus three columns that are not
  # outcomes of the fit:
  #   runtime_min  a wall-clock measurement of two different runs; comparing it
  #                would be comparing the machine's mood.
  #   rank         a FUNCTION of the columns below it, and the only one that
  #                couples the rows: one unit moving past another renumbers
  #                both, so a single difference among the retrained units would
  #                make every carried-over row look different too and destroy
  #                the diagnosis. B6-8 already asserts the ranks are a
  #                contiguous 1..n over the successes.
  #   best_epoch   a discrete argmin, see above.
  b6_outcome_cols <- setdiff(
    intersect(names(final_cmp), names(ctrl_cmp)),
    unique(c(b6_identity_cols, "runtime_min", "rank", "best_epoch")))
  b6_outcome_cols <- b6_outcome_cols[vapply(
    b6_outcome_cols, function(cc) is.numeric(final_cmp[[cc]]), logical(1))]

  b6_check("B6-10", "identities match the control run exactly", {
    a <- final_cmp[order(final_cmp$unit_id), ]
    b <- ctrl_cmp[order(ctrl_cmp$unit_id), ]
    cols <- intersect(b6_identity_cols, intersect(names(a), names(b)))
    differ <- cols[!vapply(cols, function(cc) identical(a[[cc]], b[[cc]]),
                           logical(1))]
    list(ok = length(differ) == 0L, measured = sprintf(
      "%d identity column(s) identical across %d unit(s)%s",
      length(cols) - length(differ), nrow(a),
      if (length(differ) > 0L)
        paste0(" | differ: ", paste(utils::head(differ, 6), collapse = ", "))
      else ""))
  })

  b6_check("B6-11", "outcomes sit inside the control's own seed spread", {
    a <- final_cmp[order(final_cmp$unit_id), ]
    b <- ctrl_cmp[order(ctrl_cmp$unit_id), ]

    # The rows are matched POSITIONALLY after the sort, so this has to hold or
    # every difference below is a difference between two different units. B6-9
    # asserts it too; repeating it here costs nothing and keeps this check from
    # reporting nonsense when that one fails.
    if (!identical(a$unit_id, b$unit_id)) {
      list(ok = FALSE, measured = sprintf(
        "the two runs do not hold the same units (%d against %d) -- see B6-9",
        nrow(a), nrow(b)))
    } else {
      cols <- b6_outcome_cols

      # (1) THE NA PATTERN IS ASSERTED FOR EVERY OUTCOME COLUMN, EXACTLY, and it
      # is the part of this check that is genuinely deterministic. Under
      # evaluate_test = FALSE every test_* column is NA in both runs by
      # construction; a number appearing in one of them, or a val_* metric going
      # missing, is a defect no tolerance should be asked about.
      na_differ <- cols[!vapply(cols, function(cc)
        identical(is.na(a[[cc]]), is.na(b[[cc]])), logical(1))]

      # (2) THE NUMBERS ARE ASSERTED WHERE THERE ARE NUMBERS. The all-NA columns
      # drop out here and are not quietly excused: their NA pattern was just
      # asserted above, which is the only claim that can be made about them.
      have <- cols[vapply(cols, function(cc)
        any(!is.na(a[[cc]]) & !is.na(b[[cc]])), logical(1))]

      # The yardstick, per column, measured on the CONTROL run: the widest gap
      # between two seeds of the same config on the same fold. Computed from the
      # control alone, so the thing being tested never sets its own bar.
      #
      # max_range, NOT median_sd, and the difference is what keeps this check
      # off the user's back. Two particular seeds can land near each other on a
      # particular metric by luck; the MAX over every (config, fold) group means
      # one such coincidence cannot shrink the yardstick, whereas the median
      # would move with it. The sd of a two-seed sample is also |difference| /
      # sqrt(2), i.e. tighter than the gap it describes for no reason at all.
      floor_of <- vapply(have, function(cc) {
        f <- seed_noise_floor(b, metric = cc)$max_range
        if (length(f) != 1L || !is.finite(f)) NA_real_ else f
      }, numeric(1))
      gap_of <- vapply(have, function(cc)
        max(abs(a[[cc]] - b[[cc]]), na.rm = TRUE), numeric(1))

      # A column the control cannot supply a yardstick for FAILS rather than
      # dropping out. A seed spread of zero would mean two different seeds
      # produced the same number, which is not a reason to stop checking -- it
      # is a finding in its own right, and a loud one.
      blind <- have[is.na(floor_of) | floor_of <= 0]
      ratio <- gap_of / floor_of
      ratio[is.na(floor_of) | floor_of <= 0] <- Inf

      worst <- if (length(ratio) == 0L) "nothing comparable" else sprintf(
        "%s at %.3g of the seed spread", names(ratio)[which.max(ratio)],
        max(ratio))
      list(
        # length(have) > 0L: a check that passes by finding nothing to compare
        # is the quiet pass this project keeps being bitten by.
        ok = length(na_differ) == 0L && length(blind) == 0L &&
             length(have) > 0L && all(ratio <= 1),
        measured = sprintf(
          "%d/%d outcome column(s) comparable | worst %s%s%s",
          length(have), length(cols), worst,
          if (length(na_differ) > 0L)
            paste0(" | NA pattern differs: ",
                   paste(utils::head(na_differ, 4), collapse = ", ")) else "",
          if (length(blind) > 0L)
            paste0(" | no seed spread to measure against: ",
                   paste(utils::head(blind, 4), collapse = ", ")) else ""))
    }
  })

  # ── How far the two trainings landed apart -- context, not a check ─────────
  #
  # The most useful output in this script, and deliberately NOT part of the
  # ledger. It is a measurement of THIS MACHINE, and B6's verdict is about the
  # framework; the two were a single `ok` once and that is what made a correct
  # run report a resume defect.
  #
  # READ THE TWO GROUPS WITH THE RIGHT QUESTION IN MIND. They do NOT separate
  # "resume" from "the machine" -- an earlier version of this comment claimed
  # they did, and it was wrong. Both groups are independent trainings relative
  # to the control: the carried-over units came out of phase 1's second run, the
  # retrained ones out of phase 2's. What the split does separate is a run
  # inside the same session as the control from one after a full restart, so a
  # retrained group that is markedly further out than the carried-over one
  # points at something the restart rebuilt -- the per-fold cache first of all.
  # The resume claim itself is asserted, exactly and elsewhere: B6-3 says the
  # carried rows ARE the rows that were on disk before the restart, and B6-2
  # says their checkpoints were never reopened.
  #
  # An error here is printed and does not stop the script: this block has no
  # verdict to lose, and the ledger below is what the user came for.
  b6_drift <- tryCatch({
    a <- final_cmp[order(final_cmp$unit_id), ]
    b <- ctrl_cmp[order(ctrl_cmp$unit_id), ]
    kept_rows <- a$unit_id %in% done_before
    new_rows  <- a$unit_id %in% todo_before
    gap <- function(cc, rows) {
      d <- abs(a[[cc]][rows] - b[[cc]][rows])
      if (length(d) == 0L || all(is.na(d))) NA_real_ else max(d, na.rm = TRUE)
    }
    sp <- vapply(b6_outcome_cols, function(cc) {
      f <- seed_noise_floor(b, metric = cc)$max_range
      if (length(f) != 1L) NA_real_ else as.numeric(f)
    }, numeric(1))
    gk <- vapply(b6_outcome_cols, gap, numeric(1), rows = kept_rows)
    gn <- vapply(b6_outcome_cols, gap, numeric(1), rows = new_rows)
    list(
      table = tibble::tibble(
        column       = b6_outcome_cols,
        seed_spread  = sp,
        carried_over = gk,
        retrained    = gn,
        of_spread    = pmax(gk, gn, na.rm = TRUE) / sp),
      moved_kept = sum(a$best_epoch[kept_rows] != b$best_epoch[kept_rows],
                       na.rm = TRUE),
      moved_new  = sum(a$best_epoch[new_rows] != b$best_epoch[new_rows],
                       na.rm = TRUE),
      n_kept = sum(kept_rows), n_new = sum(new_rows))
  }, error = function(e) e)

  message("\n", strrep("-", 78))
  message("HOW FAR THE RESUMED RUN LANDED FROM THE CONTROL (context, not a check)")
  message(strrep("-", 78))
  if (inherits(b6_drift, "error")) {
    message("  NOT MEASURED -- this diagnosis errored: ",
            conditionMessage(b6_drift))
    message("  B6-11 above is the assertion; this block only explains it.")
  } else {
    message("  seed_spread is the control run's own widest same-config,",
            " same-fold,")
    message("  different-seed gap. of_spread is the worse of the two columns",
            " over it,")
    message("  and B6-11 asserts it stays at or below 1.")
    print_wide(b6_drift$table, n = Inf)
    message(sprintf(
      "  best_epoch moved in %d of %d carried-over and %d of %d retrained unit(s).",
      b6_drift$moved_kept, b6_drift$n_kept,
      b6_drift$moved_new,  b6_drift$n_new))
    message("  A moved best epoch is a different checkpoint, so every val_*",
            " metric moves")
    message("  with it. It is an argmin over near-tied epochs, not a fault.")
  }
  message(strrep("-", 78))

  # ── The negative case ──────────────────────────────────────────────────────
  #
  # THE HALF THAT PROTECTS THE USER. Everything above shows resume working.
  # Nothing above would notice if the plan guard were deleted -- the units would
  # still be reused, the table would still be complete, and every check would
  # still pass, on a run whose rows were fitted to different training sets.
  # This project's own suite had exactly that hole and passed for weeks.
  #
  # The changed plan imitates the event that produced the defect: apply_buffer()
  # was fixed to protect the test set as well, and every fold lost a rim of
  # training points while every config_id kept its name.
  changed_plan <- plan
  for (j in seq_along(changed_plan$folds)) {
    tr <- changed_plan$folds[[j]]$train
    changed_plan$folds[[j]]$train <- tr[seq_len(max(1L, length(tr) - 25L))]
  }

  b6_check("B6-12", "the changed plan really is different", {
    # A negative test built on a plan that turned out to be identical passes for
    # the wrong reason, and looks exactly like one that worked.
    a <- b6_fold_membership(plan)
    b <- b6_fold_membership(changed_plan)
    n_of <- function(p) sum(vapply(p$folds, function(f) length(f$train),
                                   integer(1)))
    list(ok = !identical(a, b) && n_of(changed_plan) < n_of(plan),
         measured = sprintf("training rows %d -> %d across %d fold(s)",
                            n_of(plan), n_of(changed_plan), length(plan$folds)))
  })

  b6_check("B6-13", "check_plan_unchanged() refuses a changed plan", {
    # Read-only: the function only reads fold_plan.rds, so this cannot damage
    # the run directory it is pointed at.
    e <- tryCatch({
      check_plan_unchanged(changed_plan, resume_run_dir, resume = TRUE)
      NULL
    }, error = function(err) err)
    refused <- inherits(e, "error") &&
      grepl("NOT THE PLAN BEING ASKED FOR", conditionMessage(e), fixed = TRUE)
    list(ok = refused, measured = if (is.null(e))
      "IT RETURNED QUIETLY -- a resume onto a different split would be allowed"
      else paste("refused:", gsub("\\s+", " ",
                                  substr(conditionMessage(e), 1, 90))))
  })

  b6_check("B6-14", "dsm_train() refuses it too, before training anything", {
    # A CORRECT GUARD THAT NOTHING CALLS IS NOT A GUARD, and this project has
    # shipped one: a block that sat behind a gate and never ran. B6-13 tests the
    # function; this tests the wiring.
    #
    # ON A COPY, AND ON A COMPLETE ONE. On a copy because a broken guard would
    # write its changed plan into the run directory and overwrite the record of
    # the experiment just verified. On a COMPLETE copy because every unit is
    # then already done: if the guard is broken, the runner skips all of them
    # and this costs seconds instead of retraining the grid to prove a point.
    unlink(negative_run_dir, recursive = TRUE)
    dir.create(negative_run_dir, recursive = TRUE, showWarnings = FALSE)
    file.copy(list.files(resume_run_dir, full.names = TRUE), negative_run_dir,
              recursive = TRUE, copy.date = TRUE)
    copied_plan <- file.path(negative_run_dir, "fold_plan.rds")
    if (!file.exists(copied_plan) ||
        length(b6_done_units(negative_run_dir)) != n_expected) {
      # Without a complete copy this check is not cheap and not conclusive: the
      # runner would have units left to train, and a broken guard would train
      # them. Stop the check rather than run a different one under its name.
      stop("the copy of the run directory is incomplete", call. = FALSE)
    }
    before <- readRDS(copied_plan)

    e <- tryCatch({
      b6_train(b6_negative_run_id, resampling = changed_plan)
      NULL
    }, error = function(err) err)
    refused <- inherits(e, "error") &&
      grepl("NOT THE PLAN BEING ASKED FOR", conditionMessage(e), fixed = TRUE)
    # It must refuse BEFORE overwriting the plan on disk: a guard that stops
    # after writing has already destroyed the evidence of what was cached.
    intact <- identical(b6_fold_membership(before),
                        b6_fold_membership(readRDS(copied_plan)))
    if (refused && intact) unlink(negative_run_dir, recursive = TRUE)
    list(ok = refused && intact, measured = sprintf(
      "%s | cached fold_plan.rds left intact: %s%s",
      if (is.null(e)) "dsm_train() RESUMED ONTO THE CHANGED PLAN"
      else paste("refused:", gsub("\\s+", " ",
                                 substr(conditionMessage(e), 1, 60))),
      intact,
      if (refused && intact) "" else paste0(" | evidence kept in ",
                                            negative_run_dir)))
  })

  b6_check("B6-15", ".resumable_units() drops a relabelled config", {
    # The second guard's negative case, free because it is a pure function over
    # two tables. This is the rf_grid() run in miniature: a config_id that still
    # exists and no longer means what it meant when the unit was fitted. Resume
    # must refit it rather than read its rows back.
    bad_grid <- tune_grid
    bad_grid$base_lr[1] <- 0.0001234          # a value no config ever had
    heard <- character(0)
    kept <- withCallingHandlers(
      .resumable_units(done_before, snap_cmp, bad_grid),
      message = function(m) heard <<- c(heard, conditionMessage(m)))
    dropped <- setdiff(done_before, kept)
    want <- snap_cmp$unit_id[snap_cmp$config_id %in% tune_grid$config_id[1] &
                             snap_cmp$unit_id %in% done_before]
    list(ok = setequal(dropped, want) && length(want) > 0L &&
              any(grepl("REFITTED", heard, fixed = TRUE)),
         measured = sprintf(
      "%d cached unit(s) of %s dropped (expected %d) | said REFITTED: %s",
      length(dropped), tune_grid$config_id[1], length(want),
      any(grepl("REFITTED", heard, fixed = TRUE))))
  })

  # ── Report ─────────────────────────────────────────────────────────────────

  message("\n", strrep("-", 78))
  message(sprintf(
    "The interrupt split the %d units %d before / %d after -- %.0f%% of the run was reused,",
    n_expected, length(done_before), length(todo_before),
    100 * length(done_before) / n_expected))
  message(sprintf(
    "which at %.2f min per unit is %.1f min of training not done twice.",
    state$per_unit_min, state$per_unit_min * length(done_before)))
  message("  resumed : ", resume_run_dir)
  message("  control : ", control_run_dir)
  message(strrep("-", 78))

  b6_report(file.path(tuning_base, "capability_sweep",
                      "capability_sweep_b6.csv"))
}
