# ══════════════════════════════════════════════════════════════════════════════
# B2 -- the two-config branch of stage 04, exercised without touching the record
#
# WHAT HAS NEVER RUN.
#
# 04_final_model.R writes comparison/paired_by_seed.csv, which compares two
# configurations seed by seed:
#
#   d_ccc = ccc[cfg_A, seed_i] - ccc[cfg_B, seed_i]
#
# The same seed means the same initial RNG state, so the difference WITHIN a
# seed isolates the architecture from the draw. It is the same reasoning
# paired_family_test() rests on, and this project has measured what pairing is
# worth: 0.00 standard error against 0.12 unpaired, on a 0.02 shift under a 0.30
# fold spread (tests/test_resample.R).
#
# That branch has never executed. Every run so far fitted ONE config, and
# `length(selected_config_ids) == 2` was never true. Code that has never run is
# not code that works -- the gate itself was found to skip THREE configs in
# silence, by reading, not by running.
#
# WHY A COPY, AND WHY THAT IS THE WHOLE POINT.
#
# No selection rule produces a pair. one_se returns one config; rank1 returns
# one config; at any tune_length. So the two-config branch is reachable only by
# naming two ids by hand -- and naming them means freezing them, and the tuning
# run already carries a frozen selection:
#
#   comparison/selection.rds -- cfg_003, one_se on val_ccc, 2026-09-17 10:31
#
# That file is what makes the published selection optimism of +0.0000 mean
# anything: it is the evidence that nobody chose the final model after seeing
# the test set. freeze_selection() refuses to change it, deliberately.
#
# Spending that evidence to exercise a CSV writer would be an absurd trade. So
# this runs against a throwaway COPY of the tuning run, with the copy's frozen
# selection removed, and asserts at the end that the original is byte-identical
# and its timestamp has not moved. The copy is then deleted.
#
# WHY A SUBPROCESS, AND NOT source().
#
# 04_final_model.R begins with rm(list = ls()). Sourcing it from here would
# erase this script's own variables halfway through -- the paths it must clean
# up, the fingerprint it must compare against. Running the REAL script in a
# fresh R process is also the only way this checks the pipeline rather than a
# copy of the pipeline pasted into a test.
#
# The environment overrides that make it drivable are Option 3 of
# docs/b2_two_configs_decision.md, and they are worth having on their own: stage
# 04 was the only example script that could not be driven without editing it.
#
# COST. Three seeds, not ten. The branch is a pivot_wider over `seed` and four
# subtractions; it executes identically at n = 3 and n = 10. Ten seeds buy
# statistical power in a paired comparison nobody is allowed to act on, because
# it is computed on the test set. Measured on this run: cfg_003 takes about
# 3.2 min/seed in stage 04, cfg_002 is the cheapest config in the grid at
# 1.42 min/unit during tuning. Budget about 20 minutes of fitting plus a few
# for the patch store, against the 1 to 3.5 hours that 2 x 10 seeds would cost.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_b2_two_config_check.R")
# ══════════════════════════════════════════════════════════════════════════════

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
source(file.path(project_root, "R", "load_all.R"))

suppressMessages({
  library(dplyr)
  library(readr)
  library(tibble)
})

# tidyr is used namespace-qualified, for one pivot in the independent
# recomputation below. Checked here rather than at that line, which is twenty
# minutes of training away.
if (!requireNamespace("tidyr", quietly = TRUE)) {
  stop("tidyr is needed to recompute the paired differences independently. ",
       "install.packages(\"tidyr\")", call. = FALSE)
}

target_label <- "soc_stock_0_5cm"

# ── Settings ──────────────────────────────────────────────────────────────────

# The two configs. cfg_003 is the one the pipeline deploys; cfg_002 is the
# cheapest in the grid at 1.42 min/unit against cfg_001's 5.83. The branch does
# not care which two ids it pivots on, so the second one should be the cheap
# one -- this is a test of a CSV writer, not a comparison anybody may act on.
b2_config_ids <- c("cfg_003", "cfg_002")

# Three seeds. See the note at the top: the branch executes identically at ten.
b2_seeds <- c(7L, 28L, 42L)

# Which tuning run to copy. "latest" resolves the same way stage 04 does.
b2_source_run <- "latest"

# ── Paths ─────────────────────────────────────────────────────────────────────

tuning_base <- file.path(project_root, "outputs", "tuning",
                         "soc_stock_modeling", target_label)
final_base  <- file.path(project_root, "outputs", "final_model",
                         "soc_stock_modeling", target_label)

if (identical(b2_source_run, "latest")) {
  # A FINISHED run, by time. The copy this script makes has to be of a run
  # whose comparison tables exist, or stage 04 has nothing to select from.
  b2_source_run <- latest_run_dir(tuning_base, prefix = "soc_",
                                  require_file = file.path("comparison", "comparison_ranked.csv"),
                                  label = "b2 source run")
}
source_dir <- file.path(tuning_base, b2_source_run)

# The copy lives beside the real runs rather than in tempdir(), for two reasons
# that are not about disk: it is on the same volume, so copying 182 MB is a
# metadata-speed operation rather than a cross-device stream; and it is inside
# the git work tree, so freeze_selection() can still record a commit in the
# copy's own selection.rds -- which is one of the things asserted below.
#
# The name cannot start with "soc_": stage 04's "latest" and this script's own
# resolution both filter on that prefix, and a leftover copy must never be
# mistaken for a tuning run.
b2_stamp   <- format(Sys.time(), "%Y%m%d_%H%M%S")
copy_dir   <- file.path(tuning_base, paste0("b2_copy_", b2_stamp))
report_dir <- file.path(tuning_base, "capability_sweep", "b2_two_config")
create_output_dirs(report_dir)

message("\n", strrep("=", 78))
message("B2 -- the two-config branch of stage 04")
message(strrep("=", 78))
message("Estimated runtime: about 25 minutes. Stage 04 is run for real, in a ",
        "separate R\nprocess, on ", length(b2_config_ids), " configs x ",
        length(b2_seeds), " seeds. Its output appears below as it trains.")
message("Source tuning run : ", b2_source_run)
message("Throwaway copy    : ", basename(copy_dir))
message("Configs           : ", paste(b2_config_ids, collapse = ", "))
message("Seeds             : ", paste(b2_seeds, collapse = ", "))
message(strrep("=", 78), "\n")

# ── The ledger ────────────────────────────────────────────────────────────────

.b2_checks <- list()

check_that <- function(id, what, ok, measured) {
  ok <- isTRUE(ok)
  .b2_checks[[id]] <<- tibble::tibble(
    id = id, check = what, ok = ok, measured = as.character(measured)[1])
  message(sprintf("  [%s] %-8s %-52s %s", if (ok) "PASS" else "FAIL", id, what,
                  substr(as.character(measured)[1], 1, 90)))
  invisible(ok)
}

# EVERY CHECK THIS SCRIPT PROMISES, WRITTEN OUT.
#
# The verdict is not "nothing in the ledger is FALSE" -- an empty ledger
# satisfies that, and so does one missing the expensive check. tests/helper.R
# carries the same guard, and it was added because .report() counted a
# zero-length accumulator as a pass.
required_checks <- c(
  "b2_01",  # the copy carried no frozen selection, so stage 04 was free
  "b2_02",  # stage 04 exited 0
  "b2_03",  # the copy's record names BOTH configs
  "b2_04",  # ...and says the rule was "manual", not one_se
  "b2_05",  # paired_by_seed.csv exists at all -- the branch ran
  "b2_06",  # one row per seed, and they are the seeds that were asked for
  "b2_07",  # it carries the four difference columns
  "b2_08",  # all four differences ARE cfg_A - cfg_B, recomputed independently
  "b2_09",  # the original selection.rds is byte-identical
  "b2_10",  # ...and its modification time never moved
  "b2_11",  # the copy is gone
  "b2_12",  # the throwaway final model is gone
  "b2_13"   # the copy's frozen record carries a git commit
)

# ── Step 0: preconditions ─────────────────────────────────────────────────────

if (!dir.exists(source_dir)) stop("No such tuning run: ", source_dir, call. = FALSE)

orig_sel_path <- file.path(source_dir, "comparison", "selection.rds")
if (!file.exists(orig_sel_path)) {
  stop("The source run has no frozen selection at\n  ", orig_sel_path,
       "\n  This script exists to prove that file survives untouched, so ",
       "there is nothing\n  to prove if it does not exist. Run ",
       "04_final_model.R against this tuning run first.", call. = FALSE)
}

grid <- readRDS(file.path(source_dir, "tune_grid.rds"))
missing_cfg <- setdiff(b2_config_ids, grid$config_id)
if (length(missing_cfg) > 0L) {
  stop("config_id(s) not in this run's grid: ", paste(missing_cfg, collapse = ", "),
       "\n  config ids are PER RUN -- the same \"cfg_002\" is a different ",
       "architecture in\n  every tuning run. Available: ",
       paste(utils::head(grid$config_id, 12), collapse = ", "), call. = FALSE)
}
if (length(b2_config_ids) != 2L) {
  stop("B2 tests the TWO-config branch; ", length(b2_config_ids), " given.",
       call. = FALSE)
}

# THE FINGERPRINT, TAKEN BEFORE ANYTHING ELSE.
#
# Bytes and mtime both, because they fail differently: a rewrite with identical
# content moves the mtime, and a same-second rewrite with different content
# moves the bytes. The claim being defended is that stage 04, run against a
# copy, cannot reach the original at all -- so neither may move.
orig_bytes <- readBin(orig_sel_path, "raw", file.info(orig_sel_path)$size)
orig_mtime <- file.info(orig_sel_path)$mtime
orig_sel   <- readRDS(orig_sel_path)
message("Original frozen selection: ", paste(orig_sel$config_id, collapse = ", "),
        " (", orig_sel$rule, ", ", format(orig_mtime), ")")
message("  ", length(orig_bytes), " bytes fingerprinted\n")

finals_before <- list.dirs(final_base, recursive = FALSE, full.names = FALSE)

# ── Everything from here can leave rubbish behind, so it is wrapped ───────────
#
# A failed assertion must not cost 182 MB of copy plus a final model on disk;
# a check nobody can afford to re-run is a check nobody re-runs. The cleanup
# below runs whether the body succeeded or threw, and the verdict is reported
# afterwards.

b2_error <- NULL
final_dir_made <- NA_character_

tryCatch({

  # ── Step 1: the copy, without its frozen selection ──────────────────────────
  message("-- Copying the tuning run (182 MB or so) --")
  t0 <- Sys.time()
  dir.create(copy_dir, recursive = TRUE, showWarnings = FALSE)
  ok_copy <- all(file.copy(list.files(source_dir, full.names = TRUE), copy_dir,
                           recursive = TRUE, copy.date = TRUE))
  if (!ok_copy) stop("The copy did not complete into ", copy_dir, call. = FALSE)
  message("   copied in ", round(difftime(Sys.time(), t0, units = "mins"), 1),
          " min")

  copy_sel <- file.path(copy_dir, "comparison", "selection.rds")
  if (file.exists(copy_sel)) file.remove(copy_sel)
  check_that("b2_01", "the copy carries no frozen selection",
             !file.exists(copy_sel),
             if (file.exists(copy_sel)) "STILL PRESENT" else "removed")

  # ── Step 2: run the real stage 04, in its own process ───────────────────────
  #
  # The environment is what drives it (Option 3). Sys.setenv() reaches the child
  # because system2() inherits this process's environment, and it survives
  # 04's rm(list = ls()), which a workspace variable would not.
  Sys.setenv(soc_final_tuning_run_id = basename(copy_dir),
             soc_final_config_ids    = paste(b2_config_ids, collapse = ","),
             soc_final_seeds         = paste(b2_seeds, collapse = ","))

  rscript <- file.path(R.home("bin"), "Rscript")
  stage04 <- file.path(project_root, "examples", "soc_stock_0_5cm",
                       "04_final_model.R")
  message("\n-- Running 04_final_model.R in a separate process --")
  message("   ", rscript)
  message("   its output follows, live\n")

  # NOT --vanilla. It implies --no-environ and --no-init-file, so the child
  # would run with different library paths and a different startup from the one
  # Cassio gets when he sources this stage himself. The whole value of driving
  # the real script is that it runs under the real conditions; a check that
  # runs it under different ones is checking something else. --no-save
  # --no-restore only keeps a workspace from travelling in either direction.
  #
  # (Today there is no .Rprofile and no .Renviron here, so --vanilla would have
  # worked -- by accident of the current setup, and silently broken the day one
  # appeared.)
  t0 <- Sys.time()
  status <- system2(rscript, c("--no-save", "--no-restore", shQuote(stage04)))
  mins <- round(difftime(Sys.time(), t0, units = "mins"), 1)

  check_that("b2_02", "stage 04 exited 0", identical(as.integer(status), 0L),
             paste0("status ", status, ", ", mins, " min"))
  if (!identical(as.integer(status), 0L)) {
    stop("Stage 04 failed; nothing below would mean anything.", call. = FALSE)
  }

  # ── Step 3: find what it produced ───────────────────────────────────────────
  finals_after <- list.dirs(final_base, recursive = FALSE, full.names = FALSE)
  made <- setdiff(finals_after, finals_before)
  if (length(made) != 1L) {
    stop("Expected exactly one new final_model directory, found ", length(made),
         if (length(made)) paste0(": ", paste(made, collapse = ", ")) else "",
         call. = FALSE)
  }
  final_dir_made <- file.path(final_base, made)
  message("\nStage 04 wrote: ", made)

  # ── Step 4: the record the copy froze ───────────────────────────────────────
  copy_rec <- readRDS(copy_sel)
  check_that("b2_03", "the copy's record names both configs",
             setequal(copy_rec$config_id, b2_config_ids),
             paste(copy_rec$config_id, collapse = ", "))
  # "manual" is the point: naming ids by hand is not a rule, and the record has
  # to say which happened. Before 08c5b78 this field always read "one_se",
  # because the is.null() test that chose it could never be true.
  check_that("b2_04", "...and the rule reads 'manual'",
             identical(copy_rec$rule, "manual"), copy_rec$rule)

  # THE ONE RUN THAT EXERCISES THE git -C FIX END TO END.
  #
  # freeze_selection() used to resolve its commit against the PROCESS's working
  # directory, so the field was NA whenever R sat outside the repo. B2 is the
  # only place in the project where that path runs in a child process writing
  # into a directory inside the work tree -- so if this is not asserted, a green
  # B2 reads as covering that fix and covers none of it.
  #
  # It is also why the copy lives under outputs/ rather than in tempdir(): a
  # copy outside the work tree could not carry a commit at all.
  check_that("b2_13", "the copy's record carries a git commit",
             !is.na(copy_rec$git_commit) && nzchar(copy_rec$git_commit),
             if (is.na(copy_rec$git_commit)) "NA" else copy_rec$git_commit)

  # ── Step 5: the branch under test ───────────────────────────────────────────
  paired_path <- file.path(final_dir_made, "comparison", "paired_by_seed.csv")
  check_that("b2_05", "paired_by_seed.csv exists -- the branch ran",
             file.exists(paired_path), basename(paired_path))
  if (!file.exists(paired_path)) {
    stop("The branch this script exists to exercise did not run.", call. = FALSE)
  }

  paired <- safe_read_csv2(paired_path)
  check_that("b2_06", "one row per seed, and the seeds asked for",
             nrow(paired) == length(b2_seeds) &&
               setequal(as.integer(paired$seed), b2_seeds),
             paste0(nrow(paired), " row(s): ",
                    paste(sort(paired$seed), collapse = ", ")))

  want_cols <- c("d_ccc", "d_mae", "d_rmse", "d_mqi")
  check_that("b2_07", "the four difference columns are present",
             all(want_cols %in% names(paired)),
             paste(intersect(want_cols, names(paired)), collapse = ", "))

  # THE ONE CHECK THAT COULD FAIL FOR A REASON WORTH KNOWING.
  #
  # Everything above says the file exists and has the right shape. This says the
  # numbers in it are the subtraction they claim to be, recomputed from
  # all_seed_results without going through the pivot -- so a pivot that paired
  # the wrong rows, or subtracted in the other direction, is visible here and
  # nowhere else. The direction matters: the header of that CSV names c1 - c2,
  # and a sign flip would invert every conclusion drawn from it.
  summ <- readRDS(file.path(final_dir_made, "comparison",
                            "final_run_summary.rds"))
  asr  <- summ$all_seed_results
  c1   <- summ$selected_config_ids[1]
  c2   <- summ$selected_config_ids[2]
  # ALL FOUR COLUMNS, not just d_ccc. 04_final_model.R computes them as four
  # near-identical paste0() lines, which is exactly the shape a copy-paste sign
  # inversion or a swapped config id survives in -- and b2_07 only asserts the
  # four NAMES are present, which no arithmetic error can fail. Checking one of
  # four while the ledger says "the differences" would have been the overclaim
  # this script's own required_checks exists to prevent.
  # THE DIRECTION IS ONLY MEANINGFUL IF c1 IS THE CONFIG THIS SCRIPT ASKED FOR
  # FIRST. Stage 04 takes c1 <- selected_config_ids[1] and so does this, so a
  # subtraction in the wrong direction is catchable while a mis-ORDERED list
  # would be agreed upon by both. Tying the order back to b2_config_ids is what
  # closes that loop.
  order_ok <- identical(as.character(summ$selected_config_ids), b2_config_ids)

  # seed may arrive as text: paired_by_seed.csv is read back through
  # read_csv2(), and a column that parses as character joins to nothing against
  # all_seed_results' integer. The join would return 0 rows and b2_08 would fail
  # for a reason that is not the defect it is looking for.
  metrics <- c("ccc", "mae", "rmse", "mqi")
  paired$seed <- as.integer(paired$seed)
  asr$seed    <- as.integer(asr$seed)
  recomputed <- asr %>%
    dplyr::filter(.data$config_id %in% c(c1, c2)) %>%
    dplyr::select(seed, config_id, dplyr::all_of(metrics)) %>%
    tidyr::pivot_wider(names_from = config_id, values_from = dplyr::all_of(metrics))
  for (m in metrics) {
    recomputed[[paste0("r_", m)]] <-
      recomputed[[paste0(m, "_", c1)]] - recomputed[[paste0(m, "_", c2)]]
  }
  recomputed <- dplyr::select(recomputed, seed, dplyr::starts_with("r_"))
  cmp <- dplyr::inner_join(
    dplyr::select(paired, seed, dplyr::all_of(paste0("d_", metrics))),
    recomputed, by = "seed")
  worst <- if (nrow(cmp) > 0L) {
    max(vapply(metrics, function(m) {
      max(abs(cmp[[paste0("d_", m)]] - cmp[[paste0("r_", m)]]))
    }, numeric(1)))
  } else NA_real_
  check_that("b2_08",
             sprintf("d_* are %s minus %s, all four recomputed", c1, c2),
             order_ok && nrow(cmp) == length(b2_seeds) &&
               is.finite(worst) && worst < 1e-9,
             sprintf("worst |difference| %.3e over %d seed(s) x %d metric(s)%s",
                     worst, nrow(cmp), length(metrics),
                     if (order_ok) "" else "  [CONFIG ORDER DID NOT SURVIVE]"))

  safe_write_csv2(cmp, file.path(report_dir, "b2_paired_recomputed.csv"))

}, error = function(e) {
  b2_error <<- conditionMessage(e)
  message("\n!! B2 stopped: ", conditionMessage(e))
})

# ── Cleanup, whether the body succeeded or not ────────────────────────────────

Sys.unsetenv(c("soc_final_tuning_run_id", "soc_final_config_ids",
               "soc_final_seeds"))

message("\n-- Cleanup --")
if (dir.exists(copy_dir)) {
  unlink(copy_dir, recursive = TRUE, force = TRUE)
}
check_that("b2_11", "the throwaway copy is gone", !dir.exists(copy_dir),
           basename(copy_dir))

# THE TARGET IS DERIVED FROM DISK, NOT FROM THE HAPPY PATH.
#
# final_dir_made is only assigned after stage 04 exits 0 and after the "exactly
# one new directory" check. But 04_final_model.R creates its output directory at
# line 197, BEFORE its own validations at 201-203 and long before any training
# -- so any failure past that line leaves a final_<timestamp> behind while
# final_dir_made is still NA. This check used to report "none was made" and PASS
# in exactly that case, with the directory sitting there.
#
# That leftover is not inert: 05_predict_spatial.R resolves final_run_id =
# "latest" by globbing ^final_ and taking the newest, so a half-built directory
# from a failed B2 would become the model the next map is drawn from.
#
# finals_before was captured outside the tryCatch, so the difference is
# computable here whatever happened. Nothing outside that set is touched --
# a concurrent stage 04 of Cassio's must never be swept up by a cleanup.
leftover <- setdiff(
  list.dirs(final_base, recursive = FALSE, full.names = FALSE), finals_before)
for (d in leftover) {
  unlink(file.path(final_base, d), recursive = TRUE, force = TRUE)
}
still_there <- setdiff(
  list.dirs(final_base, recursive = FALSE, full.names = FALSE), finals_before)
check_that("b2_12", "every final_model directory this run created is gone",
           length(still_there) == 0L,
           if (length(leftover) == 0L) "none was made"
           else if (length(still_there) == 0L)
             paste0("removed: ", paste(leftover, collapse = ", "))
           else paste0("STILL PRESENT: ", paste(still_there, collapse = ", ")))

# ── The claim this whole design exists to defend ──────────────────────────────

# THE WORST OUTCOME MUST BE REPORTED, NOT THROWN.
#
# If the original has been deleted, readBin() on it errors -- and this code sits
# outside the tryCatch, so the script would die here: no ledger written, no
# verdict printed, and the guidance below naming b2_09/b2_10 as the serious
# failure never reached. The single catastrophe this script exists to detect
# would have surfaced as a cryptic connection error.
if (!file.exists(orig_sel_path)) {
  check_that("b2_09", "the original frozen selection is byte-identical",
             FALSE, "DELETED")
  check_that("b2_10", "...and its modification time never moved",
             FALSE, "DELETED")
} else {
  now_bytes <- readBin(orig_sel_path, "raw", file.info(orig_sel_path)$size)
  now_mtime <- file.info(orig_sel_path)$mtime
  check_that("b2_09", "the original frozen selection is byte-identical",
             identical(orig_bytes, now_bytes),
             paste0(length(now_bytes), " bytes"))
  # EXACTLY EQUAL, and all.equal() is the trap here rather than the tool.
  #
  # all.equal.numeric switches to a RELATIVE comparison once mean(abs(target))
  # exceeds the tolerance. An mtime as a number is ~1.79e9 and the default
  # tolerance is 1.49e-8, so the effective window was 1.49e-8 * 1.79e9 = 26.7
  # SECONDS. A check named "never moved" that tolerates half a minute of
  # movement is worse than no check, and this is precisely the failure b2_09
  # cannot see: a rewrite with identical content moves only the timestamp.
  #
  # file.info()$mtime read twice with nothing writing in between returns the
  # same double, so identical() is the right instrument.
  check_that("b2_10", "...and its modification time never moved",
             identical(as.numeric(orig_mtime), as.numeric(now_mtime)),
             format(now_mtime))
}

# ── Verdict ───────────────────────────────────────────────────────────────────

checks <- dplyr::bind_rows(.b2_checks)
safe_write_csv2(checks, file.path(report_dir, "b2_checks.csv"))

missing_checks <- setdiff(required_checks, checks$id)
failed_checks  <- checks$id[!checks$ok]

message("\n-- Checks --")
print_wide(checks, n = Inf)

verdict <- length(missing_checks) == 0L && length(failed_checks) == 0L

message("\n", strrep("=", 78))
message(sprintf("B2: %s | %d of %d checks present and true",
                if (verdict) "PASS" else "FAIL",
                sum(checks$ok), length(required_checks)))
message(strrep("=", 78))

if (length(missing_checks) > 0L) {
  message("\nCHECKS THAT NEVER RAN: ", paste(missing_checks, collapse = ", "))
  message("  A check that did not run is not a check that passed. The body ",
          "above stopped\n  early -- see the message, if any.")
}
if (length(failed_checks) > 0L) {
  message("\nFAILED: ", paste(failed_checks, collapse = ", "))
  message("  b2_09 and b2_10 failing is the serious one: it would mean stage ",
          "04, pointed at\n  a copy, reached the original frozen selection. ",
          "Everything else is the CSV\n  writer being wrong about its own ",
          "output.")
}
if (!is.null(b2_error)) {
  message("\nThe run stopped with: ", b2_error)
}

message("\nReport: ", report_dir)
