# ══════════════════════════════════════════════════════════════════════════════
# C1 -- the same grid under two validation designs
#
# WHAT B1 FOUND, AND WHY THIS EXISTS.
#
# _b1_knndm_folds.R measured, on these 3,728 points against the prediction grid,
# the median distance from a point being scored to the nearest training point:
#
#   what the map actually does        824 km
#   kNNDM folds                       837 km
#   block folds                        16 km
#
# Over the whole distribution (CAST's W1): kNNDM 128 km against the blocks'
# 1,048 km. The block design validates on a job 52x closer in than the one the
# map performs.
#
# That is a statement about GEOMETRY. It says nothing yet about whether the
# choice changes any number a reader would act on. This script answers that, by
# running the same grid, the same seeds and the same frozen test set under both
# designs and asking three questions in order of increasing importance:
#
#   1. THE LEVEL. How far does validation CCC fall under kNNDM? This is the
#      least interesting answer -- it is expected to fall, and a number that
#      falls when the task gets harder is not news.
#
#   2. THE RANKING. Do the two designs put the configs in the same order? If
#      they do, the design decides what you REPORT and not what you DEPLOY, and
#      every selection made under blocks stands. If they do not, then the block
#      CV has been choosing architectures for a job the map does not do, and
#      that is a finding about this project's own history.
#
#   3. THE WINNER. Does the config selected by one_se() change? This is the
#      question with a consequence attached, and it is the narrow case of (2)
#      that actually reaches a map.
#
# WHAT THIS SCRIPT IS NOT. It is not a test-set comparison. Both runs are scored
# on the same frozen test set, and those numbers exist, but choosing between the
# designs by which gives a better test score would be selection on the test set
# using the design as the knob. The test columns are read only to confirm the
# two runs held out the SAME points.
#
# It trains nothing. Both runs must already be on disk.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_c1_design_comparison.R")
# ══════════════════════════════════════════════════════════════════════════════

rm(list = ls())
gc()

options(width = 200)

# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/cnn_architecture.R.
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
    if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
pkgload::load_all(project_root)

suppressMessages({
  library(dplyr)
  library(readr)
  library(tibble)
})

target_label <- "soc_stock_0_5cm"

# ── Settings ──────────────────────────────────────────────────────────────────
#
# The two run ids. Name them explicitly rather than resolving "latest": this
# script's whole output is a comparison between two particular runs, and a
# comparison whose operands were guessed is not reproducible.
c1_run_block <- "soc_0_5cm_design_spatial"
c1_run_knndm <- "soc_0_5cm_design_knndm"

c1_metric <- "val_ccc"

# ── Paths ─────────────────────────────────────────────────────────────────────

tuning_base <- file.path(project_root, "outputs", "tuning",
                         "soc_stock_modeling", target_label)
report_dir  <- file.path(tuning_base, "capability_sweep", "c1_design")
create_output_dirs(report_dir)

message("\n", strrep("=", 78))
message("C1 -- the same grid under two validation designs")
message(strrep("=", 78))
message("Reads two finished tuning runs. Trains nothing; takes seconds.")
message("  block design : ", c1_run_block)
message("  kNNDM design : ", c1_run_knndm)
message(strrep("=", 78), "\n")

# ── The ledger ────────────────────────────────────────────────────────────────

.c1_checks <- list()

check_that <- function(id, what, ok, measured) {
  ok <- isTRUE(ok)
  .c1_checks[[id]] <<- tibble::tibble(
    id = id, check = what, ok = ok, measured = as.character(measured)[1])
  message(sprintf("  [%s] %-8s %-46s %s", if (ok) "PASS" else "FAIL", id, what,
                  substr(as.character(measured)[1], 1, 96)))
  invisible(ok)
}

# EVERY CHECK THIS SCRIPT PROMISES.
#
# The verdict is not "nothing failed" -- an empty ledger satisfies that.
# tests/helper.R carries the same guard, added after .report() counted a
# zero-length accumulator as a pass.
#
# c1_01..c1_05 are COMPARABILITY. If any of them fails, everything below is a
# comparison of two things that differ in more than the design, and the numbers
# mean nothing. They are checked first and stop the script.
required_checks <- c(
  "c1_01",  # both runs exist and finished
  "c1_02",  # the same configs, by hyperparameters and not by label
  "c1_03",  # the same seeds
  "c1_04",  # the same frozen test set
  "c1_05",  # the designs really are different (a guard against running one twice)
  "c1_06",  # every config has repetitions under both designs
  "c1_07",  # the noise floor is estimable under both, or the ranking is unreadable
  "c1_08",  # the ranking's own reliability is measurable, or rho means nothing
  "c1_09"   # at least one design separates a pair of configs, or there is no order
)

# ── 1. Read, and refuse anything that is not comparable ──────────────────────

read_run <- function(run_id, label) {
  d <- file.path(tuning_base, run_id)
  if (!dir.exists(d)) {
    stop("No such tuning run for the ", label, " design:\n  ", d,
         "\n  Launch it with soc_tuning_design and soc_tuning_run_id -- see the ",
         "command at the\n  foot of this file.", call. = FALSE)
  }
  by_cfg <- file.path(d, "comparison", "comparison_by_config.csv")
  allrow <- file.path(d, "comparison", "comparison_all.csv")
  if (!file.exists(by_cfg) || !file.exists(allrow)) {
    stop("The ", label, " run at\n  ", d,
         "\n  has no comparison tables. It did not finish.", call. = FALSE)
  }
  list(run_id = run_id, dir = d,
       by_config = safe_read_csv2(by_cfg),
       units     = safe_read_csv2(allrow),
       plan      = readRDS(file.path(d, "fold_plan.rds")),
       grid      = readRDS(file.path(d, "tune_grid.rds")))
}

blk <- read_run(c1_run_block, "block")
knn <- read_run(c1_run_knndm, "kNNDM")

check_that("c1_01", "both runs exist and finished", TRUE,
           sprintf("%d and %d config(s)", nrow(blk$by_config),
                   nrow(knn$by_config)))

# THE CONFIGS MUST MATCH BY HYPERPARAMETERS, NOT BY LABEL.
#
# config_id is assigned per run: "cfg_007" is a different architecture in every
# tuning run, and this project has written that warning into three files. Two
# runs drawn with the same seed and the same tune_length produce the same grid,
# but comparing on the label alone would silently survive the day one of them
# was drawn differently -- which is exactly the run whose comparison would be
# worthless.
arch_cols <- intersect(
  c("window_sizes", "conv_channels", "gate_type", "embedding_dim",
    "conv_padding", "base_lr", "dropout", "use_residual", "use_se"),
  intersect(names(blk$grid), names(knn$grid)))
fingerprint <- function(g) {
  apply(vapply(arch_cols, function(cc) {
    vapply(g[[cc]], function(z) paste(unlist(z), collapse = "x"), character(1))
  }, character(nrow(g))), 1, paste, collapse = "|")
}
fp_blk <- fingerprint(blk$grid)
fp_knn <- fingerprint(knn$grid)
check_that("c1_02", "the same configs, by hyperparameters",
           length(arch_cols) >= 3L && setequal(fp_blk, fp_knn) &&
             length(fp_blk) == length(fp_knn),
           sprintf("%d config(s) matched on %d field(s): %s",
                   length(intersect(fp_blk, fp_knn)), length(arch_cols),
                   paste(arch_cols, collapse = ", ")))

seeds_blk <- sort(unique(blk$units$seed))
seeds_knn <- sort(unique(knn$units$seed))
check_that("c1_03", "the same seeds", setequal(seeds_blk, seeds_knn),
           paste(seeds_blk, collapse = ", "))

test_blk <- sort(blk$plan$folds[[1]]$test)
test_knn <- sort(knn$plan$folds[[1]]$test)
check_that("c1_04", "the same frozen test set",
           length(test_blk) > 0L && identical(test_blk, test_knn),
           sprintf("%d point(s), identical positions", length(test_blk)))

# AND THE DESIGNS MUST ACTUALLY DIFFER. Launching the same design twice, by a
# forgotten environment variable, would produce a beautifully consistent
# comparison of nothing. The method name is the cheap check; the fold membership
# is the real one.
same_membership <- identical(
  lapply(blk$plan$folds, function(f) sort(f$validation)),
  lapply(knn$plan$folds, function(f) sort(f$validation)))
check_that("c1_05", "the two designs really are different",
           !identical(blk$plan$method, knn$plan$method) && !same_membership,
           sprintf("'%s' vs '%s'", blk$plan$method, knn$plan$method))

stop_if <- vapply(.c1_checks[c("c1_01", "c1_02", "c1_03", "c1_04", "c1_05")],
                  function(z) !z$ok, logical(1))
if (any(stop_if)) {
  stop("The two runs differ in more than the validation design, so nothing ",
       "below would mean\n  anything. See the FAILed check(s) above.",
       call. = FALSE)
}

# ── 2. The level ──────────────────────────────────────────────────────────────

mean_col <- paste0(c1_metric, "_mean")
se_col   <- paste0(c1_metric, "_se")

join_key <- tibble::tibble(config_id = blk$grid$config_id, fp = fp_blk) %>%
  dplyr::inner_join(
    tibble::tibble(config_id_knn = knn$grid$config_id, fp = fp_knn), by = "fp")

cmp <- blk$by_config %>%
  dplyr::select(config_id, dplyr::any_of(c(mean_col, se_col, "n_units"))) %>%
  dplyr::rename_with(~ paste0("blk_", .x), -config_id) %>%
  dplyr::inner_join(join_key, by = "config_id") %>%
  dplyr::inner_join(
    knn$by_config %>%
      dplyr::select(config_id, dplyr::any_of(c(mean_col, se_col, "n_units"))) %>%
      dplyr::rename_with(~ paste0("knn_", .x), -config_id) %>%
      dplyr::rename(config_id_knn = config_id),
    by = "config_id_knn") %>%
  dplyr::mutate(delta = .data[[paste0("knn_", mean_col)]] -
                  .data[[paste0("blk_", mean_col)]]) %>%
  dplyr::arrange(dplyr::desc(.data[[paste0("blk_", mean_col)]]))

check_that("c1_06", "every config has repetitions under both",
           nrow(cmp) == nrow(blk$by_config) &&
             all(cmp$blk_n_units > 0) && all(cmp$knn_n_units > 0),
           sprintf("%d config(s) paired, %d..%d units each",
                   nrow(cmp), min(c(cmp$blk_n_units, cmp$knn_n_units)),
                   max(c(cmp$blk_n_units, cmp$knn_n_units))))

nf_blk <- seed_noise_floor(blk$units, metric = c1_metric)
nf_knn <- seed_noise_floor(knn$units, metric = c1_metric)
check_that("c1_07", "the noise floor is estimable under both",
           is.finite(nf_blk$median_sd) && is.finite(nf_knn$median_sd),
           sprintf("block %.4f | kNNDM %.4f", nf_blk$median_sd, nf_knn$median_sd))

# ── 2b. SEPARABILITY: does either design tell ANY two configs apart? ─────────
#
# This is the question the ranking silently assumes has been answered. Two
# configs are separated when the gap between their means exceeds 2 SE of that
# gap -- the SEs being over (fold x seed) units, which is what by_config
# reports. It is a count, not a threshold somebody chose: 0 of 28 pairs means
# the design produced no order at all, and a rho computed on that order is
# reading noise whatever its value.
separable_pairs <- function(m, se) {
  n <- length(m)
  tot <- 0L; sep <- 0L
  for (i in seq_len(n - 1L)) {
    for (j in (i + 1L):n) {
      tot <- tot + 1L
      if (is.finite(m[i]) && is.finite(m[j]) && is.finite(se[i]) && is.finite(se[j])) {
        if (abs(m[i] - m[j]) > 2 * sqrt(se[i]^2 + se[j]^2)) sep <- sep + 1L
      }
    }
  }
  list(sep = sep, total = tot)
}

sep_blk <- separable_pairs(cmp[[paste0("blk_", mean_col)]], cmp[[paste0("blk_", se_col)]])
sep_knn <- separable_pairs(cmp[[paste0("knn_", mean_col)]], cmp[[paste0("knn_", se_col)]])

check_that("c1_09", "at least one design separates a pair of configs",
           sep_blk$sep > 0L || sep_knn$sep > 0L,
           sprintf("block %d/%d | kNNDM %d/%d pair(s) apart at 2 SE",
                   sep_blk$sep, sep_blk$total, sep_knn$sep, sep_knn$total))

message("\n-- The level: how far the number moves --")
message(sprintf("  block  %s: %.4f (best) .. %.4f (worst)", c1_metric,
                max(cmp[[paste0("blk_", mean_col)]]),
                min(cmp[[paste0("blk_", mean_col)]])))
message(sprintf("  kNNDM  %s: %.4f (best) .. %.4f (worst)", c1_metric,
                max(cmp[[paste0("knn_", mean_col)]]),
                min(cmp[[paste0("knn_", mean_col)]])))
message(sprintf("  median shift across %d config(s): %+.4f", nrow(cmp),
                stats::median(cmp$delta)))
message("  A drop here is EXPECTED and is not the finding -- kNNDM scores a ",
        "harder job.\n  What matters is whether the ORDER moved, below.")

# ── 3. The ranking ────────────────────────────────────────────────────────────
#
# Spearman, because the question is about order and not about distance. Kendall
# alongside it because tau is the one that says "how often would these two
# disagree about a PAIR of configs", which is the operational reading.

cmp$rank_blk <- rank(-cmp[[paste0("blk_", mean_col)]], ties.method = "min")
cmp$rank_knn <- rank(-cmp[[paste0("knn_", mean_col)]], ties.method = "min")

rho <- suppressWarnings(stats::cor(cmp$rank_blk, cmp$rank_knn, method = "spearman"))
tau <- suppressWarnings(stats::cor(cmp$rank_blk, cmp$rank_knn, method = "kendall"))

# THE CEILING, WITHOUT WHICH rho IS UNINTERPRETABLE.
#
# A low between-design rho reads as "the designs disagree". It reads identically
# when NEITHER design can order these configs -- and that is the live case here:
# the 3x3x3 run measured a seed noise floor of 0.031 against a 0.024 gap between
# the top two configs.
#
# So rank the configs from one seed, rank them again from another, within the
# SAME design, and correlate. That is the reproducibility of the ranking when
# only the draw changes, and the between-design rho cannot exceed it. Averaged
# over every pair of seeds, then projected by Spearman-Brown to the n-seed mean
# the comparison actually used -- because rho above is computed on means over
# n seeds, not on one.
rank_reliability <- function(units, label) {
  sds <- sort(unique(units$seed))
  if (length(sds) < 2L) return(list(r1 = NA_real_, rk = NA_real_, pairs = 0L))
  per_seed <- lapply(sds, function(sd) {
    units %>%
      dplyr::filter(.data$seed == sd) %>%
      dplyr::group_by(.data$config_id) %>%
      dplyr::summarise(m = mean(.data[[c1_metric]], na.rm = TRUE),
                       .groups = "drop")
  })
  rr <- c()
  for (i in seq_len(length(sds) - 1L)) {
    for (j in (i + 1L):length(sds)) {
      j2 <- dplyr::inner_join(per_seed[[i]], per_seed[[j]], by = "config_id",
                              suffix = c("_a", "_b"))
      if (nrow(j2) >= 3L) {
        rr <- c(rr, suppressWarnings(
          stats::cor(j2$m_a, j2$m_b, method = "spearman")))
      }
    }
  }
  rr <- rr[is.finite(rr)]
  if (length(rr) == 0L) return(list(r1 = NA_real_, rk = NA_real_, pairs = 0L))
  r1 <- mean(rr)
  k  <- length(sds)
  # Spearman-Brown. Negative or zero r1 means single-seed rankings are
  # unrelated; projecting that is meaningless, so it is not projected.
  rk <- if (r1 > 0) k * r1 / (1 + (k - 1) * r1) else r1
  list(r1 = r1, rk = rk, pairs = length(rr))
}

# How many seeds a ranking would need to reproduce itself. Spearman-Brown run
# backwards: from the k-seed reliability measured above to the single-seed r1,
# then forward to the smallest k reaching `target`. This is the number that
# decides the next science run -- more seeds, or more configs.
seeds_for <- function(rk, k, target = 0.8) {
  if (!is.finite(rk) || rk <= 0 || rk >= 1) return(NA_integer_)
  r1 <- rk / (k - (k - 1) * rk)
  if (!is.finite(r1) || r1 <= 0) return(NA_integer_)
  need <- target * (1 - r1) / (r1 * (1 - target))
  as.integer(ceiling(need))
}

rel_blk <- rank_reliability(blk$units, "block")
rel_knn <- rank_reliability(knn$units, "kNNDM")
ceiling_rho <- suppressWarnings(min(rel_blk$rk, rel_knn$rk, na.rm = TRUE))

check_that("c1_08", "the ranking's own reliability is measurable",
           is.finite(rel_blk$rk) && is.finite(rel_knn$rk),
           sprintf("block %.3f | kNNDM %.3f (over %d and %d seed pair(s))",
                   rel_blk$rk, rel_knn$rk, rel_blk$pairs, rel_knn$pairs))

message("\n-- The ranking: does the order survive the design? --")
message(sprintf("  Spearman rho between designs : %+.3f", rho))
message(sprintf("  Kendall tau                  : %+.3f   (%.0f%% of config PAIRS ordered alike)",
                tau, 100 * (tau + 1) / 2))
message(sprintf("  biggest rank move            : %d place(s)",
                max(abs(cmp$rank_blk - cmp$rank_knn))))
message(sprintf("\n  CEILING -- the same design, different seeds:"))
message(sprintf("    block  : %+.3f single seed -> %+.3f at %d seeds",
                rel_blk$r1, rel_blk$rk, length(unique(blk$units$seed))))
message(sprintf("    kNNDM  : %+.3f single seed -> %+.3f at %d seeds",
                rel_knn$r1, rel_knn$rk, length(unique(knn$units$seed))))

n_blk_seeds <- length(unique(blk$units$seed))
n_knn_seeds <- length(unique(knn$units$seed))
need_blk <- seeds_for(rel_blk$rk, n_blk_seeds)
need_knn <- seeds_for(rel_knn$rk, n_knn_seeds)

message("\n-- Separability: does either design tell two configs apart? --")
message(sprintf("  block  : %d of %d pair(s) separated at 2 SE", sep_blk$sep, sep_blk$total))
message(sprintf("  kNNDM  : %d of %d pair(s) separated at 2 SE", sep_knn$sep, sep_knn$total))
message(sprintf("  seeds for a ranking that reproduces at rho 0.80: block %s | kNNDM %s",
                ifelse(is.na(need_blk), "not estimable", as.character(need_blk)),
                ifelse(is.na(need_knn), "not estimable", as.character(need_knn))))
message(sprintf("  (this run used %d and %d)", n_blk_seeds, n_knn_seeds))

# THE GATE IS COUNTED, NOT CHOSEN. This was `ceiling_rho < 0.3` -- a threshold
# with nothing behind it, which a ceiling of 0.545 walked straight past while
# 0 of 28 kNNDM pairs were actually separated. The script then reported that
# the designs disagree, off a rho whose 5% critical value at n = 8 is 0.738.
if (sep_blk$sep == 0L && sep_knn$sep == 0L) {
  message("\n  NEITHER DESIGN SEPARATES ANY PAIR OF CONFIGS. There is no order ",
          "here to compare;\n  the between-design rho is reading seed noise. ",
          "Read the LEVEL, not the order --\n  and add seeds (see the line ",
          "above) before asking this grid to choose anything.")
} else if (sep_knn$sep == 0L || sep_blk$sep == 0L) {
  message("\n  ONLY ONE DESIGN PRODUCES AN ORDER (block ", sep_blk$sep, ", kNNDM ",
          sep_knn$sep, " separated pair(s)).\n  A rho between an order and a ",
          "coin toss is not a statement about the designs.\n  The LEVEL below ",
          "is the finding; the ranking needs more seeds first.")
} else if (is.finite(ceiling_rho)) {
  if (abs(rho) < 2 / sqrt(nrow(cmp) - 1)) {
    message("\n  The between-design rho is within 2 SE of zero (SE ~ ",
            sprintf("%.3f", 1 / sqrt(nrow(cmp) - 1)), " at ", nrow(cmp),
            " configs).\n  It does not distinguish agreement from disagreement.")
  } else if (rho >= ceiling_rho - 0.05) {
    message("\n  The between-design rho is AT the ceiling: the designs agree ",
            "as closely as one\n  design agrees with itself. The choice ",
            "changes what you report, not what you rank.")
  } else {
    message("\n  The between-design rho is BELOW the ceiling by ",
            sprintf("%.3f", ceiling_rho - rho),
            ", and both designs do separate\n  configs. The designs select ",
            "for different things by more than seed noise explains.")
  }
}

# ── 4. The winner ─────────────────────────────────────────────────────────────

pick <- function(run, label) {
  bc <- run$by_config
  if (!(se_col %in% names(bc)) || !("n_params" %in% names(bc))) {
    message("  ", label, ": one_se() needs ", se_col, " and n_params; ",
            "falling back to rank 1.")
    return(dplyr::filter(bc, rank == 1L)$config_id[1])
  }
  one_se(bc, metric = c1_metric, complexity = "n_params")$config_id[1]
}

win_blk <- pick(blk, "block")
win_knn <- pick(knn, "kNNDM")
win_knn_as_blk <- join_key$config_id[match(win_knn, join_key$config_id_knn)]

message("\n-- The winner: does the deployed config change? --")
message("  one_se under blocks : ", win_blk)
message("  one_se under kNNDM  : ", win_knn,
        "  (= ", win_knn_as_blk, " in the block run's numbering)")
if (identical(win_blk, win_knn_as_blk)) {
  message("  THE SAME ARCHITECTURE. The design decides what you report, not ",
          "what you deploy.")
} else if (sep_blk$sep == 0L || sep_knn$sep == 0L) {
  # A CHANGED WINNER IS NOT A FINDING WHEN NOTHING IS SEPARATED. one_se() takes
  # the simplest config within one SE of the best; where no pair clears 2 SE,
  # "within one SE of the best" is most of the grid, and which config comes
  # out is a draw. The run of 2026-09-19 changed winner with 0 of 28 kNNDM
  # pairs apart, and the sentence below used to call that a reason to revisit
  # every past selection.
  message("  The winner differs, BUT one of the designs separated no pair of ",
          "configs at 2 SE.\n  With the grid unseparated, one_se() is choosing ",
          "among ties: this is not evidence\n  that the designs prefer ",
          "different architectures. Add seeds, then ask again.")
} else {
  message("  A DIFFERENT ARCHITECTURE, and both designs do separate configs. ",
          "The block design\n  has been selecting for a job the map does not ",
          "do, and every selection made\n  under it is worth revisiting.")
}

# ── 5. Report ─────────────────────────────────────────────────────────────────

safe_write_csv2(cmp, file.path(report_dir, "c1_by_config.csv"))
safe_write_csv2(
  tibble::tibble(
    metric = c1_metric,
    run_block = c1_run_block, run_knndm = c1_run_knndm,
    n_configs = nrow(cmp),
    median_delta = stats::median(cmp$delta),
    spearman_rho = rho, kendall_tau = tau,
    rho_ceiling_block = rel_blk$rk, rho_ceiling_knndm = rel_knn$rk,
    rho_ceiling = ceiling_rho,
    max_rank_move = max(abs(cmp$rank_blk - cmp$rank_knn)),
    sep_pairs_block = sep_blk$sep, sep_pairs_knndm = sep_knn$sep,
    n_pairs = sep_blk$total,
    seeds_for_rho80_block = need_blk, seeds_for_rho80_knndm = need_knn,
    noise_floor_block = nf_blk$median_sd, noise_floor_knndm = nf_knn$median_sd,
    winner_block = win_blk, winner_knndm_in_block_ids = win_knn_as_blk,
    winner_changed = !identical(win_blk, win_knn_as_blk)),
  file.path(report_dir, "c1_summary.csv"))

message("\n\n-- Per config --")
print_wide(cmp, n = Inf)

# ── 6. Verdict ────────────────────────────────────────────────────────────────

# Sorted by id: c1_09 is MEASURED before c1_08 (separability gates how the
# ranking may be read), and a ledger printed out of order reads like a bug.
checks <- dplyr::arrange(dplyr::bind_rows(.c1_checks), .data$id)
safe_write_csv2(checks, file.path(report_dir, "c1_checks.csv"))

missing_checks <- setdiff(required_checks, checks$id)
failed_checks  <- checks$id[!checks$ok]

message("\n\n-- Checks --")
print_wide(checks, n = Inf)

verdict <- length(missing_checks) == 0L && length(failed_checks) == 0L

# THE BANNER SAYS WHAT THE REPORT CONCLUDED. It used to print rho against the
# ceiling and "winner CHANGED" whatever the separability said -- so the one
# line a reader quotes carried the claim the body had just refused. When the
# grid is unseparated there is no ranking and no winner to report, and the
# finding is the level.
unseparated <- sep_blk$sep == 0L || sep_knn$sep == 0L
headline <- if (unseparated) {
  short <- c(need_blk, need_knn)[c(sep_blk$sep == 0L, sep_knn$sep == 0L)]
  short <- ifelse(is.na(short), "?", as.character(short))
  sprintf("level %+.3f | NO RANKING: %d/%d and %d/%d pair(s) separated -- needs %s seeds",
          stats::median(cmp$delta), sep_blk$sep, sep_blk$total,
          sep_knn$sep, sep_knn$total, paste(short, collapse = "/"))
} else {
  sprintf("level %+.3f | rho %+.3f (ceiling %+.3f) | winner %s",
          stats::median(cmp$delta), rho, ceiling_rho,
          if (identical(win_blk, win_knn_as_blk)) "unchanged" else "CHANGED")
}
message("\n", strrep("=", 78))
message(sprintf("C1: %s | %d of %d checks | %s",
                if (verdict) "PASS" else "FAIL",
                sum(checks$ok), length(required_checks), headline))
message(strrep("=", 78))

if (length(missing_checks) > 0L) {
  message("\nCHECKS THAT NEVER RAN: ", paste(missing_checks, collapse = ", "))
}
message("\nReport: ", report_dir)

# ══════════════════════════════════════════════════════════════════════════════
# HOW THE TWO RUNS ARE LAUNCHED
#
# Same grid, same seeds, same frozen test set; the design is the only thing that
# differs. Run them one after the other -- both want the whole machine.
#
#   Sys.setenv(soc_tuning_design = "spatial",
#              soc_tuning_run_id = "soc_0_5cm_design_spatial",
#              soc_tune_length   = "24")
#   source(".../03_run_tuning.R")
#
#   Sys.setenv(soc_tuning_design = "knndm",
#              soc_tuning_run_id = "soc_0_5cm_design_knndm",
#              soc_tune_length   = "24")
#   source(".../03_run_tuning.R")
#
#   Sys.unsetenv(c("soc_tuning_design", "soc_tuning_run_id", "soc_tune_length"))
#
# The kNNDM run needs _b1_knndm_folds.R to have run first: it reads that
# script's predpoints.csv rather than rebuilding it, so that "where prediction
# happens" has exactly one definition in this project.
# ══════════════════════════════════════════════════════════════════════════════
