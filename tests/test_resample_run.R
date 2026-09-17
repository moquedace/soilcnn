# Smoke test: the whole resampling pipeline, end to end, on tiny data
#
# Every OTHER test here checks one function. This one checks that they are
# WIRED to each other: a synthetic patch store on disk, a real fold plan, real
# torch models trained for two epochs, and the real comparison tables coming
# out the far end.
#
# Why it exists: run_cnn_resample() is called once, by a script that costs
# hours of CPU. A typo in the fold loop, a column renamed in one place and not
# another, a checkpoint written under a name nothing looks for -- all of those
# survive every unit test in this directory and only show up after the
# expensive run. This file finds them in seconds.
#
# Verifies:
#   1. a store written by save_patch_window() loads and trains, unmodified
#   2. holdout + n_seeds: one row per (config, seed), unit_id != config_id
#   3. THE SEED IS SHARED ACROSS CONFIGS -- every config sees the same seeds
#   4. two folds produce two caches and twice the rows, all in ONE run dir
#   5. every unit has its own checkpoint, under the name the 99 looks for
#   6. comparison_by_config.csv aggregates to one row per config
#   7. the per-fold scaling really differs between folds (fold-dependent
#      scaling is the reason the patches are stored raw at all)
#   8. resume skips finished units instead of retraining them
#   9. a config that cannot build is recorded as failed, and does not abort
#      the other configs
#
# Run: source("D:/.../tests/test_resample_run.R")    (CPU, ~1 min)

suppressMessages({
  library(torch)
  library(coro)
  library(tibble)
  library(dplyr)
  library(readr)
})

# -- project root: works under source() in the console AND under Rscript ------

root <- (function() {
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
  for (d in cand) {
    for (up in c(".", "..")) {
      r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
      if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
    }
  }
  stop("Project root not found. setwd() to the deep_learning_caret root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

source(file.path(root, "tests", "helper.R"))
for (m in c("utils", "patches", "preprocess", "dataset", "metrics",
            "cnn_architecture", "tune_grid", "resample", "train_cnn")) {
  source(file.path(root, "R", paste0(m, ".R")))
}

ok <- c()

# -- a synthetic patch store, written the way stage 02 writes one ------------

set.seed(11)
n_pts <- 96L
n_ch  <- 4L
win   <- 3L
preds <- paste0("pred_", seq_len(n_ch))

# Two spatial clusters, so a fold plan has something real to separate.
x <- rep(c(0, 30000), each = n_pts / 2L) + runif(n_pts, -300, 300)
y <- rep(c(0, 20000), each = n_pts / 2L) + runif(n_pts, -300, 300)

arr <- array(rnorm(n_pts * n_ch * win * win, mean = 10, sd = 3),
             dim = c(n_pts, n_ch, win, win))
# Give channel 1 real signal, so training has something to learn and the loss
# is not pure noise -- a model that cannot move at all would hide wiring bugs
# behind a flat history.
centre <- (win + 1L) %/% 2L
y_true <- 2 * arr[, 1, centre, centre] + rnorm(n_pts, sd = 0.5)
y_true <- y_true - min(y_true) + 1          # positive, so log1p is defined

meta <- tibble::tibble(
  profile_id       = seq_len(n_pts),
  sample_id        = seq_len(n_pts),
  x                = x,
  y                = y,
  target_native    = y_true,
  target_transform = log1p(y_true)
)

# Limpeza na ENTRADA, nao so na saida.
#
# tempdir() sobrevive entre source() na mesma sessao. Se uma execucao anterior
# abortou antes do unlink do fim, a proxima RETOMA os modelos velhos em vez de
# treinar: o resume funcionando exatamente como deveria, contra o teste. Foi o
# que aconteceu -- quatro assercoes falharam medindo modelos de outra rodada.
# Um teste que retoma o proprio run anterior nao e reproduzivel.
store_dir <- file.path(tempdir(), "dlc_smoke_store")
out_root  <- file.path(tempdir(), "dlc_smoke_out")
unlink(store_dir, recursive = TRUE)
unlink(out_root,  recursive = TRUE)
dir.create(store_dir, recursive = TRUE, showWarnings = FALSE)

invisible(save_patch_window(arr, store_dir, win))
readr::write_csv2(meta, file.path(store_dir, "patch_meta.csv"))
saveRDS(tibble::tibble(
  scaling_applied       = FALSE,
  predictor_cols_final  = paste(preds, collapse = ";"),
  n_channels            = n_ch,
  windows_extracted     = as.character(win),
  n_points_valid        = n_pts
), file.path(store_dir, "patch_manifest.rds"))

store <- load_patch_store(store_dir, verbose = FALSE)

# Point values: the centre cell of each patch, which is what stage 01 holds.
points <- meta
for (j in seq_len(n_ch)) points[[preds[j]]] <- arr[, j, centre, centre]

type_table <- tibble::tibble(
  predictor     = preds,
  is_dummy      = FALSE,
  is_percentage = FALSE
)

cat("  synthetic store     : ", n_pts, " points x ", n_ch, " channels, window ",
    win, "x", win, "\n", sep = "")

# -- a two-config grid, as small as a CNN can be ------------------------------

grid <- make_manual_tune_grid(
  window_sizes  = list(c(win)),
  conv_channels = list(c(4L), c(8L)),
  embedding_dim = 8L,
  embed_pool    = "gap",
  use_residual  = FALSE,
  use_se_block  = FALSE,
  dropout       = 0.0,
  base_lr       = 1e-2,
  # 1e-4 de proposito: readr::write_csv2() grava isso como "1e-04", que o
  # read_csv2() com decimal virgula NAO parseia como numero -- devolve texto, e
  # o bind_rows da retomada aborta. Com um valor "redondo" aqui o teste
  # atravessaria o bug sem ve-lo.
  weight_decay  = 1e-4,
  batch_size    = 16L,
  loss_fn       = "mse"
)
ok["grid_has_two_configs"] <- nrow(grid) == 2L

device    <- torch::torch_device("cpu")
# 30 epocas, nao 2. Com 2 o modelo preve praticamente uma constante, a
# variancia das predicoes e zero e o CCC sai NaN -- e ai a agregacao e o piso
# de ruido nao sao exercitados de verdade, so atravessados. O alvo aqui e
# linear no centro do canal 1 de proposito, entao 30 epocas bastam e ainda
# custam segundos.
train_args <- list(n_epochs = 30L, patience = 30L, print_every = 100L,
                   augment = FALSE)

quiet_run <- function(...) {
  suppressWarnings(suppressMessages(
    do.call(run_cnn_resample, c(list(...), train_args))
  ))
}

# -- 1. holdout with 2 seeds --------------------------------------------------

res1 <- quiet_run(
  tune_grid = grid, store = store, points = points, type_table = type_table,
  plan = holdout(store$meta, validation_frac = 0.2, test_frac = 0.2, seed = 3L),
  transform = expm1, output_dir = out_root,
  device = device, run_id = "smoke_holdout", n_seeds = 2L,
  release_store = FALSE
)

cmp1 <- res1$comparison
ok["holdout_rows_configs_x_seeds"] <- nrow(cmp1) == 2L * 2L
ok["unit_id_present"]  <- "unit_id" %in% names(cmp1)
ok["unit_id_distinct"] <- dplyr::n_distinct(cmp1$unit_id) == nrow(cmp1)
ok["unit_id_names_the_fold_and_seed"] <- all(grepl("_f1_s[12]$", cmp1$unit_id))
ok["all_units_succeeded"] <- all(cmp1$status == "success")
# Metrica finita e, por si so, uma checagem de fiacao: NaN aqui significa que
# as predicoes chegaram constantes ao calculo de CCC.
ok["metrics_are_finite"] <- all(is.finite(cmp1$val_ccc)) &&
  all(is.finite(cmp1$val_mae))

# 3. THE assertion: both configs were trained under the SAME seeds. If the
# seed tracked the config, comparing the two would compare hyperparameters
# AND luck, with no way to separate them.
per_cfg <- tapply(cmp1$seed, cmp1$config_id, function(z) paste(sort(z), collapse = ","))
ok["seed_set_shared_across_configs"] <- length(unique(per_cfg)) == 1L
ok["two_distinct_seeds"] <- dplyr::n_distinct(cmp1$seed) == 2L

# ── The test set is not scored during tuning ─────────────────────────────────
#
# THIS PLAN HAS A TEST SET (test_frac = 0.2) and the columns must still be NA.
# That is the whole point: a frozen test set stops being frozen once its score
# sits in the tuning table beside the validation score, because a human reading
# the table selects on it without any argmax being involved. Stage 04 scores it
# once, on the config chosen without it.
#
# The distinction this checks is the one the previous assertion could NOT: the
# older test verified NA when the plan HAD NO TEST ROLE, which would pass just
# as well if evaluate_test did nothing at all.
ok["test_columns_exist_when_not_scored"] <- "test_ccc" %in% names(cmp1)
ok["test_is_not_scored_by_default"]      <- all(is.na(cmp1$test_ccc))
ok["the_plan_really_had_a_test_set"]     <- length(res1$plan$folds[[1]]$test) > 0L
ok["validation_is_still_scored"]         <- all(is.finite(cmp1$val_ccc))

# n_params must reach the comparison table, or one_se() has no notion of
# "simplest" on a CNN run. It is a config property, so it is constant within
# a config and positive for every unit.
ok["cnn_records_n_params"] <- "n_params" %in% names(cmp1) &&
  all(is.finite(cmp1$n_params)) && all(cmp1$n_params > 0)
ok["n_params_constant_within_config"] <-
  all(tapply(cmp1$n_params, cmp1$config_id, dplyr::n_distinct) == 1L)
ok["n_params_survives_to_by_config"] <- "n_params" %in% names(res1$by_config)

# The test rows must not reach disk either. evaluate_test = FALSE blanked the
# metrics and left the per-unit predictions writing test residuals to a CSV --
# a second door to the set score_test_grid() exists to keep shut, and the only
# one with no ordering check on it. Two doors and one lock is one door.
# EVERY artefact, not the one that was remembered.
#
# This block used to read predictions/ only, and it passed while
# metrics/*_perf.csv carried a `test` row for all 27 units of the real stage-03
# run -- the per-unit test CCC in plain text, in the very run whose selection
# was later frozen. A test that checks one of three doors reports that the house
# is locked.
#
# So the check now walks EVERY csv the runner writes that has a dataset_role
# column, discovered by reading them, not by naming the two a person recalled.
.roles_on_disk <- function(run_dir) {
  f <- list.files(run_dir, pattern = "[.]csv$", recursive = TRUE,
                  full.names = TRUE)
  f <- f[!grepl("comparison", f, fixed = TRUE)]   # the table is blanked, not filtered
  out <- lapply(f, function(p) {
    d <- suppressMessages(readr::read_csv2(p, show_col_types = FALSE))
    if ("dataset_role" %in% names(d)) unique(as.character(d$dataset_role)) else NULL
  })
  list(files = f[!vapply(out, is.null, logical(1))],
       roles = unique(unlist(out)))
}

.on_disk <- .roles_on_disk(res1$run_dir)
ok["some_artefact_carries_a_role"] <- length(.on_disk$files) > 0L
ok["no_artefact_carries_a_test_row"] <- !("test" %in% .on_disk$roles)
ok["validation_rows_are_still_written"] <- "validation" %in% .on_disk$roles

# ...and the rule itself, directly: it must drop test, keep the rest, and be a
# no-op both when evaluate_test is TRUE and when the frame has no role column.
ok["drop_test_rows_drops_only_test"] <- {
  d <- tibble::tibble(dataset_role = c("train", "validation", "test"), v = 1:3)
  identical(.drop_test_rows(d, FALSE)$dataset_role, c("train", "validation"))
}
ok["drop_test_rows_is_a_no_op_when_evaluating"] <- {
  d <- tibble::tibble(dataset_role = c("train", "test"), v = 1:2)
  identical(.drop_test_rows(d, TRUE), d)
}
ok["drop_test_rows_ignores_a_roleless_frame"] <- {
  d <- tibble::tibble(a = 1:3)
  identical(.drop_test_rows(d, FALSE), d)
}
ok["drop_test_rows_survives_null"] <- is.null(.drop_test_rows(NULL, FALSE))

# 5. every unit has the checkpoint the 99 looks for
ckpt <- file.path(res1$run_dir, "models", paste0(cmp1$unit_id, "_best.pt"))
ok["every_unit_has_checkpoint"] <- all(file.exists(ckpt))

# 6. aggregation collapses seeds into one row per config
ok["by_config_one_row_each"] <- nrow(res1$by_config) == 2L
ok["by_config_counts_units"]  <- all(res1$by_config$n_units == 2L)
ok["by_config_has_spread"]    <- "val_ccc_sd" %in% names(res1$by_config) &&
  all(is.finite(res1$by_config$val_ccc_sd))
ok["by_config_file_written"]  <- file.exists(
  file.path(res1$run_dir, "comparison", "comparison_by_config.csv"))

# the noise floor is estimable precisely because there were two seeds
nf <- seed_noise_floor(cmp1)
ok["noise_floor_estimable"] <- nf$n_comparable == 2L && is.finite(nf$median_sd)

cat("  holdout x 2 seeds   : ", nrow(cmp1), " units | sd between seeds ",
    sprintf("%.4f", nf$median_sd), "\n", sep = "")

# -- 8. resume: finished units are skipped, not retrained --------------------
#
# Same run_id, same grid, SAME PLAN. Nothing should train again, and the table
# must come back with the same rows -- not doubled, not empty.
#
# The plan used to be `holdout(store$meta)` here -- the defaults, 0.15/0.15 and
# seed 42 -- while the run being resumed used 0.2/0.2 and seed 3. So the test
# named "resume skips finished units" was resuming onto a DIFFERENT SPLIT, and
# passed anyway, because skipping is keyed on unit_id. The defect the guard
# exists to catch was hiding inside the test meant to protect against it.
#
# check_plan_unchanged() now refuses that, which is how this was found.

same_plan <- holdout(store$meta, validation_frac = 0.2, test_frac = 0.2,
                     seed = 3L)

mtimes_before <- file.mtime(ckpt)
res1b <- quiet_run(
  tune_grid = grid, store = store, points = points, type_table = type_table,
  plan = same_plan, transform = expm1, output_dir = out_root,
  device = device, run_id = "smoke_holdout", n_seeds = 2L,
  release_store = FALSE
)
ok["resume_keeps_row_count"] <- nrow(res1b$comparison) == nrow(cmp1)
ok["resume_retrains_nothing"] <- identical(file.mtime(ckpt), mtimes_before)

# ...and resuming onto a DIFFERENT split must be refused, end to end. The unit
# test of check_plan_unchanged() lives in test_resample.R; this one proves the
# runner actually calls it, which is the half that silently rots.
ok["runner_refuses_a_changed_plan"] <- inherits(
  try(quiet_run(
        tune_grid = grid, store = store, points = points,
        type_table = type_table,
        plan = holdout(store$meta, validation_frac = 0.3, test_frac = 0.1,
                       seed = 11L),
        transform = expm1, output_dir = out_root, device = device,
        run_id = "smoke_holdout", n_seeds = 2L, release_store = FALSE),
      silent = TRUE), "try-error")

# ...and the checkpoints must still be there afterwards: a refusal that deleted
# the work it refused to mix would be worse than the mixing.
ok["the_refusal_destroys_nothing"] <-
  all(file.exists(ckpt)) && identical(file.mtime(ckpt), mtimes_before)

# RETOMADA PARCIAL -- o caso que a retomada completa nao exercita.
#
# Com tudo pronto, toda unidade e pulada ANTES do bind_rows, e a tabela relida
# nunca encontra uma linha nova. Basta uma unidade faltando para os dois
# caminhos se cruzarem, e foi exatamente ai que o tipo adivinhado pelo
# read_csv2 derrubou o run (`window_sizes` "3" lido como numero).
#
# Aqui: um run de 1 semente, retomado pedindo 2. A semente 1 e pulada, a 2
# treina, e as linhas tem que empilhar.
res_p1 <- quiet_run(
  tune_grid = grid, store = store, points = points, type_table = type_table,
  plan = holdout(store$meta, validation_frac = 0.2, test_frac = 0.2, seed = 3L),
  transform = expm1, output_dir = out_root,
  device = device, run_id = "smoke_partial", n_seeds = 1L,
  release_store = FALSE
)
res_p2 <- quiet_run(
  tune_grid = grid, store = store, points = points, type_table = type_table,
  plan = holdout(store$meta, validation_frac = 0.2, test_frac = 0.2, seed = 3L),
  transform = expm1, output_dir = out_root,
  device = device, run_id = "smoke_partial", n_seeds = 2L,
  release_store = FALSE
)
ok["partial_resume_appends"] <- nrow(res_p2$comparison) == 4L
ok["partial_resume_all_success"] <- all(res_p2$comparison$status == "success")

# TODA coluna volta com o tipo que tinha -- nao so as que eu lembrei de listar.
#
# A primeira tentativa de conserto forcava uma LISTA de colunas a character, e
# a lista estava incompleta: weight_decay quebrou logo depois, num run de 3
# horas. Comparar o conjunto INTEIRO de classes e a unica assercao que nao
# depende de eu ter lembrado da coluna certa.
cls_before <- vapply(res_p1$comparison, function(z) class(z)[1], character(1))
cls_after  <- vapply(res_p2$comparison, function(z) class(z)[1], character(1))
shared     <- intersect(names(cls_before), names(cls_after))
ok["partial_resume_preserves_every_type"] <-
  identical(cls_before[shared], cls_after[shared])
ok["resume_reads_rds_not_csv"] <- file.exists(
  file.path(res_p2$run_dir, "comparison", "comparison_all.rds"))

bad_cols <- shared[cls_before[shared] != cls_after[shared]]
if (length(bad_cols)) {
  cat("  columns with a swapped type: ",
      paste(sprintf("%s (%s -> %s)", bad_cols, cls_before[bad_cols],
                    cls_after[bad_cols]), collapse = ", "), "
", sep = "")
}

# -- 4/7. two folds -----------------------------------------------------------

# No test set on purpose: a plan is allowed to carve none, and training must
# work under that. This used to crash -- train_one_cnn assumed three roles
# always existed, so "the framework does not invent a test set nobody asked
# for" was a promise the training path could not keep.
plan2 <- random_folds(store$meta, k = 2L, seed = 5L)
res2 <- quiet_run(
  tune_grid = grid, store = store, points = points, type_table = type_table,
  plan = plan2, transform = expm1, output_dir = out_root,
  device = device, run_id = "smoke_kfold", n_seeds = 1L,
  release_store = FALSE
)

cmp2 <- res2$comparison
ok["kfold_rows_configs_x_folds"] <- nrow(cmp2) == 2L * 2L
ok["kfold_both_folds_present"]   <- setequal(unique(cmp2$fold), c(1L, 2L))
ok["kfold_single_run_dir"] <- dplyr::n_distinct(basename(res2$run_dir)) == 1L
ok["kfold_by_config_counts_folds"] <- all(res2$by_config$n_folds == 2L)

# Without a test set the columns still exist, holding NA: one table shape
# whatever the plan was, so nothing downstream has to branch on it.
ok["no_test_still_trains"]        <- all(res2$comparison$status == "success")
ok["no_test_columns_exist"]       <- "test_ccc" %in% names(res2$comparison)
ok["no_test_metrics_are_na"]      <- all(is.na(res2$comparison$test_ccc))
ok["no_test_val_metrics_are_not"] <- all(is.finite(res2$comparison$val_ccc))

# the plan travels with the results
ok["fold_plan_saved"] <- file.exists(file.path(res2$run_dir, "fold_plan.rds"))
ok["fold_sizes_saved"] <- file.exists(file.path(res2$run_dir, "fold_sizes.csv"))
ok["fold_plan_roundtrips"] <- {
  p <- readRDS(file.path(res2$run_dir, "fold_plan.rds"))
  inherits(p, "fold_plan") && p$n_folds == 2L && p$method == "random_folds"
}

# 7. the scaling is fitted PER FOLD, and really differs. If these two files
# were identical, the whole raw-storage design would be buying nothing.
sc1 <- readr::read_csv2(file.path(res2$run_dir, "scaling_fold01.csv"),
                        show_col_types = FALSE)
sc2 <- readr::read_csv2(file.path(res2$run_dir, "scaling_fold02.csv"),
                        show_col_types = FALSE)
ok["scaling_written_per_fold"] <- nrow(sc1) == n_ch && nrow(sc2) == n_ch
ok["scaling_differs_between_folds"] <- !isTRUE(all.equal(sc1$center,
                                                         sc2$center))

cat("  2 folds x 2 configs : ", nrow(cmp2), " units | channel 1 mu: ",
    sprintf("%.3f vs %.3f", sc1$center[1], sc2$center[1]), "\n", sep = "")

# -- 9. a broken config is recorded, not fatal --------------------------------
#
# embedding_dim = 0 cannot build a layer. The OTHER config must still train,
# and the failure must leave a row -- a config that vanishes is
# indistinguishable from one that was never run, which is what the status
# column exists to prevent.

grid_bad <- grid
grid_bad$embedding_dim[1] <- 0L

res3 <- quiet_run(
  tune_grid = grid_bad, store = store, points = points, type_table = type_table,
  plan = holdout(store$meta, validation_frac = 0.2, test_frac = 0.2, seed = 3L),
  transform = expm1, output_dir = out_root,
  device = device, run_id = "smoke_failure", n_seeds = 1L,
  release_store = FALSE
)

cmp3 <- res3$comparison
ok["failed_config_has_a_row"] <- nrow(cmp3) == 2L
ok["failed_config_marked"]    <- any(cmp3$status == "failed")
ok["failed_config_has_message"] <- {
  fr <- dplyr::filter(cmp3, status == "failed")
  nrow(fr) == 1L && !is.na(fr$error_message[1]) && nzchar(fr$error_message[1])
}
ok["good_config_still_trained"] <- sum(cmp3$status == "success") == 1L
ok["failed_config_excluded_from_summary"] <-
  nrow(res3$by_config) == 1L && res3$by_config$n_failed[1] == 0L

cat("  broken config       : recorded as failed, the other one trained\n")

# -- cleanup ------------------------------------------------------------------

unlink(store_dir, recursive = TRUE)
unlink(out_root,  recursive = TRUE)

.report(ok, "test_resample_run")
