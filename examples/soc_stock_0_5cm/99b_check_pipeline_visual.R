
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
source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c("dplyr", "readr", "tibble", "purrr", "ggplot2", "tidyr", "DescTools")
install_load_pkg(pkg)

rm(list = setdiff(ls(), "project_root"))  # keep the root found above
gc()

options(width = 200)

setwd(project_root)

# The framework, for latest_run_dir() and the patch store. It is one package
# now, and loading it is one line whatever part of it a script uses.
pkgload::load_all(project_root)

# ══════════════════════════════════════════════════════════════════════════════
# 99b -- VISUAL pipeline checkpoint (companion to 99_check_pipeline.R)
#
# 99 checks numbers (counts, percentages, consistency between files).
# This one puts SOMETHING REAL on the screen: where the profiles are on the map,
# whether tuning actually improved anything, whether the final model gets the
# real SOC value right (not just the aggregated metric), and what the patches
# the CNN receives look like -- a cut-out of elevation, vegetation and (on
# purpose, it is the villain of the story) a PNV layer, extracted around real
# profiles.
#
# Four parts, increasing cost:
#   PART 1 (light, seconds): world map of the store's profiles by split.
#     Reads the tuning run's fold_plan.rds and the store's patch_meta.csv.
#     It used to read split_metadata.csv, which nothing writes any more -- see
#     the note above PART 1. The count is the store's (3,728 here), not the
#     ~37,000 of the full WOSIS extract this text was written against.
#   PART 2 (light, seconds): stage 03 (tuning) -- leaderboard of the configs and
#     "does the winner get the real value right?" (obs x pred, validation).
#   PART 3 (light, seconds): stage 04 (final model) -- training curves per seed,
#     stability between seeds, and obs x pred on TEST (the final ensemble, the
#     number that actually goes onto the map).
#   PART 4 (heavy, minutes): loads the patch tensors (0.85 GB) into memory just
#     to pull out some 6 example patches. It is the only way to reach the file
#     (RDS does not support partial reads) -- run it when you are not short on
#     RAM. Comment out PART 4 if you do not want to wait.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"

metadata_dir   <- file.path(project_root, "outputs", "metadata", "soc_stock_modeling", target_label)
patch_dir      <- file.path(project_root, "outputs", "patches", "soc_stock_modeling", target_label)
patch_meta_dir <- file.path(metadata_dir, "patches")
tuning_base    <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling", target_label)
final_base     <- file.path(project_root, "outputs", "final_model", "soc_stock_modeling", target_label)
fig_dir        <- file.path(project_root, "outputs", "qc", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Resolve the most recent run inside a folder of timestamped runs (same
# criterion used in 99_check_pipeline.R and in 03/04/05 themselves): sorts the
# subfolder names and takes the first in decreasing order.
.latest_run <- function(base_dir, prefix) {
  # By time and only if finished; NULL rather than a crash, because this is a
  # check. Which file says "finished" depends on the family.
  marker <- if (identical(prefix, "final_")) {
    file.path("comparison", "final_run_summary.rds")
  } else {
    file.path("comparison", "comparison_ranked.csv")
  }
  latest_run_dir(base_dir, prefix = prefix, require_file = marker,
                 label = paste0(prefix, "run"), on_none = "null")
}

# Lin's CCC, computed exactly as in R/metrics.R (DescTools::CCC) -- just to
# annotate the obs-x-pred plots with the same number the rest of the pipeline
# would report, without having to source() the whole of metrics.R.
.lin_ccc <- function(obs, pred) {
  as.numeric(DescTools::CCC(obs, pred, conf.level = 0.95)$rho.c$est)[1]
}

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 -- World map of the profiles (light)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Part 1: map of the profiles by split --\n")

# THE SPLIT IS NOT A PROPERTY OF THE DATA ANY MORE.
#
# This read split_metadata.csv, a file NOTHING writes -- it was produced by the
# stage 01 that stamped a role onto every point, and that stage stopped existing
# when the split became an index rather than a column. The script has stopped on
# this line ever since, which is why nobody noticed the three further breakages
# below it.
#
# The split now lives in a fold_plan, so the map is drawn from the plan the
# tuning run actually used, joined onto the store's coordinates. That is also a
# better picture: it shows the plan that produced the numbers rather than a
# stale file that once described a different one.
tuning_base <- file.path(project_root, "outputs", "tuning",
                         "soc_stock_modeling", target_label)
# The newest tuning run that has a plan, by the plan's own time -- not the
# last one alphabetically, which is what a NAMED run (soc_0_5cm_design_*)
# turned into the day it appeared.
plan_dir <- file.path(tuning_base, latest_run_dir(
  tuning_base, prefix = "soc_", require_file = "fold_plan.rds",
  label = "fold plan source"))
fold_plan <- readRDS(file.path(plan_dir, "fold_plan.rds"))
message("Fold plan: ", basename(plan_dir))
print(fold_plan)

store_meta <- readr::read_csv2(file.path(patch_dir, "patch_meta.csv"),
                               show_col_types = FALSE)

# ONE ROLE PER POINT, FOR A MAP. A point is training in one fold and validation
# in another, so "the role" is only well defined for the test set -- which is
# the same in every fold -- and for fold 1 otherwise. The label says so.
role_of <- rep("train", nrow(store_meta))
role_of[fold_plan$folds[[1]]$validation] <- "validation (fold 1)"
role_of[fold_plan$folds[[1]]$test]       <- "test (all folds)"
split_meta <- dplyr::mutate(store_meta, dataset_role = role_of)

message("Total profiles: ", nrow(split_meta))
print(dplyr::count(split_meta, dataset_role))

p_map <- ggplot2::ggplot(split_meta, ggplot2::aes(x = x, y = y, color = dataset_role)) +
  ggplot2::geom_point(size = 2, alpha = 0.5) +
  ggplot2::coord_fixed() +
  ggplot2::scale_color_manual(values = c(train = "#1b9e77", validation = "#d95f02", test = "#7570b3")) +
  ggplot2::labs(
    title = paste0(target_label, " -- ", format(nrow(split_meta), big.mark = ","),
                  " WOSIS profiles, by split"),
    subtitle = "No base map on purpose: the shape of the continents should appear on its own from the point density",
    x = "Longitude", y = "Latitude", color = "Split"
  ) +
  ggplot2::theme_bw() +
  ggplot2::guides(color = ggplot2::guide_legend(override.aes = list(size = 3, alpha = 1)))

print(p_map)
ggplot2::ggsave(file.path(fig_dir, "99b_world_map_profiles.png"), p_map,
                width = 12, height = 6, dpi = 150)
message("Saved: ", file.path(fig_dir, "99b_world_map_profiles.png"))

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 -- Stage 03: tuning (leaderboard + does the winner get the real value right?)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Part 2: stage 03 (tuning) --\n")

tuning_run_id <- .latest_run(tuning_base, "soc_")

if (is.null(tuning_run_id)) {
  message("Stage 03 not found in ", tuning_base, " -- skipping Part 2.")
} else {

  tuning_run_dir <- file.path(tuning_base, tuning_run_id)
  message("Tuning run: ", tuning_run_id)

  # ONE ROW PER CONFIG, AGGREGATED FROM THE UNITS.
  #
  # comparison_ranked.csv is one row per (config, fold, seed) and has been since
  # repetitions arrived. Everything below treats it as one row per config: it
  # builds a factor whose levels are config_id -- which fails outright on the
  # duplicates -- draws one lollipop per row, and says "N configs tested".
  #
  # comparison_by_config.csv is the aggregated table and is read in preference,
  # because it carries the standard error the leaderboard should have been
  # showing all along: a leaderboard without it invites reading a 0.02 gap as a
  # result when the seed noise floor here is 0.03. The per-unit file is the
  # fallback for older runs that predate it.
  by_cfg_file <- file.path(tuning_run_dir, "comparison",
                           "comparison_by_config.csv")
  units_file  <- file.path(tuning_run_dir, "comparison",
                           "comparison_ranked.csv")

  ranking <- if (file.exists(by_cfg_file)) {
    bc <- readr::read_csv2(by_cfg_file, show_col_types = FALSE)
    # The architecture fields live only in the per-unit table; take them from
    # the first unit of each config, where they are identical by construction.
    un <- readr::read_csv2(units_file, show_col_types = FALSE)
    arch_cols <- intersect(c("window_sizes", "conv_channels", "gate_type",
                             "embedding_dim", "conv_padding", "base_lr",
                             "runtime_min"), names(un))
    arch <- un %>%
      dplyr::group_by(config_id) %>%
      dplyr::summarise(dplyr::across(dplyr::all_of(arch_cols),
                                     ~ dplyr::first(.x)), .groups = "drop")
    bc %>%
      dplyr::rename(val_ccc = val_ccc_mean) %>%
      dplyr::left_join(arch, by = "config_id")
  } else {
    un <- readr::read_csv2(units_file, show_col_types = FALSE)
    un %>%
      dplyr::group_by(config_id) %>%
      # -config_id in both selections: it is the group key, and summarising the
      # key alongside itself is either an error or a duplicated column depending
      # on the dplyr version -- neither is what this wants.
      dplyr::summarise(dplyr::across(dplyr::where(is.numeric) & !config_id,
                                     ~ mean(.x, na.rm = TRUE)),
                       dplyr::across(!dplyr::where(is.numeric) & !config_id,
                                     ~ dplyr::first(.x)),
                       n_units = dplyr::n(), .groups = "drop") %>%
      dplyr::arrange(dplyr::desc(val_ccc)) %>%
      dplyr::mutate(rank = dplyr::row_number())
  }
  stopifnot(!anyDuplicated(ranking$config_id))
  message("configs on the leaderboard: ", nrow(ranking))

  # ── Leaderboard: every config, ordered by val_ccc ─────────────────────────
  # Lollipop instead of a plain bar: the horizontal rule makes it easier to see
  # the DIFFERENCE between neighbours in the ranking (it is this that decides
  # which config becomes the final model), not just each one's absolute value.
  ranking_plot <- ranking %>%
    dplyr::mutate(
      config_id = factor(config_id, levels = config_id[order(val_ccc)]),
      is_winner = rank == 1L
    )

  # THE ERROR BAR IS THE POINT OF THIS PANEL.
  #
  # The comment above says the lollipop exists to make the DIFFERENCE between
  # neighbours readable, "it is this that decides which config becomes the final
  # model". Without a spread that difference cannot be judged: on this run the
  # gap between first and second is 0.024 and the seed noise floor is 0.031, so
  # the leaderboard's order is not evidence and the plot should show why.
  has_se <- "val_ccc_se" %in% names(ranking_plot) &&
    any(is.finite(ranking_plot$val_ccc_se))

  p_leaderboard <- ggplot2::ggplot(ranking_plot,
                                   ggplot2::aes(x = val_ccc, y = config_id)) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, xend = val_ccc, yend = config_id,
                                       color = window_sizes), linewidth = 1) +
    {if (has_se) ggplot2::geom_errorbarh(
       ggplot2::aes(xmin = val_ccc - val_ccc_se, xmax = val_ccc + val_ccc_se),
       height = 0.25, color = "grey30") else NULL} +
    ggplot2::geom_point(ggplot2::aes(color = window_sizes, size = is_winner)) +
    ggplot2::geom_point(data = dplyr::filter(ranking_plot, is_winner),
                        shape = 21, size = 5, stroke = 1.2, color = "black", fill = NA) +
    ggplot2::scale_size_manual(values = c(`TRUE` = 4, `FALSE` = 2.5), guide = "none") +
    ggplot2::coord_cartesian(xlim = c(0, max(ranking_plot$val_ccc) * 1.05)) +
    ggplot2::labs(
      title = paste0(target_label, " -- tuning leaderboard (", tuning_run_id, ")"),
      subtitle = paste0(nrow(ranking), " configs | bar = 1 standard error over the repetitions | ",
                        "black-outlined circle = rank 1"),
      x = "Validation CCC", y = NULL, color = "Window(s)"
    ) +
    ggplot2::theme_bw()

  print(p_leaderboard)
  ggplot2::ggsave(file.path(fig_dir, "99b_tuning_leaderboard.png"), p_leaderboard,
                  width = 9, height = max(4, 0.35 * nrow(ranking)), dpi = 150)
  message("Saved: ", file.path(fig_dir, "99b_tuning_leaderboard.png"))

  # ── Does the winner get the real value right? Obs x Pred on VALIDATION ────
  # The CCC=0.xx metric in the table is abstract; seeing the points around the
  # 1:1 line is what actually convinces you that the "best config" learned
  # something physical, not just a number that came out good by chance in the
  # aggregation.
  # THE FILENAME CHANGED WHEN REPETITIONS ARRIVED, and this block has been
  # skipping itself ever since. The tuning run writes ONE FILE PER UNIT --
  # cfg_003_f1_s1_pred_all.csv -- not one per config, so file.exists() on
  # "<config>_pred_all.csv" was FALSE and the whole panel silently did not
  # appear. A guarded read that quietly produces nothing is worse than a hard
  # stop: nothing in the output says a check was skipped.
  #
  # rank == 1 is also per UNIT now, so it names the luckiest single (fold, seed)
  # of the best config. The config is taken from the unit that ranks first and
  # then ALL of its units are pooled, which is what the panel claims to show.
  best_id   <- ranking$config_id[ranking$rank == 1L][1]
  unit_pat  <- sprintf("^%s_f[0-9]+_s[0-9]+_pred_all\\.csv$", best_id)
  unit_files <- list.files(file.path(tuning_run_dir, "predictions"),
                           pattern = unit_pat, full.names = TRUE)
  if (length(unit_files) == 0L) {
    message("  [SKIP] no per-unit predictions for ", best_id,
            " under ", file.path(tuning_run_dir, "predictions"),
            " -- the panel below needs them.")
  }
  pred_file <- unit_files

  if (length(pred_file) > 0L) {
    # Every unit of the config, pooled. Each point appears once per seed, so the
    # ensemble median is taken per point -- the same statistic the map uses.
    pred_best <- purrr::map_dfr(pred_file, function(f) {
      readr::read_csv2(f, show_col_types = FALSE) %>%
        dplyr::filter(dataset_role == "validation")
    }) %>%
      dplyr::group_by(sample_id) %>%
      dplyr::summarise(obs  = dplyr::first(obs),
                       pred = stats::median(pred), .groups = "drop")

    ccc_val <- .lin_ccc(pred_best$obs, pred_best$pred)
    lims    <- range(c(pred_best$obs, pred_best$pred))

    p_best_val <- ggplot2::ggplot(pred_best, ggplot2::aes(x = obs, y = pred)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
      ggplot2::geom_point(alpha = 0.4, color = "#1b9e77") +
      ggplot2::coord_fixed(xlim = lims, ylim = lims) +
      ggplot2::annotate("text", x = lims[1], y = lims[2], hjust = 0, vjust = 1,
                        label = sprintf("CCC = %.3f\nn = %d", ccc_val, nrow(pred_best))) +
      ggplot2::labs(
        title = paste0(target_label, " -- winning config (", best_id, "): observed vs predicted"),
        subtitle = "Validation split | dashed line = 1:1 (perfect agreement)",
        x = "Observed SOC (t/ha)", y = "Predicted SOC (t/ha)"
      ) +
      ggplot2::theme_bw()

    print(p_best_val)
    ggplot2::ggsave(file.path(fig_dir, "99b_tuning_best_obs_vs_pred.png"), p_best_val,
                    width = 7, height = 7, dpi = 150)
    message("Saved: ", file.path(fig_dir, "99b_tuning_best_obs_vs_pred.png"))
  } else {
    message("Predictions for the winning config not found: ", pred_file)
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 -- Stage 04: final model (convergence, stability, accuracy on test)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Part 3: stage 04 (final model, multi-seed ensemble) --\n")

final_run_id <- .latest_run(final_base, "final_")

if (is.null(final_run_id)) {
  message("Stage 04 not found in ", final_base, " -- skipping Part 3.")
} else {

  final_run_dir <- file.path(final_base, final_run_id)
  message("Final model run: ", final_run_id)

  summary_rds <- readRDS(file.path(final_run_dir, "comparison", "final_run_summary.rds"))
  selected_cfgs <- summary_rds$selected_cfgs
  seeds         <- summary_rds$seeds

  # ── Training curves, one line per seed ─────────────────────────────────────
  # If the seeds converge to similar curves, training is stable (it is not
  # initialisation luck). A curve well outside the bundle of the others is the
  # kind of thing that stays invisible in a mean +- sd, but jumps out at you
  # here -- it is the "physical" version of the numeric relative-SD check on
  # the CCC.
  history_long <- purrr::map_dfr(selected_cfgs$config_id, function(cid) {
    purrr::map_dfr(seeds, function(sd_val) {
      f <- file.path(final_run_dir, cid, "history", sprintf("seed%04d_history.csv", sd_val))
      if (!file.exists(f)) return(NULL)
      readr::read_csv2(f, show_col_types = FALSE) %>%
        dplyr::mutate(config_id = cid, seed = factor(sd_val))
    })
  })

  if (nrow(history_long) > 0) {
    p_curves <- ggplot2::ggplot(history_long,
                                ggplot2::aes(x = epoch, y = validation_loss, color = seed)) +
      ggplot2::geom_line(alpha = 0.8) +
      ggplot2::facet_wrap(~ config_id, scales = "free_y") +
      ggplot2::labs(
        title = paste0(target_label, " -- training curves per seed (", final_run_id, ")"),
        subtitle = "Validation loss per epoch | tight bundle = training stable across seeds",
        x = "Epoch", y = "Validation loss", color = "Seed"
      ) +
      ggplot2::theme_bw()

    print(p_curves)
    ggplot2::ggsave(file.path(fig_dir, "99b_final_training_curves.png"), p_curves,
                    width = 9, height = 6, dpi = 150)
    message("Saved: ", file.path(fig_dir, "99b_final_training_curves.png"))
  }

  # ── Stability between seeds: each seed's CCC around the mean ──────────────
  # Direct visual companion to the "CCC SD relative" check in
  # 99_check_pipeline.R -- quoted verbatim so grep finds both ends.
  # here you get to see the cloud of points (each seed is an independent
  # training run from scratch), not just the SD number.
  all_seed_results <- readr::read_csv2(
    file.path(final_run_dir, "comparison", "all_seed_results_test.csv"),
    show_col_types = FALSE
  )
  config_summary <- readr::read_csv2(
    file.path(final_run_dir, "comparison", "config_summary_test.csv"),
    show_col_types = FALSE
  )

  p_stability <- ggplot2::ggplot(all_seed_results, ggplot2::aes(x = config_id, y = ccc)) +
    ggplot2::geom_jitter(width = 0.08, height = 0, size = 2.5, alpha = 0.6, color = "#7570b3") +
    ggplot2::geom_pointrange(
      data = config_summary,
      ggplot2::aes(x = config_id, y = ccc_mean,
                  ymin = ccc_mean - ccc_sd, ymax = ccc_mean + ccc_sd),
      color = "black", linewidth = 0.6, size = 0.4
    ) +
    ggplot2::labs(
      title = paste0(target_label, " -- stability across seeds (test, ", final_run_id, ")"),
      subtitle = "Purple points = each individual seed | black bar = mean +- sd",
      x = NULL, y = "CCC (test)"
    ) +
    ggplot2::theme_bw()

  print(p_stability)
  ggplot2::ggsave(file.path(fig_dir, "99b_final_seed_stability.png"), p_stability,
                  width = 7, height = 5, dpi = 150)
  message("Saved: ", file.path(fig_dir, "99b_final_seed_stability.png"))

  # ── Does the final ensemble get the real value right? Obs x Pred on TEST ──
  # This is the number that actually becomes a map in 05: the median of the N
  # seeds' predictions per profile (same aggregation logic used in the spatial
  # prediction). The grey fan behind each point shows the spread between seeds
  # for that specific profile -- a tangible way to see the ensemble's
  # uncertainty, profile by profile, not just as a mean error number.
  pred_test_all <- purrr::map_dfr(selected_cfgs$config_id, function(cid) {
    purrr::map_dfr(seeds, function(sd_val) {
      f <- file.path(final_run_dir, cid, "predictions", sprintf("seed%04d_pred_all.csv", sd_val))
      if (!file.exists(f)) return(NULL)
      readr::read_csv2(f, show_col_types = FALSE) %>%
        dplyr::filter(dataset_role == "test") %>%
        dplyr::mutate(config_id = cid, seed = sd_val)
    })
  })

  if (nrow(pred_test_all) > 0) {
    pred_test_ens <- pred_test_all %>%
      dplyr::group_by(config_id, profile_id) %>%
      dplyr::summarise(
        obs        = obs[1],
        pred_median = median(pred),
        pred_min    = min(pred),
        pred_max    = max(pred),
        .groups = "drop"
      )

    for (cid in unique(pred_test_ens$config_id)) {
      df_cid  <- dplyr::filter(pred_test_ens, config_id == cid)
      ccc_ens <- .lin_ccc(df_cid$obs, df_cid$pred_median)
      lims    <- range(c(df_cid$obs, df_cid$pred_min, df_cid$pred_max))

      p_test <- ggplot2::ggplot(df_cid, ggplot2::aes(x = obs, y = pred_median)) +
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
        ggplot2::geom_linerange(ggplot2::aes(ymin = pred_min, ymax = pred_max),
                                color = "grey70", alpha = 0.6) +
        ggplot2::geom_point(color = "#d95f02", alpha = 0.6, size = 1.8) +
        ggplot2::coord_fixed(xlim = lims, ylim = lims) +
        ggplot2::annotate("text", x = lims[1], y = lims[2], hjust = 0, vjust = 1,
                          label = sprintf("CCC (ensemble median) = %.3f\nn = %d profiles | %d seeds",
                                          ccc_ens, nrow(df_cid), length(seeds))) +
        ggplot2::labs(
          title = paste0(target_label, " -- final ensemble (", cid, "): observed vs predicted"),
          subtitle = "TEST split (never seen in training) | grey line = min-max across seeds for that profile",
          x = "Observed SOC (t/ha)", y = "Predicted SOC (seed median, t/ha)"
        ) +
        ggplot2::theme_bw()

      print(p_test)
      out_png <- file.path(fig_dir, paste0("99b_final_test_obs_vs_pred_", cid, ".png"))
      ggplot2::ggsave(out_png, p_test, width = 7, height = 7, dpi = 150)
      message("Saved: ", out_png)
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# PART 4 -- Real patches around profiles (heavy -- loads the whole store, 0.85 GB)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Part 4: real patches (loading the patch store, this may take a while) --\n")

manifest <- readRDS(file.path(patch_dir, "patch_manifest.rds"))
# predictor_cols_final is saved as a single ";"-separated string (same format
# as patch_manifest.csv), not as a vector -- it needs a split.
predictor_cols <- strsplit(manifest$predictor_cols_final, ";")[[1]]

# Channels chosen for being visually recognisable. The store holds RAW values
# (see the note below, above load_patch_store), so these are the numbers as they
# came off the raster -- the comment here used to say z-score, which contradicted
# that note and mislabelled the figure's own legend:
#   - elevation: relief should show up sharply (ridges/valleys)
#   - NDVI: vegetation pattern
#   - pnv_open_forest_evergreen_broadleaf: the very PNV layer we investigated
#     in depth (~33% loss in the interior before the fix) -- seeing this layer
#     come out clean here closes the loop on that investigation
channels_to_show <- c(
  elevation = "ensemble_digital_terrain_model_v1_1",
  ndvi     = "landsat_2020_2025_ndvi",
  pnv      = "pnv_open_forest_evergreen_broadleaf"
)

missing_ch <- setdiff(channels_to_show, predictor_cols)
if (length(missing_ch) > 0) {
  stop("Channel(s) not found in predictor_cols_final: ", paste(missing_ch, collapse = ", "))
}
channel_idx <- match(channels_to_show, predictor_cols)
names(channel_idx) <- names(channels_to_show)

message("Chosen channels and their indices: ")
print(tibble::tibble(channel = names(channel_idx), predictor = channels_to_show, index = channel_idx))

# The store keeps the RAW patches (unscaled) -- which is exactly what this part
# wants to show: the data as it came off the raster, before any statistical
# transformation.

t0 <- Sys.time()
store <- load_patch_store(patch_dir)
message("patch store loaded in ",
        round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

set.seed(42)
n_examples <- 6
# THE STORE HAS NO dataset_role, and has not since the split became an index.
# sample.int(n_train, 6) then drew from a count of zero rows. The examples are
# drawn from the fold plan's own training index, which is where the answer lives.
train_idx   <- fold_plan$folds[[1]]$train
stopifnot(length(train_idx) >= n_examples)
example_idx <- sort(sample(train_idx, n_examples))

# THE WINDOWS ARE TORCH TENSORS, NOT R ARRAYS.
#
# `arr[idx, ch, , ]` below is base R indexing, and on a torch tensor it returns
# a tensor -- as.vector() on which is not the numbers. Only the six example rows
# are converted, so this costs a few MB rather than the 0.85 GB the whole store
# would.
#
# This is the same mistake spatial_occlusion() made on 2026-09-17: dim() works
# on both representations and hides the difference until an operation does not.
as_r_array <- function(x, rows) {
  if (inherits(x, "torch_tensor")) {
    as.array(x[rows, , , , drop = FALSE]$to(device = "cpu"))
  } else {
    x[rows, , , , drop = FALSE]
  }
}
window_arrays <- list(
  `3x3`   = as_r_array(store$windows$w03, example_idx),
  `9x9`   = as_r_array(store$windows$w09, example_idx),
  `15x15` = as_r_array(store$windows$w15, example_idx)
)
# The arrays now hold ONLY the examples, in the order example_idx gave, so the
# row index inside them is the example number and not the store row.
example_row <- seq_along(example_idx)

# example_idx already holds STORE ROW INDICES drawn from the plan's training
# fold, so this indexes the store directly. The old form filtered on a
# dataset_role column the store has not had since the split became an index,
# and then indexed the empty result -- the same hard stop as above, twice.
meta_examples <- store$meta[example_idx, ] %>%
  dplyr::mutate(example_id = paste0("profile ", dplyr::row_number(),
                                    "\nSOC=", round(target_native, 1), " t/ha"))

# Builds a long data.frame: one row per (example, window, channel, pixel)
patch_long <- purrr::map_dfr(seq_along(window_arrays), function(w_i) {
  w_name <- names(window_arrays)[w_i]
  arr    <- window_arrays[[w_i]]  # [N, C, w, w]
  w_size <- dim(arr)[3]

  purrr::map_dfr(seq_along(example_idx), function(e_i) {
    idx <- example_row[e_i]          # row inside the extracted subset
    purrr::map_dfr(names(channel_idx), function(ch_name) {
      ch <- channel_idx[[ch_name]]
      mat <- arr[idx, ch, , ]
      grid <- expand.grid(row = seq_len(w_size), col = seq_len(w_size))
      tibble::tibble(
        example    = meta_examples$example_id[e_i],
        window     = factor(w_name, levels = c("3x3", "9x9", "15x15")),
        channel    = ch_name,
        row        = grid$row,
        col        = grid$col,
        value      = as.vector(mat)
      )
    })
  })
})

message("\nExtracted values (quick sanity check -- there should be no NA/Inf):")
print(dplyr::summarise(patch_long,
                       n = dplyr::n(),
                       n_na = sum(is.na(value)),
                       n_inf = sum(!is.finite(value) & !is.na(value)),
                       min = min(value, na.rm = TRUE),
                       max = max(value, na.rm = TRUE)))

for (ch_name in names(channel_idx)) {
  df_ch <- dplyr::filter(patch_long, channel == ch_name)

  p <- ggplot2::ggplot(df_ch, ggplot2::aes(x = col, y = row, fill = value)) +
    ggplot2::geom_raster() +
    ggplot2::scale_fill_viridis_c() +
    ggplot2::coord_fixed() +
    ggplot2::scale_y_reverse() +
    ggplot2::facet_grid(example ~ window, switch = "y") +
    ggplot2::labs(
      title = paste0(target_label, " -- real patches: ", ch_name,
                    " (", channels_to_show[[ch_name]], ")"),
      subtitle = "Raw values, as they come off the raster (the store keeps no scaled patches)",
      x = NULL, y = NULL, fill = "raw value"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      strip.text.y.left = ggplot2::element_text(angle = 0, hjust = 1)
    )

  print(p)
  out_png <- file.path(fig_dir, paste0("99b_patches_", ch_name, ".png"))
  ggplot2::ggsave(out_png, p, width = 9, height = 11, dpi = 150)
  message("Saved: ", out_png)
}

message("\nDone. Figures in: ", fig_dir)
message("Patches (Part 4): look at whether elevation/ndvi show a coherent spatial")
message("pattern (not random noise) and whether the larger window (15x15) shows more")
message("context around the same centre than the smaller one (3x3) -- they should line up.")
message("Tuning/final model (Parts 2-3): look at whether the leaderboard winner lands")
message("close to the 1:1 line in validation AND in test, whether the bundle of training")
message("curves is tight between seeds, and whether the stability cloud has no isolated outlier.")
