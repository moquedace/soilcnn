# ── The column names dplyr resolves at run time, declared to R CMD check ──────
#
# WHY THIS FILE EXISTS. A dplyr verb reads a column by its bare name --
# dplyr::filter(perf, dataset_role == "test") -- and finds it inside the table
# when the line runs. R CMD check reads the same line without the table, finds
# no variable called dataset_role anywhere, and reports "no visible binding
# for global variable". The first check of the package (2026-09-28) reported
# 121 of them, over the 70 names below, and nothing else.
#
# EVERY NAME WAS CHECKED AGAINST ITS USE BEFORE IT CAME IN. A pass over the
# check's own list found, for each (function, name) pair, the lines of that
# function where the name occurs bare: every one sits inside a dplyr verb
# reading a table's column (select, filter, group_by, summarise, mutate,
# arrange, relocate, rename), or is one of rlang's pronouns (.data, .env).
# None is a variable nobody defined.
#
# That is also why the list is written out rather than generated. A name
# declared here is a name the check stops looking at, so a real undefined
# variable spelled like one of these -- a typo of `mae` in a function that has
# no table -- would go unreported. A NOTE the check prints later is read
# before its name is added here.
#
# REJECTED:
#   * rewriting every such call with .data$column. It is what dplyr's
#     documentation recommends for packages, and it touches some 30 functions
#     to change nothing they compute.
#   * importFrom(dplyr, .data). It covers the pronoun and none of the bare
#     names, and this package's NAMESPACE is exactly what its @export tags say
#     (tests/test_package_metadata.R holds it to that).
utils::globalVariables(c(
  # rlang's pronouns, which dplyr re-exports
  ".data", ".env",
  # the point table
  "profile_id", "sample_id", "x", "y", "target_native", "target_transform",
  # predictions and their roles
  "dataset_role", "obs", "pred", "obs_transform", "pred_transform", "model",
  "target_version", "obs_q_group",
  # metrics (calc_metrics()'s columns, and the tables built on them)
  "n", "r2", "mae", "nse", "rmse", "rpd", "mqi", "bias", "bias_pct",
  "val_ccc", "val_mae", "ccc_mean", "delta_ccc", "pct_of_ccc",
  # the units of a run, and the configurations they fit
  "config_id", "window_sizes", "conv_channels", "status", "n_units", "n_folds",
  "n_failed", "best_epoch", "runtime_min",
  # the predictor table dsm_prepare() writes, and its quality report
  "predictor", "type", "is_dummy", "is_percentage", "n_unique", "min_value",
  "max_value", "n_na_at_points", "pct_na", "risk", "n_invalidated", "n_sole_cause",
  # the diagnostics' own tables
  "key", "value", "old", "new", "n_mismatch", "scope", "n_pixels_hidden",
  "plateau", "median_bias", "median_bias_rel", "median_bias_flat", "max_bias_rel",
  "pct_descending", "flat_only", "worst", "descending", "block_size",
  "largest_share", "cv_di"
))
