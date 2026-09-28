# Unit test: what dsm_train() takes from the store and the machine when it is
# not told -- the grid's windows, the target's inverse, the cores
#
# WHY THIS FILE EXISTS.
#
# dsm_train(data, tune_length = 30) is the call a new user makes, and until
# 2026-09-27 three of its defaults came from the SOC example rather than from
# the data it was given:
#
#   the grid      drew 3/9/15, the SOC windows at 250 m, from ANY store -- a
#                 store of 5 and 11 stopped with "this grid needs 3, 9, 15"
#   the inverse   was identity, so a log1p store trained without
#                 transform = expm1 reported every "native" metric in log space
#   the cores     were whatever the device was built with; the examples typed
#                 30 by hand, and the forest took every logical core
#   the batches   were drawn from 128/256/512 whatever the fold held -- and a
#                 batch larger than the fold's training set takes NO gradient
#                 step (the loader drops the incomplete batch), so the unit
#                 "succeeded" with an untrained network
#
# Each is now read from the store or the machine, and each is checked here
# against the rule it follows. Nothing trains: all of it is decided before the
# first tensor. The end-to-end path -- a default grid actually trained on a
# store of one window, and the store's inverse actually applied -- is in
# test_api_run.R.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_train_defaults.R")

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
      if (file.exists(file.path(r, "R", "load_all.R"))) return(r)
    }
  }
  stop("Project root not found.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
suppressMessages(source(file.path(root, "R", "load_all.R")))

ok <- c()
err <- function(expr) {
  e <- tryCatch({ suppressMessages(expr); NULL }, error = function(e) conditionMessage(e))
  if (is.null(e)) "" else e
}

# ── 1. the window options a store can serve ──────────────────────────────────
#
# Every window alone, then every pair, smaller first. For 3, 9 and 15 that is
# the list the SOC example wrote by hand, in the same order -- which is what
# keeps a seed drawing the same grid it drew before.
ok["three_windows_give_the_soc_list_exactly"] <-
  identical(.window_options(c(3L, 9L, 15L)), .cnn_param_space$window_sizes)
ok["options_are_sorted_and_deduplicated"] <-
  identical(.window_options(c(15, 3, 9, 9)), .window_options(c(3L, 9L, 15L)))
ok["two_windows_give_two_singles_and_their_pair"] <-
  identical(.window_options(c(11L, 5L)), list(5L, 11L, c(5L, 11L)))
ok["one_window_gives_one_option"] <- identical(.window_options(7L), list(7L))
ok["four_windows_give_four_singles_and_six_pairs"] <-
  length(.window_options(c(3L, 5L, 7L, 9L))) == 10L
ok["an_even_fractional_or_non_positive_window_is_refused"] <-
  all(vapply(list(4, 2.5, 0, -3), function(w) grepl("odd", err(.window_options(w))),
             logical(1)))

# ── 2. the grid, drawn over the store's windows ──────────────────────────────
ok["a_store_of_3_9_15_draws_the_grid_it_always_drew"] <-
  identical(make_tune_grid(tune_length = 20L, seed = 11L),
            make_tune_grid(tune_length = 20L, seed = 11L, windows = c(3L, 9L, 15L)))

g5 <- make_tune_grid(tune_length = 30L, seed = 5L, windows = c(5L, 11L))
drawn5 <- vapply(g5$window_sizes, paste, character(1), collapse = "+")
ok["the_grid_draws_only_the_stores_windows"] <- all(drawn5 %in% c("5", "11", "5+11"))
ok["every_window_option_is_drawn"] <- all(c("5", "11", "5+11") %in% drawn5)
ok["a_single_window_config_has_no_gate"] <-
  all(g5$gate_type[lengths(g5$window_sizes) == 1L] == "no_gate_concat")

ok["a_fixed_window_the_store_lacks_is_refused_and_named"] <-
  grepl("21", err(make_tune_grid(5L, seed = 1L, windows = c(3L, 9L),
                                 fixed = list(window_sizes = list(c(3L, 21L))))))
ok["a_fixed_window_the_store_holds_is_kept"] <- {
  g <- make_tune_grid(5L, seed = 1L, windows = c(3L, 9L),
                      fixed = list(window_sizes = list(9L)))
  all(vapply(g$window_sizes, identical, logical(1), 9L))
}

# ── 2b. batch sizes a fold can fill ──────────────────────────────────────────
#
# An epoch takes floor(n_train / batch) steps. At least four, and never a
# batch below two (BatchNorm). On the SOC folds (~2,100 training points)
# nothing is left out, so the SOC grid is drawn as it always was.
soc_b <- .cnn_param_space$batch_size
ok["on_the_soc_folds_every_batch_size_stays"] <-
  identical(.batch_options(soc_b, 2100), soc_b)
ok["a_batch_that_gives_under_four_steps_is_left_out"] <-
  identical(.batch_options(soc_b, 1000), 128L)
ok["with_none_left_the_largest_power_of_two_that_gives_four"] <-
  identical(.batch_options(soc_b, 25), 4L)
ok["never_a_batch_below_two"] <- identical(.batch_options(soc_b, 5), 2L)
ok["every_batch_offered_gives_at_least_four_steps"] <-
  all(vapply(c(8, 10, 40, 100, 600, 3000), function(n)
    all(floor(n / .batch_options(soc_b, n)) >= 4), logical(1)))
ok["a_fold_of_one_point_is_refused"] <- grepl("n_train", err(.batch_options(soc_b, 1)))
ok["a_grid_on_the_soc_folds_is_the_grid_it_was"] <-
  identical(make_tune_grid(tune_length = 20L, seed = 11L),
            make_tune_grid(tune_length = 20L, seed = 11L, n_train = 2100))
ok["a_grid_on_small_folds_draws_only_batches_they_can_fill"] <-
  all(make_tune_grid(tune_length = 10L, seed = 3L, n_train = 25)$batch_size == 4L)

# The loader, the backstop: a batch larger than the training set is refused
# instead of training nothing.
if (requireNamespace("torch", quietly = TRUE)) {
  mk_role <- function(n) list(w03 = torch::torch_zeros(n, 2L, 3L, 3L),
                              y = torch::torch_zeros(n))
  cache <- list(train = mk_role(10L), validation = mk_role(4L))
  cfg_big <- tibble::tibble(config_id = "big", window_sizes = list(3L),
                            batch_size = 16L)
  ok["the_loader_refuses_a_batch_larger_than_the_training_set"] <-
    grepl("no gradient step", err(.make_loaders_from_cache(cache, cfg_big)))
  cfg_fit <- cfg_big
  cfg_fit$batch_size <- 8L
  ok["the_loader_builds_a_batch_that_fits"] <-
    is.list(.make_loaders_from_cache(cache, cfg_fit))
}

# Through the registry, the way dsm_train() asks.
cnn <- get_model("cnn")
ok["the_cnn_generator_declares_windows"] <- "windows" %in% names(formals(cnn$default_grid))
g_reg <- .draw_default_grid(cnn, 12L, 42L, c(5L, 11L), n_train = 600, verbose = FALSE)
ok["the_front_end_draws_over_the_windows_it_is_given"] <-
  all(unlist(g_reg$window_sizes) %in% c(5L, 11L))
ok["the_front_end_sizes_the_batches_to_the_smallest_fold"] <-
  all(g_reg$batch_size == 128L)
# A generator that does not declare `windows` is called exactly as before.
legacy <- model_spec(
  "legacy_patch", "patches",
  fit = function(x, y, cfg, ...) NULL, predict = function(object, x, ...) 0,
  default_grid = function(tune_length, seed, x = NULL, y = NULL)
    tibble::tibble(config_id = "only", window_sizes = list(3L)))
ok["a_generator_without_windows_is_called_as_before"] <-
  nrow(.draw_default_grid(legacy, 1L, 1L, 5L, verbose = FALSE)) == 1L

# ── 3. the inverse of the target transform ───────────────────────────────────
#
# A dsm_data carries the transform dsm_load() read from the store. Built by
# hand here: this is about the rule, not about the store.
mk_data <- function(tr) {
  structure(list(transform = if (is.null(tr)) NULL else target_transform_spec(tr)),
            class = "dsm_data")
}
d_log  <- mk_data("log1p")
d_none <- mk_data("none")
d_bare <- mk_data(NULL)

ok["null_takes_the_stores_inverse"] <-
  identical(.resolve_train_transform(NULL, d_log, verbose = FALSE), expm1)
ok["the_same_inverse_given_by_hand_passes"] <-
  identical(.resolve_train_transform(expm1, d_log), expm1)
ok["an_equivalent_inverse_written_differently_passes"] <-
  is.function(.resolve_train_transform(function(z) exp(z) - 1, d_log))
ok["an_inverse_that_disagrees_is_refused_and_the_store_named"] <- {
  m <- err(.resolve_train_transform(identity, d_log))
  grepl("disagrees", m) && grepl("log1p", m)
}
ok["a_store_without_the_log_refuses_expm1"] <-
  grepl("disagrees", err(.resolve_train_transform(expm1, d_none)))
ok["a_store_without_the_log_takes_identity"] <-
  identical(.resolve_train_transform(NULL, d_none, verbose = FALSE), identity)
ok["with_nothing_recorded_null_is_identity"] <-
  identical(.resolve_train_transform(NULL, d_bare, verbose = FALSE), identity)
ok["with_nothing_recorded_a_function_is_taken_as_given"] <-
  identical(.resolve_train_transform(expm1, d_bare), expm1)
ok["a_transform_that_is_not_a_function_is_refused"] <-
  grepl("function", err(.resolve_train_transform("expm1", d_log)))

# Both are checked AT THE DOOR -- before a plan is resolved, which on these
# hand-built objects would itself fail. The error must be the one about the
# argument, not one from deeper in.
ok["dsm_train_refuses_a_disagreeing_inverse_before_anything_runs"] <-
  grepl("disagrees", err(dsm_train(d_log, model = "rf", transform = identity)))

# ── 4. the cores ─────────────────────────────────────────────────────────────
ok["dsm_train_refuses_a_fractional_n_cores_before_anything_runs"] <-
  grepl("whole number", err(dsm_train(d_bare, model = "rf", n_cores = 2.5)))
ok["the_forest_takes_n_cores"] <- "n_cores" %in% names(formals(get_model("rf")$fit))

if (requireNamespace("torch", quietly = TRUE)) {
  before <- torch::torch_get_num_threads()
  n_set  <- suppressMessages(set_torch_threads(2L))
  ok["set_torch_threads_sets_them"] <- n_set == 2L && torch::torch_get_num_threads() == 2L
  invisible(suppressMessages(set_torch_threads(before)))
}

# ── 5. a grid given by hand, checked against the parameter space ─────────────
#
# A missing or misspelt column used to surface at the first unit -- after the
# plan and the fold cache -- or, for an optional one, never. dsm_train() now
# checks a grid it is given, at the door; the columns come from the space.
g_ok <- make_manual_tune_grid(window_sizes = list(c(3L, 9L)), base_lr = c(1e-3, 3e-4))
ok["a_grid_from_the_space_passes_unchanged"] <-
  identical(.check_cnn_grid(g_ok, verbose = FALSE), g_ok)
ok["the_grid_columns_come_from_the_space"] <-
  setequal(.cnn_grid_columns(), c("config_id", names(.cnn_param_space), names(.expand_dropout(0))))
g_typo <- g_ok; names(g_typo)[names(g_typo) == "base_lr"] <- "base_rl"
ok["a_misspelt_required_column_is_refused_and_named"] <- {
  m <- err(.check_cnn_grid(g_typo, verbose = FALSE))
  grepl("lacks column", m) && grepl("base_lr", m) && grepl("base_rl", m)
}
g_opt <- g_ok; names(g_opt)[names(g_opt) == "embed_pool"] <- "embed_pol"
ok["a_misspelt_optional_column_is_refused"] <-
  grepl("look misspelt", err(.check_cnn_grid(g_opt, verbose = FALSE)))
g_extra <- g_ok; g_extra$val_ccc_mean <- c(0.5, 0.6)
ok["a_column_that_is_no_parameter_is_carried_along"] <-
  identical(.check_cnn_grid(g_extra, verbose = FALSE)$val_ccc_mean, c(0.5, 0.6))
g_knob <- g_ok[, setdiff(names(g_ok), names(.expand_dropout(0)))]
g_knob$dropout <- c(0.2, 0)
g_knob_out <- .check_cnn_grid(g_knob, verbose = FALSE)
ok["the_dropout_knob_alone_is_expanded_as_the_generator_does"] <-
  isTRUE(all.equal(g_knob_out$head_dropout_1, c(0.2, 0))) &&
  isTRUE(all.equal(g_knob_out$gate_dropout, c(0.1, 0)))
g_part <- g_ok[, setdiff(names(g_ok), "gate_dropout")]
ok["some_dropout_sites_without_the_others_are_refused"] <-
  grepl("some of the five", err(.check_cnn_grid(g_part, verbose = FALSE)))
g_dup <- g_ok; g_dup$config_id <- c("a", "a")
ok["a_duplicated_config_id_is_refused"] <- grepl("unique", err(.check_cnn_grid(g_dup, verbose = FALSE)))
g_even <- g_ok; g_even$window_sizes <- list(4L, c(3L, 9L))
ok["an_even_window_is_refused"] <- grepl("odd", err(.check_cnn_grid(g_even, verbose = FALSE)))
g_gate <- g_ok; g_gate$gate_type <- c("vector_feature", "vector_featurewise")
ok["an_unknown_gate_is_refused_with_the_known_ones"] <- {
  m <- err(.check_cnn_grid(g_gate, verbose = FALSE))
  grepl("vector_feature,|vector_feature\\.", m) && grepl("no_gate_concat", m)
}
g_lr <- g_ok; g_lr$base_lr <- c(0, 1e-3)
ok["a_learning_rate_of_zero_is_refused"] <- grepl("base_lr", err(.check_cnn_grid(g_lr, verbose = FALSE)))
ok["dsm_train_refuses_a_bad_grid_before_the_plan"] <-
  grepl("lacks column", err(dsm_train(d_bare, model = "cnn", tune_grid = g_typo, verbose = FALSE)))

cat(sprintf("  window options 3/9/15    : %s\n",
            paste(vapply(.window_options(c(3L, 9L, 15L)), paste, character(1),
                         collapse = "+"), collapse = ", ")))
cat(sprintf("  window options 5/11      : drawn %s\n",
            paste(sort(unique(drawn5)), collapse = ", ")))

.report(ok, "test_train_defaults")
