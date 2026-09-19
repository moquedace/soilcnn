# Unit test: the I/O helpers everything else stands on, and "latest"
#
# WHY THIS FILE EXISTS.
#
# Every artefact this project writes goes through safe_write_csv2() and
# safe_save_rds(); every artefact it reads back goes through safe_read_csv2();
# and every stage that says "latest" now goes through latest_run_dir(). None of
# the four had a test. Two of them have already produced real defects:
#
#   - "latest" resolved by NAME, so a run called soc_0_5cm_design_spatial --
#     which had died on its first unit -- sorted ahead of the timestamped run
#     everything downstream was built on, and stage 04 would have refit the
#     final model against it without a word.
#   - .resumable_units() once keyed resume on unit_id alone, so a test named
#     "resume skips finished units" passed while resuming onto a DIFFERENT split.
#
# The fixtures here are built on disk, in a temporary directory, with file
# times set explicitly -- because the defect was in how files were chosen, and a
# fixture that hands the function a tidy vector would test the fixture.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_run_dirs.R")

suppressMessages({
  library(tibble)
  library(dplyr)
  library(readr)
})

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
  stop("Project root not found.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
source(file.path(root, "R", "utils.R"))

ok <- logical(0)

scratch <- file.path(tempdir(), "test_run_dirs")
unlink(scratch, recursive = TRUE, force = TRUE)
dir.create(scratch, recursive = TRUE, showWarnings = FALSE)

# ── 1. latest_run_dir(): TIME, NOT NAME, AND ONLY IF FINISHED ─────────────────
#
# Three runs, with names chosen so that alphabetical order disagrees with time
# order in every way that matters:
#
#   run_b_named      finished, OLDEST, sorts LAST alphabetically
#   run_a_timestamp  finished, NEWEST, sorts first
#   run_z_broken     unfinished, sorts... anywhere; it has no marker file
#
# The old expression, sort(dirs, decreasing = TRUE)[1], returns run_z_broken.
# The right answer is run_a_timestamp.
base <- file.path(scratch, "runs")
mk <- function(id, finished, age_hours) {
  d <- file.path(base, id)
  dir.create(file.path(d, "comparison"), recursive = TRUE, showWarnings = FALSE)
  if (finished) {
    f <- file.path(d, "comparison", "comparison_ranked.csv")
    writeLines("unit_id;config_id", f)
    Sys.setFileTime(f, Sys.time() - age_hours * 3600)
  }
  invisible(d)
}
mk("run_b_named",     finished = TRUE,  age_hours = 48)
mk("run_a_timestamp", finished = TRUE,  age_hours = 1)
mk("run_z_broken",    finished = FALSE, age_hours = 0)

marker <- file.path("comparison", "comparison_ranked.csv")
picked <- suppressMessages(latest_run_dir(base, prefix = "run_",
                                          require_file = marker))
ok["latest_is_newest_by_time_not_by_name"] <- identical(picked, "run_a_timestamp")

# and the old expression really does give the wrong answer on this fixture,
# or the fixture is not testing the defect
ok["fixture_would_have_fooled_the_old_expression"] <-
  identical(sort(list.dirs(base, recursive = FALSE, full.names = FALSE),
                 decreasing = TRUE)[1], "run_z_broken")

# resume makes an old run the newest again: touching its marker moves it up
Sys.setFileTime(file.path(base, "run_b_named", marker), Sys.time())
ok["a_resumed_run_becomes_latest_again"] <- identical(
  suppressMessages(latest_run_dir(base, "run_", require_file = marker)),
  "run_b_named")

# the prefix is honoured: nothing under a different prefix is a candidate
dir.create(file.path(base, "other_newest", "comparison"), recursive = TRUE)
writeLines("x", file.path(base, "other_newest", marker))
ok["prefix_excludes_other_families"] <- identical(
  suppressMessages(latest_run_dir(base, "run_", require_file = marker)),
  "run_b_named")

# the unfinished run is named in the message, not silently dropped
msgs <- character(0)
withCallingHandlers(
  latest_run_dir(base, "run_", require_file = marker),
  message = function(m) { msgs <<- c(msgs, conditionMessage(m)); invokeRestart("muffleMessage") })
ok["unfinished_runs_are_named_in_the_message"] <-
  any(grepl("run_z_broken", msgs)) && any(grepl("skipped as unfinished", msgs))

# ── 2. WHEN NOTHING QUALIFIES ─────────────────────────────────────────────────

# a stage cannot proceed: stop, and the message says which run is newest and
# what to do
empty <- file.path(scratch, "unfinished_only")
dir.create(file.path(empty, "run_1"), recursive = TRUE)
err <- tryCatch(latest_run_dir(empty, "run_", require_file = marker),
                error = function(e) conditionMessage(e))
ok["no_finished_run_stops"] <- is.character(err) && grepl("No FINISHED", err)
ok["the_refusal_says_what_to_do"] <- grepl("name the run explicitly", err)

# a CHECK must not crash: on_none = "null" returns NULL and says so
got <- suppressMessages(latest_run_dir(empty, "run_", require_file = marker,
                                       on_none = "null"))
ok["on_none_null_returns_null"] <- is.null(got)

# a base that does not exist: same two behaviours
ok["missing_base_stops"] <- inherits(
  try(latest_run_dir(file.path(scratch, "nowhere"), "run_"), silent = TRUE),
  "try-error")
ok["missing_base_is_null_when_asked"] <- is.null(suppressMessages(
  latest_run_dir(file.path(scratch, "nowhere"), "run_", on_none = "null")))

# both selectors at once is a caller error, refused up front
ok["file_and_pattern_together_are_refused"] <- inherits(
  try(latest_run_dir(base, "run_", require_file = marker,
                     require_pattern = "x"), silent = TRUE), "try-error")

# ── 3. require_pattern: A FAMILY OF FILES, NEWEST FILE WINS ───────────────────
#
# 05c wants the newest run that has shard logs, and a run still being written
# has the freshest log. The mtime that counts is the newest MATCHING file, not
# the directory's.
logs <- file.path(scratch, "logs")
for (id in c("old_run", "live_run")) {
  dir.create(file.path(logs, id), recursive = TRUE, showWarnings = FALSE)
}
writeLines("a", file.path(logs, "old_run", "shard_r001of002_c001of002.log"))
Sys.setFileTime(file.path(logs, "old_run", "shard_r001of002_c001of002.log"),
                Sys.time() - 7200)
writeLines("b", file.path(logs, "live_run", "shard_r001of001_c001of001.log"))
dir.create(file.path(logs, "no_logs_but_newest"))
ok["pattern_picks_the_run_with_the_newest_matching_file"] <- identical(
  suppressMessages(latest_run_dir(logs, prefix = "",
                                  require_pattern = "^shard_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+[.]log$")),
  "live_run")

# ── 4. safe_write_csv2 / safe_read_csv2: THE PT-BR ROUND TRIP ─────────────────
#
# Every table the project writes uses ';' and ','. A value must survive the
# trip; a string holding the separator or the decimal mark must survive it too,
# and so must NA -- because the metrics tables carry NA in the test columns on
# purpose (evaluate_test = FALSE) and a reader that turned NA into "NA" or 0
# would score a phantom.
tbl <- tibble::tibble(
  id    = c("a", "b", "c"),
  value = c(31.190645, -0.5, NA_real_),
  note  = c("plain", "has; semicolon", "has, comma and 1,5"),
  flag  = c(TRUE, FALSE, NA)
)
csv <- file.path(scratch, "round", "trip.csv")
p <- safe_write_csv2(tbl, csv)
back <- safe_read_csv2(csv)

ok["csv_returns_the_path_it_wrote"] <- identical(normalizePath(p),
                                                normalizePath(csv))
ok["csv_creates_missing_directories"] <- file.exists(csv)
ok["csv_numeric_survives_the_decimal_comma"] <-
  isTRUE(all.equal(back$value[1:2], tbl$value[1:2], tolerance = 1e-9))
ok["csv_na_is_still_na"] <- is.na(back$value[3]) && is.na(back$flag[3])
ok["csv_strings_with_separators_survive"] <- identical(back$note, tbl$note)
ok["csv_logical_survives"] <- identical(back$flag[1:2], c(TRUE, FALSE))

# the file really is the PT-BR dialect on disk
raw <- readLines(csv, n = 2)
ok["csv_on_disk_uses_semicolon_and_comma"] <-
  grepl(";", raw[1], fixed = TRUE) && grepl("31,190645", raw[2], fixed = TRUE)

# overwriting an existing file replaces it rather than appending
safe_write_csv2(tbl[1, ], csv)
ok["csv_overwrite_replaces"] <- nrow(safe_read_csv2(csv)) == 1L

# ── 5. safe_save_rds and the timestamped fallback ────────────────────────────
rds <- file.path(scratch, "round", "obj.rds")
p2 <- safe_save_rds(list(a = 1, b = "x"), rds)
ok["rds_round_trip"] <- identical(readRDS(rds), list(a = 1, b = "x"))
ok["rds_returns_the_path"] <- identical(normalizePath(p2), normalizePath(rds))

# .timestamped_path is the escape hatch when a file cannot be removed (Windows
# holds a handle). Its shape is what other code greps for, so it is asserted.
tp <- .timestamped_path("D:/some/dir/report.csv", "csv")
ok["timestamped_path_keeps_dir_and_stem"] <-
  identical(dirname(tp), "D:/some/dir") && startsWith(basename(tp), "report_")
ok["timestamped_path_has_a_timestamp_and_the_ext"] <-
  grepl("^report_[0-9]{8}_[0-9]{6}[.]csv$", basename(tp))

# ── 6. .drop_test_rows: ONE RULE, THREE SHAPES ───────────────────────────────
#
# It gates evaluate_test = FALSE across three artefacts, and one of those doors
# was once found unlocked while the test read only the other two.
df <- tibble::tibble(dataset_role = c("train", "validation", "test"), v = 1:3)
ok["drop_test_removes_test_rows"] <-
  identical(.drop_test_rows(df, evaluate_test = FALSE)$dataset_role,
            c("train", "validation"))
ok["drop_test_keeps_everything_when_evaluating"] <-
  identical(.drop_test_rows(df, evaluate_test = TRUE), df)
ok["drop_test_passes_through_non_tables"] <-
  is.null(.drop_test_rows(NULL, FALSE)) &&
  identical(.drop_test_rows(list(x = 1), FALSE), list(x = 1))
ok["drop_test_passes_through_tables_without_a_role"] <-
  identical(.drop_test_rows(tibble::tibble(v = 1:2), FALSE), tibble::tibble(v = 1:2))

# ── 7. .resumable_units: A config_id IS A LABEL, NOT AN IDENTITY ─────────────
#
# The grid stores window_sizes as a list-column c(9L, 15L); the record flattens
# it as "9x15" and conv_channels as "64_128". Compared literally, every cached
# unit looks stale and the guard forces a pointless full retrain -- the worse
# failure, because it costs hours and looks like it worked. Compared not at all,
# a cfg_003 that now names a different architecture is resumed under the old
# weights. Both directions are asserted.
grid <- tibble::tibble(
  config_id     = c("cfg_001", "cfg_002"),
  window_sizes  = list(c(9L, 15L), c(15L)),
  conv_channels = list(c(64L, 128L), c(64L, 128L)),
  base_lr       = c(1e-4, 3e-4)
)
rec <- tibble::tibble(
  unit_id       = c("cfg_001_f1_s1", "cfg_002_f1_s1", "cfg_009_f1_s1"),
  config_id     = c("cfg_001", "cfg_002", "cfg_009"),
  window_sizes  = c("9x15", "15", "3"),
  conv_channels = c("64_128", "64_128", "32"),
  base_lr       = c("1e-04", "3e-04", "1e-03"),
  val_ccc       = c(0.4, 0.5, 0.1)
)
done <- rec$unit_id

kept <- suppressMessages(.resumable_units(done, rec, grid))
ok["resume_keeps_units_whose_hyperparameters_match"] <-
  all(c("cfg_001_f1_s1", "cfg_002_f1_s1") %in% kept)
ok["resume_keeps_a_cached_config_the_grid_no_longer_names"] <-
  "cfg_009_f1_s1" %in% kept

# now cfg_002 means something else: same label, different window
grid2 <- grid
grid2$window_sizes[[2]] <- c(3L, 9L)
kept2 <- suppressMessages(.resumable_units(done, rec, grid2))
ok["resume_refits_a_relabelled_config"] <- !("cfg_002_f1_s1" %in% kept2)
ok["resume_leaves_the_matching_one_alone"] <- "cfg_001_f1_s1" %in% kept2

# 7 and 7L and "7" are one value; a numeric written either way must match
grid3 <- grid
grid3$base_lr <- c(0.0001, 0.0003)      # same numbers, different spelling
ok["resume_does_not_care_how_a_number_was_spelt"] <-
  setequal(suppressMessages(.resumable_units(done, rec, grid3)), kept)

# degenerate inputs pass straight through instead of erroring
ok["resume_with_nothing_done_is_a_no_op"] <-
  identical(.resumable_units(character(0), rec, grid), character(0))
ok["resume_with_an_empty_record_is_a_no_op"] <-
  identical(.resumable_units(done, rec[0, ], grid), done)

# ── 8. print_wide: returns its input invisibly and does not truncate columns ──
wide <- tibble::as_tibble(as.list(stats::setNames(seq_len(40), paste0("c", 1:40))))
out <- utils::capture.output(res <- print_wide(wide))
ok["print_wide_returns_the_table"] <- identical(res, wide)
ok["print_wide_shows_the_last_column"] <- any(grepl("c40", out))

# ── 9. env_chr / env_int / env_csv: ONE READER FOR THE OVERRIDES ─────────────
#
# Three scripts each carried a private copy and they had already drifted: one
# stopped trimming whitespace. The value "  8 " must read as 8 everywhere, a
# seed list with one bad entry must refuse rather than shrink, and an override
# must announce itself -- a variable left set in a session is how stage 04
# nearly trained the wrong configs.
Sys.unsetenv(c("dlc_test_chr", "dlc_test_int", "dlc_test_csv"))
ok["env_unset_returns_the_default"] <-
  identical(env_chr("dlc_test_chr", "dflt"), "dflt") &&
  identical(env_int("dlc_test_int", 7L), 7L) &&
  identical(env_csv("dlc_test_csv", c("a", "b")), c("a", "b"))

Sys.setenv(dlc_test_chr = "  spatial ", dlc_test_int = " 8 ",
           dlc_test_csv = " cfg_003, cfg_002,, ")
ok["env_values_are_trimmed"] <-
  identical(suppressMessages(env_chr("dlc_test_chr", "x")), "spatial") &&
  identical(suppressMessages(env_int("dlc_test_int", 1L)), 8L)
ok["env_csv_drops_empty_items_and_trims"] <-
  identical(suppressMessages(env_csv("dlc_test_csv", NULL)), c("cfg_003", "cfg_002"))

# an override announces itself, so a forgotten one cannot be silent
msg <- character(0)
withCallingHandlers(env_int("dlc_test_int", 1L),
  message = function(m) { msg <<- c(msg, conditionMessage(m)); invokeRestart("muffleMessage") })
ok["env_override_is_announced"] <- any(grepl("from the environment", msg))

Sys.setenv(dlc_test_int = "eight", dlc_test_csv = "7,28,x")
ok["env_int_refuses_a_non_integer"] <- inherits(
  try(env_int("dlc_test_int", 1L), silent = TRUE), "try-error")
ok["env_int_refuses_below_min"] <- {
  Sys.setenv(dlc_test_int = "0")
  inherits(try(env_int("dlc_test_int", 1L), silent = TRUE), "try-error")
}
ok["env_csv_names_the_bad_item_and_refuses"] <- {
  e <- tryCatch(env_csv("dlc_test_csv", NULL, as_int = TRUE),
                error = function(e) conditionMessage(e))
  is.character(e) && grepl("x", e, fixed = TRUE)
}
Sys.setenv(dlc_test_csv = "7, 28 ,42")
ok["env_csv_as_int_parses"] <-
  identical(suppressMessages(env_csv("dlc_test_csv", NULL, as_int = TRUE)),
            c(7L, 28L, 42L))
Sys.setenv(dlc_test_csv = " , ,")
ok["env_csv_all_empty_is_refused"] <- inherits(
  try(env_csv("dlc_test_csv", NULL), silent = TRUE), "try-error")
Sys.unsetenv(c("dlc_test_chr", "dlc_test_int", "dlc_test_csv"))

unlink(scratch, recursive = TRUE, force = TRUE)

cat(sprintf("  latest_run_dir           : picks by time (%s), skips unfinished, names them\n",
            picked))
cat(sprintf("  pt-br csv round trip     : %.6f survives; NA stays NA; '%s' intact\n",
            back$value[1], tbl$note[2]))
cat("  resume identity          : relabelled cfg_002 is refitted, spelling of 1e-4 is not\n")

.report(ok, "test_run_dirs")
