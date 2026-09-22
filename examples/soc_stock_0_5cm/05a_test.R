
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
source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c("processx")
install_load_pkg(pkg)

# ══════════════════════════════════════════════════════════════════════════════
# 05a_test -- test of the 2D pipeline before running the full job
#
# Runs only test_n_shards shards chosen from the n_row_shards x n_col_shards grid.
# The point is to check:
#   1. Geometry of the output tiles (extent, resolution, nrows/ncols)
#   2. Values produced (median, sd, mask) -- plausibility
#   3. RSS per process (the real RAM floor with 2D)
#   4. Throughput (s/block) to estimate the ETA of the real job
#
# At the end it prints a diagnostic comparing RAM and throughput against 05
# (reference: ~7-8 GB floor, ~130 s/block after optimisation).
#
# It does NOT run the merge -- the test tiles stay in raster/parts_2d/ for
# manual inspection. Delete them before running the real 05a.
# ══════════════════════════════════════════════════════════════════════════════

# ── TEST configuration ─────────────────────────────────────────────────────────
# These values must be the SAME ones you plan to use in the real 05a.
# Only test_n_shards and max_concurrent may be smaller in the test.

n_row_shards <- 250      # same as what you will use in the real 05a
n_col_shards <- 4        # same as what you will use in the real 05a

# How many shards to run in the test. 4 = 1 per column (tests the full width).
# Raise it to 8 if you want to test 2 rows of shards.
test_n_shards <- 4

# Which shards to run. "auto" = spread over the 4 column tiles in row 1
# (tests the geometry in every column). Or set them by hand, e.g.:
#   test_shards <- list(c(1,1), c(1,2), c(1,3), c(1,4))
test_shards <- "auto"

# Concurrency and RAM (may be smaller than the real ones for the test)
max_concurrent <- 2
poll_interval_s <- 15

# ── Paths ──────────────────────────────────────────────────────────────────────

script_dir    <- file.path(project_root, "examples", "soc_stock_0_5cm")
worker_script <- file.path(script_dir, "05_predict_spatial.R")

log_dir <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs",
                     paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_TEST2D"))
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

rscript_bin <- file.path(R.home("bin"), "Rscript.exe")
if (!file.exists(rscript_bin)) stop("Rscript.exe not found: ", rscript_bin)

# ── Build the list of test shards ────────────────────────────────────────────

if (identical(test_shards, "auto")) {
  # Bug fixed: the previous version only added a sample from the middle row
  # (denser/tropical -- the one that really calibrates real throughput/RAM)
  # when test_n_shards > n_col_shards. With this file's default values
  # (both 4) that condition was NEVER TRUE, so the test only saw row 1
  # (almost all ocean/polar ice, except where it crosses land) and the
  # max_concurrent/ETA recommendation came out of a distorted mean (nearly
  # empty shards with RSS~1.7 GB and predict~2s, hiding the one real shard
  # with RSS~12.7 GB and predict~550s -- mean = meaningless number, dangerous
  # if used for RAM: a max_concurrent so high that concurrent dense shards
  # would blow up the machine's RAM).
  # Now it ALWAYS reserves at least 1 shard from the middle row, even when
  # it has to reduce row 1's width coverage.
  n_row1 <- max(1L, min(test_n_shards - 1L, n_col_shards))
  n_mid  <- max(1L, test_n_shards - n_row1)
  mid_row <- ceiling(n_row_shards / 2)
  test_shards <- c(
    lapply(seq_len(n_row1), function(cs) c(1L, cs)),
    lapply(seq_len(min(n_mid, n_col_shards)), function(cs) c(mid_row, cs))
  )
}

n_test <- length(test_shards)
message(sprintf("2D test: %d x %d grid | running %d shard(s) | %d at a time",
                n_row_shards, n_col_shards, n_test, max_concurrent))
message(sprintf("Test shards: %s",
                paste(sapply(test_shards, function(x) sprintf("[r%d/c%d]", x[1], x[2])),
                      collapse = " ")))
message("Logs: ", log_dir, "\n")

# ── Queue ──────────────────────────────────────────────────────────────────────

pending    <- seq_len(n_test)
active     <- list()
exit_codes <- integer(n_test)
rss_peak   <- numeric(n_test)   # max RSS read from each shard's log

launch_shard <- function(idx) {
  rs <- test_shards[[idx]][1]
  cs <- test_shards[[idx]][2]
  log_file <- file.path(log_dir,
    sprintf("test_r%03dof%03d_c%03dof%03d.log", rs, n_row_shards, cs, n_col_shards))
  p <- processx::process$new(
    rscript_bin,
    args = c(worker_script,
             as.character(rs), as.character(cs),
             as.character(n_row_shards), as.character(n_col_shards),
             as.character(max_concurrent)),
    # Passed explicitly, as 05a does: the worker reads soc_predict_raster_dir
    # to choose its grid, and a test on the 20 km grid must not silently
    # measure the 250 m one.
    env = if (nzchar(Sys.getenv("soc_predict_raster_dir"))) {
      c("current", soc_predict_raster_dir = Sys.getenv("soc_predict_raster_dir"))
    } else NULL,
    stdout  = log_file,
    stderr  = log_file,
    cleanup = TRUE
  )
  message(sprintf("  Shard [r%03d/c%03d] PID %d -> %s",
                  rs, cs, p$get_pid(), basename(log_file)))
  p
}

while (length(active) < max_concurrent && length(pending) > 0) {
  idx <- pending[1]; pending <- pending[-1]
  active[[as.character(idx)]] <- launch_shard(idx)
}

t0 <- Sys.time()
n_done <- 0L

while (length(active) > 0 || length(pending) > 0) {
  Sys.sleep(poll_interval_s)

  finished_ids <- character(0)
  for (idx_chr in names(active)) {
    p <- active[[idx_chr]]
    if (!p$is_alive()) {
      idx <- as.integer(idx_chr)
      rs  <- test_shards[[idx]][1]
      cs  <- test_shards[[idx]][2]
      exit_codes[idx] <- p$get_exit_status()
      n_done <- n_done + 1L
      status <- if (exit_codes[idx] == 0L) "OK" else paste0("FAILED (exit ", exit_codes[idx], ")")
      message(sprintf("  [r%03d/c%03d] finished: %s", rs, cs, status))
      finished_ids <- c(finished_ids, idx_chr)
    }
  }
  active[finished_ids] <- NULL

  while (length(active) < max_concurrent && length(pending) > 0) {
    idx <- pending[1]; pending <- pending[-1]
    active[[as.character(idx)]] <- launch_shard(idx)
  }

  el <- Sys.time() - t0
  message(sprintf("  [%.1f %s] %d/%d done, %d running, %d waiting",
                  as.numeric(el), units(el), n_done, n_test, length(active), length(pending)))
}

el_total <- Sys.time() - t0

# ── Log diagnostics ────────────────────────────────────────────────────────────

message("\n", strrep("═", 70))
message("2D TEST DIAGNOSTICS")
message(strrep("═", 70))

parse_worker_log <- function(idx) {
  rs <- test_shards[[idx]][1]
  cs <- test_shards[[idx]][2]
  log_file <- file.path(log_dir,
    sprintf("test_r%03dof%03d_c%03dof%03d.log", rs, n_row_shards, cs, n_col_shards))
  if (!file.exists(log_file)) return(NULL)
  lines <- readLines(log_file, warn = FALSE)

  # strip_ncol vs r_ncol
  strip_line <- lines[grepl("strip_ncol", lines)]
  strip_ncol <- NA_integer_; r_ncol <- NA_integer_; ram_factor <- NA_real_
  if (length(strip_line) > 0) {
    m <- regmatches(strip_line[1], regexpr("strip_ncol (\\d+) vs r_ncol (\\d+)", strip_line[1]))
    if (length(m) > 0) {
      nums <- as.integer(regmatches(m, gregexpr("\\d+", m))[[1]])
      if (length(nums) >= 2) { strip_ncol <- nums[1]; r_ncol <- nums[2] }
      ram_factor <- if (!is.na(r_ncol) && strip_ncol > 0) r_ncol / strip_ncol else NA_real_
    }
  }

  # output_block_rows
  blk_line <- lines[grepl("Auto output_block_rows", lines)]
  output_block_rows <- NA_integer_
  if (length(blk_line) > 0) {
    m <- regmatches(blk_line[1], regexpr("= (\\d+)", blk_line[1]))
    if (length(m) > 0) output_block_rows <- as.integer(sub("= ", "", m))
  }

  # Maximum RSS
  # Bug fixed: the previous version left the " MB" suffix on the extracted
  # value (sub() only stripped the "RSS " prefix), so as.numeric("12345 MB")
  # always gave NA -- rss_peak_mb was always NA and the max_concurrent
  # recommendation fell to -Inf/absurd. Lookbehind/lookahead avoids
  # capturing the text.
  rss_lines <- lines[grepl("RSS [0-9]+(?:\\.[0-9]+)? MB", lines, perl = TRUE)]
  rss_vals  <- as.numeric(regmatches(rss_lines,
    regexpr("(?<=RSS )[0-9]+(?:\\.[0-9]+)?(?= MB)", rss_lines, perl = TRUE)))
  rss_peak_mb <- if (length(rss_vals) > 0) max(rss_vals, na.rm = TRUE) else NA_real_

  # s/block (predict)
  block_lines <- lines[grepl("predict [0-9]+\\.[0-9]+s", lines)]
  predict_times <- as.numeric(unlist(regmatches(block_lines,
    gregexpr("[0-9]+\\.[0-9]+(?=s \\| RSS)", block_lines, perl = TRUE))))
  median_predict_s <- if (length(predict_times) > 0) median(predict_times) else NA_real_

  # total n_valid (last Block line)
  n_valid <- NA_integer_
  blk_summary <- lines[grepl("^Block [0-9]+/[0-9]+", lines)]
  if (length(blk_summary) > 0) {
    m <- regmatches(blk_summary[length(blk_summary)],
                    regexpr("valid ([0-9,]+)", blk_summary[length(blk_summary)]))
    if (length(m) > 0) n_valid <- as.integer(gsub(",", "", sub("valid ", "", m)))
  }

  # exit
  ok <- exit_codes[idx] == 0L

  list(rs = rs, cs = cs, ok = ok,
       strip_ncol = strip_ncol, r_ncol = r_ncol, ram_factor = ram_factor,
       output_block_rows = output_block_rows,
       rss_peak_mb = rss_peak_mb, median_predict_s = median_predict_s,
       n_valid = n_valid, log = log_file)
}

results <- lapply(seq_len(n_test), parse_worker_log)

message(sprintf("\n%-20s %10s %10s %10s %12s %12s %8s",
                "shard", "strip_ncol", "ram_factor", "blk_rows", "RSS_peak_MB", "pred_s/blk", "status"))
message(strrep("-", 85))

for (r in results) {
  if (is.null(r)) next
  message(sprintf("[r%03d/c%03d]          %10s %10s %10s %12s %12s %8s",
                  r$rs, r$cs,
                  ifelse(is.na(r$strip_ncol), "?", format(r$strip_ncol, big.mark = ",")),
                  ifelse(is.na(r$ram_factor),  "?", sprintf("%.1fx", r$ram_factor)),
                  ifelse(is.na(r$output_block_rows), "?", r$output_block_rows),
                  ifelse(is.na(r$rss_peak_mb), "?", sprintf("%.0f", r$rss_peak_mb)),
                  ifelse(is.na(r$median_predict_s), "?", sprintf("%.1f", r$median_predict_s)),
                  if (r$ok) "OK" else "FAILED"))
}

message(strrep("-", 85))

# Checks the geometry of the tiles produced
message("\n── Tile geometry check ─────────────────────────────────────────────")
target_label <- "soc_stock_0_5cm"

output_dir <- file.path(project_root, "outputs", "spatial_prediction",
                        "soc_stock_modeling", target_label)
parts_dir  <- file.path(output_dir, "raster", "parts_2d")
config_id  <- "auto"

if (identical(config_id, "auto")) {
  final_model_base <- file.path(project_root, "outputs", "final_model",
                                "soc_stock_modeling", target_label)
  # THIS CARRIED THE DEFECT B2 FOUND, IN FULL: newest final_ by NAME, and when
  # its summary was missing, config_id stayed "auto" and travelled on to a
  # model directory that does not exist. Now: newest FINISHED run by time, the
  # config in SELECTION order (selected_cfgs is the grid's order -- with two
  # configs, [1] can be the runner-up), and a refusal rather than a silent
  # "auto" if nothing resolves.
# latest_run_dir() lives in R/utils.R; this script deliberately loads no
# more of the framework than it uses.
source(file.path(project_root, "R", "utils.R"))
  final_run_id <- latest_run_dir(
    final_model_base, prefix = "final_",
    require_file = file.path("comparison", "final_run_summary.rds"),
    label = "final_run_id")
  fs <- readRDS(file.path(final_model_base, final_run_id, "comparison",
                          "final_run_summary.rds"))
  config_id <- selected_config_id(fs, final_run_id)
  message("config_id resolved to: ", config_id)
}

test_tiles <- list.files(parts_dir,
  pattern = paste0("^", target_label, "_", config_id,
                   "_ensemble_median.*_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.tif$"),
  full.names = TRUE)

if (length(test_tiles) > 0) {
  metadata_dir  <- file.path(project_root, "outputs", "metadata",
                             "soc_stock_modeling", target_label)
  raster_table  <- readr::read_csv2(file.path(metadata_dir, "raster_table_used.csv"),
                                    show_col_types = FALSE)
  full_template <- terra::rast(raster_table$raster_file[1])

  for (tf in sort(test_tiles)) {
    r <- terra::rast(tf)
    geom_ok <- terra::compareGeom(r, full_template, stopOnError = FALSE,
                                  res = TRUE, crs = TRUE, ext = FALSE)
    message(sprintf("  %s", basename(tf)))
    message(sprintf("    nrow=%d ncol=%d | ext [%.4f %.4f %.4f %.4f] | res OK=%s",
                    terra::nrow(r), terra::ncol(r),
                    terra::xmin(r), terra::xmax(r), terra::ymin(r), terra::ymax(r),
                    geom_ok))
  }
} else {
  message("  No test tile found in: ", parts_dir)
}

# Final recommendations
message("\n── Recommendations for the real job (05a_run_parallel.R) ──────────")
ok_results <- Filter(function(r) !is.null(r) && r$ok, results)
if (length(ok_results) > 0) {
  rss_vals    <- sapply(ok_results, function(r) r$rss_peak_mb)
  pred_vals   <- sapply(ok_results, function(r) r$median_predict_s)
  ram_factors <- sapply(ok_results, function(r) r$ram_factor)

  avg_rss  <- mean(rss_vals,  na.rm = TRUE)
  avg_pred <- mean(pred_vals, na.rm = TRUE)
  avg_rf   <- mean(ram_factors, na.rm = TRUE)

  # RAM available: 64 GB = 64000 MB; 30% margin
  ram_total_mb    <- 64000
  ram_budget_mb   <- ram_total_mb * 0.70   # 70% for shards
  safe_concurrent <- floor(ram_budget_mb / max(avg_rss, 1))
  safe_concurrent <- min(safe_concurrent, parallel::detectCores())

  message(sprintf("  Peak RAM per shard  : %.0f MB (mean of the %d test shards)",
                  avg_rss, length(ok_results)))
  message(sprintf("  RAM reduction vs 05 : %.1fx (strip_ncol / r_ncol)",
                  avg_rf))
  message(sprintf("  Throughput          : %.1f s/block (predict)",
                  avg_pred))
  message(sprintf("  safe max_concurrent (70%% of 64 GB): %d processes",
                  safe_concurrent))

  n_blocks_est <- n_row_shards * n_col_shards * 32   # ~32 blocks/shard (estimate)
  eta_h <- (n_blocks_est * avg_pred) / safe_concurrent / 3600
  message(sprintf("  Estimated ETA (real job, %d x %d = %d shards): ~%.0f h (~%.1f days)",
                  n_row_shards, n_col_shards, n_row_shards * n_col_shards,
                  eta_h, eta_h / 24))

  message(sprintf("\n  -> Edit 05a_run_parallel.R: max_concurrent <- %d", safe_concurrent))
} else {
  message("  No shard completed successfully -- check the logs in: ", log_dir)
}

message(sprintf("\nTest finished in %.1f %s. Logs in: %s",
                as.numeric(el_total), units(el_total), log_dir))
message("Test tiles in: ", parts_dir)
message("Delete the test tiles before running the real 05a.")
