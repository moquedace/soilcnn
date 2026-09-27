# ══════════════════════════════════════════════════════════════════════════════
# T3 -- the map on the 250 m grid: how many workers, how many threads, and how
# much RAM, measured on full-width rows
#
# THE QUESTION.
#
# P4 proved dsm_predict() makes stage 05's map and that the chain holds on the
# 250 m grid. What it could not measure is the global run's cost: its speed
# came from the 20 km grid (rows of 2,004 columns) and from a probe unit
# ~2,000 columns wide. At 250 m a step reads rows of 160,298 columns: R
# converts every one of them, the step's buffer alone is ~5 GB, and several
# workers read the same HDD at once. Those are the numbers the global run
# depends on, and they are measured here, for two layouts of the same cores
# (the first run's three, and why the second asks about two, are below the
# settings):
#
#   a  2 workers x 7 threads    (what the RAM allows, with every core)
#   b  1 worker  x 15 threads
#
# Each layout maps ITS OWN band of full-width rows near 47 N, next to each
# other -- the same latitude, so about the same land, and none of them in the
# operating system's file cache when its turn comes. 32 rows a unit, every
# band the global map writes, both calibration sources.
#
# WHAT IS COMPARED: valid pixels per second of wall time; where a unit's time
# goes (read / network / DI); seconds per full row read and per valid pixel;
# and each worker's peak RAM against the estimate the work plan sized the
# steps with.
#
# WHAT IT DECIDES: the layout the global run uses, and its ETA. The checks
# only say the measurement is sound and the RAM model held.
#
# WHAT IT WRITES: the maps under outputs/.../spatial_prediction/.../
# dsm_predict/t3_<layout>_<commit> (full-width, ~1-3 GB each) and the tables
# in outputs/.../tuning/.../capability_sweep/t3_predict/. A second run of the
# same commit resumes the maps and only repeats the tables.
#
# COST: ~25-35 min.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_t3_predict_layouts.R")
# ══════════════════════════════════════════════════════════════════════════════

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
install_load_pkg(c("torch", "terra", "dplyr", "readr", "tibble", "purrr", "matrixStats",
                   "ps", "callr"))
source(file.path(project_root, "R", "load_all.R"))
options(width = 200)

target_label <- "soc_stock_0_5cm"
base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)
metadata_dir <- base("outputs", "metadata")
data_dir     <- base("data", "processed")
patch_dir    <- base("outputs", "patches")
final_base   <- base("outputs", "final_model")
tuning_base  <- base("outputs", "tuning")
sp_dir       <- base("outputs", "spatial_prediction")
maps_dir     <- file.path(sp_dir, "dsm_predict")
t3_dir       <- file.path(tuning_base, "capability_sweep", "t3_predict")
code_tag     <- .git_commit_at(project_root)
n_cores      <- env_int("soc_t3_n_cores", 15L)
unit_rows    <- 32L

# The layouts: the same cores, split two ways.
#
# THE FIRST RUN (commit 811beae) tried 5, 15 and 3 threads a worker. The RAM
# cut the first to 2 workers and the third to 3, and every layout computed
# ~20,000 valid px/s whatever its split: 2 x 5, 1 x 15 and 3 x 3 threads gave
# 21.3k, 20.4k and 20.0k px/s of compute. Beyond ~10 threads the machine's
# memory bandwidth, not its cores, sets the pace. Its workers also peaked at
# 24-32 GB (the reader's garbage, since fixed). So the second run asks what is
# left: 2 workers x 7 threads -- what the RAM allows, with every core --
# against 1 x 15.
layouts <- tibble::tibble(layout = c("a", "b"), threads = c(7L, 15L), units = c(4L, 3L))
# Rows the earlier runs mapped may still be in the file cache: the first
# mapped 448 from the first profile's, the second (commit dda346b) the 224
# after those. This run maps the bands after both.
#
# THE SECOND RUN found the leak this one checks for: a worker's peak climbed
# ~5 GB a unit (a: 15.9 -> 21.1 GB; b: 16.9 -> 21.9 -> 25.9), while the first
# unit's peak sat on the estimate (15-17 GB against 16.2). oneDNN compiles and
# keeps a primitive per input shape, and the strips were cropped to a new
# shape at nearly every coastal chunk. t3_05 asks that the peak stop growing.
offset_rows <- env_int("soc_t3_offset_rows", 672L, min = 0L)

final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
final_dir  <- file.path(final_base, final_run_id)
summ       <- readRDS(file.path(final_dir, "comparison", "final_run_summary.rds"))
config_id  <- selected_config_id(summ, final_run_id)
cfg_row    <- summ$selected_cfgs[summ$selected_cfgs$config_id == config_id, , drop = FALSE]
tuning_dir <- file.path(tuning_base, summ$tuning_run_id)
calibration <- c(block = tuning_dir)
for (cand in file.path(tuning_base, c("soc_0_5cm_knndm_cfg003", "soc_0_5cm_design_knndm"))) {
  if (dir.exists(cand) &&
      !is.null(suppressMessages(cv_residuals_for_config(cand, cfg_row, required = FALSE)))) {
    calibration <- c(calibration, knndm = cand)
    break
  }
}

data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = 3L,
  target_col   = safe_read_csv2(file.path(metadata_dir, "target_config.csv"))$target_col[1],
  verbose      = FALSE)
rt <- safe_read_csv2(file.path(metadata_dir, "raster_table_used.csv"))
qc_path <- file.path(metadata_dir, "qc_table.csv")
g250 <- terra::rast(rt$raster_file[1])
n_row_250 <- as.integer(terra::nrow(g250)); n_col_250 <- as.integer(terra::ncol(g250))
h <- (max(unlist(cfg_row$window_sizes)) - 1L) %/% 2L

# The bands of rows: side by side from the first profile's row (47 N).
r_first <- as.integer(terra::rowFromCell(g250, terra::cellFromXY(
  g250, cbind(data$store$meta$x[1], data$store$meta$y[1]))))
starts <- r_first - 3L + offset_rows + c(0L, cumsum(layouts$units[-nrow(layouts)])) * unit_rows
layouts$r0 <- as.integer(starts)
layouts$r1 <- as.integer(starts + layouts$units * unit_rows - 1L)

message("\n", strrep("=", 78))
message("T3 -- the map on the 250 m grid: workers x threads, full-width rows")
message(strrep("=", 78))
message("  final run   : ", final_dir, "  (", config_id, ", ", length(summ$seeds), " seeds)")
message("  grid        : ", format(n_row_250, big.mark = ","), " x ", format(n_col_250, big.mark = ","),
        " | ", n_cores, " cores | units of ", unit_rows, " rows, full width")
for (i in seq_len(nrow(layouts))) {
  message(sprintf("  layout %s   : %d thread(s) a worker, rows %d-%d (%d units)", layouts$layout[i],
                  layouts$threads[i], layouts$r0[i], layouts$r1[i], layouts$units[i]))
}
message(strrep("=", 78), "\n")

required <- sprintf("t3_%02d", 1:5)
L <- check_ledger("T3")

res <- list()
maps <- list()
for (i in seq_len(nrow(layouts))) {
  ly <- layouts[i, ]
  message("\n-- layout ", ly$layout, ": ", ly$threads, " thread(s) a worker --")
  m <- dsm_predict(final_dir, data, rasters = rt, qc_table = qc_path, calibration = calibration,
                   extent = list(rows = c(ly$r0, ly$r1), cols = c(1L, n_col_250)),
                   n_cores = n_cores, threads_per_worker = ly$threads,
                   unit_rows = unit_rows, step_rows = unit_rows,
                   probe = identical(ly$layout, "a"), output_dir = maps_dir,
                   run_id = sprintf("t3_%s_%s", ly$layout, code_tag))
  maps[[ly$layout]] <- m
  u <- m$units
  rows_read <- sum(u$r1 - u$r0 + 1L + 2L * h)
  busy <- sum(u$total_s)
  # The workers that did the units, from their records: a resumed map ran
  # none in this call, and its wall time is then the units' own, shared.
  workers <- length(unique(u$worker))
  rate <- if (is.finite(m$map_minutes)) sum(u$n_valid) / (60 * m$map_minutes) else
    sum(u$n_valid) / (busy / workers)
  res[[i]] <- tibble::tibble(
    layout = ly$layout, workers = workers, threads = ly$threads,
    rows = sum(u$r1 - u$r0 + 1L), n_cells = sum(u$n_cells), n_valid = sum(u$n_valid),
    map_min = m$map_minutes, valid_px_per_s = rate,
    read_s_per_row = sum(u$read_s) / rows_read,
    s_per_valid_px = sum(u$net_s + u$di_s + u$bands_s + u$write_s) / max(1, sum(u$n_valid)),
    share_read = sum(u$read_s) / busy, share_net = sum(u$net_s) / busy,
    share_di = sum(u$di_s) / busy,
    peak_gb = max(u$peak_gb, na.rm = TRUE), estimate_gb = m$work$per_worker_gb,
    budget_gb = m$work$budget_gb, step_rows = m$work$step_rows, probe = m$probe$status)
}
tab <- dplyr::bind_rows(res)

# ── the checks: the measurement is sound, and the RAM model held ─────────────
ledger_check(L, "t3_01", "every layout mapped every unit and wrote every band",
             all(vapply(seq_len(nrow(layouts)), function(i) {
               m <- maps[[layouts$layout[i]]]
               nrow(m$units) == layouts$units[i] && all(file.exists(m$vrt))
             }, logical(1))),
             paste(sprintf("%s: %d unit(s), %d band(s)", layouts$layout,
                           vapply(maps, function(m) nrow(m$units), integer(1)),
                           vapply(maps, function(m) nrow(m$bands), integer(1))), collapse = " | "))
ledger_check(L, "t3_02", "the probe passes on the 250 m grid (layout a)",
             identical(maps$a$probe$status, "pass"),
             sprintf("%s -- %s", maps$a$probe$status, maps$a$probe$reason %||% ""))
ledger_check(L, "t3_03", "no worker's peak exceeded the RAM model's estimate by more than 25%",
             all(tab$peak_gb <= 1.25 * tab$estimate_gb),
             paste(sprintf("%s: %.1f GB of %.1f estimated", tab$layout, tab$peak_gb, tab$estimate_gb),
                   collapse = " | "))
ledger_check(L, "t3_04", "the workers together stayed inside the RAM budget",
             all(is.na(tab$budget_gb) | tab$workers * tab$peak_gb <= tab$budget_gb),
             paste(sprintf("%s: %d x %.1f GB against %.1f", tab$layout, tab$workers, tab$peak_gb,
                           tab$budget_gb), collapse = " | "))
# A unit record carries its worker's peak so far: along one worker's units it
# may rise by the first steps' shapes, and then must stop -- the global run is
# ~250 units a worker, and a climb of a GB a unit ends it.
growth <- dplyr::bind_rows(lapply(names(maps), function(ly) {
  u <- maps[[ly]]$units
  dplyr::bind_rows(lapply(split(u, u$worker), function(w) {
    w <- w[order(w$finished_at), , drop = FALSE]
    tibble::tibble(layout = ly, worker = w$worker[1], units = nrow(w),
                   first_gb = w$peak_gb[1], last_gb = w$peak_gb[nrow(w)])
  }))
}))
ledger_check(L, "t3_05", "a worker's peak stops growing from unit to unit",
             all(growth$units < 2L | growth$last_gb - growth$first_gb <= 1),
             paste(sprintf("%s/w%d: %.1f -> %.1f GB over %d unit(s)", growth$layout, growth$worker,
                           growth$first_gb, growth$last_gb, growth$units), collapse = " | "))

# ── what the global run costs, per layout ────────────────────────────────────
#
# A unit's time is its rows read plus its valid pixels computed; the global
# run cuts units of 256 rows, so it reads 256 + 2h rows per 256, and the
# workers share both. The valid pixels are the 20 km map's share of the grid
# (22.4%): a first-order number, since a band of 47 N is more land than the
# globe's average.
valid_global <- as.numeric(n_row_250) * n_col_250 * 0.2245
rows_global  <- n_row_250 * (256 + 2 * h) / 256
tab$eta_h <- (rows_global * tab$read_s_per_row + valid_global * tab$s_per_valid_px) /
  tab$workers / 3600

message("\n-- The layouts on full-width 250 m rows --")
print_wide(dplyr::mutate(tab, map_min = round(map_min, 1), valid_px_per_s = round(valid_px_per_s),
                         read_s_per_row = round(read_s_per_row, 3),
                         s_per_valid_px = signif(s_per_valid_px, 3),
                         share_read = round(100 * share_read), share_net = round(100 * share_net),
                         share_di = round(100 * share_di), peak_gb = round(peak_gb, 1),
                         estimate_gb = round(estimate_gb, 1), budget_gb = round(budget_gb, 1),
                         eta_h = round(eta_h, 1)), n = Inf)

best <- tab[which.max(tab$valid_px_per_s), , drop = FALSE]
message(sprintf(paste0(
  "\nFastest here: layout %s -- %d worker(s) x %d thread(s), %s valid px/s.\n",
  "The global map at that rate: ~%.0f h (%.1f days), %s valid pixels, %s rows read.\n",
  "An estimate from %s rows near 47 N, not a measurement of the globe."),
  best$layout, best$workers, best$threads, format(round(best$valid_px_per_s), big.mark = ","),
  best$eta_h, best$eta_h / 24, format(signif(valid_global, 3), big.mark = ","),
  format(round(rows_global), big.mark = ","), format(sum(tab$rows), big.mark = ",")))

create_output_dirs(t3_dir)
safe_write_csv2(tab, file.path(t3_dir, "t3_layouts.csv"))
v <- ledger_verdict(L, required, file.path(t3_dir, "t3_checks.csv"))
message("\nTables: ", t3_dir)
