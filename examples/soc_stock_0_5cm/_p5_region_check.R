# ══════════════════════════════════════════════════════════════════════════════
# P5 -- the global map's rehearsal: dsm_predict() over Brazil at 250 m
#
# THE QUESTION.
#
# P4 showed dsm_predict() makes stage 05's map on the 20 km grid, and that the
# chain holds on the 250 m rasters (the probe). T3 measured its speed and
# memory on full-width 250 m rows, over two or three units a worker. What
# neither did is what the global map will do: hours of work, dozens of units
# a worker, on the grid the model was trained on. P5 does that over Brazil --
# ~17,500 x 17,500 cells, 69 units of 256 rows -- with the settings
# 05_dsm_predict_global.R uses, and checks:
#
#   p5_01  the probe passes: the profiles' own pixels, every seed, before the map
#   p5_02  every unit finished and every band has its mosaic
#   p5_03  no worker's peak exceeded the RAM model's estimate by more than 25%
#   p5_04  memory stays flat: after a worker process's second unit its peak
#          rises by at most 1 GB, and no worker had to give its memory back
#   p5_05  a part across two seams -- between two units and between two
#          chunks of columns -- mapped on its own, is that part of the whole
#   p5_06  every band keeps its order at every pixel of 64 rows drawn at
#          random: min <= median <= max, each interval around the median, DI
#          >= 0, the AOA the DI against its threshold, and nothing where the
#          mask says no prediction
#
# and prints what a person should look at: where the map is (one VRT per
# band, for QGIS), the share inside the AOA, the spread of the median, the
# speed. The checks say the code did what it should; whether the map looks
# like Brazil's soils is for someone who knows them.
#
# WHAT IT WRITES: outputs/.../spatial_prediction/.../dsm_predict/
# p5_brasil_<commit> (~10 GB) and p5_brasil_seam_<commit>, and the checks in
# outputs/.../tuning/.../capability_sweep/p5_region/. The maps are named by
# the code that made them: a second run of the same commit RESUMES the map --
# a reboot costs only the units in flight -- and repeats the checks; a new
# commit maps it anew.
#
# COST: ~3-5 h at T3's rates, 2 workers x 7 threads when two fit in the RAM
# (else 1 x 15, as in the global script). Pause Windows Update and keep the
# machine from sleeping.
#
# SETTINGS (environment variables, optional):
#   soc_p5_extent    "xmin,xmax,ymin,ymax" in degrees; Brazil's mainland by default
#   soc_p5_threads   7 or 15 -- chosen from the free RAM when unset
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_p5_region_check.R")
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
maps_dir     <- file.path(base("outputs", "spatial_prediction"), "dsm_predict")
p5_dir       <- file.path(tuning_base, "capability_sweep", "p5_region")
code_tag     <- .git_commit_at(project_root)

# Brazil's mainland with a margin: Monte Caburai (5.27 N), Arroio Chui
# (33.75 S), Serra do Divisor (73.99 W), Ponta do Seixas (34.79 W). A
# rectangle, so it holds some of every neighbour, and the sea to the east.
box <- suppressWarnings(as.numeric(env_csv("soc_p5_extent", c("-74.1", "-34.7", "-33.9", "5.4"))))
if (length(box) != 4L || anyNA(box) || box[1] >= box[2] || box[3] >= box[4]) {
  stop("soc_p5_extent must be \"xmin,xmax,ymin,ymax\" in degrees, xmin < xmax and ymin < ymax.",
       call. = FALSE)
}
n_cores   <- env_int("soc_p5_n_cores", 15L)
unit_rows <- 256L                # as the global map
step_rows <- 32L

# ── the model and its calibration, as the global script ──────────────────────
final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
final_dir  <- file.path(final_base, final_run_id)
summ       <- readRDS(file.path(final_dir, "comparison", "final_run_summary.rds"))
config_id  <- selected_config_id(summ, final_run_id)
cfg_row    <- summ$selected_cfgs[summ$selected_cfgs$config_id == config_id, , drop = FALSE]
tuning_dir <- file.path(tuning_base, summ$tuning_run_id)
knndm_dir  <- NULL
for (cand in file.path(tuning_base, c("soc_0_5cm_knndm_cfg003", "soc_0_5cm_design_knndm"))) {
  if (dir.exists(cand) &&
      !is.null(suppressMessages(cv_residuals_for_config(cand, cfg_row, required = FALSE)))) {
    knndm_dir <- cand
    break
  }
}
if (is.null(knndm_dir)) {
  stop("No kNNDM tuning run with cross-validated residuals for ", config_id, " under ",
       tuning_base, ": the rehearsal maps what the global map will, both sources.",
       call. = FALSE)
}
calibration <- c(block = tuning_dir, knndm = knndm_dir)

data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = 3L,                                   # no patch is needed; the smallest file
  target_col   = safe_read_csv2(file.path(metadata_dir, "target_config.csv"))$target_col[1],
  verbose      = FALSE)
rt <- safe_read_csv2(file.path(metadata_dir, "raster_table_used.csv"))
qc_path <- file.path(metadata_dir, "qc_table.csv")

# 2 x 7 when two workers fit in the RAM dsm_predict() budgets, else 1 x 15:
# the global script's rule, so the rehearsal runs as the map will. A region
# narrower than the world has a smaller window, so two fit more easily here.
ram_free <- tryCatch(ps::ps_system_memory()$avail / 1e9, error = function(e) NA_real_)
two_fit  <- is.finite(ram_free) && 0.7 * ram_free >= 2 * 14.6
threads  <- env_int("soc_p5_threads", if (two_fit) 7L else 15L)

message("\n", strrep("=", 78))
message("P5 -- the global map's rehearsal: dsm_predict() over Brazil at 250 m")
message(strrep("=", 78))
message("  final run   : ", final_dir, "  (", config_id, ", ", length(summ$seeds), " seeds)")
message("  calibration : block = ", basename(tuning_dir), " | knndm = ", basename(knndm_dir))
message(sprintf("  extent      : %.2f to %.2f E, %.2f to %.2f N | units of %d rows in steps of %d",
                box[1], box[2], box[3], box[4], unit_rows, step_rows))
message(sprintf("  layout      : %d thread(s) a worker on %d cores | RAM free %.1f GB", threads,
                n_cores, ram_free))
message(strrep("=", 78), "\n")

required <- sprintf("p5_%02d", 1:6)
L <- check_ledger("P5")

m <- dsm_predict(final_dir, data, rasters = rt, qc_table = qc_path, calibration = calibration,
                 extent = box, n_cores = n_cores, threads_per_worker = threads,
                 unit_rows = unit_rows, step_rows = step_rows, probe = TRUE,
                 output_dir = maps_dir, run_id = paste0("p5_brasil_", code_tag))

ledger_check(L, "p5_01", "the probe passes on the 250 m grid, before the map",
             identical(m$probe$status, "pass"),
             sprintf("%s -- %d profile(s), max relative difference %.2e", m$probe$status,
                     m$probe$n %||% 0L, m$probe$max_rel_diff %||% NA_real_))

u <- m$units
n_expected <- as.integer(ceiling(m$grid$n_rows_out / m$work$unit_rows))
ledger_check(L, "p5_02", "every unit finished and every band has its mosaic",
             nrow(u) == n_expected && length(m$vrt) == nrow(m$bands) && all(file.exists(m$vrt)),
             sprintf("%d of %d unit(s) | %d band(s), %d mosaic(s) | rows %d-%d, cols %d-%d",
                     nrow(u), n_expected, nrow(m$bands), sum(file.exists(m$vrt)),
                     m$grid$rows[1], m$grid$rows[2], m$grid$cols[1], m$grid$cols[2]))

# ── memory: by process, over the whole run ────────────────────────────────────
#
# A unit's record carries its worker's peak so far and its process: a slot's
# worker is replaced after a restart and a resumed map starts new ones, so
# the climb is measured per process. The first two units of a process may
# raise its peak (the first strips, the first DI batches); after that it must
# stop.
est <- m$work$per_worker_gb
pk  <- max(u$peak_gb, na.rm = TRUE)
ledger_check(L, "p5_03", "no worker's peak exceeded the RAM model's estimate by more than 25%",
             is.finite(pk) && pk <= 1.25 * est,
             sprintf("%.1f GB of %.1f estimated (%d worker(s) x %d thread(s))", pk, est,
                     m$work$n_workers, m$work$threads))

proc <- if (all(is.na(u$pid))) paste0("w", u$worker) else paste0("p", u$pid)
rise <- vapply(split(u, proc), function(w) {
  w <- w[order(w$finished_at), , drop = FALSE]
  if (nrow(w) < 3L) NA_real_ else max(w$peak_gb[-(1:2)]) - w$peak_gb[2]
}, numeric(1))
per_proc <- table(proc)
notes <- Sys.glob(file.path(m$run_dir, "units", "*", "recycled.rds"))
ledger_check(L, "p5_04", "memory stays flat: a process's peak rises <= 1 GB after its second unit, and none left over its memory",
             any(is.finite(rise)) && all(rise[is.finite(rise)] <= 1) && length(notes) == 0L,
             sprintf("%d process(es), %d-%d unit(s) each | worst rise %.2f GB | %d exit(s) over %.1f GB",
                     length(per_proc), min(per_proc), max(per_proc),
                     if (any(is.finite(rise))) max(rise, na.rm = TRUE) else NA_real_,
                     length(notes), m$manifest$recycle_gb))

# ── the seams: a part across a unit boundary and a chunk boundary ─────────────
#
# The whole map's units meet every 256 rows, and within a unit its chunks of
# 2,048 columns meet. A part of 96 x 600 cells centred on one meeting of
# each, mapped as a map of its own, has its own units and chunks -- its rows
# there sit inside one step, its columns inside one chunk -- so any seam the
# whole left would show as a difference. The part is placed where the land
# is: among the meetings near the middle, the one whose four quadrants hold
# the most valid pixels, the fewest of them counting.
R0 <- m$grid$rows[1]; C0 <- m$grid$cols[1]
win <- function(vrt, rows, cols, origin) {
  r <- terra::rast(vrt)
  terra::readStart(r)
  on.exit(terra::readStop(r))
  v <- terra::readValues(r, row = rows[1] - origin[1] + 1L, nrows = rows[2] - rows[1] + 1L,
                         col = cols[1] - origin[2] + 1L, ncols = cols[2] - cols[1] + 1L)
  matrix(v, nrow = rows[2] - rows[1] + 1L, byrow = TRUE)
}
near_middle <- function(x, k) x[order(abs(seq_along(x) - (length(x) + 1) / 2))][seq_len(min(k, length(x)))]
uo <- u[order(u$r0), , drop = FALSE]
row_seams <- near_middle(uo$r1[-nrow(uo)], 8L)          # a unit's last row
cc <- m$work$chunk_cols
col_seams <- C0 + cc * seq_len(max(0L, (m$grid$n_cols_out - 1L) %/% cc))  # a chunk's first column
col_seams <- near_middle(col_seams[col_seams - 300L >= C0 & col_seams + 299L <= m$grid$cols[2]], 4L)
best <- NULL
for (b in row_seams) for (k in col_seams) {
  if (b - 47L < R0 || b + 48L > m$grid$rows[2]) next
  vm <- win(m$vrt[["valid_mask"]], c(b - 47L, b + 48L), c(k - 300L, k + 299L), c(R0, C0)) == 1
  q <- c(sum(vm[1:48, 1:300], na.rm = TRUE), sum(vm[1:48, 301:600], na.rm = TRUE),
         sum(vm[49:96, 1:300], na.rm = TRUE), sum(vm[49:96, 301:600], na.rm = TRUE))
  if (is.null(best) || min(q) > best$score) best <- list(b = b, k = k, score = min(q), n = sum(q))
}
seam_ok <- FALSE
seam_msg <- "no unit boundary and chunk boundary with land around them"
if (!is.null(best) && best$score >= 100L) {
  pr <- c(best$b - 47L, best$b + 48L); pc <- c(best$k - 300L, best$k + 299L)
  part <- dsm_predict(final_dir, data, rasters = rt, qc_table = qc_path, calibration = calibration,
                      extent = list(rows = pr, cols = pc), n_cores = n_cores,
                      threads_per_worker = threads, unit_rows = unit_rows, step_rows = step_rows,
                      probe = FALSE, output_dir = maps_dir,
                      run_id = paste0("p5_brasil_seam_", code_tag), verbose = FALSE)
  ct <- m$calibration
  worst <- c(rel = 0, di = 0); flips <- 0L; same_na <- TRUE
  for (i in seq_len(nrow(m$bands))) {
    bd <- m$bands[i, ]
    a <- win(part$vrt[[bd$band]], pr, pc, c(pr[1], pc[1]))
    w <- win(m$vrt[[bd$band]], pr, pc, c(R0, C0))
    same_na <- same_na && identical(is.na(a), is.na(w))
    f <- is.finite(a) & is.finite(w)
    if (bd$kind %in% c("valid_mask")) {
      flips <- flips + sum(a[f] != w[f])
    } else if (identical(bd$kind, "aoa")) {
      d <- win(m$vrt[[ct$di_band[ct$source == bd$source][1]]], pr, pc, c(R0, C0))
      away <- f & is.finite(d) & abs(d - ct$aoa_threshold[ct$source == bd$source][1]) > 1e-6
      flips <- flips + sum(a[away] != w[away])
    } else if (identical(bd$kind, "di")) {
      if (any(f)) worst[["di"]] <- max(worst[["di"]], max(abs(a[f] - w[f])))
    } else if (any(f)) {
      worst[["rel"]] <- max(worst[["rel"]], max(abs(a[f] - w[f]) / (1 + abs(w[f]))))
    }
  }
  seam_ok <- same_na && flips == 0L && worst[["rel"]] < 1e-5 && worst[["di"]] < 1e-6
  seam_msg <- sprintf("rows %d-%d (units meet after %d), cols %d-%d (chunks meet at %d): %s valid px, >= %d a quadrant | worst %.1e relative, DI %.1e | %d mask/AOA pixel(s) differ%s",
                      pr[1], pr[2], best$b, pc[1], pc[2], best$k, format(best$n, big.mark = ","),
                      best$score, worst[["rel"]], worst[["di"]], flips,
                      if (same_na) "" else " | the NA pattern differs")
}
ledger_check(L, "p5_05", "a part across a unit seam and a chunk seam is that part of the whole",
             seam_ok, seam_msg)

# ── every band in its order, at 64 rows drawn at random ──────────────────────
set.seed(20260927)
rows_s <- sort(sample(seq_len(m$grid$n_rows_out), min(64L, m$grid$n_rows_out)))
read_rows <- function(vrt) {
  r <- terra::rast(vrt)
  terra::readStart(r)
  on.exit(terra::readStop(r))
  unlist(lapply(rows_s, function(i) terra::readValues(r, row = i, nrows = 1L)))
}
S <- lapply(stats::setNames(m$bands$band, m$bands$band), function(b) read_rows(m$vrt[[b]]))
bt <- m$bands
v <- S$valid_mask == 1
v[is.na(v)] <- FALSE
bad <- character(0)
need <- function(ok, what) if (!isTRUE(ok)) bad <<- c(bad, what)
need(!anyNA(S$valid_mask), "valid_mask has NA inside the extent")
for (b in setdiff(bt$band, "valid_mask")) {
  need(all(is.finite(S[[b]][v])), paste0(b, " is missing where the mask says predicted"))
  need(all(is.na(S[[b]][!v])), paste0(b, " has values where the mask says not predicted"))
}
md <- S$ensemble_median[v]
need(all(S$ensemble_min[v] <= md & md <= S$ensemble_max[v]), "min <= median <= max")
need(all(S$ensemble_min[v] <= S$ensemble_mean[v] & S$ensemble_mean[v] <= S$ensemble_max[v]),
     "min <= mean <= max")
need(all(S$ensemble_sd[v] >= 0) && all(S$ensemble_mad[v] >= 0), "sd and mad >= 0")
ivs <- unique(bt[bt$kind == "interval", c("source", "label", "method"), drop = FALSE])
for (i in seq_len(nrow(ivs))) {
  lo <- bt$band[bt$kind == "interval" & bt$source == ivs$source[i] & bt$label == ivs$label[i] &
                  bt$method == ivs$method[i] & bt$side == "lower"]
  hi <- sub("_lower_", "_upper_", lo, fixed = TRUE)
  need(all(S[[lo]][v] >= 0 & S[[lo]][v] <= md & md <= S[[hi]][v]),
       paste0(lo, " <= median <= ", hi, ", lower >= 0"))
}
for (b in bt$band[bt$kind == "di"]) need(all(S[[b]][v] >= 0), paste0(b, " >= 0"))
ct <- m$calibration
inside <- c()
for (s in ct$source) {
  a <- S[[paste0("aoa_", s)]][v]; d <- S[[ct$di_band[ct$source == s][1]]][v]
  thr <- ct$aoa_threshold[ct$source == s][1]
  away <- abs(d - thr) > 1e-6
  need(all(a %in% c(0, 1)) && all(a[away] == as.integer(d[away] <= thr)),
       paste0("aoa_", s, " is the DI against ", signif(thr, 4)))
  inside[s] <- mean(a == 1)
}
ledger_check(L, "p5_06", "every band keeps its order at every pixel of the sampled rows",
             sum(v) > 0L && length(bad) == 0L,
             sprintf("%d row(s), %s cell(s), %s predicted%s", length(rows_s),
                     format(length(v), big.mark = ","), format(sum(v), big.mark = ","),
                     if (length(bad)) paste0(" | FAILED: ", paste(bad, collapse = "; ")) else ""))

# ── what a person should look at ──────────────────────────────────────────────
tot <- colSums(u[, c("read_s", "net_s", "di_s", "bands_s", "write_s")])
message(sprintf("\n(info) the map: %s of %s cell(s) predicted (%.1f%%) | %d unit(s) | %.1f h of mapping%s",
                format(sum(u$n_valid), big.mark = ","), format(sum(u$n_cells), big.mark = ","),
                100 * sum(u$n_valid) / max(1, sum(u$n_cells)), nrow(u), m$map_minutes / 60,
                if (is.finite(m$map_minutes)) "" else " (resumed: see the first run)"))
message(sprintf("(info) speed: %s valid px/s over the run | time inside the units: read %.0f%% | network %.0f%% | DI %.0f%% | bands %.0f%% | write %.0f%%",
                format(round(sum(u$n_valid) / max(1e-9, 60 * m$map_minutes)), big.mark = ","),
                100 * tot[["read_s"]] / sum(tot), 100 * tot[["net_s"]] / sum(tot),
                100 * tot[["di_s"]] / sum(tot), 100 * tot[["bands_s"]] / sum(tot),
                100 * tot[["write_s"]] / sum(tot)))
message(sprintf("(info) inside the AOA, at the sampled pixels: %s",
                paste(sprintf("%s %.1f%%", names(inside), 100 * inside), collapse = " | ")))
q <- stats::quantile(md, c(0.05, 0.25, 0.5, 0.75, 0.95))
message(sprintf("(info) ensemble median (t/ha) at the sampled pixels: 5%% %.1f | 25%% %.1f | 50%% %.1f | 75%% %.1f | 95%% %.1f",
                q[1], q[2], q[3], q[4], q[5]))
sz <- sum(file.size(list.files(m$run_dir, recursive = TRUE, full.names = TRUE)), na.rm = TRUE)
message(sprintf("(info) on disk: %.1f GB in %s", sz / 1e9, m$run_dir))
message("(info) to look at it: open ", m$vrt[["ensemble_median"]], " and ",
        m$vrt[[paste0("aoa_", ct$source[1])]], " in QGIS.")

create_output_dirs(p5_dir)
verdict <- ledger_verdict(L, required, file.path(p5_dir, "p5_checks.csv"))
if (verdict$pass) {
  message("\nP5 passed: hours of the global map's work over Brazil at 250 m, memory flat, no seams,",
          "\nevery band in its order. The global map is 05_dsm_predict_global.R.")
}
