# ══════════════════════════════════════════════════════════════════════════════
# T5 -- which part of the map's reader keeps memory: torch's assignment, the
# R-to-torch conversion, or terra's read
#
# THE QUESTION.
#
# T4 found the map worker's leak in the READ phase: the level a full
# collection leaves rose ~3.5 GB a unit -- the size of one full-width step's
# rows tensor (181 x 32 x 160,312 float32 = 3.71 GB) -- the same with GDAL on
# one thread and with MKL's memory manager off. torch 0.17.0's allocator does
# not cache (src/lantern/src/Allocator.cpp at v0.17.0: alloc_cpu / free_cpu),
# so memory that stays is memory still referenced, or never finalized.
#
# The reader does three things that could hold it:
#
#   torch_assign  x <- torch_empty(); x[k, , ] <- a tensor, band by band
#   torch_from_r  each band's tensor made from an R vector
#   terra_read    terra::readValues() of every band, no torch at all
#
# and then the map's own reader, and the same reader built without `[<-`:
#
#   read_rows     .predict_read_rows(), as the workers call it
#   stack         each band's tensor kept in a list, torch_stack() at the end
#
# Each runs in a fresh R process (15 torch threads, as the map), three times
# over the same 32 full-width rows, and after each time drops everything,
# runs a full collection and reads the process's working set. One that
# climbs ~3.7 GB a time is the leak; one that stays flat is not.
#
# COST: ~6-8 min. Writes nothing but its table (capability_sweep/t5_read).
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_t5_read_memory.R")
# ══════════════════════════════════════════════════════════════════════════════

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
install_load_pkg(c("torch", "terra", "dplyr", "readr", "tibble", "ps", "callr"))
pkgload::load_all(project_root)
options(width = 200)

target_label <- "soc_stock_0_5cm"
base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)
metadata_dir <- base("outputs", "metadata")
final_base   <- base("outputs", "final_model")
tuning_base  <- base("outputs", "tuning")
t5_dir       <- file.path(tuning_base, "capability_sweep", "t5_read")

final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
summ      <- readRDS(file.path(final_base, final_run_id, "comparison", "final_run_summary.rds"))
config_id <- selected_config_id(summ, final_run_id)
rt <- safe_read_csv2(file.path(metadata_dir, "raster_table_used.csv"))
qc <- safe_read_csv2(file.path(metadata_dir, "qc_table.csv"))
sc <- safe_read_csv2(file.path(final_base, final_run_id, config_id, "predictor_scaling.csv"))
qc <- qc[match(rt$predictor, qc$predictor), , drop = FALSE]
sc <- sc[match(rt$predictor, sc$predictor), , drop = FALSE]
g250 <- terra::rast(rt$raster_file[1])
n_row <- as.integer(terra::nrow(g250)); n_col <- as.integer(terra::ncol(g250))
rows <- env_int("soc_t5_row", 17473L) + 0:31      # after T4's rows, near 47 N
reps <- 3L

# One experiment, in a fresh R process: repeat `kind` `reps` times, and after
# each drop everything, collect fully and read the working set.
experiment <- function(kind, root, files, rules, center, scale, rows, n_row, n_col, reps) {
  suppressMessages(pkgload::load_all(root, quiet = TRUE))
  set_torch_threads(15L)
  rss <- function() as.numeric(ps::ps_memory_info(ps::ps_handle())[["rss"]]) / 1e9
  n_ch <- length(files)
  nr <- length(rows)
  srcs <- NULL
  if (kind %in% c("terra_read", "read_rows", "stack")) {
    srcs <- lapply(files, terra::rast)
    for (s in srcs) terra::readStart(s)
  }
  env <- list(srcs = srcs, n_ch = n_ch, rules = rules,
              has_rule = !is.na(rules$na_below) | !is.na(rules$clamp_lower) | !is.na(rules$clamp_upper),
              gc_hook = .predict_gc_hook(1))
  job <- list(center = center, scale = scale, grid = list(nrow = n_row))
  w_buf <- n_col + 14L
  out <- numeric(0)
  base_gb <- rss()
  for (r in seq_len(reps)) {
    if (startsWith(kind, "size_")) {
      # One flat tensor of that many GB, every page touched, then dropped.
      gb <- as.numeric(sub("size_", "", kind))
      x <- torch::torch_empty(as.integer(round(gb * 1e9 / 4)))
      x$fill_(0)
      rm(x)
    } else if (identical(kind, "torch_assign")) {
      x <- torch::torch_empty(c(n_ch, nr, w_buf))
      for (k in seq_len(n_ch)) {
        x[k, , ] <- torch::torch_zeros(c(nr, w_buf))
        env$gc_hook()
      }
      rm(x)
    } else if (identical(kind, "torch_from_r")) {
      x <- torch::torch_empty(c(n_ch, nr, w_buf))
      for (k in seq_len(n_ch)) {
        v <- stats::runif(nr * w_buf)
        x[k, , ] <- torch::torch_tensor(v, dtype = torch::torch_float32())$view(c(nr, w_buf))
        rm(v)
        env$gc_hook()
      }
      rm(x)
    } else if (identical(kind, "terra_read")) {
      for (k in seq_len(n_ch)) {
        v <- terra::readValues(srcs[[k]], row = rows[1], nrows = nr, col = 1L, ncols = n_col,
                               mat = FALSE)
        rm(v)
        env$gc_hook()
      }
    } else if (identical(kind, "read_rows")) {
      b <- .predict_read_rows(rows, 1L, n_col, 7L, 7L, w_buf, env, job)
      rm(b)
    } else if (identical(kind, "stack")) {
      parts <- vector("list", n_ch)
      for (k in seq_len(n_ch)) {
        v <- terra::readValues(srcs[[k]], row = rows[1], nrows = nr, col = 1L, ncols = n_col,
                               mat = FALSE)
        t <- torch::torch_tensor(v, dtype = torch::torch_float32())$view(c(nr, n_col))
        rm(v)
        t$sub_(center[k])$div_(scale[k])
        t$masked_fill_(torch::torch_isfinite(t)$logical_not(), 0)
        parts[[k]] <- torch::nnf_pad(t, c(7L, 7L, 0L, 0L))
        rm(t)
        env$gc_hook()
      }
      x <- torch::torch_stack(parts, dim = 1L)
      rm(parts, x)
    }
    invisible(gc(verbose = FALSE, full = TRUE))
    out[r] <- rss()
  }
  if (!is.null(srcs)) for (s in srcs) terra::readStop(s)
  list(kind = kind, base_gb = base_gb, after_gb = out,
       peak_gb = as.numeric(ps::ps_memory_info(ps::ps_handle())[["peak_wset"]] %||% NA) / 1e9)
}

# THE FIRST RUN (commit b56d6ff): torch_assign, torch_from_r, read_rows and
# stack each kept 3.7-4.3 GB a repetition; terra_read kept nothing. A 3.71 GB
# tensor is not given back however it is built, while T4 saw the 1.6 GB halo
# come back -- so the second run asks by size, across 2^31 bytes, and runs the
# reader again now that it holds a step in channel blocks under 1.5 GB.
kinds <- env_csv("soc_t5_kinds", c("size_1.0", "size_1.9", "size_2.2", "size_3.7", "read_rows"))
message("\n", strrep("=", 78))
message("T5 -- which part of the reader keeps memory | rows ", rows[1], "-", rows[length(rows)],
        ", full width (", format(n_col, big.mark = ","), " columns), ", nrow(rt), " bands")
message(strrep("=", 78))

required <- c("t5_01")
L <- check_ledger("T5")
res <- list()
for (kd in kinds) {
  message(sprintf("  %-13s ...", kd))
  t0 <- Sys.time()
  r <- tryCatch(callr::r(experiment, args = list(
    kind = kd, root = project_root, files = rt$raster_file, rules = as.data.frame(qc),
    center = as.numeric(sc$center), scale = as.numeric(sc$scale), rows = rows,
    n_row = n_row, n_col = n_col, reps = reps),
    env = c(callr::rcmd_safe_env(), OMP_NUM_THREADS = "15", MKL_NUM_THREADS = "15")),
    error = function(e) e)
  if (inherits(r, "error")) {
    message("    ERROR: ", conditionMessage(r))
    next
  }
  res[[kd]] <- tibble::tibble(experiment = kd, base_gb = r$base_gb,
                              after_1 = r$after_gb[1], after_2 = r$after_gb[2],
                              after_3 = r$after_gb[3],
                              growth_per_rep = (r$after_gb[reps] - r$after_gb[1]) / (reps - 1),
                              peak_gb = r$peak_gb,
                              minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
}
tab <- dplyr::bind_rows(res)
ledger_check(L, "t5_01", "every experiment ran its three repetitions",
             nrow(tab) == length(kinds) && all(is.finite(c(tab$after_1, tab$after_2, tab$after_3))),
             sprintf("%d of %d experiment(s)", nrow(tab), length(kinds)))

message("\n-- The working set (GB) after each repetition and a full collection --")
print_wide(dplyr::mutate(tab, dplyr::across(c(base_gb, after_1, after_2, after_3,
                                              growth_per_rep, peak_gb, minutes), ~ round(.x, 2))),
           n = Inf)
message("\n  A step's rows tensor is ", sprintf("%.2f", nrow(rt) * 32 * (n_col + 14) * 4 / 1e9),
        " GB: an experiment that grows about that much a repetition keeps it.")

create_output_dirs(t5_dir)
safe_write_csv2(tab, file.path(t5_dir, "t5_read_memory.csv"))
v <- ledger_verdict(L, required, file.path(t5_dir, "t5_checks.csv"))
