# ══════════════════════════════════════════════════════════════════════════════
# T6 -- is a large tensor freed at all, and if it is, who keeps the memory:
# R, torch, or the allocator underneath
#
# THE QUESTION.
#
# T5's second run killed the 2^31-byte explanation: every tensor it made and
# dropped was kept, whatever its size -- 1.0 GB grew 0.95 GB a repetition,
# 1.9 GB 1.9, 2.2 GB 2.2, 3.7 GB 3.7 -- and the reader, holding a step in
# blocks of channels under 1.5 GB, still kept 3.75 GB a repetition. Yet the
# 20 MB tensors the reader makes band by band, and the network's strips
# (~70 MB), are given back: their repetitions never added up.
#
# What sits under every CPU tensor here: libtorch on Windows allocates with
# mimalloc (PyTorch >= 2.1.2; c10.dll carries it -- its option names and
# messages are in the binary). mimalloc keeps a block that does not go back
# to the OS in several documented cases, and a huge block freed from a
# thread other than the one that allocated it is only reclaimed when that
# thread collects (microsoft/mimalloc issue #440). So the suspects are:
#
#   R        the tensor is never finalized (its R object stays reachable)
#   torch    it is finalized, but its storage is still referenced
#   mimalloc the storage is freed, and the allocator neither returns nor
#            reuses the memory
#
# and each experiment below asks one of them, in a fresh R process, three
# times, reading the working set after a full collection:
#
#   size_*           one flat tensor of that size, filled and dropped -- the
#                    control, and the size below which memory comes back
#   *_release        the storage dropped by hand (x$set_() on an empty
#                    tensor) before the R object: if the working set falls
#                    there, the allocator gives memory back and R's
#                    finalization is what never drops it
#   *_1thread        one torch thread: a free from another thread
#   *_purge0         MIMALLOC_PURGE_DELAY=0, freed memory decommitted at once
#   *_no_arena       MIMALLOC_DISALLOW_ARENA_ALLOC=1, every block straight
#                    from the OS and back to it
#   read_*           the map's reader with blocks of that size -- the 64 MB
#                    and one-channel ones are the fix if the limit is a size
#
# Every tensor also carries an R finalizer that counts: "finalized" says
# whether R let go of it.
#
# The first experiment runs with MIMALLOC_VERBOSE and MIMALLOC_SHOW_STATS:
# its mimalloc version, options and statistics are saved and printed.
#
# COST: ~8-10 min. Writes nothing but its tables (capability_sweep/t6_release).
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_t6_release_memory.R")
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
# The read_* experiments hold the reader's rows in channel blocks, which the
# worker's window replaced once T6 had named the cause: this is a record of a
# measurement at commit a7421ff, and runs there.
if (!exists(".predict_channel_blocks", mode = "function")) {
  stop("T6 measured the reader holding a step in channel blocks, removed once T6 named the ",
       "cause (the worker's window replaced them). Run it at commit a7421ff.", call. = FALSE)
}

target_label <- "soc_stock_0_5cm"
base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)
metadata_dir <- base("outputs", "metadata")
final_base   <- base("outputs", "final_model")
tuning_base  <- base("outputs", "tuning")
t6_dir       <- file.path(tuning_base, "capability_sweep", "t6_release")
create_output_dirs(t6_dir)

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
rows <- env_int("soc_t6_row", 17473L) + 0:31      # T5's rows: this is not a timing
reps <- 3L
step_gb <- nrow(rt) * length(rows) * (n_col + 14) * 4 / 1e9

# One experiment, in a fresh R process: repeat it `reps` times; after each,
# drop everything, collect fully and read the working set.
experiment <- function(kind, gb, block_mb, release, threads, root, files, rules, center,
                       scale, rows, n_row, n_col, reps) {
  suppressMessages(pkgload::load_all(root, quiet = TRUE))
  set_torch_threads(threads)
  rss <- function() as.numeric(ps::ps_memory_info(ps::ps_handle())[["rss"]]) / 1e9
  # A finalizer that only counts, closed over nothing but its counter, so it
  # cannot keep the tensor it watches alive.
  fin <- new.env()
  fin$n <- 0L
  counter <- eval(quote(function(e) fin$n <- fin$n + 1L), list2env(list(fin = fin), parent = baseenv()))
  empty_out <- function(tensors) invisible(lapply(tensors, function(z) z$set_(torch::torch_empty(0L))))
  n_ch <- length(files)
  nr <- length(rows)
  w_buf <- n_col + 14L
  srcs <- NULL
  if (identical(kind, "read")) {
    srcs <- lapply(files, terra::rast)
    for (s in srcs) terra::readStart(s)
  }
  env <- list(srcs = srcs, n_ch = n_ch, rules = rules,
              has_rule = !is.na(rules$na_below) | !is.na(rules$clamp_lower) | !is.na(rules$clamp_upper),
              gc_hook = .predict_gc_hook(1))
  job <- list(center = center, scale = scale, grid = list(nrow = n_row))
  blocks <- if (identical(kind, "read")) {
    .predict_channel_blocks(n_ch, nr, w_buf, max_bytes = block_mb * 1e6)
  } else NULL
  after <- numeric(0); held <- numeric(0); dropped <- numeric(0); watched <- 0L
  base_gb <- rss()
  for (r in seq_len(reps)) {
    if (identical(kind, "size")) {
      x <- list(torch::torch_empty(as.integer(round(gb * 1e9 / 4))))
      x[[1]]$fill_(0)
    } else {
      x <- .predict_read_rows(rows, 1L, n_col, 7L, 7L, w_buf, env, job, blocks = blocks)$x
    }
    invisible(lapply(x, reg.finalizer, f = counter))
    watched <- watched + length(x)
    held[r] <- rss()
    if (release) {
      empty_out(x)
      dropped[r] <- held[r] - rss()
    }
    rm(x)
    invisible(gc(verbose = FALSE, full = TRUE))
    after[r] <- rss()
  }
  if (!is.null(srcs)) for (s in srcs) terra::readStop(s)
  list(base_gb = base_gb, after_gb = after, held_gb = held,
       dropped_gb = if (release) dropped else rep(NA_real_, reps),
       finalized = fin$n, watched = watched, n_blocks = length(blocks),
       peak_gb = as.numeric(ps::ps_memory_info(ps::ps_handle())[["peak_wset"]] %||% NA) / 1e9)
}

# name, what, size (GB) or block (MB), release by hand, threads, extra environment
plan <- tibble::tribble(
  ~name,                    ~kind,  ~gb,  ~block_mb, ~release, ~threads, ~extra,
  "size_1.0",               "size", 1.0,  NA,        FALSE,    15L,      "MIMALLOC_VERBOSE=1;MIMALLOC_SHOW_STATS=1",
  "size_1.0_release",       "size", 1.0,  NA,        TRUE,     15L,      "",
  "size_1.0_1thread",       "size", 1.0,  NA,        FALSE,    1L,       "",
  "size_1.0_purge0",        "size", 1.0,  NA,        FALSE,    15L,      "MIMALLOC_PURGE_DELAY=0",
  "size_1.0_no_arena",      "size", 1.0,  NA,        FALSE,    15L,      "MIMALLOC_DISALLOW_ARENA_ALLOC=1",
  "size_0.1",               "size", 0.1,  NA,        FALSE,    15L,      "",
  "size_0.25",              "size", 0.25, NA,        FALSE,    15L,      "",
  "size_0.5",               "size", 0.5,  NA,        FALSE,    15L,      "",
  "read_1500mb",            "read", NA,   1500,      FALSE,    15L,      "",
  "read_64mb",              "read", NA,   64,        FALSE,    15L,      "",
  "read_one_channel",       "read", NA,   21,        FALSE,    15L,      "",
  "read_1500mb_release",    "read", NA,   1500,      TRUE,     15L,      "",
  "read_1500mb_1thread",    "read", NA,   1500,      FALSE,    1L,       "",
  "read_1500mb_purge0",     "read", NA,   1500,      FALSE,    15L,      "MIMALLOC_PURGE_DELAY=0",
  "read_1500mb_no_arena",   "read", NA,   1500,      FALSE,    15L,      "MIMALLOC_DISALLOW_ARENA_ALLOC=1")
only <- env_csv("soc_t6_experiments", plan$name)
bad <- setdiff(only, plan$name)
if (length(bad)) stop("soc_t6_experiments names unknown experiment(s): ", paste(bad, collapse = ", "),
                      call. = FALSE)
plan <- plan[plan$name %in% only, , drop = FALSE]

parse_extra <- function(s) {
  if (!nzchar(s)) return(character(0))
  kv <- strsplit(strsplit(s, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(vapply(kv, `[`, "", 2L), vapply(kv, `[`, "", 1L))
}

message("\n", strrep("=", 78))
message("T6 -- is a large tensor freed, and who keeps the memory | rows ", rows[1], "-",
        rows[length(rows)], ", full width (", format(n_col, big.mark = ","), " columns), ",
        nrow(rt), " bands")
message(strrep("=", 78))

required <- c("t6_01")
L <- check_ledger("T6")
res <- list()
stderr_file <- file.path(t6_dir, "mimalloc_stderr.txt")
for (i in seq_len(nrow(plan))) {
  p <- plan[i, ]
  extra <- parse_extra(p$extra)
  message(sprintf("  %-22s ...%s", p$name,
                  if (length(extra)) paste0("  [", paste(names(extra), extra, sep = "=", collapse = " "), "]") else ""))
  t0 <- Sys.time()
  err_to <- if ("MIMALLOC_VERBOSE" %in% names(extra)) stderr_file else NULL
  r <- tryCatch(callr::r(experiment, args = list(
    kind = p$kind, gb = p$gb, block_mb = p$block_mb, release = p$release, threads = p$threads,
    root = project_root, files = rt$raster_file, rules = as.data.frame(qc),
    center = as.numeric(sc$center), scale = as.numeric(sc$scale), rows = rows,
    n_row = n_row, n_col = n_col, reps = reps),
    env = c(callr::rcmd_safe_env(), OMP_NUM_THREADS = as.character(p$threads),
            MKL_NUM_THREADS = as.character(p$threads), extra),
    stderr = err_to),
    error = function(e) e)
  if (inherits(r, "error")) {
    message("    ERROR: ", conditionMessage(r))
    next
  }
  size <- if (identical(p$kind, "size")) p$gb else step_gb
  growth <- (r$after_gb[reps] - r$after_gb[1]) / (reps - 1)
  res[[p$name]] <- tibble::tibble(
    experiment = p$name, tensor_gb = size, blocks = r$n_blocks, base_gb = r$base_gb,
    after_1 = r$after_gb[1], after_2 = r$after_gb[2], after_3 = r$after_gb[3],
    growth_per_rep = growth,
    dropped_by_hand = mean(r$dropped_gb),
    finalized = sprintf("%d of %d", r$finalized, r$watched),
    verdict = if (growth > 0.5 * size) "keeps" else if (growth < 0.1 * size) "gives back" else "partly",
    peak_gb = r$peak_gb,
    minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
}
tab <- dplyr::bind_rows(res)
ledger_check(L, "t6_01", "every experiment ran its three repetitions",
             nrow(tab) == nrow(plan) && all(is.finite(c(tab$after_1, tab$after_2, tab$after_3))),
             sprintf("%d of %d experiment(s)", nrow(tab), nrow(plan)))

message("\n-- The working set (GB) after each repetition and a full collection --")
print_wide(dplyr::mutate(tab, dplyr::across(c(tensor_gb, base_gb, after_1, after_2, after_3,
                                              growth_per_rep, dropped_by_hand, peak_gb, minutes),
                                            ~ round(.x, 2))),
           n = Inf)
message("\n  keeps: grows more than half the tensor a repetition | gives back: under a tenth.",
        "\n  dropped_by_hand: the working set's fall when the storage was emptied by hand,",
        "\n  before the R object was dropped (the *_release experiments).",
        "\n  finalized: the tensors whose R object R's collector finalized.")

# Everything that process wrote to stderr: its R messages are suppressed, so
# this is mimalloc's -- the version and options at start, the statistics at
# exit (whose lines carry no "mimalloc:" prefix, hence no filter).
if (file.exists(stderr_file)) {
  lines <- readLines(stderr_file, warn = FALSE)
  message("\n-- mimalloc, as the first experiment's process reported it (", length(lines),
          " line(s) in ", stderr_file, ") --")
  message(paste(utils::head(lines, 150), collapse = "\n"))
}

safe_write_csv2(tab, file.path(t6_dir, "t6_release_memory.csv"))
v <- ledger_verdict(L, required, file.path(t6_dir, "t6_checks.csv"))
