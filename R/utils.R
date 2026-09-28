# Utility functions: safe I/O, directory helpers, device setup
# Shared helpers: file writers that refuse a locked target, the point-table
# contract, run-directory resolution, environment overrides.

# ── The pipe ──────────────────────────────────────────────────────────────────
#
# Nine modules use it, and code in a namespace finds only what the namespace
# holds, what it imports and base: a worker process, which attaches no
# package, would stop at the first `%>%`. Bound here, once, from the package
# that owns it. (It was first bound for R/load_all.R, whose source()d files
# could not count on a library(dplyr) either -- the README's own Quickstart
# failed on it.)
`%>%` <- dplyr::`%>%`

# ── I/O helpers ──────────────────────────────────────────────────────────────
#
# A LOCKED FILE IS AN ERROR, NOT A RENAME. These writers used to divert to a
# timestamped sibling when the target could not be removed (a Windows handle
# held by Excel or a viewer), and return the new path invisibly. No caller ever
# read that return value -- grep finds none -- so the authoritative file kept
# its OLD contents while a fresh one sat beside it unread, and stage 04 would
# have ranked a stale table without a word. Stopping costs a re-run; the rename
# cost a wrong result that looked like a right one.
.refuse_locked <- function(path) {
  if (!file.exists(path)) return(invisible(TRUE))
  removed <- suppressWarnings(try(file.remove(path), silent = TRUE))
  if (inherits(removed, "try-error") || isFALSE(removed)) {
    stop("Cannot overwrite ", path, " -- it is locked, most likely open in ",
         "another program.\n  Close it and re-run. Nothing was written.",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' Write a CSV (semicolon-separated) safely, removing old file first if needed.
#' @export
safe_write_csv2 <- function(data, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  .refuse_locked(path)
  readr::write_csv2(data, path)
  invisible(path)
}

#' Save an R object as RDS safely.
#' Read a CSV written by safe_write_csv2().
#'
# read_csv2() emits "Using ',' as decimal and '.' as grouping mark" on EVERY
# call. In a script that reads 22 files that is 22 lines of noise interleaved
# through the report -- and the notice is about the locale WE chose, so it
# tells nobody anything. Silenced here, once, instead of repeating
# suppressMessages() at every call site in the pipeline.
#' @export
safe_read_csv2 <- function(path, ...) {
  suppressMessages(readr::read_csv2(path, show_col_types = FALSE, ...))
}

#' @export
safe_save_rds <- function(object, path, compress = FALSE) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  .refuse_locked(path)
  saveRDS(object = object, file = path, compress = compress)
  invisible(path)
}

#' Save a torch state dict or model safely.
#'
#' Guards against a real and nasty failure mode: torch_save() in R torch 0.17.0
#' breaks above 2^31 bytes. Measured here -- 2,147,479,648 bytes writes fine,
#' 2,147,487,648 bytes kills the session, and in one case it produced a file of
#' the CORRECT SIZE whose tail was 4.3 GB of zeros. Silent corruption that
#' passes a "no non-finite values" check, because zero is finite.
#'
#' Model state dicts are far below this (tens of MB), so the guard should never
#' fire in normal use -- but if it ever does, it must be loud, because the
#' alternative is a plausible-looking wrong result. For anything large, store
#' plain R arrays with saveRDS (see save_patch_window() in R/dataset.R).
#' @export
safe_torch_save <- function(object, path) {
  .torch_save_limit <- 2^31

  n_bytes <- tryCatch({
    sizes <- vapply(
      if (inherits(object, "torch_tensor")) list(object) else as.list(object),
      function(z) if (inherits(z, "torch_tensor"))
        prod(as.numeric(z$shape)) * 4 else 0,
      numeric(1)
    )
    max(sizes, 0)
  }, error = function(e) 0)

  if (n_bytes >= .torch_save_limit) {
    stop("Refusing to torch_save(): a tensor of ",
         format(n_bytes, big.mark = ","), " bytes exceeds the 2^31 limit of ",
         "torch_save() in this torch build, which corrupts silently rather ",
         "than erroring.\n  Use saveRDS on a plain R array instead.",
         call. = FALSE)
  }

  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  .refuse_locked(path)
  torch::torch_save(object, path)
  invisible(path)
}

# ── the point table contract ───────────────────────────────────────────────
#
# The framework expects a fixed set of column names rather than an argument per
# column. Eight `*_col =` arguments would spread the same complexity across
# every signature in the package; one documented, validated contract keeps it
# in one place. What must never happen is a user DISCOVERING the contract from
# a cryptic error deep inside a training loop -- hence check_point_contract().
#
# It lives HERE, in the foundation file, because predict_loader() (train_cnn.R)
# and the patch store (dataset.R) both need it. (While the files were
# source()d one by one, putting it in either made the other fail to find it.)
#
# Rename your columns to these before calling. They are the only names the
# framework hardcodes about your data.

.point_contract <- c(
  profile_id       = "stable identifier of the observation (a site, a profile)",
  sample_id        = "integer row key, used to align tables to each other",
  target_native    = "target in its native units -- what metrics are reported in",
  target_transform = "target in training space (e.g. log1p of the above)"
)

#' Check a table against the point contract, with an actionable error.
#'
#' @param x    A data frame / tibble.
#' @param need Which contract columns are required here.
#' @param what Label used in the error message.
check_point_contract <- function(x, need = names(.point_contract),
                                 what = "points") {
  need <- intersect(need, names(.point_contract))
  gone <- need[!need %in% names(x)]
  if (length(gone) == 0L) return(invisible(TRUE))

  stop(
    what, " is missing required column(s): ", paste(gone, collapse = ", "),
    "\n\nThis framework expects fixed column names:\n",
    paste(sprintf("  %-17s %s", need, .point_contract[need]), collapse = "\n"),
    "\n\nRename the columns in your table to match before calling.",
    call. = FALSE
  )
}

# ── Directory helpers ─────────────────────────────────────────────────────────

#' Create a set of directories and verify they exist.
#' @export
create_output_dirs <- function(dirs) {
  purrr::walk(dirs, ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE))
  check <- tibble::tibble(
    path  = dirs,
    full  = normalizePath(dirs, winslash = "/", mustWork = FALSE),
    exists = file.exists(dirs)
  )
  if (any(!check$exists)) {
    stop("Could not create: ", paste(check$full[!check$exists], collapse = ", "),
         "\n  Check permissions and, on Windows, the 260-character path limit.",
         call. = FALSE)
  }
  invisible(check)
}

# ── How many cores: one answer, used by every function ────────────────────────
#
# ONE RULE, IN ONE PLACE. Before this, setup_torch_device() counted PHYSICAL
# cores minus one while stage 05 divided the LOGICAL count by its workers --
# two answers to the same question on the same machine -- and three example
# scripts typed 30 by hand. Every function that takes `n_cores` asks here.
#
# Above the physical count is allowed but said out loud: hyperthreads share a
# core's arithmetic units, and torch on CPU runs slower, not faster, when its
# threads outnumber the cores they compete for.

.physical_cores <- function() {
  n <- suppressWarnings(parallel::detectCores(logical = FALSE))
  if (is.na(n) || n < 1L) n <- suppressWarnings(parallel::detectCores(logical = TRUE))
  if (is.na(n) || n < 1L) n <- 1L
  as.integer(n)
}

#' The number of cores a call may use.
#'
#' @param n_cores NULL for the physical cores minus one (one left for the
#'   system), or a whole number >= 1.
#' @param what    What the cores are for, for the message.
#' @return An integer >= 1.
resolve_cores <- function(n_cores = NULL, what = "this step") {
  phys <- .physical_cores()
  if (is.null(n_cores)) return(max(1L, phys - 1L))
  n <- suppressWarnings(as.integer(n_cores))
  if (length(n_cores) != 1L || is.na(n) || n < 1L || n != n_cores) {
    stop("n_cores must be a whole number >= 1, got ",
         paste(deparse(n_cores), collapse = ""), ".", call. = FALSE)
  }
  if (n > phys) {
    message("  n_cores = ", n, " is above the ", phys, " physical core(s) of ",
            "this machine; ", what, " will not run faster for it, and torch ",
            "may run slower.")
  }
  n
}

# ── What a worker loads ───────────────────────────────────────────────────────
#
# dsm_final() and dsm_predict() work in fresh R processes, and each loads the
# framework itself. It must be THE SAME framework the session that started it
# runs: the source tree if the session loaded that (pkgload::load_all()), the
# installed copy if it loaded that (library()), and the same installed copy
# where two libraries hold one. A worker that loaded the installed package for
# a session working on the source tree would map with the code of the last
# install, and nothing would say so.
#
# What travels to the worker is a description, not the namespace. A function
# whose environment is the namespace makes the worker load the package while
# it READS its arguments -- before it has set what must be set before torch
# starts (dsm_predict()'s collection threshold), and from wherever the
# worker's library path finds a copy first. So open() is sent with base as its
# environment, and .pkg_check_portable() refuses a job that carries anything
# tied to the namespace.
#
# And the code must not change under a run. Recycled workers start hours into
# a global map; one that loaded files edited in the meantime would map its
# units with other code than the rest. .onLoad() fingerprints what it loaded,
# and a worker whose fingerprint differs from its session's stops.
.pkg_state <- new.env(parent = emptyenv())

# The files a load reads: the package's metadata and its R/ directory -- the
# sources in a source tree, the lazy-load database in an installed copy, which
# a reinstall rewrites.
.pkg_code_hash <- function(path) {
  files <- c(file.path(path, c("DESCRIPTION", "NAMESPACE")),
             sort(list.files(file.path(path, "R"), full.names = TRUE)))
  files <- files[file.exists(files) & !dir.exists(files)]
  paste(unname(tools::md5sum(files)), collapse = "")
}

.pkg_loader <- function() {
  # The environment of THIS function, not of whatever the name finds: a
  # source()d copy would otherwise find the attached package's and pass.
  ns <- environment(sys.function())
  if (!isNamespace(ns)) {
    stop("dsm_final() and dsm_predict() start workers that load the framework ",
         "as a package, and this session has it some other way -- its files ",
         "source()d, most likely.\n  Start a fresh R session and load it with ",
         "pkgload::load_all(\"<project root>\") or library(soilcnn).",
         call. = FALSE)
  }
  pkg  <- unname(getNamespaceName(ns))
  dev  <- requireNamespace("pkgload", quietly = TRUE) && pkgload::is_dev_package(pkg)
  path <- normalizePath(getNamespaceInfo(ns, "path"), winslash = "/", mustWork = FALSE)
  desc <- file.path(path, "DESCRIPTION")
  if (!file.exists(desc) ||
      !identical(unname(read.dcf(desc, fields = "Package")[1, 1]), pkg)) {
    stop("This session's ", pkg, " was loaded from ", path, ", and that ",
         "directory does not hold its DESCRIPTION -- a worker could not load ",
         "the same code.", call. = FALSE)
  }
  if (is.null(.pkg_state$code_hash)) {
    stop("This session's ", pkg, " has no fingerprint of the code it loaded; ",
         "its .onLoad() did not run. Load it with library() or ",
         "pkgload::load_all().", call. = FALSE)
  }
  # Runs in the worker, before the framework exists there: base R and
  # pkgload only, and base as its environment (see above).
  open <- function(loader) {
    if (loader$dev) {
      if (!requireNamespace("pkgload", quietly = TRUE)) {
        stop("This worker must load ", loader$package, " from ", loader$path,
             " as its session did, and needs pkgload: install.packages(\"pkgload\").",
             call. = FALSE)
      }
      suppressMessages(pkgload::load_all(loader$path, quiet = TRUE))
    } else {
      loadNamespace(loader$package, lib.loc = dirname(loader$path))
    }
    ns <- asNamespace(loader$package)
    here <- get(".pkg_state", envir = ns)$code_hash
    if (!identical(here, loader$code_hash)) {
      stop("This worker loaded ", loader$package, " from ", loader$path,
           " and the code there is not the code its session loaded: it changed ",
           "since. A run whose workers map with different code gives a map made ",
           "of two -- finish the run with the code it started with, or restart it.",
           call. = FALSE)
    }
    ns
  }
  environment(open) <- baseenv()
  list(package = pkg, dev = dev, path = path, code_hash = .pkg_state$code_hash,
       open = open)
}

# NOTHING OF THE NAMESPACE IN WHAT A WORKER RECEIVES: a function or an
# environment in the job whose enclosure leads to this package would make the
# worker load it while reading its arguments (see .pkg_loader()). Checked
# before any worker starts, because what it prevents fails without a word.
.pkg_check_portable <- function(x, who) {
  ns  <- environment(sys.function())
  bad <- character(0)
  walk <- function(v, where) {
    e <- if (is.function(v)) environment(v) else if (is.environment(v)) v else NULL
    if (!is.null(e) && identical(topenv(e), ns)) bad <<- c(bad, where)
    if (is.list(v)) {
      nm <- names(v)
      for (i in seq_along(v)) {
        walk(v[[i]], paste0(where, "$", if (is.null(nm) || !nzchar(nm[i])) i else nm[i]))
      }
    }
  }
  walk(x, "job")
  if (length(bad)) {
    stop(who, " would send its workers ", length(bad), " object(s) whose ",
         "environment is the package's own: ", paste(utils::head(bad, 5), collapse = ", "),
         if (length(bad) > 5L) ", ..." else "", ". A worker reading them would ",
         "load the package before it chose which copy to load.", call. = FALSE)
  }
  invisible(TRUE)
}

# ── Torch / device setup ──────────────────────────────────────────────────────

#' Set torch's thread pools.
#'
#' Split out of setup_torch_device() so that a caller who already holds a
#' device can change the threads without building a second one --
#' dsm_train(device = d, n_cores = 8) is that caller. The threads belong to
#' the R session, not to a device: whatever set them last is what every later
#' torch call gets.
#'
#' The interop pool can be sized once per session, before torch first runs
#' work in parallel; later calls leave it as it is and say so once.
#'
#' @param n_threads NULL for the physical cores minus one (see
#'   resolve_cores()), or a whole number >= 1.
#' @return The number of threads, invisibly.
#' @export
set_torch_threads <- function(n_threads = NULL) {
  n_threads <- resolve_cores(n_threads, what = "torch")
  Sys.setenv(
    OMP_NUM_THREADS = as.character(n_threads),
    MKL_NUM_THREADS = as.character(n_threads)
  )
  exports <- getNamespaceExports("torch")
  if ("torch_set_num_threads" %in% exports) {
    tryCatch(
      torch::torch_set_num_threads(n_threads),
      error = function(e) message("Could not set intra-op threads: ", e$message)
    )
  }
  opt_key <- "torch_interop_threads_set"
  if ("torch_set_num_interop_threads" %in% exports && !isTRUE(getOption(opt_key))) {
    tryCatch(
      {
        torch::torch_set_num_interop_threads(n_threads)
        options(torch_interop_threads_set = TRUE)
      },
      error = function(e) {
        options(torch_interop_threads_set = TRUE)
        message("Skipping interop threads (already started): ", e$message)
      }
    )
  }
  invisible(n_threads)
}

#' Configure torch threads and select compute device.
#'
#' @param n_threads   Number of intra-op threads; NULL for the physical cores
#'   minus one (see resolve_cores()).
#' @param use_cuda    Use GPU if available.
#' @return A torch_device object.
#' @export
setup_torch_device <- function(n_threads = NULL, use_cuda = TRUE) {
  # FROM THE MACHINE, NOT FROM A LITERAL. The default was 8 and every example
  # script overrode it with 30 -- the author's workstation -- so a user on a
  # laptop would have oversubscribed and a user on a bigger box would have
  # idled. NULL reads the physical core count and leaves one for the OS.
  n_threads <- set_torch_threads(n_threads)
  device <- if (use_cuda && torch::cuda_is_available()) {
    torch::torch_device("cuda")
  } else {
    torch::torch_device("cpu")
  }
  message("Device: ", device$type, "  (", n_threads, " intra-op thread(s))")
  device
}

# ── Misc ──────────────────────────────────────────────────────────────────────

#' Deep-clone a model state dict (detach + clone every tensor).
clone_state_dict <- function(state_dict) {
  lapply(state_dict, function(x) x$detach()$clone())
}

#' Set learning rate on all param groups of an optimizer.
set_optimizer_lr <- function(optimizer, lr) {
  for (i in seq_along(optimizer$param_groups)) {
    optimizer$param_groups[[i]]$lr <- lr
  }
  invisible(optimizer)
}

#' Return SiLU activation if available, otherwise ReLU.
make_activation <- function() {
  if ("nn_silu" %in% getNamespaceExports("torch")) torch::nn_silu() else torch::nn_relu()
}

# ── Data augmentation: dihedral (D4) symmetries ───────────────────────────────
#
# A raster patch can be rotated by 90°/180°/270° and mirrored without changing
# the value at its centre — the geographic orientation of the patch is arbitrary
# for predicting a point property. These 8 symmetries (the dihedral group D4) are
# therefore label-preserving augmentations that multiply the effective training
# data ×8 for free. This is the most natural regulariser for patch-based CNNs and
# directly counters the early overfitting seen with larger architectures.
#
# Dims of a torch tensor are 1-based here: (N=1, C=2, H=3, W=4).

#' Apply one of the 8 D4 symmetries to a 4D tensor (N, C, H, W). Square patches.
#'
#' @param x A 4D torch tensor.
#' @param k Integer 1–8 selecting the symmetry.
apply_d4 <- function(x, k) {
  switch(k,
    x,                                                    # 1: identity
    torch::torch_flip(x$transpose(3, 4), dims = 4),       # 2: rotate 90
    torch::torch_flip(x, dims = c(3, 4)),                 # 3: rotate 180
    torch::torch_flip(x$transpose(3, 4), dims = 3),       # 4: rotate 270
    torch::torch_flip(x, dims = 3),                       # 5: flip vertical
    torch::torch_flip(x, dims = 4),                       # 6: flip horizontal
    x$transpose(3, 4),                                    # 7: transpose (diagonal)
    torch::torch_flip(x$transpose(3, 4), dims = c(3, 4))  # 8: anti-diagonal
  )
}

#' Apply an INDEPENDENT random D4 transform to EACH sample in the batch.
#'
#' Per-sample (not per-batch) augmentation: every sample draws its own symmetry,
#' so a single gradient step already mixes all 8 orientations. This gives smoother
#' gradients and more representative BatchNorm statistics than rotating the whole
#' batch the same way — the difference matters most for the large windows, which
#' carry the most parameters and overfit first.
#'
#' Each branch tensor receives the SAME per-sample symmetry vector, keeping the
#' two spatial scales of a given sample geometrically consistent. Every D4 element
#' fixes the centre cell of an odd-sized square patch, so the centre-point label
#' is preserved. Call only during training — never on validation/test loaders.
#'
#' Implementation: sample one symmetry index per sample, group the sample indices
#' by symmetry, and apply each of the (at most 8) transforms once to its group.
#' This is 8 flip/transpose ops per batch at most — cheaper than transforming the
#' whole batch 8 times and selecting.
#'
#' @param tensor_list List of 4D torch tensors (one per branch), dims (N, C, H, W).
augment_d4_batch <- function(tensor_list) {
  n      <- tensor_list[[1]]$shape[[1]]
  ks     <- sample.int(8L, n, replace = TRUE)   # one symmetry per sample
  groups <- split(seq_len(n), ks)               # sample indices per symmetry

  lapply(tensor_list, function(x) {
    out <- x$clone()
    for (kk in names(groups)) {
      idx <- groups[[kk]]
      out[idx, , , ] <- apply_d4(x[idx, , , , drop = FALSE], as.integer(kk))
    }
    out
  })
}

# ── printing a wide table without losing columns ──────────────────────────────
#
# `print(x, width = Inf)` is a TIBBLE feature. On a plain data.frame it reaches
# print.data.frame -> print.default, where `width` is coerced to integer, Inf
# becomes NA, and the call dies with "invalid printing width" plus a coercion
# warning that names nothing.
#
# That is not a hypothetical: stage 01 builds its summary by summarise() over a
# data.frame -- because terra::extract() returns one -- so the summary was a
# data.frame, and the script stopped on the line that printed it, AFTER the
# expensive extraction and BEFORE writing anything.
#
# The fix belongs in one place rather than at each call site. Anyone using this
# framework on their own data.frames would hit the same wall, and "remember to
# pass a tibble" is not a contract a package can rely on.
#
# @param x The table to print.
# @param n Rows to show; NULL leaves the tibble default (10).
#' @export
print_wide <- function(x, n = NULL) {
  x <- tibble::as_tibble(x)
  if (is.null(n)) print(x, width = Inf) else print(x, n = n, width = Inf)
  invisible(x)
}

# ── Resume: a config_id is a LABEL, not an identity ───────────────────────────
#
# WHY THIS EXISTS.
#
# Every runner resumes by unit_id, and a unit_id is built from the config_id:
# "cfg_002_f1_s3". That is correct only while config_id means the same thing it
# meant when the unit was fitted -- and nothing was enforcing that.
#
# It broke for real. rf_grid() drew its configs with replacement and never
# de-duplicated, so 03b fitted three identical forests under the names rf_001,
# rf_003 and rf_004. Fixing the grid changed what those names denote: the new
# rf_001 is mtry = p/3, the old one was mtry = 0.1p. Resuming would have read
# the old rows, matched the names, skipped the work, and reported results for
# hyperparameters that were never fitted -- with no error and a plausible CCC.
#
# The same hazard is latent in the CNN grid. make_tune_grid() draws
# sequentially, so raising tune_length keeps the earlier configs and resume
# works; but that holds only while the parameter space is unchanged. Add one
# value to one axis and every subsequent draw shifts, silently.
#
# So: match on the HYPERPARAMETERS, and let the label follow. A cached unit is
# reusable only if the config it recorded still equals the config the grid now
# asks for under that name. Anything else is refitted.
#
# Refit rather than stop, because a changed grid is a normal thing to do and
# stopping would punish it. The message says how many and why, so a resume that
# silently retrains everything cannot be mistaken for a resume that worked.
.resumable_units <- function(done_ids, comparison, tune_grid, verbose = TRUE) {
  if (length(done_ids) == 0L || nrow(comparison) == 0L) return(done_ids)
  if (!"config_id" %in% names(comparison)) return(done_ids)

  # Only the columns the grid and the record share: the record also carries
  # outcomes (val_ccc, runtime_min, ...) which are results, not identity.
  pars <- setdiff(intersect(names(tune_grid), names(comparison)), "config_id")
  if (length(pars) == 0L) return(done_ids)

  # ONE CANONICAL FORM FOR BOTH SIDES.
  #
  # The grid and the record do not store the multi-valued parameters the same
  # way, and they never did: the grid keeps window_sizes as a list-column
  # c(5L, 7L), while the comparison row flattens it for the CSV as "5x7"
  # (conv_channels as "32_64"). Compared literally, every CNN unit ever
  # written looks stale, and a guard against a silent wrong answer becomes a
  # guarantee of a pointless full retrain -- a worse failure than the one it
  # was built to prevent, because it costs hours and looks like it worked.
  #
  # So both sides are reduced to the same tokens: split on the separators the
  # writers use, and compare the values. format() over as.character() because
  # a list-column element is a VECTOR, and as.character() on it returns one
  # string per element rather than one string per config.
  tok <- function(v) {
    z <- if (is.character(v) && length(v) == 1L) {
      strsplit(v, "[x_|,]")[[1]]
    } else {
      format(v, digits = 12, trim = TRUE)
    }
    z <- trimws(z)
    # "7" and "7L" and 7 are one value; a number written either way must match.
    num <- suppressWarnings(as.numeric(z))
    ifelse(is.na(num), z, format(num, digits = 12, trim = TRUE))
  }
  sig <- function(df) {
    vapply(seq_len(nrow(df)), function(i) {
      paste(vapply(pars, function(p) paste(tok(df[[p]][[i]]), collapse = ","),
                   character(1)), collapse = "|")
    }, character(1))
  }

  want <- stats::setNames(sig(tune_grid), tune_grid$config_id)
  have <- sig(comparison)

  keep <- vapply(seq_len(nrow(comparison)), function(i) {
    id <- comparison$config_id[i]
    # A cached config the grid no longer names is not stale, it is simply not
    # asked for; it never matches a unit_id the loop generates, so leaving it
    # alone costs nothing and keeps the record of what was run.
    if (!id %in% names(want)) return(TRUE)
    identical(have[i], unname(want[id]))
  }, logical(1))

  stale <- comparison$unit_id[!keep]
  out   <- setdiff(done_ids, stale)

  if (verbose && length(stale) > 0L) {
    ids <- unique(comparison$config_id[!keep])
    message("Resume: ", length(stale), " cached unit(s) describe hyperparameters ",
            "that no longer\n  match the grid under the same name (",
            paste(utils::head(ids, 6), collapse = ", "),
            if (length(ids) > 6) ", ..." else "", "). They will be REFITTED.\n",
            "  A config_id is a label; the configuration is the identity.")
  }
  out
}

# ── Resume: the FOLDS have to be the same folds too ───────────────────────────
#
# WHY THIS EXISTS, AND WHY .resumable_units() IS NOT ENOUGH.
#
# .resumable_units() proves a cached unit was fitted on the hyperparameters its
# name claims. It says nothing about the DATA those hyperparameters were fitted
# to -- and the fold plan is data.
#
# It came up the moment the buffer was fixed. apply_buffer() had protected only
# the validation set; once it also protected the test set, every fold lost a rim
# of training points. The cached units were still `cfg_002` with the same
# learning rate and the same window, so the hyperparameter check passed them --
# while they had been trained on a training set that no longer exists. Resuming
# would have produced a comparison table whose rows were fitted on different
# data, ranked against each other, with nothing on screen saying so.
#
# There is no safe partial answer here. A plan that differs invalidates every
# cached unit at once, so this REFUSES rather than silently discarding hours of
# training: the run directory is the record of an experiment, and quietly
# overwriting half of it with units from a different experiment is worse than
# stopping. The caller picks a new run_id, or deletes the old one on purpose.
#
# Compared by fold MEMBERSHIP, not by the plan object: params differ for
# irrelevant reasons (a new field, a rounded buffer) while the split is
# identical, and the split is what training actually consumed.
#' @export
check_plan_unchanged <- function(plan, run_dir, resume = TRUE) {
  path <- file.path(run_dir, "fold_plan.rds")
  if (!isTRUE(resume) || !file.exists(path)) return(invisible(TRUE))

  # UNREADABLE IS NOT ABSENT. This used to catch a read error, treat the run
  # as having no plan, and let the resume proceed -- the one situation in
  # which the check exists could not be performed, and it answered "fine".
  old <- tryCatch(readRDS(path), error = function(e) {
    stop("fold_plan.rds exists in ", run_dir, " but cannot be read (",
         conditionMessage(e), ").\n  A resume cannot be verified against a ",
         "plan that will not open. Use a new run_id, or delete the directory ",
         "deliberately.", call. = FALSE)
  })
  if (is.null(old$folds)) return(invisible(TRUE))

  same <- length(old$folds) == length(plan$folds) &&
    all(vapply(seq_along(plan$folds), function(j) {
      a <- old$folds[[j]]; b <- plan$folds[[j]]
      identical(sort(as.integer(a$train)),      sort(as.integer(b$train))) &&
      identical(sort(as.integer(a$validation)), sort(as.integer(b$validation))) &&
      identical(sort(as.integer(a$test)),       sort(as.integer(b$test)))
    }, logical(1)))
  if (same) return(invisible(TRUE))

  # Name the difference. "The plan changed" sends someone reading diffs; the
  # counts usually identify the cause on sight.
  n_of <- function(p, role) sum(vapply(p$folds, function(f) length(f[[role]]),
                                       integer(1)))
  stop(
    "THE FOLD PLAN IN THIS RUN DIRECTORY IS NOT THE PLAN BEING ASKED FOR.\n\n",
    "  ", run_dir, "\n\n",
    sprintf("  cached : %d fold(s) | train %d | validation %d | test %d\n",
            length(old$folds), n_of(old, "train"), n_of(old, "validation"),
            n_of(old, "test")),
    sprintf("  asked  : %d fold(s) | train %d | validation %d | test %d\n\n",
            length(plan$folds), n_of(plan, "train"), n_of(plan, "validation"),
            n_of(plan, "test")),
    "Resuming would rank units fitted on different training sets against each\n",
    "other. Use a new run_id, or delete this directory deliberately if the\n",
    "cached run is genuinely obsolete.",
    call. = FALSE)
}


# ── evaluate_test = FALSE: one rule, one place ────────────────────────────────
#
# WHY THIS IS A FUNCTION AND NOT THREE dplyr::filter() CALLS.
#
# `evaluate_test = FALSE` is supposed to mean the test set is not read while a
# config is being chosen. It was implemented three times and got fixed once:
#
#   comparison table    blanked                        (both runners)
#   predictions/*.csv   filtered in train_cnn.R only   (2026-09-16)
#   metrics/*_perf.csv  NEVER FILTERED, either runner
#
# So the number the switch exists to withhold -- the per-unit test CCC -- was
# written in plain text, 27 times, in the very run whose selection was later
# frozen. `metrics/cfg_003_f1_s1_perf.csv` line 2 begins
# `cfg_003_f1_s1;smooth_l1;test;591;0,4936...`.
#
# The comment written when the first door was closed said "two doors and one
# lock is one door". There were three. A rule spread across call sites is a rule
# that will be enforced at some of them.
#
# Every artefact that leaves a runner and carries a dataset_role goes through
# here, so adding a fourth artefact later cannot forget.
.drop_test_rows <- function(x, evaluate_test) {
  if (isTRUE(evaluate_test)) return(x)
  if (is.null(x) || !is.data.frame(x) || !"dataset_role" %in% names(x)) return(x)
  dplyr::filter(x, .data$dataset_role != "test")
}

#' The most recent run under `base`, by time, and only if it finished.
#'
#' WHY THIS IS NOT sort(dirs, decreasing = TRUE)[1].
#'
#' That expression means "last alphabetically", which equals "most recent" only
#' while every run id is a timestamp sharing one prefix. Runs given names broke
#' it silently: `soc_0_5cm_design_spatial` sorts ahead of
#' `soc_0_5cm_20260916_232318` because 'd' > '2', so "latest" started resolving
#' to a run that had died on its first unit, and stage 04 would have refit the
#' final model against it without a word.
#'
#' It also never asked whether the run FINISHED. Stage 04 creates its output
#' directory before its own validations run, so a failure leaves a
#' final_<timestamp> behind that "latest" would then deploy -- which is what
#' B2's review found from the other direction.
#'
#' @param base            directory holding the run directories.
#' @param prefix          run ids must start with this ("soc_", "final_"). ""
#'   accepts every directory.
#' @param require_file    path INSIDE a run that only exists when it finished,
#'   e.g. "comparison/comparison_ranked.csv". A run without it is not a
#'   candidate, and its mtime is what "newest" is measured on. NULL accepts any
#'   directory, which is almost never what you want.
#' @param require_pattern alternative to require_file for runs whose completion
#'   is a FAMILY of files rather than one (05c's shard logs): a regex that at
#'   least one file directly inside the run must match. "Newest" is then the
#'   newest matching file, so a run still being written wins over an old one.
#' @param label           what to call these runs in messages.
#' @param on_none         what to do when no run qualifies. "stop" (default) for
#'   a stage that cannot proceed without one; "null" for a CHECK that must
#'   report "incomplete" and carry on -- 99_check_pipeline.R does that, and a
#'   stop() there would turn a diagnosis into a crash. Returning NULL is loud
#'   only if the caller tests for it; every caller that passes "null" here
#'   prints a line saying so.
#' @return The run id (basename), or stop() / NULL when nothing qualifies.
#' @export
latest_run_dir <- function(base, prefix, require_file = NULL,
                           require_pattern = NULL, label = "run",
                           on_none = c("stop", "null")) {
  on_none <- match.arg(on_none)
  none <- function(...) {
    if (identical(on_none, "stop")) stop(..., call. = FALSE)
    message(label, ": ", paste0(..., collapse = ""))
    NULL
  }
  if (!is.null(require_file) && !is.null(require_pattern)) {
    stop("latest_run_dir(): pass require_file OR require_pattern, not both.",
         call. = FALSE)
  }

  if (!dir.exists(base)) {
    return(none("No ", label, " directory at all: ", base))
  }
  ids <- list.dirs(base, recursive = FALSE, full.names = FALSE)
  ids <- ids[startsWith(ids, prefix)]
  if (length(ids) == 0L) {
    return(none("No ", label, " under ", base,
                if (nzchar(prefix)) paste0(" starting with \"", prefix, "\"") else "",
                "."))
  }

  # THE TIME OF THE THING THAT SAYS "FINISHED", not of the directory: a
  # directory's mtime moves when anything inside is touched, including a
  # checkpoint written by a run that later died.
  when <- rep(as.POSIXct(NA), length(ids))
  if (!is.null(require_file)) {
    f <- file.path(base, ids, require_file)
    ok <- file.exists(f)
    when[ok] <- file.info(f[ok])$mtime
  } else if (!is.null(require_pattern)) {
    for (i in seq_along(ids)) {
      m <- list.files(file.path(base, ids[i]), pattern = require_pattern,
                      full.names = TRUE)
      if (length(m) > 0L) when[i] <- max(file.info(m)$mtime)
    }
  } else {
    when <- file.info(file.path(base, ids))$mtime
  }

  finished <- ids[!is.na(when)]
  if (length(finished) == 0L) {
    what <- if (!is.null(require_file)) require_file else
      if (!is.null(require_pattern)) paste0("a file matching ", require_pattern) else
      "anything"
    newest_any <- ids[which.max(file.info(file.path(base, ids))$mtime)]
    return(none("No FINISHED ", label, " under ", base, ".\n  ", length(ids),
                " director(ies) are there, but none carries ", what,
                ",\n  which is what says the run completed. The newest by time is '",
                newest_any, "'.\n  Finish it, or name the run explicitly ",
                "instead of \"latest\"."))
  }

  # MTIME, not the name. A resumed run legitimately becomes the newest again,
  # which is the behaviour wanted; a name cannot express that.
  when <- when[!is.na(when)]
  pick <- finished[which.max(when)]

  skipped <- setdiff(ids, finished)
  message(label, " resolved to: ", pick,
          sprintf("  (newest of %d finished, %s)", length(finished),
                  format(max(when), "%Y-%m-%d %H:%M")))
  if (length(skipped) > 0L) {
    message("  skipped as unfinished: ", paste(skipped, collapse = ", "))
  }
  pick
}

# ── Environment overrides: one reader, three shapes ───────────────────────────
#
# WHY A SCRIPT IS DRIVEN BY ENVIRONMENT VARIABLES AT ALL. Every example script
# opens with rm(list = ls()), deliberately, so that a stale object from an
# earlier run can never leak into a model. Sys.setenv() survives that erasure
# and a workspace object does not, so it is the only channel into a source()d
# run -- and it is what lets a check script run the REAL stage in a child
# process instead of a copy of it (see _b2_two_config_check.R).
#
# WHY ONE HOME. Three scripts each carried their own .env_int(); one of them
# had stopped trimming whitespace, one printed what it read and the others did
# not, and only stage 04 knew how to read a comma-separated list. Three readers
# of the same variable that disagree about "  8 " is a defect waiting for the
# day someone pastes a value with a space in it.
#
# Every reader prints what it took from the environment, because a value set
# in a session and forgotten is how stage 04 nearly trained the wrong configs:
# the variables persist, the file's defaults do not announce that they were
# overridden, and nothing on screen says which was used.

.env_raw <- function(name) {
  v <- trimws(Sys.getenv(name, unset = ""))
  if (nzchar(v)) v else NULL
}

#' Read a string override, or the default.
#' @export
env_chr <- function(name, default) {
  v <- .env_raw(name)
  if (is.null(v)) return(default)
  message("  ", name, " = \"", v, "\"  (from the environment)")
  v
}

#' Read a positive integer override, or the default. Refuses anything else:
#' a thread count or a seed that silently became NA is worse than not starting.
#' @export
env_int <- function(name, default, min = 1L) {
  v <- .env_raw(name)
  if (is.null(v)) return(default)
  n <- suppressWarnings(as.integer(v))
  if (is.na(n) || n < min) {
    stop(name, " is set to '", v, "', which is not an integer >= ", min, ".",
         call. = FALSE)
  }
  message("  ", name, " = ", n, "  (from the environment)")
  n
}

#' Read a comma-separated override as a character vector, or as integers.
#'
#' Empty items ("a,,b", a trailing comma) are dropped; an all-empty value is
#' refused rather than returned as character(0), which downstream code would
#' treat as "nothing selected" and proceed with. With as_int = TRUE every item
#' must parse, and the ones that do not are named -- a seed list with one bad
#' entry must not become a shorter seed list.
#' @export
env_csv <- function(name, default, as_int = FALSE) {
  v <- .env_raw(name)
  if (is.null(v)) return(default)
  parts <- trimws(strsplit(v, ",", fixed = TRUE)[[1]])
  parts <- parts[nzchar(parts)]
  if (length(parts) == 0L) {
    stop(name, " is set to '", v, "', which parses to no values.", call. = FALSE)
  }
  if (as_int) {
    n <- suppressWarnings(as.integer(parts))
    if (anyNA(n)) {
      stop(name, " is set to '", v, "' and these are not integers: ",
           paste(parts[is.na(n)], collapse = ", "), call. = FALSE)
    }
    parts <- n
  }
  message("  ", name, " = ", paste(parts, collapse = ", "),
          "  (", length(parts), ", from the environment)")
  parts
}

# ── Which config a final run deployed: the choice's order, not the grid's ─────
#
# final_run_summary.rds carries selected_cfgs, built in stage 04 as
# dplyr::filter(tune_grid_full, config_id %in% selected_config_ids). filter()
# keeps tune_grid_full's row order -- the GRID's -- so with two configs its
# first row can be the runner-up. selected_config_ids is the chosen list in the
# order it was chosen (written from 2026-09-18 on). Three copies of this rule
# existed, two of them still reading the grid order; one home.
#'
#' @param summary the list read from comparison/final_run_summary.rds.
#' @param label   what to call the run in messages.
#' @return the config id, or stop() -- "auto" must never travel on unresolved.
#' @export
selected_config_id <- function(summary, label = "this final run") {
  id <- if (!is.null(summary$selected_config_ids)) {
    summary$selected_config_ids[1]
  } else {
    message("  (", label, " predates selected_config_ids; falling back to the ",
            "grid order, which differs\n   from the selection order only when ",
            "more than one config was fitted)")
    summary$selected_cfgs$config_id[1]
  }
  if (is.null(id) || is.na(id) || !nzchar(id) || identical(id, "auto")) {
    stop("No config could be resolved from ", label, "'s final_run_summary.rds.",
         call. = FALSE)
  }
  id
}
