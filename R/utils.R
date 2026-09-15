# Utility functions: safe I/O, directory helpers, device setup
# These functions protect against file-lock issues common on Windows
# and provide consistent output for all framework components.

# ── I/O helpers ──────────────────────────────────────────────────────────────

#' Write a CSV (semicolon-separated) safely, removing old file first if needed.
safe_write_csv2 <- function(data, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(path)) {
    removed <- try(file.remove(path), silent = TRUE)
    if (inherits(removed, "try-error") || isFALSE(removed)) {
      path <- .timestamped_path(path, "csv")
    }
  }
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
safe_read_csv2 <- function(path, ...) {
  suppressMessages(readr::read_csv2(path, show_col_types = FALSE, ...))
}

safe_save_rds <- function(object, path, compress = FALSE) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(path)) {
    removed <- try(file.remove(path), silent = TRUE)
    if (inherits(removed, "try-error") || isFALSE(removed)) {
      path <- .timestamped_path(path, "rds")
    }
  }
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
         "than erroring.
Use saveRDS on a plain R array instead.",
         call. = FALSE)
  }

  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(path)) {
    removed <- try(file.remove(path), silent = TRUE)
    if (inherits(removed, "try-error") || isFALSE(removed)) {
      path <- .timestamped_path(path, "pt")
    }
  }
  torch::torch_save(object, path)
  invisible(path)
}

.timestamped_path <- function(path, ext) {
  file.path(
    dirname(path),
    paste0(
      tools::file_path_sans_ext(basename(path)),
      "_", format(Sys.time(), "%Y%m%d_%H%M%S"),
      ".", ext
    )
  )
}

# ── the point table contract ───────────────────────────────────────────────
#
# The framework expects a fixed set of column names rather than an argument per
# column. Eight `*_col =` arguments would spread the same complexity across
# every signature in the package; one documented, validated contract keeps it
# in one place. What must never happen is a user DISCOVERING the contract from
# a cryptic error deep inside a training loop -- hence check_point_contract().
#
# It lives HERE, in the foundation file every entry point loads first, because
# predict_loader() (train_cnn.R) and the patch store (dataset.R) both need it.
# Putting it in either of those made the other fail to find it under source().
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
create_output_dirs <- function(dirs) {
  purrr::walk(dirs, ~ dir.create(.x, recursive = TRUE, showWarnings = FALSE))
  check <- tibble::tibble(
    path  = dirs,
    full  = normalizePath(dirs, winslash = "/", mustWork = FALSE),
    exists = file.exists(dirs)
  )
  if (any(!check$exists)) {
    print(check)
    stop("Some output directories could not be created.")
  }
  invisible(check)
}

# ── Torch / device setup ──────────────────────────────────────────────────────

#' Configure torch threads and select compute device.
#'
#' @param n_threads   Number of intra-op threads (set to available CPU cores).
#' @param use_cuda    Use GPU if available.
#' @return A torch_device object.
setup_torch_device <- function(n_threads = 8, use_cuda = TRUE) {
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
  device <- if (use_cuda && torch::cuda_is_available()) {
    torch::torch_device("cuda")
  } else {
    torch::torch_device("cpu")
  }
  message("Device: ", device$type)
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
print_wide <- function(x, n = NULL) {
  x <- tibble::as_tibble(x)
  if (is.null(n)) print(x, width = Inf) else print(x, n = n, width = Inf)
  invisible(x)
}
