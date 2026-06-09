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
safe_torch_save <- function(object, path) {
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
