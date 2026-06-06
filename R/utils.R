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
