# ── Spatial occlusion: does the network use the neighbourhood at all? ─────────
#
# THE QUESTION THIS PROJECT EXISTS TO ANSWER.
#
# A dual-branch CNN over patches is only worth its cost if the ARRANGEMENT of
# the neighbourhood carries signal. Stage 03b answers that from the outside, by
# racing the CNN against a forest fed the same neighbourhood as per-channel
# means. This answers it from the inside, on the trained network itself:
#
#   hide part of the patch, re-predict, and see what the loss of that part
#   costs.
#
# If hiding everything except the centre pixel costs nothing, the convolution
# is not using context -- whatever else it is doing -- and a model over point
# values would do the same work for a fraction of the cost. That is a
# falsifiable statement about a trained model, it is cheap (inference only,
# nothing retrained), and it is the one this framework was built to test.
#
# HOW THE HIDING IS DONE, AND WHY NOT WITH ZEROS.
#
# The obvious move is to set the occluded pixels to zero. After scaling, zero
# is the training mean, so it reads as "the average landscape" -- but a patch
# whose rim is exactly the mean everywhere is a landscape that does not exist,
# and the network is then answering a question about data it was never shown.
# A drop measured that way confounds "this region mattered" with "this input is
# off-distribution", and the second effect grows with the area hidden -- which
# is precisely the comparison being made.
#
# So the default is PERMUTATION: the occluded region is taken from another,
# randomly chosen sample. Each channel keeps its own marginal distribution and
# its within-region spatial texture; what is destroyed is the association
# between that region and THIS point's target. That is the null hypothesis
# worth testing, and it is the same reasoning behind permutation importance
# rather than mean-imputation importance.
#
#   method = "zero" is kept, because the two disagreeing is itself informative:
#   a large zero-effect with a small permutation-effect says the network is
#   sensitive to the input being unusual, not to the neighbourhood's content.
#
# RINGS, NOT ARBITRARY SHAPES. Distance from the centre is measured in
# Chebyshev steps, because that is the shape of a patch: ring 1 is the 8 pixels
# touching the centre, ring 2 the 16 around those. A 15x15 patch has 7 rings.
# Reporting per ring answers "how far out does it still matter?", which is the
# question that sets the window size for the next run.

#' Chebyshev ring index of every pixel in a w x w patch.
#'
#' Ring 0 is the centre pixel. For even w there is no single centre, and this
#' refuses rather than silently choosing one of the four -- patch_grid() only
#' ever produces odd windows, so an even one means something upstream is wrong.
#'
#' @param w Window side.
#' @return A w x w integer matrix of ring indices.
patch_ring_index <- function(w) {
  w <- as.integer(w)
  if (w %% 2L != 1L) {
    stop("Window ", w, " is even, so it has no centre pixel. Occlusion rings ",
         "are defined around the centre.", call. = FALSE)
  }
  c0 <- (w + 1L) %/% 2L
  d  <- abs(seq_len(w) - c0)
  outer(d, d, pmax)
}

#' Replace a set of pixels with the same pixels taken from other samples.
#'
#' Operates on the fold cache's own representation: `cache[[role]][[key]]` is a
#' TORCH TENSOR of shape [n, channels, w, w], not a base R array. The first
#' version of this function assumed an R array, which fails on real data and
#' failed in the test for the same reason -- worth stating in the code, because
#' `dim()` works on both and hides the difference until an assignment does not.
#'
#' @param x      Tensor [n, channels, w, w].
#' @param mask   w x w logical matrix: TRUE where the pixels are hidden.
#' @param method "permute" (default) or "zero".
#' @param seed   Draw seed for the permutation.
#' @return A new tensor; `x` is not modified.
occlude_patch_array <- function(x, mask, method = c("permute", "zero"),
                                seed = 42L) {
  method <- match.arg(method)
  if (!inherits(x, "torch_tensor")) {
    stop("occlude_patch_array() works on the cache's tensors, got a ",
         class(x)[1], ".", call. = FALSE)
  }
  sh <- x$shape
  if (length(sh) != 4L) {
    stop("Expected [n, channels, w, w], got ", length(sh), " dimension(s).",
         call. = FALSE)
  }
  if (!identical(dim(mask), as.integer(sh[3:4]))) {
    stop("Mask is ", paste(dim(mask), collapse = "x"), " but the patch is ",
         paste(sh[3:4], collapse = "x"), ".", call. = FALSE)
  }
  if (!any(mask)) return(x)

  # torch_where WITH A [w, w] MASK, rather than flattening and indexing.
  #
  # The flattening route works and is a trap: torch tensors are row-major while
  # R arrays are column-major, so as.vector(mask) and a torch reshape walk the
  # pixels in DIFFERENT orders. For ring masks the two agree by accident --
  # patch_ring_index() is symmetric, being outer(d, d, pmax) -- so the bug
  # would hide behind exactly the masks this file uses, and appear the first
  # time someone passed an asymmetric one.
  #
  # Broadcasting [w, w] against [n, channels, w, w] sidesteps the question:
  # there is no linear index to get backwards.
  m <- torch::torch_tensor(mask, dtype = torch::torch_bool())$to(device = x$device)

  if (method == "zero") {
    return(torch::torch_where(m, torch::torch_zeros_like(x), x))
  }

  # ONE donor per sample, not one per pixel.
  #
  # Drawing a donor per pixel would scramble the hidden region internally as
  # well, so the measured drop would mix "the neighbourhood is uninformative"
  # with "the neighbourhood is no longer a landscape". One donor per sample
  # keeps the region a real piece of real terrain -- just somebody else's.
  n <- as.integer(sh[1])
  if (n < 2L) return(x)
  donor <- with_local_seed(seed, {
    p <- sample.int(n)
    # A sample donating to itself is not occluded at all, and with n small that
    # is a non-trivial share of the rows. Rotate the fixed points away.
    fixed <- which(p == seq_len(n))
    if (length(fixed)) p[fixed] <- p[(fixed %% n) + 1L]
    p
  })
  # torch_index_select() rather than x[donor, , , ]: R-level tensor indexing
  # has its own drop/recycling rules, and this is one place where being
  # explicit costs nothing and removes a class of silent shape bugs.
  di <- torch::torch_tensor(as.integer(donor), dtype = torch::torch_long())
  torch::torch_where(m, torch::torch_index_select(x, 1L, di), x)
}

#' What each part of the patch is worth to a trained model.
#'
#' @param model     A trained module (already on `device`).
#' @param cache     The fold cache (from build_fold_cache()$cache).
#' @param cfg       The config row the model was built from.
#' @param points_valid Metadata for the role, as fold_points_valid() returns.
#' @param role      Which split to measure on. Validation by default: the test
#'   set is frozen, and occlusion is a diagnostic, not a result.
#' @param transform Inverse of the target transformation.
#' @param device    torch device.
#' @param method    "permute" (default) or "zero".
#' @param seed      Permutation seed.
#' @param clamp     Plausible range of the target, as in predict_loader().
#'   The default floors at zero, which suits a stock and not a difference.
#' @return An object of class "spatial_occlusion".
spatial_occlusion <- function(model, cache, cfg, points_valid,
                              role = "validation", transform = identity,
                              device, method = c("permute", "zero"),
                              seed = 42L, clamp = c(0, Inf)) {
  method <- match.arg(method)
  ws   <- cfg$window_sizes[[1]]
  keys <- patch_window_key(ws)
  if (is.null(cache[[role]])) {
    stop("The cache has no role '", role, "'.", call. = FALSE)
  }

  score <- function(mod_windows) {
    # REPLACE THE WINDOWS, KEEP THE ROLE.
    #
    # cache[[role]] holds the window arrays AND the target vector `y`, and
    # .make_loaders_from_cache() reads `y` from there. Assigning a list of
    # windows over the whole role would drop the target and fail three frames
    # deep in tensor_dataset(), with a message about lengths.
    cache2 <- cache
    for (k in names(mod_windows)) cache2[[role]][[k]] <- mod_windows[[k]]
    loaders <- .make_loaders_from_cache(cache2, cfg)
    # THE SAME CLAMP TRAINING USED, and exposed because it is not always right.
    #
    # predict_loader() floors predictions at zero by default, which is correct
    # for a stock and wrong for anything that can go negative (a change, a
    # temperature). Left implicit, the diagnostic would floor half the
    # predictions of such a target and report a collapse that the model never
    # had -- and the collapse would look exactly like a real finding.
    pred <- predict_loader(model, loaders[[role]], points_valid, role,
                           transform = transform, device = device,
                           clamp = clamp)
    out <- calc_metrics(pred$obs, pred$pred)
    rm(loaders); invisible(gc(verbose = FALSE))
    out
  }

  # Every scope is a predicate on the ring index, applied to each window that
  # is large enough to have that ring. A 3x3 branch has rings 0 and 1 only, so
  # "ring 3" simply does not touch it -- which is the correct behaviour and the
  # reason the mask is built per window rather than once.
  max_ring <- (max(ws) - 1L) %/% 2L
  scopes <- c(
    list(list(id = "baseline",    keep = function(r) rep(FALSE, length(r)),
              what = "nothing hidden")),
    list(list(id = "context_all", keep = function(r) r > 0L,
              what = "everything except the centre pixel")),
    list(list(id = "centre_only_hidden", keep = function(r) r == 0L,
              what = "the centre pixel alone")),
    lapply(seq_len(max_ring), function(d) {
      list(id = sprintf("ring_%02d", d), keep = function(r) r == d,
           what = sprintf("the ring %d pixel(s) from the centre", d))
    })
  )

  rows <- lapply(scopes, function(sc) {
    mod <- lapply(seq_along(keys), function(i) {
      a <- cache[[role]][[keys[i]]]
      if (is.null(a)) {
        stop("Cache for role '", role, "' is missing window ", keys[i], ".",
             call. = FALSE)
      }
      ring <- patch_ring_index(ws[i])
      mask <- matrix(sc$keep(as.vector(ring)), nrow = nrow(ring))
      # The seed depends on the WINDOW, not on the scope: two scopes that hide
      # the same pixels must hide them the same way, or their difference is
      # partly the draw.
      occlude_patch_array(a, mask, method = method, seed = seed + ws[i])
    })
    names(mod) <- keys
    n_hidden <- sum(vapply(seq_along(keys), function(i) {
      ring <- patch_ring_index(ws[i]); sum(sc$keep(as.vector(ring)))
    }, numeric(1)))
    dplyr::bind_cols(
      tibble::tibble(scope = sc$id, hidden = sc$what,
                     n_pixels_hidden = as.integer(n_hidden)),
      score(mod))
  })

  res <- dplyr::bind_rows(rows)
  base <- res$ccc[res$scope == "baseline"][1]
  res$delta_ccc <- res$ccc - base
  res$pct_of_ccc <- round(100 * res$delta_ccc / base, 1)

  structure(list(table = res, method = method, role = role,
                 window_sizes = ws, baseline_ccc = base),
            class = "spatial_occlusion")
}

#' Say what the occlusion found, including when it found nothing.
print.spatial_occlusion <- function(x, ...) {
  cat("\nSpatial occlusion --", x$role, "| windows",
      paste(x$window_sizes, collapse = "+"), "| method", x$method, "\n")
  cat(strrep("-", 72), "\n")
  print(dplyr::select(x$table, scope, n_pixels_hidden, ccc, mae, delta_ccc,
                      pct_of_ccc), n = Inf)

  ctx <- x$table$delta_ccc[x$table$scope == "context_all"][1]
  ctr <- x$table$delta_ccc[x$table$scope == "centre_only_hidden"][1]
  cat("\n")
  cat(sprintf("  hiding all context  : %+.4f CCC (%.1f%% of %.4f)\n",
              ctx, 100 * ctx / x$baseline_ccc, x$baseline_ccc))
  cat(sprintf("  hiding the centre   : %+.4f CCC\n", ctr))

  # THE COMPARISON THAT MATTERS is context against centre. A network that loses
  # more by losing one pixel than by losing the other 224 is a network doing
  # point prediction with expensive extra steps.
  if (is.finite(ctx) && is.finite(ctr)) {
    if (abs(ctx) < abs(ctr)) {
      cat("\n  -> THE CENTRE PIXEL CARRIES MORE THAN THE WHOLE NEIGHBOURHOOD.\n")
      cat("     On this evidence the convolution is not using spatial context,\n")
      cat("     and a model over point values answers the same question for a\n")
      cat("     fraction of the cost. Compare with 03b before concluding: the\n")
      cat("     two are independent routes to the same question.\n")
    } else {
      cat("\n  -> The neighbourhood is doing work: hiding it costs more than\n")
      cat("     hiding the centre. The per-ring rows say how far out it reaches,\n")
      cat("     which is what should set the window size of the next run.\n")
    }
  }
  invisible(x)
}

#' Run the occlusion for one trained unit of a tuning run.
#'
#' A convenience over spatial_occlusion(): finds the checkpoint, rebuilds the
#' architecture from the grid and the fold's cache from the plan, so the caller
#' names a unit rather than assembling one.
#'
#' @param run_dir   Tuning run directory.
#' @param data      dsm_data, or a list with store, points, type_table.
#' @param config_id Which config.
#' @param fold,seed_i Which unit of it.
#' @param ...       Passed to spatial_occlusion().
occlusion_report <- function(run_dir, data, config_id, fold = 1L, seed_i = 1L,
                             transform = identity, device, ...) {
  tune_grid <- readRDS(file.path(run_dir, "tune_grid.rds"))
  plan      <- readRDS(file.path(run_dir, "fold_plan.rds"))
  cfg <- tune_grid[tune_grid$config_id == config_id, , drop = FALSE]
  if (nrow(cfg) != 1L) {
    stop("config_id '", config_id, "' is not in this run's grid.", call. = FALSE)
  }
  ck <- file.path(run_dir, "models",
                  sprintf("%s_f%d_s%d_best.pt", config_id, fold, seed_i))
  if (!file.exists(ck)) stop("No checkpoint: ", ck, call. = FALSE)

  idx   <- plan$folds[[fold]]
  cache <- build_fold_cache(data$store, data$points, data$type_table, idx,
                            data$store$window_sizes, verbose = FALSE)
  m <- build_cnn_from_config(cfg, data$store$n_channels)
  m$load_state_dict(torch::torch_load(ck))
  m$to(device = device)

  out <- spatial_occlusion(m, cache$cache, cfg, fold_points_valid(data$store, idx),
                           transform = transform, device = device, ...)
  rm(m, cache); invisible(gc(verbose = FALSE))
  out
}
