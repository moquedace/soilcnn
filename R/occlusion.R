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
#' @noRd
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
#' TORCH TENSOR of shape `[n, channels, w, w]`, not a base R array. The first
#' version of this function assumed an R array, which fails on real data and
#' failed in the test for the same reason -- worth stating in the code, because
#' `dim()` works on both and hides the difference until an assignment does not.
#'
#' @param x      Tensor `[n, channels, w, w]`.
#' @param mask   w x w logical matrix: TRUE where the pixels are hidden.
#' @param method "permute" (default) or "zero".
#' @param seed   Draw seed for the permutation.
#' @return A new tensor; `x` is not modified.
#' @noRd
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
#' @param points_valid ONE role's metadata tibble -- fold_points_valid() returns
#'   a named list, so this is `fold_points_valid(store, index)[[role]]`.
#' @param role      Which split to measure on. Validation by default: the test
#'   set is frozen, and occlusion is a diagnostic, not a result.
#' @param transform Inverse of the target transformation.
#' @param device    torch device.
#' @param method    "permute" (default) or "zero".
#' @param seed      Permutation seed.
#' @param clamp     Plausible range of the target, as in predict_loader().
#'   The default floors at zero, which suits a stock and not a difference.
#' @return An object of class "spatial_occlusion".
#' @export
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

  # ONE ROLE'S TIBBLE, NOT THE WHOLE NAMED LIST.
  #
  # fold_points_valid() returns list(train =, validation =, test =), and this
  # wants the one tibble for `role`. occlusion_report() passed the whole list,
  # so the wrapper could never have worked -- and the unit tests did not catch
  # it because they call THIS function directly with a tibble. The wrapper had
  # never been called by anything.
  #
  # Guarded here rather than fixed silently at the call site: the failure
  # without it is check_point_contract() complaining that profile_id is
  # missing, three frames down, which sends the reader to the point table
  # instead of to the argument.
  if (!is.data.frame(points_valid) && is.list(points_valid) &&
      all(c("train", "validation") %in% names(points_valid))) {
    stop("`points_valid` is the whole fold_points_valid() list, not one role. ",
         "Pass\n  fold_points_valid(store, index)[[\"", role, "\"]]",
         call. = FALSE)
  }
  check_point_contract(points_valid, what = "points_valid")

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
#'
#' @param x   A `spatial_occlusion`, from [spatial_occlusion()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.spatial_occlusion <- function(x, ...) {
  cat("\nSpatial occlusion --", x$role, "| windows",
      paste(x$window_sizes, collapse = "+"), "| method", x$method, "\n")
  cat(strrep("-", 72), "\n")
  print(dplyr::select(x$table, scope, n_pixels_hidden, ccc, mae, delta_ccc,
                      pct_of_ccc), n = Inf)

  ctx <- x$table$delta_ccc[x$table$scope == "context_all"][1]
  ctr <- x$table$delta_ccc[x$table$scope == "centre_only_hidden"][1]

  # AREA IS A CONFOUND, AND THE FIRST VERSION OF THIS BLOCK IGNORED IT.
  #
  # It compared |context_all| against |centre_only_hidden| and announced a
  # winner. Those hide 224 pixels and 1 pixel. On the real cfg_003 that gave
  # -0.513 against -0.002 and the line read "the neighbourhood is doing work"
  # -- which is true of almost any convolution over almost any patch, because
  # 224 permuted pixels destroy every feature map while one permuted pixel
  # perturbs them. A comparison that a working network cannot fail is not a
  # measurement.
  #
  # So the per-pixel cost is reported beside the total, and the verdict is
  # phrased on the RINGS, which have their own sizes but differ from each other
  # by DISTANCE rather than by being the whole patch against one pixel.
  per_px <- function(scope) {
    i <- which(x$table$scope == scope)[1]
    n <- x$table$n_pixels_hidden[i]
    if (!length(i) || is.na(n) || n == 0) return(NA_real_)
    x$table$delta_ccc[i] / n
  }
  n_px <- function(scope) x$table$n_pixels_hidden[x$table$scope == scope][1]
  cat(sprintf("\n  hiding all context  : %+.4f CCC over %3d px  (%+.5f per px)\n",
              ctx, n_px("context_all"), per_px("context_all")))
  cat(sprintf("  hiding the centre   : %+.4f CCC over %3d px  (%+.5f per px)\n",
              ctr, n_px("centre_only_hidden"), per_px("centre_only_hidden")))

  rings <- x$table[grepl("^ring_", x$table$scope), , drop = FALSE]
  if (nrow(rings) > 1L) {
    rings$per_px <- rings$delta_ccc / rings$n_pixels_hidden
    worst <- rings$scope[which.min(rings$delta_ccc)][1]
    cat(sprintf("  costliest ring      : %s (%+.4f CCC over %d px)\n",
                worst, min(rings$delta_ccc),
                rings$n_pixels_hidden[which.min(rings$delta_ccc)]))
    cat("\n  Per ring, cost per pixel hidden:\n")
    cat("   ", paste(sprintf("%s %+.5f", sub("ring_", "r", rings$scope),
                             rings$per_px), collapse = "  "), "\n")
  }

  # WHAT A LARGE CONTEXT EFFECT DOES AND DOES NOT MEAN.
  #
  # It means this NETWORK reads the neighbourhood. It does NOT mean the
  # neighbourhood carries information the centre pixel lacks. At 250 m most
  # covariates are smooth, so a pixel three cells away is close to a copy of the
  # centre; a model can lean entirely on the rim and learn nothing the centre
  # would not have told it.
  #
  # The two questions are separable and this framework answers both:
  #
  #   does the network use the neighbourhood?      <- this report
  #   does the neighbourhood ADD anything?         <- stage 03b, rf_centre
  #                                                   against rf_context
  #
  # They can disagree without either being wrong, and when they do the answer is
  # redundancy rather than contradiction. Say which one is being quoted.
  if (is.finite(ctx) && is.finite(ctr)) {
    if (abs(ctr) > abs(ctx)) {
      cat("\n  -> One pixel costs more than the other 224. This network is doing\n")
      cat("     point prediction with convolutional machinery around it.\n")
    } else {
      cat("\n  -> This network reads the neighbourhood: the rim carries its\n")
      cat("     prediction and the centre pixel barely moves it.\n")
      cat("     THIS IS NOT EVIDENCE THAT THE NEIGHBOURHOOD ADDS INFORMATION.\n")
      cat("     Smooth covariates make the rim a near-copy of the centre, so a\n")
      cat("     network can depend on it entirely and still learn nothing a\n")
      cat("     centre-only model would have missed. Stage 03b answers that\n")
      cat("     question -- rf_centre against rf_context -- and this one does not.\n")
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
#' @param transform NULL (the default) for the inverse of the transform the
#'   store was built under, as in dsm_train(); a function to use instead,
#'   refused if it disagrees with the store's.
#' @inheritParams spatial_occlusion
#' @return A `spatial_occlusion`, as spatial_occlusion() returns.
#' @export
occlusion_report <- function(run_dir, data, config_id, fold = 1L, seed_i = 1L,
                             role = "validation", transform = NULL,
                             device, ...) {
  # The store's inverse by default, as dsm_train() and score_test_grid().
  transform <- .resolve_train_transform(transform, data, verbose = FALSE)
  tune_grid <- readRDS(file.path(run_dir, "tune_grid.rds"))
  plan      <- readRDS(file.path(run_dir, "fold_plan.rds"))
  cfg <- tune_grid[tune_grid$config_id == config_id, , drop = FALSE]
  if (nrow(cfg) != 1L) {
    stop("config_id '", config_id, "' is not in this run's grid.", call. = FALSE)
  }
  ck <- file.path(run_dir, "models",
                  sprintf("%s_f%d_s%d_best.pt", config_id, fold, seed_i))
  if (!file.exists(ck)) stop("No checkpoint: ", ck, call. = FALSE)

  idx <- plan$folds[[fold]]

  # ONLY THE WINDOWS THIS CONFIG USES. The store may hold 3, 9 and 15 while the
  # config reads 15 alone; caching all three scales and copies tensors nothing
  # will look at -- 0.85 GB instead of 0.61 GB here, and worse on a larger store.
  ws_needed <- sort(unique(unlist(cfg$window_sizes)))
  cache <- build_fold_cache(data$store, data$points, data$type_table, idx,
                            ws_needed, verbose = FALSE)
  m <- build_cnn_from_config(cfg, data$store$n_channels)
  m$load_state_dict(torch::torch_load(ck))
  m$to(device = device)

  out <- spatial_occlusion(m, cache$cache, cfg,
                           fold_points_valid(data$store, idx)[[role]],
                           role = role, transform = transform,
                           device = device, ...)
  rm(m, cache); invisible(gc(verbose = FALSE))
  out
}
