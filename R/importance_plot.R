# ── Figures of an importance ──────────────────────────────────────────────────
#
# WHAT EACH ONE SHOWS. Every importance answers its own question, so each kind
# gets the figure that answers it, not one figure for all:
#
#   permutation   the variables ranked, with the spread between models -- a
#                 variable whose bar crosses zero in some models is not a
#                 finding;
#   context       by ring, the cost PER PIXEL from the centre out (the totals
#                 grow with the ring's area, R/occlusion.R's lesson); per
#                 variable, the context's cost per pixel against the centre's;
#                 by window, each branch's cost, with the gate beside it;
#   shap          the summary plot of Lundberg & Lee: one row per variable, a
#                 point per profile at its SHAP value, coloured by the
#                 variable's value there -- the direction is read from the
#                 colours, the spread from the width;
#   ale           the curves, with the band between models, and the class
#                 effects as bars;
#   sage          the loss each variable takes away, ranked, with the spread
#                 between models -- below zero, it costs skill;
#   refit         the skill a network trained without each variable loses,
#                 with the spread between seeds;
#   a map        importance_map()'s layers: SHAP per variable on a diverging
#                 scale centred on zero, the dominant variable, the prediction.
#
# BASE GRAPHICS, NOT A PLOTTING PACKAGE: the vignette's figures are drawn so,
# and a figure is not worth a dependency every user would install.

#' Plot a variable importance.
#'
#' The figure that answers the importance's own question: bars for a
#' permutation, the cost per pixel by ring (or per variable, or per window)
#' for a context, the summary plot for SHAP -- a point per profile at its SHAP
#' value, coloured by the variable's value there -- the curves for ALE, for
#' SAGE the loss each variable takes away, and for a refit the skill a network
#' trained without it loses.
#'
#' @param x A `dsm_importance`, from [dsm_importance()].
#' @param n How many variables to show.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
plot.dsm_importance <- function(x, n = 20L, ...) {
  m <- x$method
  switch(m$kind,
    permutation = .importance_plot_bars(
      x$table, n, sprintf("Permutation importance -- %s", basename(x$run_dir)),
      c(ccc = "drop in CCC", rmse = "rise in RMSE",
        rmse_transform = "rise in RMSE (trained space)")[[m$metric]]),
    context = .importance_plot_context(x, n),
    shap = .importance_plot_shap(x, n),
    ale = .importance_plot_ale(x, n),
    sage = .importance_plot_bars(x$table, n, sprintf("SAGE -- %s", basename(x$run_dir)),
                                 "loss explained (trained space)"),
    refit = .importance_plot_bars(
      x$table, n, sprintf("Refit without the variable -- %s", basename(x$run_dir)),
      c(ccc = "drop in CCC, refitted without it", rmse = "rise in RMSE, refitted without it",
        rmse_transform = "rise in RMSE (trained space), refitted without it")[[m$metric]]),
    stop("No figure for an importance of kind '", m$kind, "'.", call. = FALSE))
  invisible(x)
}

# The house colours, from the vignette's figures.
.importance_col <- c(blue = "#246589", earth = "#ab6645", olive = "#60734f",
                     ink = "#253840", muted = "#63767d", line = "#dce4e3")

# A left margin wide enough for the longest label shown.
.importance_left_margin <- function(labels) {
  max(6, min(22, 0.55 * max(nchar(labels), 1L) + 1))
}

# Ranked horizontal bars, largest at the top, with the spread between models.
.importance_plot_bars <- function(tab, n, main, xlab) {
  show <- utils::head(tab[order(-tab$importance), , drop = FALSE], n)
  show <- show[rev(seq_len(nrow(show))), , drop = FALSE]
  lab <- show$variable %||% show$target
  op <- graphics::par(mar = c(4.5, .importance_left_margin(lab), 3, 1.5))
  on.exit(graphics::par(op))
  sdv <- show$sd_models %||% rep(NA_real_, nrow(show))
  sdv[is.na(sdv)] <- 0
  lim <- range(c(0, show$importance - sdv, show$importance + sdv), na.rm = TRUE)
  col <- ifelse(show$importance >= 0, .importance_col[["blue"]], .importance_col[["earth"]])
  y <- graphics::barplot(show$importance, horiz = TRUE, names.arg = lab, las = 1, col = col,
                         border = NA, xlim = lim, main = main, xlab = xlab, cex.names = 0.8)
  if (any(sdv > 0)) {
    graphics::arrows(show$importance - sdv, y, show$importance + sdv, y, angle = 90,
                     code = 3, length = 0.03, col = .importance_col[["ink"]])
  }
  graphics::abline(v = 0, col = .importance_col[["muted"]])
  invisible(NULL)
}

.importance_plot_context <- function(x, n) {
  m <- x$method
  t <- x$table
  if (identical(m$by, "window")) {
    t <- t[order(t$window), , drop = FALSE]
    sub <- if (is.null(x$gate)) "no gate" else
      sprintf("gate on %s: %.2f on average", patch_window_key(x$window_sizes[1]),
              mean(x$gate$gate))
    graphics::barplot(t$importance, names.arg = t$target, col = .importance_col[["blue"]],
                      border = NA, main = "Each window's whole input permuted",
                      ylab = "drop in skill", sub = sub)
    graphics::abline(h = 0, col = .importance_col[["muted"]])
    return(invisible(NULL))
  }
  if (isTRUE(m$per_variable)) {
    cols <- c("variable", "per_pixel")
    j <- merge(t[t$band == "centre", cols], t[t$band == "context", cols], by = "variable",
               suffixes = c("_centre", "_context"))
    tot <- merge(t[t$band == "centre", c("variable", "importance")],
                 t[t$band == "context", c("variable", "importance")], by = "variable")
    j$total <- tot$importance.x[match(j$variable, tot$variable)] +
      tot$importance.y[match(j$variable, tot$variable)]
    j$px_ratio <- ifelse(j$per_pixel_centre > 0 & j$per_pixel_context >= 0,
                         j$per_pixel_context / j$per_pixel_centre, NA_real_)
    j <- utils::head(j[order(-j$total), , drop = FALSE], n)
    j <- j[rev(seq_len(nrow(j))), , drop = FALSE]
    op <- graphics::par(mar = c(4.5, .importance_left_margin(j$variable), 3, 1.5))
    on.exit(graphics::par(op))
    graphics::barplot(j$px_ratio, horiz = TRUE, names.arg = j$variable, las = 1,
                      col = .importance_col[["olive"]], border = NA, cex.names = 0.8,
                      main = "Read around the point, or at it?",
                      xlab = "cost of a context pixel / cost of the centre pixel")
    graphics::abline(v = c(0, 1), col = .importance_col[["muted"]], lty = c(1, 2))
    return(invisible(NULL))
  }
  t <- t[match(x$targets$target, t$target), , drop = FALSE]
  t <- t[t$band != "context", , drop = FALSE]
  graphics::barplot(t$per_pixel, names.arg = sub("ring_0?", "ring ", t$band),
                    col = .importance_col[["blue"]], border = NA, las = 2,
                    main = "How far from the point the model reads",
                    ylab = "drop in skill per pixel hidden")
  graphics::abline(h = 0, col = .importance_col[["muted"]])
  invisible(NULL)
}

# The summary plot: a row per variable, a point per profile at its SHAP value,
# coloured from low (blue) to high (earth) by the variable's value there; grey
# for a variable of several channels, which has no one value.
.importance_plot_shap <- function(x, n) {
  show <- utils::head(x$table[order(-x$table$importance), , drop = FALSE], n)
  vars <- rev(show$variable)
  op <- graphics::par(mar = c(4.5, .importance_left_margin(vars), 3, 4))
  on.exit(graphics::par(op))
  P <- x$points
  vals <- x$values
  lim <- range(unlist(lapply(vars, function(v) P[[v]])), na.rm = TRUE)
  graphics::plot(NA, xlim = lim, ylim = c(0.5, length(vars) + 0.5), yaxt = "n",
                 xlab = sprintf("SHAP value (%s)", if (identical(x$transform_name, "log1p"))
                   "log1p: +0.1 is about +10%" else "the network's units"),
                 ylab = "", main = .importance_label(x$method), cex.main = 0.9)
  graphics::axis(2, at = seq_along(vars), labels = vars, las = 1, cex.axis = 0.8)
  graphics::abline(v = 0, col = .importance_col[["muted"]])
  pal <- grDevices::colorRampPalette(c(.importance_col[["blue"]], "#d9d9d9",
                                       .importance_col[["earth"]]))(64)
  for (i in seq_along(vars)) {
    s <- P[[vars[i]]]
    keep <- if (length(s) > 2000L) {
      with_local_seed(i, sort(sample.int(length(s), 2000L)))
    } else seq_along(s)
    v <- if (!is.null(vals) && vars[i] %in% colnames(vals)) vals[keep, vars[i]] else NULL
    col <- if (is.null(v) || all(!is.finite(v))) {
      grDevices::adjustcolor(.importance_col[["muted"]], 0.5)
    } else {
      r <- rank(v, na.last = "keep") / sum(is.finite(v))
      idx <- pmax(1L, ceiling(r * 64))
      idx[is.na(idx)] <- 32L                     # no value at the point: the middle grey
      grDevices::adjustcolor(pal[idx], 0.7)
    }
    # The jitter by a seed of its own: the same importance draws the same
    # figure, and a plot leaves the session's random numbers as they were.
    yy <- i + with_local_seed(1000L + i, stats::runif(length(keep), -0.3, 0.3))
    graphics::points(s[keep], yy, pch = 16, cex = 0.5, col = col)
  }
  graphics::mtext("value: low", side = 4, line = 0.5, at = 0.6, las = 1, cex = 0.7,
                  col = .importance_col[["blue"]])
  graphics::mtext("high", side = 4, line = 0.5, at = length(vars) + 0.4, las = 1, cex = 0.7,
                  col = .importance_col[["earth"]])
  invisible(NULL)
}

# The curves of the continuous variables, the bars of the categorical ones, as
# small multiples, the most important first.
.importance_plot_ale <- function(x, n) {
  show <- utils::head(x$table[order(-x$table$importance), , drop = FALSE], min(n, 12L))
  k <- nrow(show)
  nc <- min(4L, k)
  nr <- ceiling(k / nc)
  op <- graphics::par(mfrow = c(nr, nc), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 2, 0))
  on.exit(graphics::par(op))
  for (v in show$variable) {
    if (identical(show$kind[show$variable == v], "continuous")) {
      cu <- x$curves[x$curves$variable == v, , drop = FALSE]
      sdv <- cu$ale_sd
      sdv[is.na(sdv)] <- 0
      graphics::plot(cu$value, cu$ale, type = "n", xlab = v, ylab = "effect",
                     ylim = range(c(cu$ale - sdv, cu$ale + sdv)), main = v, cex.main = 0.9)
      graphics::polygon(c(cu$value, rev(cu$value)), c(cu$ale - sdv, rev(cu$ale + sdv)),
                        col = grDevices::adjustcolor(.importance_col[["blue"]], 0.2), border = NA)
      graphics::lines(cu$value, cu$ale, col = .importance_col[["blue"]], lwd = 2)
      graphics::rug(cu$value, col = .importance_col[["muted"]])
      graphics::abline(h = 0, col = .importance_col[["muted"]], lty = 2)
    } else {
      cl <- x$classes[x$classes$variable == v, , drop = FALSE]
      cl <- utils::head(cl[order(-abs(cl$effect)), , drop = FALSE], 12L)
      lab <- sub(paste0("^", v, "_"), "", cl$class)
      graphics::barplot(rev(cl$effect), names.arg = rev(lab), horiz = TRUE, las = 1,
                        cex.names = 0.6, border = NA, main = v, cex.main = 0.9,
                        col = ifelse(rev(cl$effect) >= 0, .importance_col[["earth"]],
                                     .importance_col[["blue"]]))
      graphics::abline(v = 0, col = .importance_col[["muted"]])
    }
  }
  graphics::mtext(sprintf("ALE -- %s, in %s", basename(x$run_dir),
                          if (identical(x$transform_name, "log1p")) "log1p" else "the target's units"),
                  outer = TRUE, cex = 0.95)
  invisible(NULL)
}

#' Plot maps of SHAP values.
#'
#' One panel per variable on a diverging scale centred on zero -- where the
#' variable raises the prediction and where it lowers it -- then the dominant
#' variable and the prediction.
#'
#' @param x An `importance_map`, from [importance_map()].
#' @param n How many variables, the most important first.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
plot.importance_map <- function(x, n = 12L, ...) {
  v <- names(x$shap)
  mean_abs <- vapply(v, function(l) {
    z <- terra::values(x$shap[[l]], mat = FALSE)
    mean(abs(z), na.rm = TRUE)
  }, numeric(1))
  v <- utils::head(v[order(-mean_abs)], n)
  k <- length(v) + 2L
  nc <- min(4L, k)
  nr <- ceiling(k / nc)
  op <- graphics::par(mfrow = c(nr, nc), mar = c(1, 1, 2, 1), oma = c(0, 0, 2, 0))
  on.exit(graphics::par(op))
  lim <- max(vapply(v, function(l) max(abs(terra::values(x$shap[[l]], mat = FALSE)), na.rm = TRUE),
                    numeric(1)))
  pal <- grDevices::colorRampPalette(c(.importance_col[["blue"]], "#f7f7f7",
                                       .importance_col[["earth"]]))(65)
  for (l in v) .importance_image(x$shap[[l]], pal, c(-lim, lim), l)
  k_dom <- nrow(x$legend)
  pal_dom <- grDevices::hcl.colors(max(k_dom, 3L), "Set 3")[seq_len(k_dom)]
  .importance_image(x$dominant, pal_dom, c(0.5, k_dom + 0.5), "dominant")
  present <- sort(unique(stats::na.omit(terra::values(x$dominant, mat = FALSE))))
  graphics::legend("bottomleft", legend = x$legend$variable[present], fill = pal_dom[present],
                   cex = 0.6, bg = grDevices::adjustcolor("white", 0.8), border = NA)
  .importance_image(x$prediction, grDevices::hcl.colors(64, "viridis"), NULL, "prediction")
  graphics::mtext(sprintf("SHAP on a diverging scale, +/- %.3g", lim), outer = TRUE, cex = 0.9)
  invisible(x)
}

# One layer as an image, north up, in its own coordinates.
.importance_image <- function(r, col, zlim, main) {
  z <- terra::as.matrix(r, wide = TRUE)
  e <- as.vector(terra::ext(r))
  xs <- seq(e[1], e[2], length.out = ncol(z) + 1L)
  ys <- seq(e[3], e[4], length.out = nrow(z) + 1L)
  xs <- (xs[-1] + xs[-length(xs)]) / 2
  ys <- (ys[-1] + ys[-length(ys)]) / 2
  graphics::image(xs, ys, t(z[nrow(z):1, , drop = FALSE]), col = col,
                  zlim = zlim %||% range(z, na.rm = TRUE), asp = 1, axes = FALSE,
                  xlab = "", ylab = "", main = main, cex.main = 0.85)
  graphics::box(col = .importance_col[["line"]])
  invisible(NULL)
}

#' Plot importances side by side.
#'
#' The first two importances' shares of their largest, one against the other,
#' each variable a point on a 1:1 line; and, for two SHAP importances of the
#' same points, how well they agree variable by variable.
#'
#' @param x An `importance_comparison`, from [compare_importance()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
plot.importance_comparison <- function(x, ...) {
  cols <- names(x$labels)
  if (length(cols) < 2L) return(invisible(x))
  a <- x$table[[paste0("share_", cols[1])]]
  b <- x$table[[paste0("share_", cols[2])]]
  pa <- x$points_agreement
  op <- graphics::par(mfrow = c(1L, if (is.null(pa)) 1L else 2L), mar = c(4.5, 4.5, 3, 1))
  on.exit(graphics::par(op))
  graphics::plot(a, b, pch = 16, col = .importance_col[["blue"]], xlim = c(min(0, a, na.rm = TRUE), 1),
                 ylim = c(min(0, b, na.rm = TRUE), 1),
                 xlab = sprintf("%s (share of its largest)", x$labels[[1]]),
                 ylab = sprintf("%s (share of its largest)", x$labels[[2]]),
                 main = sprintf("Agreement: Spearman %.2f", x$agreement[1, 2]), cex.lab = 0.75)
  graphics::abline(0, 1, col = .importance_col[["muted"]], lty = 2)
  top <- utils::head(order(-(pmax(a, b, na.rm = TRUE))), 8L)
  graphics::text(a[top], b[top], x$table$target[top], pos = 4, cex = 0.6,
                 col = .importance_col[["ink"]])
  if (!is.null(pa)) {
    pa <- pa[order(pa$r), , drop = FALSE]
    op2 <- graphics::par(mar = c(4.5, .importance_left_margin(pa$variable), 3, 1))
    graphics::barplot(pa$r, names.arg = pa$variable, horiz = TRUE, las = 1, border = NA,
                      col = .importance_col[["olive"]], xlim = c(min(0, pa$r, na.rm = TRUE), 1),
                      cex.names = 0.7, main = "SHAP point by point", xlab = "correlation")
    graphics::par(op2)
  }
  invisible(x)
}
