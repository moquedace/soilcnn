# Patch geometry — cell indexing and array assembly
#
# Single source of truth for how a (row, col) centre becomes a [N, C, H, W]
# patch array. Both extraction paths go through here:
#   • training side   — one band at a time over a raster strip
#   • prediction side — all bands at once over a tile strip
# Keeping one implementation is the point: if the two drift apart, the model
# is fed one geometry and the map is built from another, and the result looks
# entirely plausible while being wrong. tests/test_patch_geometry.R pins it.
#
# Index convention — do not change without updating the test:
#   offsets <- expand.grid(dr, dc)      dr varies fastest (R column-major)
#   position j = dr_idx + (dc_idx - 1) * w
#   therefore array dim 1 is the ROW offset (H), dim 2 the COLUMN offset (W)
#
# All row/col inputs are LOCAL to the strip that was read (1-based), never
# global raster coordinates. Callers do the global -> local conversion.
#
# Cost note: gathering cells out of the strip is the expensive step, so every
# function here indexes the strip AT MOST ONCE and reuses the gathered values
# for both the validity check and the assembly. Splitting that into two passes
# would double the cost of the hottest loop in the pipeline.

# ── cell indexing ─────────────────────────────────────────────────────────────

#' Linear cell indices of every patch position, for every centre.
#'
#' @param centre_row  Integer vector, row of each centre, local to the strip.
#' @param centre_col  Integer vector, column of each centre, local to the strip.
#' @param n_cols      Columns in the strip (row-major stride).
#' @param window_size Odd integer patch side.
#' @return An (n_centres x window_size^2) integer matrix of linear indices.
#' @noRd
patch_cell_index <- function(centre_row, centre_col, n_cols, window_size) {
  if (window_size %% 2L != 1L) {
    stop("window_size must be odd (a patch has one centre pixel), got ",
         window_size, ".", call. = FALSE)
  }
  if (length(centre_row) != length(centre_col)) {
    stop("centre_row and centre_col must have the same length -- got ",
         length(centre_row), " and ", length(centre_col), ".", call. = FALSE)
  }

  half_w  <- (window_size - 1L) %/% 2L
  n       <- length(centre_row)
  offsets <- expand.grid(dr = (-half_w):half_w, dc = (-half_w):half_w)
  n_pos   <- nrow(offsets)

  cm <- matrix(0L, nrow = n, ncol = n_pos)
  for (j in seq_len(n_pos)) {
    cm[, j] <- (centre_row + offsets$dr[j] - 1L) * n_cols +
               (centre_col + offsets$dc[j])
  }
  cm
}

# ── training side: one band at a time ─────────────────────────────────────────

#' Gather and assemble one band's patches in a single pass.
#'
#' @param band_values Numeric vector of one band's cells, row-major.
#' @param cell_index  Matrix from patch_cell_index().
#' @param window_size Odd integer patch side.
#' @return list(array = [n_centres, w, w], valid = logical vector per centre).
#' @noRd
patch_band_assemble <- function(band_values, cell_index, window_size) {
  n     <- nrow(cell_index)
  n_pos <- ncol(cell_index)
  vals  <- band_values[as.vector(t(cell_index))]

  list(
    array = aperm(array(vals, dim = c(window_size, window_size, n)),
                  c(3L, 1L, 2L)),
    valid = colSums(!is.finite(matrix(vals, nrow = n_pos, ncol = n))) == 0L
  )
}

# ── prediction side: all bands at once, in two stages ─────────────────────────
# Stage 1 gathers once and reports validity; stage 2 only reshapes and slices.
# The stages are separate because validity must be intersected across ALL
# window sizes before any array is built.

#' Gather every channel's patch cells once, and report per-centre validity.
#'
#' @param cell_values Numeric matrix, cells (row-major) x channels.
#' @param cell_index  Matrix from patch_cell_index().
#' @param n_channels  Number of channels (columns of cell_values).
#' @return list(values = array [n_pos, n_centres, n_channels], valid = logical).
#' @noRd
patch_gather <- function(cell_values, cell_index, n_channels) {
  n     <- nrow(cell_index)
  n_pos <- ncol(cell_index)
  vals  <- cell_values[as.vector(t(cell_index)), , drop = FALSE]

  row_finite <- rowSums(!is.finite(vals)) == 0
  valid      <- colSums(matrix(row_finite, nrow = n_pos, ncol = n)) == n_pos

  dim(vals) <- c(n_pos, n, n_channels)
  list(values = vals, valid = valid)
}

#' Reshape gathered values into [n_keep, n_channels, w, w]. No re-indexing.
#'
#' @param values     The `values` element of patch_gather().
#' @param keep       Integer vector of centre positions to keep.
#' @param n_channels Number of channels.
#' @param window_size Odd integer patch side.
#' @noRd
patch_finish <- function(values, keep, n_channels, window_size) {
  if (length(keep) == 0L) {
    return(array(0, dim = c(0L, n_channels, window_size, window_size)))
  }
  step <- aperm(values[, keep, , drop = FALSE], c(2L, 3L, 1L))
  array(step, dim = c(length(keep), n_channels, window_size, window_size))
}

# ── edge rule ─────────────────────────────────────────────────────────────────

#' Centres far enough from the strip edge for a full window.
#'
#' The framework's rule everywhere: a centre needs half_w cells of margin on
#' every side, otherwise the patch would read outside the strip and silently
#' wrap into the adjacent row.
#' @noRd
patch_centre_in_bounds <- function(centre_row, centre_col, n_rows, n_cols,
                                   window_size) {
  half_w <- (window_size - 1L) %/% 2L
  centre_row - half_w >= 1L & centre_row + half_w <= n_rows &
    centre_col - half_w >= 1L & centre_col + half_w <= n_cols
}
