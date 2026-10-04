# ── Variable importance: which variables a fitted model relies on ─────────────
#
# ONE FRAME, SEVERAL QUESTIONS. dsm_train() takes the validation design as an
# argument -- resampling = spatial_cv(), random_cv(), ... -- and changing that
# one argument changes the question the run answers. dsm_importance() is built
# the same way: method = permutation_importance(), context_importance(),
# shap_importance(), sage_importance(), refit_importance() or ale_effect() --
# one slot, six questions. They are not rival estimates of one number. Each
# answers its own question
# (Ewald et al. 2024, "A guide to feature importance methods for scientific
# inference"), and where they disagree is where the reading is.
#
# PERMUTATION (permutation_importance()). Each point takes one variable's whole patch
# from another point, the model predicts again, and the loss of skill is the
# variable's importance to THIS model (Breiman 2001; Fisher, Rudin & Dominici
# 2019 call it model reliance). It retrains nothing, so it costs forward passes
# only. Two variants answer two more questions:
#
#   within = blocks or classes   the donor comes from the same block or class
#                                (an ecoregion). What the variable adds beyond
#                                where the point is: a regional gradient loses
#                                its importance, a local discriminator keeps
#                                it. Unrestricted permutation builds points
#                                that do not exist when variables are
#                                correlated (Hooker, Mentch & Zhou 2021); a
#                                donor from the same place is a plausible one.
#                                Conditioning on strata is Strobl et al.
#                                (2008)'s idea; strata made of SPACE are this
#                                package's adaptation of it.
#   fill = "mean"                the variable is set to its training mean, an
#                                ablation. Not the default, for the reason
#                                R/occlusion.R gives at length: a patch flat at
#                                the mean is a landscape that does not exist,
#                                so the drop mixes "uninformative" with "off
#                                the data". Against the permutation, the
#                                difference is itself the finding.
#
# CONTEXT (context_importance()). The same permutation, of PLACES in the patch
# rather than of variables: a ring of pixels at one distance from the point,
# every ring but the centre, or a window's whole input. It is R/occlusion.R's
# question -- does the network read the neighbourhood, and how far out --
# asked of the final model and its folds, with every model checked and every
# draw shared. Per variable, it says which variables are read through the
# neighbourhood and which at the point alone; per window, how much each branch
# carries, beside what the gate between them says it weighs. Hiding a part of
# the input is occlusion (Zeiler & Fergus 2014); by ring, per variable and per
# window is this package's use of it on concentric patches.
#
# SHAP (shap_importance()). Not a perturbation but an attribution: each
# point's prediction, less a reference, shared among the variables with a sign
# (Lundberg & Lee 2017), from the network's gradients -- expected gradients
# (Erion et al. 2021) over a background of the model's own training points, or
# integrated gradients (Sundararajan et al. 2017) from the training mean -- or,
# over a few themes, the exact Shapley values of the coalitions, with no
# gradient at all (the kernel: Lundberg & Lee 2017's game). The gradient
# estimators and the kernel share the same total and split it alike when the
# themes do not interact, so where they part is how much the themes do.
# Mean |SHAP| says how much a variable MOVES the predictions; the permutation
# says how much the model's skill rests on it. The two need not agree, and
# where they do not is a reading. The values add up to the prediction less the
# reference, and that is checked, as the score is.
#
# SAGE (sage_importance()). The Shapley values of the LOSS rather than of the
# prediction (Covert, Lundberg & Lee 2020), in the kernel's game: how much of
# the model's skill each theme carries, a signal two themes share split between
# them. The values add up exactly to the loss of the mean prediction less the
# model's own, and that is checked.
#
# REFIT (refit_importance(), R/importance_refit.R). Whether a variable is
# needed at all: the final run's seeds trained again with it left out (LOCO;
# Lei et al. 2018). The only method here that retrains, and the only one that
# answers for a network that never saw the variable.
#
# ALE (ale_effect()). Not how much, but how: the prediction along each
# variable's range (Apley & Zhu 2020), each point moved only across its own bin
# of the range, so the model is asked about values near those it saw. The
# whole patch moves, keeping its texture; a categorical's effect is the patch
# made each class.
#
# ONE MASK FOR EVERYTHING. Whatever is perturbed -- a variable, a ring, a
# variable at a ring, a window -- is a mask over each window's channels and
# pixels, and a point takes the masked values from ONE donor in every window.
# The windows are cut around the same point, so ring d is the same ground in
# the 3x3 and in the 15x15; one donor keeps it the same ground after the swap.
#
# THE UNIT IS THE VARIABLE, NOT THE CHANNEL. A categorical raster arrives as
# 0/1 channels, one per class. Permuting one of them alone makes points that
# are in two classes or in none, and the drop would measure that absurdity. So
# the channels of one categorical move together, with one donor. The package is
# not told which raster a dummy came from -- prepared rasters arrive already
# one-hot -- so importance_groups() infers the sets, and says what it found:
# dummies that share a name prefix AND are never 1 together at the points. Any
# grouping of the user's own (themes: climate, relief, ...) replaces it.
#
# WHERE IT IS MEASURED. rows = "test": the final model's seeds, on the test
# set the whole run held out -- the importance of the model that draws the
# map. rows = "folds": the tuning run's models of the same configuration, each
# on its own fold's validation rows -- the importance as the run's validation
# design sees it, which is how two designs (random against spatial: near the
# profiles against far from them) are compared (Brenning 2023 measures
# importance against prediction distance; Meyer et al. 2019 showed that which
# variables help depends on the validation).
#
# EVERY MODEL IS AN ESTIMATE. A final run has N seeds and a tuning run k folds
# times its seeds; each is one equally good model, and they need not rely on
# the same variables (Fisher et al.'s point: many good models, different
# reliance). Each is scored on its own, and the table reports the mean, the
# spread between models and in how many of them the variable mattered at all.
# Every model is given the SAME permutations of the same rows -- common random
# numbers -- so two variables, or two models, differ by what they are and not
# by the draw.
#
# THE CHECK THAT STOPS. Before anything is permuted, each model's unperturbed
# predictions must reproduce, point by point, the ones its run wrote for it.
# That one comparison covers the checkpoint, the configuration, the rows, the
# scaling, the windows' order and the inverse transform: if any of them is
# wrong, the predictions differ, and every importance computed after it would
# have been measured against a model that is not the run's.

# ── the method ────────────────────────────────────────────────────────────────

#' Permutation importance: how much a fitted model relies on each variable.
#'
#' Each point takes one variable's whole patch -- every pixel, every window --
#' from another point; the model predicts again, and the loss of skill is the
#' variable's importance to that model. Nothing is retrained.
#'
#' A variable whose permutation costs nothing is one the model does not use.
#' It may still carry information other variables also carry: permutation
#' measures reliance, not necessity, and a model that can find the same thing
#' elsewhere leans on whichever it found first.
#'
#' @param draws  Permutations per variable and per model. The importance is
#'   their mean; their spread is reported beside the spread between models.
#' @param within Where a donor may come from. NULL (the default): any point
#'   of the rows scored. A number: square blocks of that side, in the
#'   coordinate units of the store's `x` and `y` -- the importance beyond the
#'   region. A column name of the point table (an ecoregion, a soil region):
#'   the same class. Or a vector with one label per point of the store. A
#'   point alone in its block or class keeps its own values; how many did is
#'   reported.
#' @param fill   "permute" (the default), or "mean": the variable is set to
#'   its training mean everywhere in the patch -- an ablation, which puts the
#'   patch off the data (see Details) and is read against the permutation.
#' @param metric What the table is ranked by: "ccc" (the drop in Lin's
#'   concordance, native units), "rmse" (the rise in RMSE, native units) or
#'   "rmse_transform" (the rise in RMSE in the space the model was trained
#'   in, e.g. log1p). All three are computed and kept.
#' @param seed   Seed of the permutations. The same draws are given to every
#'   variable and every model.
#' @return An `importance_spec`, for [dsm_importance()].
#' @details The mean of a z-scored channel is 0 and of a dummy its class
#'   frequency, so `fill = "mean"` gives every point the average landscape,
#'   flat. A large drop under it with a small one under permutation says the
#'   model reacts to the input being unusual rather than to its content.
#' @export
permutation_importance <- function(draws = 5L, within = NULL,
                                   fill = c("permute", "mean"),
                                   metric = c("ccc", "rmse", "rmse_transform"),
                                   seed = 42L) {
  fill   <- match.arg(fill)
  metric <- match.arg(metric)
  if (!is.numeric(draws) || length(draws) != 1L || !is.finite(draws) ||
      draws < 1 || draws != round(draws)) {
    stop("draws must be one whole number of 1 or more.", call. = FALSE)
  }
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed)) {
    stop("seed must be one number.", call. = FALSE)
  }
  if (!is.null(within)) {
    if (identical(fill, "mean")) {
      stop("within and fill = \"mean\" do not go together: a mean has no donor to ",
           "draw from a block. Use one or the other.", call. = FALSE)
    }
    is_block <- is.numeric(within) && length(within) == 1L
    if (is_block && (!is.finite(within) || within <= 0)) {
      stop("within as a block size must be one positive number, in the units of ",
           "the store's x and y.", call. = FALSE)
    }
    if (!is_block && !(is.atomic(within) && length(within) >= 1L)) {
      stop("within must be NULL, a block size, a column name of the point table, ",
           "or one label per point of the store.", call. = FALSE)
    }
  }
  # A MEAN HAS NO DRAWS. Five identical passes would cost five times and
  # report a spread of zero that reads as precision.
  if (identical(fill, "mean")) draws <- 1L
  structure(list(kind = "permutation", draws = as.integer(draws), within = within,
                 fill = fill, metric = metric, seed = as.integer(seed)),
            class = "importance_spec")
}

#' Print an importance method.
#'
#' @param x   An `importance_spec`, from [permutation_importance()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.importance_spec <- function(x, ...) {
  # Only a permutation, a context and a refit are scored by a metric; the
  # others carry one for the frame and rank by their own value.
  ranked <- if (x$kind %in% c("permutation", "context", "refit")) paste(" | ranked by", x$metric) else ""
  cat("<importance_spec> ", .importance_label(x), ranked, "\n", sep = "")
  invisible(x)
}

#' Context importance: how far from the point, and at which scale, the model reads.
#'
#' The patch is a neighbourhood. This permutes part of it -- the point's own
#' pixel, a ring of pixels at one distance from it, every ring but the centre,
#' or one window's whole input -- and measures the loss of skill, as
#' [permutation_importance()] does for a variable. The windows are cut around
#' the same point, so a ring is the same ground in every window, and it takes
#' its values from one donor in all of them.
#'
#' A large importance of the context says the network reads the neighbourhood.
#' It does not say the neighbourhood adds what the centre lacks: at 250 m most
#' covariates are smooth, a pixel three cells away is close to a copy of the
#' centre, and a network can lean on the rim and learn nothing the centre would
#' not have told it. Ring d holds 8d pixels, so the table gives the cost per
#' pixel beside the total: a wider ring costs more for its area alone.
#'
#' @param by "ring" (the default): one row per distance band. "window": one row
#'   per window the model reads, that window's whole input permuted -- how much
#'   the branch carries -- and, when the model has a gate between its two
#'   branches, the gate read point by point beside it (`$gate`).
#' @param bands For `by = "ring"`: the bands, a named list of ring numbers -- 0
#'   is the point's own pixel, ring d the 8d pixels d steps from it. NULL: the
#'   centre, each ring alone, and the context (every ring but the centre); with
#'   `per_variable`, the centre and the context.
#' @param per_variable For `by = "ring"`: one row per variable and band --
#'   which variables the model reads through the neighbourhood, and which at
#'   the point alone. The variables are those of `groups` in [dsm_importance()].
#' @param draws  Permutations per band (or variable and band, or window) and
#'   per model.
#' @param metric What the table is ranked by, as in [permutation_importance()].
#' @param seed   Seed of the permutations, shared by every model.
#' @return An `importance_spec`, for [dsm_importance()].
#' @export
context_importance <- function(by = c("ring", "window"), bands = NULL, per_variable = FALSE,
                               draws = 5L, metric = c("ccc", "rmse", "rmse_transform"),
                               seed = 42L) {
  by     <- match.arg(by)
  metric <- match.arg(metric)
  if (!is.numeric(draws) || length(draws) != 1L || !is.finite(draws) ||
      draws < 1 || draws != round(draws)) {
    stop("draws must be one whole number of 1 or more.", call. = FALSE)
  }
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed)) {
    stop("seed must be one number.", call. = FALSE)
  }
  if (!is.logical(per_variable) || length(per_variable) != 1L || is.na(per_variable)) {
    stop("per_variable must be TRUE or FALSE.", call. = FALSE)
  }
  if (identical(by, "window") && (!is.null(bands) || isTRUE(per_variable))) {
    stop("bands and per_variable belong to by = \"ring\"; by = \"window\" permutes each ",
         "window's whole input.", call. = FALSE)
  }
  if (!is.null(bands)) {
    nm <- names(bands)
    if (!is.list(bands) || length(bands) == 0L || is.null(nm) || any(!nzchar(nm)) ||
        anyDuplicated(nm)) {
      stop("bands must be a named list of ring numbers, one name each, e.g. ",
           "list(near = 0:1, far = 2:7).", call. = FALSE)
    }
    good <- vapply(bands, function(b) is.numeric(b) && length(b) >= 1L && all(is.finite(b)) &&
                     all(b >= 0) && all(b == round(b)), logical(1))
    if (!all(good)) {
      stop("A band is a set of ring numbers, whole and 0 or more (0 is the centre): ",
           paste(nm[!good], collapse = ", "), ".", call. = FALSE)
    }
    bands <- lapply(bands, function(b) sort(unique(as.integer(b))))
  }
  structure(list(kind = "context", by = by, bands = bands, per_variable = per_variable,
                 draws = as.integer(draws), metric = metric, seed = as.integer(seed),
                 within = NULL, fill = "permute"),
            class = "importance_spec")
}

#' SHAP importance: how each variable pushes each prediction, up or down.
#'
#' Shapley values (Lundberg & Lee 2017) of the network's predictions, estimated
#' from its gradients. Each point's prediction, less a reference prediction, is
#' shared among the variables, with a sign: this one raised it, that one
#' lowered it. Averaged in absolute value over the points they are a global
#' importance -- of how much each variable MOVES the predictions, which is not
#' how much it improves them ([permutation_importance()] measures that); kept
#' point by point (`$points`) they are the map of what drives the prediction
#' where (Padarian et al. 2020; Wadoux et al. 2023 for soil carbon).
#'
#' @param estimator "expected_gradients" (the default; Erion et al. 2021): the
#'   reference is a background of the model's own training points, drawn at
#'   random, so no baseline is chosen by hand -- the values are SHAP values of
#'   the network, and the same estimator as SHAP's GradientExplainer.
#'   "integrated_gradients" (Sundararajan et al. 2017): one baseline, the
#'   training mean of every channel -- the average landscape, flat, which no
#'   point is -- kept because its sum is exact rather than sampled.
#'   "kernel": the Shapley values of the variables themselves, computed from
#'   coalitions, with no gradient (Lundberg & Lee 2017's KernelSHAP game): the
#'   value of a set of variables is the mean prediction, over a background of
#'   training points, with the variables outside it taken from the background
#'   point -- the whole patch, the same point in every window. Exact over
#'   every coalition up to 14 variables; by sampled permutations above, up to
#'   40. Meant for a few variables -- give `groups` themes -- and a few points.
#'   Read against the gradient estimators: they share the total and split it
#'   alike when the variables do not interact, so where they part is how much
#'   the variables do.
#' @param samples For expected gradients: draws per point, each a background
#'   point and a place on the straight path to it. The noise they leave is
#'   reported.
#' @param background How many of the model's training points the references
#'   are drawn from. NULL: 200 for expected gradients, 16 for the kernel --
#'   each coalition costs one pass per background point.
#' @param steps For integrated gradients: points on the path.
#' @param permutations For the kernel above 14 variables: permutations
#'   sampled (in pairs, each with its reverse).
#' @param max_points Explain this many of the rows only, drawn at random --
#'   the same rows for every estimator, so two can be compared point by point.
#'   NULL: every row for the gradient estimators, 200 for the kernel.
#' @param seed Seed of the draws, shared by every model.
#' @return An `importance_spec`, for [dsm_importance()].
#' @details The values are in the units the network predicts in: for a log1p
#'   target, a SHAP value of +0.1 raises the predicted value by about 10%. Each
#'   point's values add up to its prediction less the background's mean
#'   prediction (expected gradients) or less the prediction at the baseline
#'   (integrated gradients); [dsm_importance()] checks that they do and stops
#'   if they do not. SHAP by deep-network rules (DeepSHAP) is not offered: it
#'   needs a propagation rule written for every layer of this architecture.
#' @export
shap_importance <- function(estimator = c("expected_gradients", "integrated_gradients", "kernel"),
                            samples = 100L, background = NULL, steps = 50L,
                            permutations = 64L, max_points = NULL, seed = 42L) {
  estimator <- match.arg(estimator)
  whole <- function(v, what, min = 1) {
    if (!is.numeric(v) || length(v) != 1L || !is.finite(v) || v < min || v != round(v)) {
      stop(what, " must be one whole number of ", min, " or more.", call. = FALSE)
    }
    as.integer(v)
  }
  is_kernel  <- identical(estimator, "kernel")
  samples    <- whole(samples, "samples")
  background <- whole(background %||% if (is_kernel) 16L else 200L, "background", 2)
  steps      <- whole(steps, "steps", 2)
  permutations <- whole(permutations, "permutations", 2)
  max_points <- max_points %||% if (is_kernel) 200L else NULL
  if (!is.null(max_points)) max_points <- whole(max_points, "max_points", 2)
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed)) {
    stop("seed must be one number.", call. = FALSE)
  }
  # draws, metric, within and fill are what the frame reads of every method; a
  # SHAP value is one pass of the model per sample, scored against nothing.
  structure(list(kind = "shap", estimator = estimator, samples = samples,
                 background = background, steps = steps, permutations = permutations,
                 max_points = max_points, seed = as.integer(seed),
                 draws = 1L, metric = "ccc", within = NULL, fill = "permute"),
            class = "importance_spec")
}

#' SAGE: how much of the model's skill each variable carries, shared fairly.
#'
#' Shapley Additive Global importancE (Covert, Lundberg & Lee 2020): the
#' Shapley values of the loss the model explains. The value of a set of
#' variables is the loss of the model's predictions when only they are known
#' -- the others taken from background points of the model's own training
#' data, the whole patch, the same point in every window, as in
#' [shap_importance()]`("kernel")`. A variable's SAGE value is the loss it
#' takes away; variables that carry the same information split it, where a
#' permutation credits the shared part to neither and a refit without one
#' gives each nothing (the other stands in). The values add up exactly to the
#' loss of the mean prediction -- no variable known -- less the model's own.
#'
#' Exact over every coalition up to 14 variables, by sampled permutations up
#' to 40: meant for a few variables -- give `groups` themes.
#'
#' @param loss "mse" (the default) or "mae", in the space the model was
#'   trained in (log1p for a log1p target).
#' @param background How many of the model's training points the absent
#'   variables are drawn from; each coalition costs one pass per point each.
#' @param max_points How many of the rows the loss is taken over, drawn by the
#'   seed -- the kernel's default, so the two read the same points at the same
#'   cost. More points, a steadier loss, and a longer run.
#' @param permutations Above 14 variables: permutations sampled, in pairs.
#' @param seed Seed of the draws, shared by every model.
#' @return An `importance_spec`, for [dsm_importance()].
#' @export
sage_importance <- function(loss = c("mse", "mae"), background = 16L, max_points = 200L,
                            permutations = 64L, seed = 42L) {
  loss <- match.arg(loss)
  whole <- function(v, what, min = 1) {
    if (!is.numeric(v) || length(v) != 1L || !is.finite(v) || v < min || v != round(v)) {
      stop(what, " must be one whole number of ", min, " or more.", call. = FALSE)
    }
    as.integer(v)
  }
  if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed)) {
    stop("seed must be one number.", call. = FALSE)
  }
  structure(list(kind = "sage", loss = loss, background = whole(background, "background", 2),
                 max_points = whole(max_points, "max_points", 2),
                 permutations = whole(permutations, "permutations", 2), seed = as.integer(seed),
                 draws = 1L, metric = "ccc", within = NULL, fill = "permute"),
            class = "importance_spec")
}

#' ALE: how the prediction changes along each variable's range.
#'
#' Accumulated local effects (Apley & Zhu 2020). A variable's range is cut
#' into bins by its quantiles at the points; within each bin, every point is
#' moved to the bin's lower edge and to its upper edge, and the difference of
#' the two predictions is the local effect. Accumulated across the bins and
#' centred, it is the curve. A point is moved only across its own bin, so the
#' model is asked about values close to the ones it saw -- which is why ALE is
#' read with correlated predictors, where a partial dependence plot would ask
#' about combinations that do not occur.
#'
#' A point's whole patch is moved, every pixel and every window by the same
#' amount: the neighbourhood keeps its texture and changes its level. Moving
#' the centre pixel alone would build a spike no landscape has. A categorical
#' -- a one-hot set, or a binary map alone -- has no range: its effect is the
#' prediction with the whole patch made each class, less the prediction as it
#' is. That is a partial dependence, which asks about a class also where it
#' does not occur; the number of points in each class says where it does.
#'
#' @param variables Which variables, named as [importance_groups()] names them
#'   (a channel, or a one-hot set). NULL: all of them. Groups of your own are
#'   not used: an effect needs one variable's values.
#' @param bins Bins of a continuous variable's range, cut at its quantiles at
#'   the points -- fewer where it has fewer distinct values.
#' @return An `importance_spec`, for [dsm_importance()]. Its importance is the
#'   spread of the curve over the points -- the standard deviation of the
#'   effect at their values; flat is no effect (Greenwell et al. 2018 for the
#'   idea, on partial dependence).
#' @export
ale_effect <- function(variables = NULL, bins = 20L) {
  if (!is.null(variables) && (!is.character(variables) || length(variables) == 0L ||
                              anyNA(variables) || anyDuplicated(variables))) {
    stop("variables must be NULL or distinct variable names, as importance_groups() ",
         "names them.", call. = FALSE)
  }
  if (!is.numeric(bins) || length(bins) != 1L || !is.finite(bins) || bins < 2 ||
      bins != round(bins)) {
    stop("bins must be one whole number of 2 or more.", call. = FALSE)
  }
  structure(list(kind = "ale", variables = variables, bins = as.integer(bins),
                 draws = 1L, metric = "ccc", within = NULL, fill = "permute"),
            class = "importance_spec")
}

# A method in a few words, for a print and for the column of a comparison.
.importance_label <- function(spec) {
  if (identical(spec$kind, "refit")) {
    return(sprintf("Refit without each variable (LOCO), %d seed(s) checked first",
                   spec$check_seeds))
  }
  if (identical(spec$kind, "sage")) {
    return(sprintf("SAGE, %s, %d background point(s), at most %d point(s)", spec$loss,
                   spec$background, spec$max_points))
  }
  if (identical(spec$kind, "ale")) {
    return(sprintf("ALE, %d bin(s)%s", spec$bins,
                   if (is.null(spec$variables)) "" else
                     sprintf(", %d variable(s)", length(spec$variables))))
  }
  if (identical(spec$kind, "shap")) {
    pts <- if (is.null(spec$max_points)) "" else sprintf(", at most %d point(s)", spec$max_points)
    return(switch(spec$estimator,
      expected_gradients = sprintf("SHAP by expected gradients, %d sample(s) over %d background point(s)%s",
                                   spec$samples, spec$background, pts),
      integrated_gradients = sprintf("SHAP by integrated gradients, %d step(s) from the training mean%s",
                                     spec$steps, pts),
      kernel = sprintf("SHAP by kernel, Shapley over the variables, %d background point(s)%s",
                       spec$background, pts)))
  }
  if (identical(spec$kind, "context")) {
    what <- if (identical(spec$by, "window")) "by window (each branch's whole input)" else
      if (isTRUE(spec$per_variable)) "by ring, per variable" else "by ring"
    return(sprintf("context %s, %d draw(s)", what, spec$draws))
  }
  if (identical(spec$fill, "mean")) return("permutation, mean fill (ablation)")
  w <- spec$within
  where <- if (is.null(w)) {
    "over all rows"
  } else if (is.numeric(w) && length(w) == 1L) {
    paste0("within blocks of ", format(w))
  } else if (is.character(w) && length(w) == 1L) {
    paste0("within ", w)
  } else {
    "within the labels given"
  }
  sprintf("permutation %s, %d draw(s)", where, spec$draws)
}

# ── which channels are one variable ───────────────────────────────────────────

#' Which channels the importance treats as one variable.
#'
#' A categorical raster arrives as several 0/1 channels, one per class.
#' Permuting one of them alone makes points that belong to two classes or to
#' none, so the channels of one categorical are one variable. The package is
#' not told which raster a dummy came from, so it infers the sets: dummies that
#' share a name prefix (cut at "_") and are never 1 together at the points.
#' Every other channel is a variable of its own. The sets found are listed;
#' check them, and give your own grouping when they are not what the rasters
#' were.
#'
#' A grouping of your own can be anything that makes sense to perturb as one:
#' the classes of a categorical named outright, or themes (climate, relief,
#' vegetation...), which answers how much the model relies on each theme as a
#' block -- the way to read many correlated variables.
#'
#' @param data   The `dsm_data` the model was fitted on, from [dsm_load()].
#' @param groups "auto" (the default), "channel" (every channel alone, dummies
#'   too -- see above for what that measures), or your grouping: a data frame
#'   with columns `channel` and `variable`, or a named list (variable =
#'   channel names). Channels it does not name keep the automatic rule.
#' @return A tibble in channel order: `channel`, `variable`, and `rule`
#'   ("alone", "one-hot set" or "yours").
#' @export
importance_groups <- function(data, groups = "auto") {
  .importance_check_data(data)
  ch <- as.character(data$store$predictors)
  out <- tibble::tibble(channel = ch, variable = ch, rule = "alone")
  if (identical(groups, "channel")) return(out)

  dummies <- ch[as.logical(data$type_table$is_dummy)]
  sets <- .importance_onehot_sets(dummies, data$points)
  # A SET NEVER TAKES A CHANNEL'S NAME. wet_inland and wet_coastal make the set
  # "wet", and a binary map called wet may stand beside them, alone: the two
  # would share one name in every table after this.
  clash <- names(sets) %in% ch
  names(sets)[clash] <- paste0(names(sets)[clash], "_classes")
  for (nm in names(sets)) {
    i <- match(sets[[nm]], ch)
    out$variable[i] <- nm
    out$rule[i] <- "one-hot set"
  }
  if (identical(groups, "auto")) return(out)
  if (is.character(groups) && length(groups) == 1L) {
    stop("groups must be \"auto\", \"channel\", a data frame with columns channel ",
         "and variable, or a named list; got \"", groups, "\".", call. = FALSE)
  }

  user <- .importance_user_groups(groups, ch)
  # A GROUP OF YOURS THAT CUTS A ONE-HOT SET IN TWO perturbs part of a
  # categorical, which is the absurd point again. Said, not refused: the
  # inference may have joined two rasters that are not one, and you may know.
  for (nm in names(sets)) {
    v <- unique(user$variable[user$channel %in% sets[[nm]]])
    named <- sum(sets[[nm]] %in% user$channel)
    if (length(v) > 1L || (named > 0L && named < length(sets[[nm]]))) {
      # SAID IN COUNTS. The first version listed every channel of the set, and
      # a soil-class set is 33 of them: a warning nobody could read. What it
      # means is how the set was cut, and which channels fell out of it.
      left <- setdiff(sets[[nm]], user$channel)
      warning(sprintf(paste0(
        "Your grouping splits the one-hot set '%s' (%d channels): %d in %d variable(s) of yours",
        "%s. Permuting part of a categorical makes points in two classes or none -- put ",
        "the whole set in one variable, unless it is not one categorical."),
        nm, length(sets[[nm]]), named, length(v),
        if (length(left)) sprintf(", %d left out (%s%s), which stay a variable of their own",
                                  length(left), paste(utils::head(left, 3L), collapse = ", "),
                                  if (length(left) > 3L) ", ..." else "") else ""),
        call. = FALSE)
    }
  }
  i <- match(user$channel, ch)
  out$variable[i] <- user$variable
  out$rule[i] <- "yours"

  # ONE NAME, ONE THING. A variable of yours named like a channel it does not
  # contain -- "clay" holding sand and silt, while the channel clay stays
  # alone -- would merge the two silently.
  mixed <- tapply(out$rule, out$variable, function(r) length(unique(r)) > 1L)
  if (any(mixed)) {
    stop("The variable name(s) ", paste(names(mixed)[mixed], collapse = ", "),
         " are both a group of yours and a channel left out of it. Give the group ",
         "another name, or put the channel in it.", call. = FALSE)
  }
  out
}

# The one-hot sets among the dummies: share a name prefix, never 1 together.
#
# BY PREFIX, THEN BY THE DATA. Names alone are not enough -- soil_class_* and
# soil_drainage_* share "soil" and are two categoricals -- and the data alone is
# not either: two rare binary maps can simply never meet at the points. So the
# dummies are split by their first name token; a group whose members are
# mutually exclusive at the points is a set, and one that is not is split again
# by the next token. soil_* is not exclusive (a point has a class AND a
# drainage), so it splits into soil_class_* and soil_drainage_*, each of which
# is. A dummy whose whole name is the prefix (wetlands beside wetlands_inland)
# is not a class of that set and stays alone.
.importance_onehot_sets <- function(dummies, points) {
  if (length(dummies) < 2L) return(list())
  tokens <- stats::setNames(strsplit(dummies, "_", fixed = TRUE), dummies)
  exclusive <- function(chs) {
    m <- as.matrix(points[, chs, drop = FALSE])
    all(rowSums(m == 1, na.rm = TRUE) <= 1L)
  }
  # The longest prefix every member extends: a set's name, and never the whole
  # name of one of its members.
  prefix_of <- function(chs) {
    tk <- tokens[chs]
    k <- 0L
    repeat {
      if (any(lengths(tk) <= k + 1L)) break
      nxt <- vapply(tk, `[`, character(1), k + 1L)
      if (length(unique(nxt)) != 1L) break
      k <- k + 1L
    }
    if (k == 0L) NA_character_ else paste(tk[[1]][seq_len(k)], collapse = "_")
  }
  out <- list()
  split_at <- function(chs, depth) {
    bare <- chs[lengths(tokens[chs]) <= depth]
    rest <- setdiff(chs, bare)
    if (length(rest) < 2L) return(invisible(NULL))
    if (exclusive(rest)) {
      nm <- prefix_of(rest)
      if (!is.na(nm)) out[[nm]] <<- rest
      return(invisible(NULL))
    }
    nxt <- vapply(tokens[rest], `[`, character(1), depth + 1L)
    for (g in split(rest, nxt)) split_at(g, depth + 1L)
    invisible(NULL)
  }
  first <- vapply(tokens, `[`, character(1), 1L)
  for (g in split(dummies, first)) split_at(g, 1L)
  out
}

# A grouping of the user's, as a table channel -> variable.
.importance_user_groups <- function(groups, ch) {
  if (is.data.frame(groups)) {
    if (!all(c("channel", "variable") %in% names(groups))) {
      stop("A grouping given as a data frame needs the columns channel and variable; ",
           "it has ", paste(names(groups), collapse = ", "), ".", call. = FALSE)
    }
    tab <- tibble::tibble(channel = as.character(groups$channel),
                          variable = as.character(groups$variable))
  } else if (is.list(groups) && !is.null(names(groups)) && all(nzchar(names(groups)))) {
    tab <- tibble::tibble(channel = as.character(unlist(groups, use.names = FALSE)),
                          variable = rep(names(groups), lengths(groups)))
  } else {
    stop("groups must be \"auto\", \"channel\", a data frame with columns channel and ",
         "variable, or a named list (variable = channel names).", call. = FALSE)
  }
  if (anyNA(tab$channel) || anyNA(tab$variable) || any(!nzchar(tab$variable))) {
    stop("The grouping has an empty channel or variable name.", call. = FALSE)
  }
  unknown <- setdiff(tab$channel, ch)
  if (length(unknown) > 0L) {
    stop("The grouping names channel(s) the store does not have: ",
         paste(utils::head(unknown, 10L), collapse = ", "),
         if (length(unknown) > 10L) sprintf(" (and %d more)", length(unknown) - 10L),
         ".", call. = FALSE)
  }
  twice <- unique(tab$channel[duplicated(tab$channel)])
  if (length(twice) > 0L) {
    stop("A channel can be in one variable only; in more than one: ",
         paste(twice, collapse = ", "), ".", call. = FALSE)
  }
  tab
}

# What the grouping amounts to, in one line.
.importance_groups_line <- function(grp) {
  vars <- unique(grp$variable)
  sets <- unique(grp$variable[grp$rule == "one-hot set"])
  mine <- unique(grp$variable[grp$rule == "yours"])
  alone <- sum(grp$rule == "alone")
  parts <- sprintf("%d channel(s) alone", alone)
  if (length(sets)) {
    n_in <- vapply(sets, function(s) sum(grp$variable == s), integer(1))
    parts <- c(parts, sprintf("%d one-hot set(s): %s", length(sets),
                              paste(sprintf("%s (%d)", sets, n_in), collapse = ", ")))
  }
  if (length(mine)) parts <- c(parts, sprintf("%d group(s) of yours", length(mine)))
  sprintf("%d variable(s): %s", length(vars), paste(parts, collapse = "; "))
}

.importance_check_data <- function(data) {
  if (!is.list(data) || is.null(data$store) || is.null(data$points) ||
      is.null(data$type_table)) {
    stop("data must be the dsm_data the model was fitted on, from dsm_load().",
         call. = FALSE)
  }
  invisible(TRUE)
}

# ── the importance of a fitted model ──────────────────────────────────────────

#' Variable importance of a fitted model, by the method you choose.
#'
#' The method is an argument, as the validation design is in [dsm_train()]:
#' [permutation_importance()] measures how much the model relies on each
#' variable, over all rows or within blocks or classes; [context_importance()]
#' how far from the point, and through which window, it reads;
#' [shap_importance()] how each variable pushes each prediction, up or down;
#' [sage_importance()] how much of the model's skill each variable carries,
#' shared fairly among variables that carry the same information;
#' [refit_importance()] how much skill a network trained without the
#' variable loses; [ale_effect()] how the prediction changes along each
#' variable's range. Each variable is a channel, or the channels of one
#' categorical (see [importance_groups()]), or a group of yours.
#'
#' Before anything is perturbed, each model's own score is computed again and
#' must reproduce the one its run wrote; if it does not, nothing is measured.
#'
#' @param final  A `dsm_final`, or the directory of a final run.
#' @param data   The `dsm_data` the run was fitted on, from [dsm_load()]. Its
#'   windows need not be loaded: one that is not is read from the store.
#' @param method An importance method: [permutation_importance()],
#'   [context_importance()], [shap_importance()], [sage_importance()],
#'   [refit_importance()] or [ale_effect()].
#' @param rows   "test" (the default): the final run's seeds, on the test set
#'   the whole run held out -- the importance of the model that draws the map.
#'   "folds": the tuning run's models of the same configuration, each on its
#'   own fold's validation rows -- the importance as that validation design
#'   sees it; run it for each design to compare them.
#' @param groups Which channels form one variable: "auto", "channel", or your
#'   grouping, as in [importance_groups()].
#' @param config Which of the final run's configurations, when it fitted more
#'   than one. NULL: the selected one.
#' @param threads torch threads for this session; NULL leaves them as set.
#' @param batch_size Points per forward pass.
#' @param verbose Say what is done, model by model.
#' @param at     Points of the map instead of rows of the store: a data frame
#'   of `x`, `y`, from [importance_points()] or of your own. Their patches are
#'   cut from the rasters as the profiles' were, and only [shap_importance()]
#'   runs there -- a map point has no observation to score against. The
#'   values become maps with [importance_map()].
#' @param rasters With `at`: where the rasters are, as in [dsm_predict()]; NULL
#'   for the store's own raster table.
#' @param seeds  Which of the final run's seeds to use; NULL for all. A map of
#'   many points may take a few: each seed is one model explained.
#' @param chunk_points With `at`: points cut and explained at a time. Their
#'   patches are held in memory together.
#' @return A `dsm_importance`: `table` (one row per target -- a variable, a
#'   band, a variable at a band or a window -- ranked), `by_model` (one row per
#'   target and model), `baseline` (each model's unperturbed score), `groups`,
#'   `targets` (what each target is), `gate` (by window, for a model with a
#'   gate: the gate point by point), `units` (the models scored) and the
#'   settings. For SHAP: `table` (mean |SHAP|, its share and direction per
#'   variable), `points` (every point's values), `patch` and `pixels` (where
#'   in the patch they sit) and `completeness` (how well they add up). For
#'   ALE: `table` (the spread, trend and extremes of each effect), `curves`
#'   (each continuous variable's curve, with its spread between models) and
#'   `classes` (each categorical's class effects). For SAGE: `table` (the loss
#'   each variable takes away, and its share), `by_model` and `losses` (each
#'   model's loss, and the mean prediction's, which SAGE splits). For a refit:
#'   `table`, `by_model`, `raw` (each unit's scores), `check` (the seeds
#'   trained again with nothing left out, against the run), `noise` (what a
#'   refit moves a score by with nothing to lose) and `refit_dir`.
#' @export
dsm_importance <- function(final, data, method = permutation_importance(),
                           rows = c("test", "folds"), groups = "auto", config = NULL,
                           threads = NULL, batch_size = 512L, verbose = TRUE,
                           at = NULL, rasters = NULL, seeds = NULL, chunk_points = 5000L) {
  rows <- match.arg(rows)
  say  <- function(...) if (verbose) message(...)
  if (!inherits(method, "importance_spec")) {
    stop("method must be an importance method, such as permutation_importance().",
         call. = FALSE)
  }
  if (!method$kind %in% c("permutation", "context", "shap", "ale", "sage", "refit")) {
    stop("Importance method '", method$kind, "' is not implemented yet.", call. = FALSE)
  }
  # A REFIT TRAINS THE FINAL RUN'S SEEDS AGAIN, as run_spec.rds recorded them.
  # On the folds it would train every fold model of the tuning run again, and
  # a tuning run does not record the schedule its units trained with: the
  # check that a refit is the run's training could not be made.
  if (identical(method$kind, "refit") && rows != "test") {
    stop("A refit trains the final run's seeds again and scores them on its test set; the ",
         "tuning run does not record the schedule its fold models trained with, so they ",
         "cannot be refitted as they were. Use rows = \"test\".", call. = FALSE)
  }
  if (!is.numeric(batch_size) || length(batch_size) != 1L || batch_size < 1) {
    stop("batch_size must be one positive whole number.", call. = FALSE)
  }
  batch_size <- as.integer(batch_size)
  .importance_check_data(data)

  fr <- .predict_final(final, config)
  ch <- as.character(data$store$predictors)
  if (!identical(as.character(fr$scaling$predictor), ch)) {
    stop("The store's channels are not the ones the final run was fitted on (", nrow(fr$scaling),
         " in its scaling, ", length(ch), " in the store, or another order). Pass the dsm_data ",
         "the run was fitted on.", call. = FALSE)
  }
  transform <- .resolve_train_transform(NULL, data, verbose = FALSE)
  clamp     <- .predict_clamp(NULL, fr$summ)
  grp  <- importance_groups(data, groups)
  vars <- lapply(split(seq_along(ch), factor(grp$variable, levels = unique(grp$variable))),
                 as.integer)
  if (!is.null(threads)) set_torch_threads(threads)
  if (!is.null(at)) {
    if (!identical(method$kind, "shap")) {
      stop("At map points there is no observation to score a perturbation against: ",
           "only shap_importance() runs there.", call. = FALSE)
    }
    return(.importance_shap_map_run(fr, data, at, grp, vars, method, transform, clamp,
                                    batch_size, say, rasters, seeds, chunk_points))
  }
  units <- .importance_units(fr, data, rows)
  units <- .importance_pick_seeds(units, seeds)
  if (identical(method$kind, "shap")) {
    return(.importance_shap_run(fr, data, units, grp, vars, method, transform, clamp,
                                batch_size, say, rows))
  }
  if (identical(method$kind, "ale")) {
    return(.importance_ale_run(fr, data, units, groups, method, transform, clamp,
                               batch_size, say, rows))
  }
  if (identical(method$kind, "sage")) {
    return(.importance_sage_run(fr, data, units, grp, vars, method, transform, clamp,
                                batch_size, say, rows))
  }
  if (identical(method$kind, "refit")) {
    return(.importance_refit_run(fr, data, units, grp, vars, method, transform, clamp,
                                 batch_size, say))
  }

  # WHAT IS PERTURBED: a variable, a band of rings, a variable at a band, or a
  # window -- each a target with its masks. Every model of a call is the same
  # configuration, so the targets are built once, from its windows.
  ws_model <- as.integer(units$cfg[[1]]$window_sizes[[1]])
  tg <- if (identical(method$kind, "context")) {
    .importance_context_targets(method, ws_model, vars)
  } else {
    list(targets = vars,
         info = tibble::tibble(target = names(vars), variable = names(vars),
                               band = NA_character_, window = NA_integer_,
                               pixels = NA_real_, n_channels = as.integer(lengths(vars))))
  }
  is_window <- identical(method$kind, "context") && identical(method$by, "window")

  n_pass <- length(tg$targets) * method$draws
  say(sprintf("Importance -- %s | %s of final run '%s' (%s)", .importance_label(method),
              if (rows == "test") "test set" else "fold validation rows",
              basename(fr$run_dir), fr$config_id))
  if (identical(method$kind, "permutation") || isTRUE(method$per_variable)) {
    say("  ", .importance_groups_line(grp))
  }
  say(sprintf("  %d model(s) x %d target(s) x %d draw(s) = %d forward pass(es) over each model's rows",
              nrow(units), length(tg$targets), method$draws, nrow(units) * n_pass))

  device <- torch::torch_device("cpu")
  baseline <- list(); raw <- list(); alone <- list(); gate <- list()
  cache <- NULL; cache_key <- NULL; donors <- NULL; fill <- NULL
  t0 <- Sys.time()
  for (u in seq_len(nrow(units))) {
    un <- units[u, , drop = FALSE]
    rows_u <- un$rows[[1]]
    # ONE CACHE PER ROW SET. The seeds of a final run share their rows and
    # their scaling, and a fold's seeds share theirs: the patches are cut and
    # scaled once for all of them, and the draws are made once, which is what
    # gives every model the same permutations.
    if (!identical(cache_key, un$cache_key)) {
      cache <- NULL; invisible(gc(verbose = FALSE))
      ws_all <- sort(unique(unlist(un$cfg[[1]]$window_sizes)))
      idx <- stats::setNames(list(rows_u), un$role)
      cache <- build_fold_cache(data$store, data$points, data$type_table, idx, ws_all,
                                scaling = un$scaling[[1]], verbose = FALSE)$cache[[un$role]]
      cache_key <- un$cache_key
      strata <- .importance_strata(method$within, data, rows_u)
      if (identical(method$fill, "permute")) {
        donors <- lapply(seq_len(method$draws), function(d) {
          .importance_donors(length(rows_u), strata, method$seed + d - 1L)
        })
        fill <- NULL
      } else {
        donors <- NULL
        fill <- .importance_fill_values(data, un$scaling[[1]], un$train_rows[[1]])
      }
      alone[[un$cache_key]] <- if (is.null(strata)) 0 else .importance_alone_share(strata)
    }
    cfg  <- un$cfg[[1]]
    keys <- patch_window_key(cfg$window_sizes[[1]])
    inputs <- lapply(keys, function(k) cache[[k]])
    meta_u <- data$store$meta[rows_u, , drop = FALSE]

    model <- build_cnn_from_config(cfg, data$store$n_channels)
    model$load_state_dict(torch::torch_load(un$model_file))
    model$to(device = device)

    res <- .importance_unit(model, inputs, as.numeric(meta_u$target_native),
                            as.numeric(meta_u$target_transform), tg$targets, donors = donors,
                            fill = fill, transform = transform, clamp = clamp,
                            batch_size = batch_size)
    .importance_check_baseline(res$baseline, un, pred_t = res$pred_t,
                               sample_id = meta_u$sample_id,
                               obs = as.numeric(meta_u$target_native))
    baseline[[u]] <- dplyr::mutate(res$baseline, unit = un$unit, .before = 1)
    raw[[u]]      <- dplyr::mutate(res$raw, unit = un$unit, .before = 1)
    if (is_window) {
      gp <- .importance_gate_points(model, inputs, batch_size)
      if (!is.null(gp)) {
        gate[[u]] <- dplyr::mutate(gp, unit = un$unit, sample_id = meta_u$sample_id, .before = 1)
      }
    }
    rm(model); invisible(gc(verbose = FALSE))

    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    say(sprintf("  [%d/%d] %s: CCC %.4f, as its run wrote | %.1f min%s",
                u, nrow(units), un$unit, res$baseline$ccc, el,
                .importance_eta(el / u * (nrow(units) - u))))
  }
  rm(cache); invisible(gc(verbose = FALSE))

  baseline <- dplyr::bind_rows(baseline)
  raw      <- dplyr::bind_rows(raw)
  agg <- .importance_aggregate(raw, baseline, method$metric)
  # The aggregation keys on the target; what the target IS -- its variable,
  # band, window and how many pixels and channels it moved -- joins it here.
  tab <- dplyr::left_join(dplyr::rename(agg$table, target = "variable"), tg$info, by = "target")
  if (identical(method$kind, "context") && identical(method$by, "ring")) {
    tab$per_pixel <- tab$importance / tab$pixels
  }
  tab <- dplyr::relocate(tab, "rank", "target", "variable", "band", "window", "pixels",
                         "n_channels")
  by_model <- dplyr::left_join(dplyr::rename(agg$by_model, target = "variable"),
                               tg$info[, c("target", "variable", "band", "window")],
                               by = "target")

  out <- structure(list(
    table = tab, by_model = by_model, baseline = baseline,
    raw = dplyr::rename(raw, target = "variable"),
    groups = grp, targets = tg$info, method = method, rows = rows,
    gate = if (length(gate)) dplyr::bind_rows(gate) else NULL,
    units = units[, c("unit", "kind", "fold", "seed", "role", "n_rows", "model_file")],
    run_dir = fr$run_dir, config_id = fr$config_id, window_sizes = ws_model,
    rows_alone_share = unlist(alone), label = NULL),
    class = "dsm_importance")
  out$label <- .importance_object_label(out)
  share <- mean(out$rows_alone_share)
  if (is.finite(share) && share > 0.2) {
    warning(sprintf(paste0("%.0f%% of the rows were alone in their block or class: they kept ",
                           "their own values, and their share of each importance is zero. ",
                           "Larger blocks, or fewer classes, give them donors."), 100 * share),
            call. = FALSE)
  }
  out
}

# What is left, in words: nothing once done, and never "about 0 min".
.importance_eta <- function(left_min) {
  if (left_min <= 0) return("")
  if (left_min < 1) return(", under a minute to go")
  sprintf(", about %.0f min to go", left_min)
}

# The models a call scores: the final run's seeds on its test set, or the
# tuning run's fold models of the same configuration on their validation rows.
.importance_units <- function(fr, data, rows) {
  n_store <- nrow(data$store$meta)
  if (rows == "test") {
    spec_path <- file.path(fr$run_dir, "run_spec.rds")
    if (!file.exists(spec_path)) {
      stop("The final run records no split (", spec_path, "): its test rows are unknown.",
           call. = FALSE)
    }
    split <- readRDS(spec_path)$split
    test  <- as.integer(split$test)
    if (length(test) == 0L) {
      stop("The final run has no test set. Use rows = \"folds\": the tuning run's models ",
           "on their validation rows.", call. = FALSE)
    }
    if (max(test) > n_store || max(split$train) > n_store) {
      stop("The final run's rows go past the store's ", n_store, " points: this is not ",
           "the store it was fitted on.", call. = FALSE)
    }
    perf <- file.path(fr$run_dir, fr$config_id, "metrics", sprintf("seed%04d_perf.csv", fr$seeds))
    return(tibble::tibble(
      unit = sprintf("seed%04d", fr$seeds), kind = "final seed", fold = NA_integer_,
      seed = fr$seeds, role = "test", rows = rep(list(test), length(fr$seeds)),
      train_rows = rep(list(as.integer(split$train)), length(fr$seeds)),
      scaling = rep(list(fr$scaling), length(fr$seeds)),
      cfg = rep(list(fr$cfg), length(fr$seeds)), model_file = fr$model_files,
      perf_file = perf,
      pred_file = file.path(fr$run_dir, fr$config_id, "predictions",
                            sprintf("seed%04d_pred_all.csv", fr$seeds)),
      cache_key = "test", n_rows = length(test)))
  }

  td <- fr$tuning_dir
  if (is.null(td) || !dir.exists(td)) {
    stop("rows = \"folds\" needs the tuning run the final run was chosen from, and it is ",
         "not found", if (!is.null(td)) paste0(" (", td, ")") else "", ".", call. = FALSE)
  }
  plan <- readRDS(file.path(td, "fold_plan.rds"))
  grid <- readRDS(file.path(td, "tune_grid.rds"))
  cfg  <- grid[grid$config_id == fr$config_id, , drop = FALSE]
  if (nrow(cfg) != 1L) {
    stop("Configuration ", fr$config_id, " is not in the tuning run's grid (", td, ").",
         call. = FALSE)
  }
  pat <- paste0("^", fr$config_id, "_f([0-9]+)_s([0-9]+)_best\\.pt$")
  files <- list.files(file.path(td, "models"), pattern = pat)
  if (length(files) == 0L) {
    stop("The tuning run kept no model of ", fr$config_id, " (", file.path(td, "models"), ").",
         call. = FALSE)
  }
  fold <- as.integer(sub(pat, "\\1", files))
  seed <- as.integer(sub(pat, "\\2", files))
  o <- order(fold, seed)
  fold <- fold[o]; seed <- seed[o]; files <- files[o]
  if (max(fold) > length(plan$folds)) {
    stop("A model of fold ", max(fold), " in a plan of ", length(plan$folds), " fold(s).",
         call. = FALSE)
  }
  scal <- lapply(seq_along(plan$folds), function(k) {
    if (!k %in% fold) return(NULL)
    s <- fit_scaling(data$points, data$type_table, plan$folds[[k]]$train)
    if (any(s$degenerate)) {
      stop("Degenerate scaling on fold ", k, "'s training rows -- this is not the store ",
           "the tuning run was fitted on.", call. = FALSE)
    }
    s
  })
  tibble::tibble(
    unit = sprintf("f%02d_s%d", fold, seed), kind = "fold model", fold = fold, seed = seed,
    role = "validation", rows = lapply(fold, function(k) as.integer(plan$folds[[k]]$validation)),
    train_rows = lapply(fold, function(k) as.integer(plan$folds[[k]]$train)),
    scaling = scal[fold], cfg = rep(list(cfg), length(fold)),
    model_file = file.path(td, "models", files),
    perf_file = file.path(td, "metrics", sprintf("%s_f%d_s%d_perf.csv", fr$config_id, fold, seed)),
    pred_file = file.path(td, "predictions", sprintf("%s_f%d_s%d_pred_all.csv", fr$config_id,
                                                     fold, seed)),
    cache_key = sprintf("fold_%02d", fold),
    n_rows = vapply(fold, function(k) length(plan$folds[[k]]$validation), integer(1)))
}

# The model, unperturbed, against what its run wrote for it.
#
# THE PREDICTIONS, POINT BY POINT, NOT ONLY THE SCORE. The first check compared
# the CCC alone. It did refuse the model the SHAP run once took in training
# mode (0.0017 against -0.0141 written; 2026-10-03), but a summary of a model
# that predicts almost one value -- the test fixture's untrained seeds -- moves
# little when the model is another, and says nothing of where. Every
# prediction must be the run's, to 1e-4 relative (float differences between
# thread counts are ~1e-7; T1), and the observations too -- a store with other
# targets predicts the same and scores otherwise. The CCC stays the check for a
# run that wrote no predictions.
.importance_check_baseline <- function(baseline, un, pred_t = NULL, sample_id = NULL,
                                       obs = NULL) {
  pf <- un$pred_file %||% NA_character_
  if (!is.null(pred_t) && !is.na(pf) && file.exists(pf)) {
    pa <- safe_read_csv2(pf)
    pa <- pa[pa$dataset_role == un$role, , drop = FALSE]
    at <- match(as.character(sample_id), as.character(pa$sample_id))
    if (anyNA(at) || !all(c("pred_transform", "obs") %in% names(pa))) {
      stop("Model ", un$unit, " does not reproduce its run: ", sum(is.na(at)), " of its ",
           length(at), " row(s) are not among the '", un$role, "' predictions it wrote (",
           pf, "). The rows or the store are not the run's.", call. = FALSE)
    }
    dev_p <- max(abs(pred_t - pa$pred_transform[at]) / pmax(1, abs(pa$pred_transform[at])))
    dev_o <- max(abs(obs - pa$obs[at]) / pmax(1, abs(pa$obs[at])))
    if (!is.finite(dev_p) || dev_p > 1e-4 || !is.finite(dev_o) || dev_o > 1e-6) {
      stop(sprintf(paste0("Model %s does not reproduce its run's own %s: they differ by %.2g ",
                          "(relative) at its %d point(s), against %s. The checkpoint, the store, ",
                          "the rows, the scaling or the transform is not the run's, and every ",
                          "importance would be measured against another model."),
                   un$unit, if (!is.finite(dev_o) || dev_o > 1e-6) "observations" else "predictions",
                   if (!is.finite(dev_o) || dev_o > 1e-6) dev_o else dev_p, length(at), pf),
           call. = FALSE)
    }
    return(invisible(dev_p))
  }
  if (!file.exists(un$perf_file)) {
    stop("Cannot check model ", un$unit, " against its run: ", un$perf_file, " is missing. ",
         "Nothing is measured against a model whose score cannot be reproduced.",
         call. = FALSE)
  }
  perf <- safe_read_csv2(un$perf_file)
  rec  <- perf[perf$dataset_role == un$role, , drop = FALSE]
  if (nrow(rec) != 1L) {
    stop("The run wrote no single '", un$role, "' score for model ", un$unit, " (",
         un$perf_file, ").", call. = FALSE)
  }
  same_n   <- as.integer(rec$n) == as.integer(baseline$n)
  same_ccc <- isTRUE(abs(as.numeric(rec$ccc) - baseline$ccc) < 1e-4)
  if (!same_n || !same_ccc) {
    stop(sprintf(paste0("Model %s does not reproduce its run's own score: CCC %.6f on %d ",
                        "point(s) here, %.6f on %d in %s. The checkpoint, the store, the rows, ",
                        "the scaling or the transform is not the run's, and every importance ",
                        "would be measured against another model."),
                 un$unit, baseline$ccc, as.integer(baseline$n), as.numeric(rec$ccc),
                 as.integer(rec$n), un$perf_file), call. = FALSE)
  }
  invisible(TRUE)
}

# ── the work, for one model ───────────────────────────────────────────────────

# One model, its rows already cut and scaled: the score unperturbed, then once
# per target and draw. Kept apart from the files so the tests can hand it a
# model whose answer is known.
#
# inputs   the model's windows, in its order: tensors [n, channels, w, w]
# targets  named list, one element per thing perturbed: channel positions (a
#          variable), or list(channels =, rings =, windows =) -- see
#          .importance_target_masks(); the names become the `variable` column
# donors   one vector of row positions per draw (the permutations), or NULL
# fill     one value per channel (fill = "mean"), or NULL
.importance_unit <- function(model, inputs, obs_native, obs_transform, targets,
                             donors = NULL, fill = NULL, transform = identity,
                             clamp = c(0, Inf), batch_size = 512L) {
  if (is.null(donors) == is.null(fill)) {
    stop("Give donors (a permutation) or fill (a mean), one of the two.", call. = FALSE)
  }
  n_ch <- as.integer(inputs[[1]]$shape[2])
  n    <- as.integer(inputs[[1]]$shape[1])
  ws   <- vapply(inputs, function(t) as.integer(t$shape[3]), integer(1))
  if (length(obs_native) != n || length(obs_transform) != n) {
    stop("The rows scored and the observations differ in number.", call. = FALSE)
  }
  model$eval()
  base_t <- .importance_forward(model, inputs, batch_size)
  base   <- .importance_scores(base_t, obs_native, obs_transform, transform, clamp)

  draws <- if (is.null(donors)) 1L else length(donors)
  out <- vector("list", length(targets) * draws)
  k <- 0L
  for (v in seq_along(targets)) {
    masks <- .importance_target_masks(targets[[v]], n_ch, ws)
    for (d in seq_len(draws)) {
      pt <- .importance_forward(model, inputs, batch_size, mask = masks,
                                donor = if (is.null(donors)) NULL else donors[[d]],
                                fill = fill)
      sc <- .importance_scores(pt, obs_native, obs_transform, transform, clamp)
      k <- k + 1L
      out[[k]] <- tibble::tibble(variable = names(targets)[v], draw = d,
                                 ccc = sc$ccc, rmse = sc$rmse,
                                 rmse_transform = sc$rmse_transform)
    }
  }
  list(baseline = tibble::as_tibble(base), raw = dplyr::bind_rows(out), pred_t = base_t)
}

# A target's masks: one per window, over its channels and pixels, or NULL for
# a window the target leaves alone.
#
#   channels  positions perturbed (NULL: all of them)
#   rings     ring numbers perturbed (NULL: every pixel). Ring d of window w is
#             the same ground in every window, and a window without ring d is
#             left alone -- the 3x3 has no ring 2.
#   windows   positions of the windows perturbed (NULL: all of them)
.importance_target_masks <- function(target, n_ch, ws) {
  if (is.numeric(target)) target <- list(channels = target)
  ch <- if (is.null(target$channels)) seq_len(n_ch) else target$channels
  ch_mask <- torch::torch_tensor(seq_len(n_ch) %in% ch,
                                 dtype = torch::torch_bool())$view(c(1L, n_ch, 1L, 1L))
  lapply(seq_along(ws), function(k) {
    if (!is.null(target$windows) && !(k %in% target$windows)) return(NULL)
    if (is.null(target$rings)) return(ch_mask)
    ring <- patch_ring_index(ws[k])
    # Built as occlusion builds its masks: an R matrix in the ring's own layout,
    # made a tensor whole, so no linear index is walked in two orders.
    px <- matrix(as.vector(ring) %in% target$rings, nrow = nrow(ring))
    if (!any(px)) return(NULL)
    px_t <- torch::torch_tensor(px, dtype = torch::torch_bool())$view(c(1L, 1L, ws[k], ws[k]))
    torch::torch_logical_and(ch_mask, px_t)
  })
}

# The model's predictions, in the space it was trained in, batch by batch.
#
# PERMUTED IN EVERY WINDOW WITH THE SAME DONOR. A point's windows are concentric
# patches around one place; a variable taken from point j in the 3x3 window and
# from point k in the 15x15 would hand the network two places at once, and the
# drop would measure that. One donor vector serves every window.
#
# Batches keep every temporary small: the whole test set's 15x15 window, copied
# once per variable and draw, would be blocks of hundreds of MB that mimalloc
# never gives back (T6).
#
# `mask` is NULL (nothing perturbed), one mask for every window, or a list of
# one per window, NULL where that window is left alone.
#
# `shift` (ALE) is list(channel =, values =): every pixel of that channel, in
# every window, moved by the point's own value -- the whole patch raised or
# lowered, its texture kept.
.importance_forward <- function(model, inputs, batch_size, mask = NULL, donor = NULL,
                                 fill = NULL, shift = NULL) {
  # EVALUATION MODE, HERE, FOR EVERY CALLER. A module is built in training mode:
  # dropout on, batch norm on each batch's own statistics. The SHAP run, once
  # reorganised, took the model's reference predictions before anything had
  # called eval() -- the check refused it, 4.4x off the run's own (2026-10-03).
  # A forward that is only ever for importance sets the mode itself.
  model$eval()
  n <- as.integer(inputs[[1]]$shape[1])
  out <- numeric(n)
  masks <- if (is.null(mask)) NULL else if (is.list(mask)) mask else rep(list(mask), length(inputs))
  fill_t <- if (is.null(fill)) NULL else
    torch::torch_tensor(as.numeric(fill), dtype = torch::torch_float())$view(c(1L, length(fill), 1L, 1L))
  shift_m <- if (is.null(shift)) NULL else {
    n_ch <- as.integer(inputs[[1]]$shape[2])
    torch::torch_tensor(as.numeric(seq_len(n_ch) == shift$channel),
                        dtype = torch::torch_float())$view(c(1L, n_ch, 1L, 1L))
  }
  torch::with_no_grad({
    for (s in seq.int(1L, n, by = batch_size)) {
      e <- min(n, s + batch_size - 1L)
      di <- if (is.null(donor)) NULL else
        torch::torch_tensor(as.integer(donor[s:e]), dtype = torch::torch_long())
      sv <- if (is.null(shift)) NULL else
        torch::torch_tensor(as.numeric(shift$values[s:e]), dtype = torch::torch_float())$view(c(-1L, 1L, 1L, 1L))
      xb <- lapply(seq_along(inputs), function(i) {
        t <- inputs[[i]]
        b <- t[s:e, , , , drop = FALSE]
        if (!is.null(sv)) return(b + shift_m * sv)
        m <- if (is.null(masks)) NULL else masks[[i]]
        if (is.null(m)) return(b)
        if (!is.null(di)) return(torch::torch_where(m, torch::torch_index_select(t, 1L, di), b))
        torch::torch_where(m, fill_t, b)
      })
      out[s:e] <- as.numeric(do.call(model, unname(xb))$to(device = "cpu"))
    }
  })
  out
}

# The three scores, computed as the run computed its own: the inverse
# transform, the clamp, then Lin's CCC and the RMSE in native units; and the
# RMSE in the transform's space, the one the model was trained in.
.importance_scores <- function(pred_t, obs_native, obs_transform, transform, clamp) {
  pn <- transform(pred_t)
  if (is.finite(clamp[1])) pn <- pmax(pn, clamp[1])
  if (is.finite(clamp[2])) pn <- pmin(pn, clamp[2])
  k  <- is.finite(pn) & is.finite(obs_native)
  kt <- is.finite(pred_t) & is.finite(obs_transform)
  list(n = sum(k), ccc = ccc(obs_native, pn),
       rmse = sqrt(mean((pn[k] - obs_native[k])^2)),
       rmse_transform = sqrt(mean((pred_t[kt] - obs_transform[kt])^2)))
}

# ── context: the bands, the windows, the gate ────────────────────────────────

# The targets of a context_importance(), for a model reading windows `ws`, and
# what each one is: its band, window, and the ground it covers in pixels.
.importance_context_targets <- function(spec, ws, vars) {
  max_ring <- (max(ws) - 1L) %/% 2L
  if (identical(spec$by, "window")) {
    keys <- patch_window_key(ws)
    tg <- stats::setNames(lapply(seq_along(ws), function(k) list(windows = k)), keys)
    info <- tibble::tibble(target = keys, variable = NA_character_, band = NA_character_,
                           window = as.integer(ws), pixels = as.numeric(ws)^2,
                           n_channels = NA_integer_)
    return(list(targets = tg, info = info))
  }
  bands <- spec$bands
  if (is.null(bands)) {
    rings <- seq_len(max_ring)
    bands <- if (isTRUE(spec$per_variable)) {
      list(centre = 0L, context = rings)
    } else {
      c(list(centre = 0L), stats::setNames(as.list(rings), sprintf("ring_%02d", rings)),
        list(context = rings))
    }
  }
  # A band no window reaches would be permuted in no window, cost nothing, and
  # read as a finding: "the far ring does not matter".
  beyond <- vapply(bands, function(b) all(b > max_ring), logical(1))
  if (any(beyond)) {
    stop("Band(s) ", paste(names(bands)[beyond], collapse = ", "), " lie beyond the largest ",
         "window this model reads (", max(ws), "x", max(ws), ", rings 0 to ", max_ring, ").",
         call. = FALSE)
  }
  # The ground a band covers: 1 pixel for the centre, 8d for ring d.
  pixels <- vapply(bands, function(b) {
    b <- b[b <= max_ring]
    sum(ifelse(b == 0L, 1, 8 * b))
  }, numeric(1))
  if (!isTRUE(spec$per_variable)) {
    tg <- lapply(bands, function(b) list(rings = b))
    info <- tibble::tibble(target = names(bands), variable = NA_character_, band = names(bands),
                           window = NA_integer_, pixels = unname(pixels), n_channels = NA_integer_)
    return(list(targets = tg, info = info))
  }
  grid <- expand.grid(v = names(vars), b = names(bands), stringsAsFactors = FALSE)
  nm <- paste(grid$v, grid$b, sep = " | ")
  tg <- stats::setNames(lapply(seq_len(nrow(grid)), function(i) {
    list(channels = vars[[grid$v[i]]], rings = bands[[grid$b[i]]])
  }), nm)
  info <- tibble::tibble(target = nm, variable = grid$v, band = grid$b, window = NA_integer_,
                         pixels = unname(pixels[grid$b]),
                         n_channels = as.integer(lengths(vars)[grid$v]))
  list(targets = tg, info = info)
}

# The gate between a model's two branches, point by point: its weight on the
# first window's branch, and each branch's embedding norm. NULL for a model
# with one branch, or two concatenated without a gate.
#
# WHAT THE GATE IS NOT. The fused embedding is gate * f1 + (1 - gate) * f2, so
# a gate of 0.6 weighs the first branch's embedding by 0.6 -- and that
# embedding may be half as long as the other's, and the head may read the two
# through weights of any size. The gate is how the model mixes, not how much
# each scale carries; by = "window" measures the second beside it.
.importance_gate_points <- function(model, inputs, batch_size) {
  if (!identical(as.integer(model$n_branches), 2L) ||
      identical(model$gate_type, "no_gate_concat")) {
    return(NULL)
  }
  n <- as.integer(inputs[[1]]$shape[1])
  g <- n1 <- n2 <- numeric(n)
  model$eval()
  torch::with_no_grad({
    for (s in seq.int(1L, n, by = batch_size)) {
      e <- min(n, s + batch_size - 1L)
      o <- do.call(model$forward_with_internals,
                   unname(lapply(inputs, function(t) t[s:e, , , , drop = FALSE])))
      g[s:e]  <- as.numeric(o$gate$mean(dim = 2L)$to(device = "cpu"))
      n1[s:e] <- as.numeric(o$f1$norm(dim = 2L)$to(device = "cpu"))
      n2[s:e] <- as.numeric(o$f2$norm(dim = 2L)$to(device = "cpu"))
    }
  })
  tibble::tibble(gate = g, norm_1 = n1, norm_2 = n2)
}

# ── SHAP: the gradients, the sums, the check ─────────────────────────────────

# shap_importance() through every model of the call: the same units, caches and
# check as the permutations, then the attributions instead of the draws.
#
# THE BATCH IS SMALL ON PURPOSE. A backward pass holds the network's activations
# for every point of the batch, on top of the inputs, the references, the
# gradients and the running sums -- five tensors of the batch's size per
# window. At 512 points a 15x15 window of 174 channels is ~80 MB a tensor, and
# blocks of that order are the ones mimalloc keeps (T6); 128 keeps each ~20 MB.
.importance_shap_run <- function(fr, data, units, grp, vars, method, transform, clamp,
                                 batch_size, say, rows) {
  ch   <- as.character(data$store$predictors)
  n_ch <- length(ch)
  G <- matrix(0, n_ch, length(vars), dimnames = list(ch, names(vars)))
  for (v in seq_along(vars)) G[vars[[v]], v] <- 1
  ws_model <- as.integer(units$cfg[[1]]$window_sizes[[1]])
  keys <- patch_window_key(ws_model)
  b_size <- min(batch_size, 128L)
  est    <- method$estimator
  is_eg  <- identical(est, "expected_gradients")
  needs_bg <- est %in% c("expected_gradients", "kernel")
  K <- switch(est, expected_gradients = method$samples, integrated_gradients = method$steps,
              kernel = NA_integer_)
  if (identical(est, "kernel")) .importance_kernel_size(ncol(G))

  say(sprintf("Importance -- %s | %s of final run '%s' (%s)", .importance_label(method),
              if (rows == "test") "test set" else "fold validation rows",
              basename(fr$run_dir), fr$config_id))
  say("  ", .importance_groups_line(grp))
  say("  ", .importance_shap_cost_line(method, ncol(G), nrow(units)))

  device <- torch::torch_device("cpu")
  baseline <- list(); per_model <- list(); comp <- list()
  phi_of <- list(); fx_of <- list(); ref_of <- list(); ring_of <- list(); pix_of <- list()
  sel_of <- list()
  cache <- NULL; cache_key <- NULL; bg <- NULL; draws <- NULL; fill <- NULL; sel <- NULL
  t0 <- Sys.time()
  for (u in seq_len(nrow(units))) {
    un <- units[u, , drop = FALSE]
    rows_u <- un$rows[[1]]
    # ONE CACHE, ONE BACKGROUND, ONE SET OF ROWS AND ONE SET OF DRAWS PER ROW
    # SET: every seed of a final run gets the same references, the same places
    # on the path, and explains the same points.
    if (!identical(cache_key, un$cache_key)) {
      cache <- NULL; bg <- NULL; invisible(gc(verbose = FALSE))
      ws_all <- sort(unique(unlist(un$cfg[[1]]$window_sizes)))
      idx <- stats::setNames(list(rows_u), un$role)
      cache <- build_fold_cache(data$store, data$points, data$type_table, idx, ws_all,
                                scaling = un$scaling[[1]], verbose = FALSE)$cache[[un$role]]
      cache_key <- un$cache_key
      train <- un$train_rows[[1]]
      # THE ROWS EXPLAINED: all, or max_points of them drawn by the seed alone,
      # so two estimators with one seed explain the same points.
      n_all <- length(rows_u)
      sel <- if (is.null(method$max_points) || method$max_points >= n_all) seq_len(n_all) else
        with_local_seed(method$seed + 31L, sort(sample.int(n_all, method$max_points)))
      if (needs_bg) {
        # THE MODEL'S OWN TRAINING POINTS are the background: the reference a
        # SHAP value is measured from is the distribution the model learned on,
        # never the rows being explained.
        bg_rows <- with_local_seed(method$seed + 7919L, {
          if (length(train) > method$background) sort(sample(train, method$background)) else train
        })
        bg <- build_fold_cache(data$store, data$points, data$type_table,
                               list(background = bg_rows), ws_all,
                               scaling = un$scaling[[1]], verbose = FALSE)$cache$background
      }
      n <- length(sel)
      draws <- if (is_eg) with_local_seed(method$seed, list(
        j = matrix(sample.int(length(bg_rows), n * K, replace = TRUE), n, K),
        a = matrix(stats::runif(n * K), n, K))) else NULL
      fill <- if (identical(est, "integrated_gradients")) {
        .importance_fill_values(data, un$scaling[[1]], train)
      } else NULL
    }
    cfg <- un$cfg[[1]]
    inputs <- lapply(keys, function(k) cache[[k]])
    meta_u <- data$store$meta[rows_u, , drop = FALSE]

    model <- build_cnn_from_config(cfg, data$store$n_channels)
    model$load_state_dict(torch::torch_load(un$model_file))
    model$to(device = device)

    # THE MODEL'S OWN SCORE, ON EVERY ROW, before any is explained: the check
    # is against what the run wrote, which is for the whole set.
    f_full <- .importance_forward(model, inputs, batch_size)
    base <- tibble::as_tibble(.importance_scores(
      f_full, as.numeric(meta_u$target_native), as.numeric(meta_u$target_transform),
      transform, clamp))
    .importance_check_baseline(base, un, pred_t = f_full, sample_id = meta_u$sample_id,
                               obs = as.numeric(meta_u$target_native))
    sub <- if (length(sel) == n_all) inputs else {
      st <- torch::torch_tensor(as.integer(sel), dtype = torch::torch_long())
      lapply(inputs, function(t) torch::torch_index_select(t, 1L, st))
    }
    res <- .importance_attribute(model, sub, G, method, K = K,
                                 bg_inputs = if (needs_bg) lapply(keys, function(k) bg[[k]]) else NULL,
                                 draws = draws, fill = fill, transform = transform, clamp = clamp,
                                 batch_size = b_size)
    comp[[u]] <- .importance_check_completeness(res, est, un$unit)
    baseline[[u]] <- dplyr::mutate(base, unit = un$unit, .before = 1)
    per_model[[u]] <- tibble::tibble(unit = un$unit, variable = names(vars),
                                     importance = colMeans(abs(res$phi)),
                                     mean_signed = colMeans(res$phi))
    phi_of[[u]] <- res$phi; fx_of[[u]] <- res$f_x; ref_of[[u]] <- res$f_x - res$delta
    ring_of[[u]] <- res$ring_abs; pix_of[[u]] <- res$pix_abs; sel_of[[u]] <- sel
    rm(model, sub); invisible(gc(verbose = FALSE))

    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    say(sprintf("  [%d/%d] %s: CCC %.4f, as its run wrote | values add up%s | %.1f min%s",
                u, nrow(units), un$unit, base$ccc,
                if (is_eg) sprintf(" (sampling noise %.1f%%)", 100 * comp[[u]]$rel_noise) else
                  sprintf(" (to %.2g%%)", 100 * comp[[u]]$rel_noise),
                el, .importance_eta(el / u * (nrow(units) - u))))
  }
  rm(cache, bg); invisible(gc(verbose = FALSE))

  # ── points: the models of one row set averaged, the row sets one after another.
  # The mean of the seeds' values is the value of their mean prediction, SHAP
  # being additive in the model.
  sets <- split(seq_len(nrow(units)), factor(units$cache_key, levels = unique(units$cache_key)))
  pts <- lapply(sets, function(us) {
    r <- units$rows[[us[1]]][sel_of[[us[1]]]]
    phi <- Reduce(`+`, phi_of[us]) / length(us)
    colnames(phi) <- names(vars)
    dplyr::bind_cols(
      tibble::tibble(sample_id = data$store$meta$sample_id[r],
                     x = data$store$meta$x[r], y = data$store$meta$y[r],
                     fold = units$fold[us[1]],
                     prediction = Reduce(`+`, fx_of[us]) / length(us),
                     reference = Reduce(`+`, ref_of[us]) / length(us)),
      tibble::as_tibble(phi, .name_repair = "minimal"))
  })
  points <- dplyr::bind_rows(pts)
  all_rows <- unlist(lapply(sets, function(us) units$rows[[us[1]]][sel_of[[us[1]]]]),
                     use.names = FALSE)
  # From the matrices, not from `points`: a variable named like one of its
  # columns (x, y) would be read as the coordinate there.
  phi_all <- do.call(rbind, lapply(sets, function(us) Reduce(`+`, phi_of[us]) / length(us)))
  colnames(phi_all) <- names(vars)
  # The value at each point explained, for every one-channel variable -- as at
  # map points: for the direction below and for plots of SHAP against value.
  single <- names(vars)[lengths(vars) == 1L]
  values <- matrix(vapply(single, function(v) as.numeric(data$points[[ch[vars[[v]]]]][all_rows]),
                          numeric(length(all_rows))),
                   nrow = length(all_rows), dimnames = list(NULL, single))

  # DIRECTION: does a higher value of the variable at the point raise the
  # prediction? Spearman between the value and the SHAP value, over the points.
  # A group of channels has no one value, so no direction.
  direction <- vapply(names(vars), function(v) {
    if (length(vars[[v]]) != 1L) return(NA_real_)
    val <- data$points[[ch[vars[[v]]]]][all_rows]
    s <- phi_all[, v]
    sv <- stats::sd(val, na.rm = TRUE)
    if (!is.finite(sv) || sv == 0 || !is.finite(stats::sd(s)) || stats::sd(s) == 0) {
      return(NA_real_)
    }
    suppressWarnings(stats::cor(val, s, method = "spearman", use = "complete.obs"))
  }, numeric(1))

  by_model <- dplyr::bind_rows(per_model)
  tab <- by_model %>%
    dplyr::rename(imp_model = "importance") %>%
    dplyr::group_by(.data$variable) %>%
    dplyr::summarise(importance = mean(.data$imp_model),
                     sd_models = if (dplyr::n() > 1L) stats::sd(.data$imp_model) else NA_real_,
                     mean_signed = mean(.data$mean_signed),
                     n_models = dplyr::n(), .groups = "drop")
  tab$share <- tab$importance / sum(tab$importance)
  tab$direction <- unname(direction[tab$variable])
  tab$n_channels <- as.integer(lengths(vars)[tab$variable])
  tab <- tab[order(-tab$importance, tab$variable), , drop = FALSE]
  tab$rank <- seq_len(nrow(tab))
  tab$target <- tab$variable
  tab <- dplyr::relocate(tab, "rank", "target", "variable", "n_channels")

  # WHERE IN THE PATCH: per variable, window and ring, the mean |SHAP| of the
  # ring's pixels summed -- and per window, every channel's |SHAP| pixel by
  # pixel, the classic attribution map of the input. The kernel has none: its
  # players are variables, whole patches, not pixels.
  patch <- NULL; pixels <- NULL
  if (!identical(est, "kernel")) {
    ring_mean <- Reduce(`+`, ring_of) / length(ring_of)
    patch <- dplyr::bind_rows(lapply(seq_along(ws_model), function(i) {
      r_max <- (ws_model[i] - 1L) %/% 2L
      tibble::tibble(variable = rep(names(vars), times = r_max + 1L),
                     window = keys[i], ring = rep(0:r_max, each = length(vars)),
                     mean_abs = as.vector(ring_mean[, i, seq_len(r_max + 1L)]))
    }))
    pixels <- stats::setNames(lapply(seq_along(ws_model), function(i) {
      Reduce(`+`, lapply(pix_of, `[[`, i)) / length(pix_of)
    }), keys)
  }

  out <- structure(list(
    table = tab, by_model = by_model, baseline = dplyr::bind_rows(baseline),
    points = points, values = values, patch = patch, pixels = pixels,
    completeness = dplyr::bind_rows(comp), groups = grp, method = method, rows = rows,
    units = units[, c("unit", "kind", "fold", "seed", "role", "n_rows", "model_file")],
    run_dir = fr$run_dir, config_id = fr$config_id, window_sizes = ws_model,
    transform_name = data$transform$name %||% "none", rows_alone_share = 0, label = NULL),
    class = "dsm_importance")
  out$label <- .importance_object_label(out)
  out
}

# One model's SHAP values, its rows already cut and scaled. Kept apart from the
# files so the tests can hand it a model whose attributions are known.
#
# G          channels x variables, 0/1: which channels each variable sums
# estimator  "expected_gradients": references drawn from bg_inputs by draws$j,
#            places on the path by draws$a (both rows x K); "integrated_gradients":
#            one reference, `fill` (a value per channel) in every pixel, and K
#            places on the path at the midpoints of K equal steps
#
# Every window takes the SAME reference point and the same place on the path:
# a point's windows are one place, and so is its reference's.
.importance_shap_unit <- function(model, inputs, obs_native, obs_transform, G, estimator,
                                  bg_inputs = NULL, draws = NULL, fill = NULL, K = 50L,
                                  transform = identity, clamp = c(0, Inf), batch_size = 128L) {
  is_eg <- identical(estimator, "expected_gradients")
  if (is_eg && (is.null(bg_inputs) || is.null(draws))) {
    stop("Expected gradients need a background and its draws.", call. = FALSE)
  }
  if (!is_eg && is.null(fill)) stop("Integrated gradients need a baseline.", call. = FALSE)
  model$eval()
  # Gradients are wanted for the inputs only; none is kept for the weights.
  for (p in model$parameters) p$requires_grad_(FALSE)
  n    <- as.integer(inputs[[1]]$shape[1])
  n_ch <- as.integer(inputs[[1]]$shape[2])
  ws   <- vapply(inputs, function(t) as.integer(t$shape[3]), integer(1))
  V    <- ncol(G)
  G_t  <- torch::torch_tensor(G, dtype = torch::torch_float())

  f_x  <- .importance_forward(model, inputs, batch_size)
  base <- .importance_scores(f_x, obs_native, obs_transform, transform, clamp)
  if (is_eg) {
    delta <- f_x - mean(.importance_forward(model, bg_inputs, batch_size))
  } else {
    base_in <- lapply(ws, function(w) {
      torch::torch_tensor(as.numeric(fill), dtype = torch::torch_float())$view(
        c(1L, n_ch, 1L, 1L))$expand(c(1L, n_ch, w, w))$contiguous()
    })
    delta <- f_x - .importance_forward(model, base_in, 1L)
  }

  ring_masks <- lapply(ws, function(w) {
    ri <- patch_ring_index(w)
    lapply(0:((w - 1L) %/% 2L), function(r) {
      torch::torch_tensor(matrix(as.numeric(as.vector(ri) == r), nrow = nrow(ri)),
                          dtype = torch::torch_float())$view(c(1L, 1L, w, w))
    })
  })
  r_top <- max((ws - 1L) %/% 2L)
  phi_var  <- matrix(0, n, V)
  ring_abs <- array(0, dim = c(V, length(ws), r_top + 1L))
  pix_abs  <- lapply(ws, function(w) matrix(0, w, w))

  for (s in seq.int(1L, n, by = batch_size)) {
    e  <- min(n, s + batch_size - 1L)
    xb <- lapply(inputs, function(t) t[s:e, , , , drop = FALSE])
    phi <- lapply(xb, torch::torch_zeros_like)
    for (k in seq_len(K)) {
      if (is_eg) {
        jt  <- torch::torch_tensor(as.integer(draws$j[s:e, k]), dtype = torch::torch_long())
        ref <- lapply(bg_inputs, function(t) torch::torch_index_select(t, 1L, jt))
        alpha <- torch::torch_tensor(draws$a[s:e, k], dtype = torch::torch_float())$view(c(-1L, 1L, 1L, 1L))
      } else {
        ref <- base_in
        alpha <- (k - 0.5) / K
      }
      xa <- lapply(seq_along(xb), function(i) {
        (ref[[i]] + alpha * (xb[[i]] - ref[[i]]))$detach()$requires_grad_(TRUE)
      })
      out <- do.call(model, xa)
      # allow_unused: a window a model ignores has no gradient, not an error.
      g <- torch::autograd_grad(out$sum(), xa, allow_unused = TRUE)
      torch::with_no_grad({
        for (i in seq_along(xb)) {
          if (!torch::is_undefined_tensor(g[[i]])) phi[[i]]$add_((xb[[i]] - ref[[i]]) * g[[i]])
        }
      })
    }
    torch::with_no_grad({
      tot <- NULL
      for (i in seq_along(xb)) {
        phi[[i]]$div_(K)
        for (r in seq_along(ring_masks[[i]])) {
          pv  <- torch::torch_matmul((phi[[i]] * ring_masks[[i]][[r]])$sum(dim = c(3L, 4L)), G_t)
          tot <- if (is.null(tot)) pv else tot + pv
          ring_abs[, i, r] <- ring_abs[, i, r] + as.numeric(pv$abs()$sum(dim = 1L))
        }
        pix_abs[[i]] <- pix_abs[[i]] + as.matrix(phi[[i]]$abs()$sum(dim = c(1L, 2L)))
      }
      phi_var[s:e, ] <- as.matrix(tot)
    })
  }
  list(baseline = tibble::as_tibble(base), phi = phi_var, f_x = f_x, delta = delta,
       gap = rowSums(phi_var) - delta, ring_abs = ring_abs / n,
       pix_abs = lapply(pix_abs, function(m) m / n))
}

# The attributions of one model, by the estimator asked for.
.importance_attribute <- function(model, inputs, G, method, K, bg_inputs = NULL, draws = NULL,
                                  fill = NULL, transform = identity, clamp = c(0, Inf),
                                  batch_size = 128L) {
  if (identical(method$estimator, "kernel")) {
    return(.importance_kernel_unit(model, inputs, G, bg_inputs, permutations = method$permutations,
                                   seed = method$seed, batch_size = max(batch_size, 512L)))
  }
  n <- as.integer(inputs[[1]]$shape[1])
  .importance_shap_unit(model, inputs, rep(NA_real_, n), rep(NA_real_, n), G, method$estimator,
                        bg_inputs = bg_inputs, draws = draws, fill = fill, K = K,
                        transform = transform, clamp = clamp, batch_size = batch_size)
}

# How many variables the kernel can take: every coalition up to 14, sampled
# permutations up to 40 -- beyond, a permutation is a pass per variable per
# background point per point, and themes are what the method is for.
.importance_kernel_size <- function(M, sage = FALSE) {
  if (M > 40L) {
    stop(if (sage) "SAGE takes" else "The kernel estimator takes", " up to 40 variables; this ",
         "grouping has ", M, ". Give groups = a table of themes", if (sage)
           ", or use permutation_importance()." else ", or use expected gradients.", call. = FALSE)
  }
  invisible(M)
}

# What a SHAP run will do, in a line.
.importance_shap_cost_line <- function(method, M, n_models) {
  switch(method$estimator,
    expected_gradients = sprintf("%d model(s) x %d sample(s), each a forward and a backward pass over each model's rows",
                                 n_models, method$samples),
    integrated_gradients = sprintf("%d model(s) x %d step(s), each a forward and a backward pass over each model's rows",
                                   n_models, method$steps),
    kernel = if (M <= 14L) {
      sprintf("%d model(s) x every one of %s coalitions of %d variable(s) x %d background point(s), forward passes",
              n_models, format(2^M, big.mark = ","), M, method$background)
    } else {
      sprintf("%d model(s) x %d sampled permutation(s) of %d variable(s) x %d background point(s), forward passes",
              n_models, 2L * (method$permutations %/% 2L), M, method$background)
    })
}

# The Shapley values of the variables, from coalitions (the KernelSHAP game,
# Lundberg & Lee 2017), for one model. Kept apart from the files so the tests
# can hand it a model whose values are known.
#
# THE VALUE OF A COALITION is the mean prediction, over the background points,
# with every channel of a variable outside the coalition taken from the
# background point: the whole patch, the same point in every window -- a
# point's windows are one place, and so are its reference's. With every
# coalition (up to `exact_max` variables) the values are the Shapley values
# themselves; above, sampled permutations, each with its reverse, estimate
# them. Either way they add up exactly, the coalition of all being the
# prediction and the empty one the background's mean.
.importance_kernel_unit <- function(model, inputs, G, bg_inputs, exact_max = 14L,
                                    permutations = 64L, seed = 42L, batch_size = 512L) {
  value <- .importance_coalition_value(model, inputs, G, bg_inputs, batch_size)
  sh <- .importance_shapley(value, ncol(G), exact_max, permutations, seed)
  delta <- sh$v_full - sh$v_empty
  list(phi = sh$phi, f_x = sh$v_full, delta = delta, gap = rowSums(sh$phi) - delta,
       ring_abs = NULL, pix_abs = NULL)
}

# The value of a coalition S of variables, for one model: at each point, the
# mean prediction over the background points with every channel of a variable
# outside S taken from the background point. A function of S, shared by the
# kernel (its Shapley values at each point) and SAGE (the Shapley values of
# the loss of these predictions).
.importance_coalition_value <- function(model, inputs, G, bg_inputs, batch_size = 512L) {
  model$eval()
  n    <- as.integer(inputs[[1]]$shape[1])
  n_ch <- as.integer(inputs[[1]]$shape[2])
  B    <- as.integer(bg_inputs[[1]]$shape[1])
  # Every (point, background point) pair is one row of the passes: in batches
  # of rows, the point's and the reference's patches picked by index -- no
  # stack of n x B patches is ever made.
  pt_idx <- rep(seq_len(n), times = B)
  bg_idx <- rep(seq_len(B), each = n)
  function(S) {
    keep <- torch::torch_tensor(as.vector(G %*% as.numeric(S)) > 0,
                                dtype = torch::torch_bool())$view(c(1L, n_ch, 1L, 1L))
    out <- numeric(n * B)
    torch::with_no_grad({
      for (s in seq.int(1L, n * B, by = batch_size)) {
        e  <- min(n * B, s + batch_size - 1L)
        pi_t <- torch::torch_tensor(pt_idx[s:e], dtype = torch::torch_long())
        bi_t <- torch::torch_tensor(bg_idx[s:e], dtype = torch::torch_long())
        xb <- lapply(seq_along(inputs), function(i) {
          torch::torch_where(keep, torch::torch_index_select(inputs[[i]], 1L, pi_t),
                             torch::torch_index_select(bg_inputs[[i]], 1L, bi_t))
        })
        out[s:e] <- as.numeric(do.call(model, unname(xb))$to(device = "cpu"))
      }
    })
    rowMeans(matrix(out, nrow = n, ncol = B))
  }
}

# The Shapley values of a game of M players whose payoff for a coalition S is
# payoff(S) -- one number per point (the kernel) or one number (SAGE). Exact
# over every coalition up to `exact_max` players; above, sampled permutations,
# each with its reverse, which add up exactly too: along a permutation the
# marginals telescope to the payoff of all less the payoff of none.
.importance_shapley <- function(payoff, M, exact_max = 14L, permutations = 64L, seed = 42L) {
  if (M <= exact_max) {
    n_coal <- 2^M
    bits <- 2^(seq_len(M) - 1L)
    member <- function(m) bitwAnd(m, bits) > 0
    first <- payoff(member(0))
    V <- matrix(NA_real_, length(first), n_coal)
    V[, 1] <- first
    for (m in seq_len(n_coal - 1)) V[, m + 1] <- payoff(member(m))
    size <- vapply(0:(n_coal - 1), function(m) sum(member(m)), numeric(1))
    w <- factorial(0:(M - 1)) * factorial((M - 1):0) / factorial(M)
    phi <- matrix(0, nrow(V), M)
    for (i in seq_len(M)) {
      without <- which(bitwAnd(0:(n_coal - 1), bits[i]) == 0) - 1
      for (m in without) {
        phi[, i] <- phi[, i] + w[size[m + 1] + 1] * (V[, m + bits[i] + 1] - V[, m + 1])
      }
    }
    return(list(phi = phi, v_empty = V[, 1], v_full = V[, n_coal]))
  }
  half  <- max(1L, permutations %/% 2L)
  perms <- with_local_seed(seed + 101L, lapply(seq_len(half), function(p) sample.int(M)))
  perms <- c(perms, lapply(perms, rev))
  v_empty <- payoff(rep(FALSE, M))
  v_full  <- payoff(rep(TRUE, M))
  phi <- matrix(0, length(v_empty), M)
  for (pm in perms) {
    S <- rep(FALSE, M)
    prev <- v_empty
    for (k in seq_len(M)) {
      S[pm[k]] <- TRUE
      cur <- if (k == M) v_full else payoff(S)
      phi[, pm[k]] <- phi[, pm[k]] + (cur - prev)
      prev <- cur
    }
  }
  list(phi = phi / length(perms), v_empty = v_empty, v_full = v_full)
}

# SAGE for one model (Covert, Lundberg & Lee 2020): the Shapley values of the
# loss the model's predictions make, in the game the kernel plays -- the value
# of a coalition the loss, over the points, of the coalition's predictions. A
# variable's value is the loss it takes away, shared fairly among variables
# that carry the same information; they add up to the loss of the mean
# prediction (no variable known) less the model's own.
.importance_sage_unit <- function(model, inputs, y, G, bg_inputs, loss = "mse",
                                  exact_max = 14L, permutations = 64L, seed = 42L,
                                  batch_size = 512L) {
  value <- .importance_coalition_value(model, inputs, G, bg_inputs, batch_size)
  lossf <- if (identical(loss, "mae")) function(p) mean(abs(p - y)) else
    function(p) mean((p - y)^2)
  sh <- .importance_shapley(function(S) -lossf(value(S)), ncol(G), exact_max, permutations,
                            seed)
  phi <- as.vector(sh$phi)
  list(phi = phi, loss_model = -sh$v_full, loss_empty = -sh$v_empty,
       gap = sum(phi) - (sh$v_full - sh$v_empty))
}

# sage_importance() through every model of the call: the same units, rows,
# background and checks as the kernel, the loss of the coalitions instead of
# their predictions.
.importance_sage_run <- function(fr, data, units, grp, vars, method, transform, clamp,
                                 batch_size, say, rows) {
  ch   <- as.character(data$store$predictors)
  n_ch <- length(ch)
  G <- matrix(0, n_ch, length(vars), dimnames = list(ch, names(vars)))
  for (v in seq_along(vars)) G[vars[[v]], v] <- 1
  M <- ncol(G)
  .importance_kernel_size(M, sage = TRUE)
  ws_model <- as.integer(units$cfg[[1]]$window_sizes[[1]])
  keys <- patch_window_key(ws_model)
  say(sprintf("Importance -- %s | %s of final run '%s' (%s)", .importance_label(method),
              if (rows == "test") "test set" else "fold validation rows",
              basename(fr$run_dir), fr$config_id))
  say("  ", .importance_groups_line(grp))
  say("  ", .importance_shap_cost_line(utils::modifyList(method, list(estimator = "kernel")),
                                       M, nrow(units)))

  device <- torch::torch_device("cpu")
  baseline <- list(); per_model <- list(); losses <- list()
  cache <- NULL; cache_key <- NULL; bg <- NULL; sel <- NULL
  t0 <- Sys.time()
  for (u in seq_len(nrow(units))) {
    un <- units[u, , drop = FALSE]
    rows_u <- un$rows[[1]]
    if (!identical(cache_key, un$cache_key)) {
      cache <- NULL; bg <- NULL; invisible(gc(verbose = FALSE))
      ws_all <- sort(unique(unlist(un$cfg[[1]]$window_sizes)))
      cache <- build_fold_cache(data$store, data$points, data$type_table,
                                stats::setNames(list(rows_u), un$role), ws_all,
                                scaling = un$scaling[[1]], verbose = FALSE)$cache[[un$role]]
      cache_key <- un$cache_key
      n_all <- length(rows_u)
      # The same draw of rows as shap_importance()'s for one seed: SAGE and
      # SHAP of one call can be read on the same points.
      sel <- if (method$max_points >= n_all) seq_len(n_all) else
        with_local_seed(method$seed + 31L, sort(sample.int(n_all, method$max_points)))
      train <- un$train_rows[[1]]
      bg_rows <- with_local_seed(method$seed + 7919L, {
        if (length(train) > method$background) sort(sample(train, method$background)) else train
      })
      bg <- build_fold_cache(data$store, data$points, data$type_table, list(background = bg_rows),
                             ws_all, scaling = un$scaling[[1]], verbose = FALSE)$cache$background
    }
    inputs <- lapply(keys, function(k) cache[[k]])
    meta_u <- data$store$meta[rows_u, , drop = FALSE]
    model <- build_cnn_from_config(un$cfg[[1]], data$store$n_channels)
    model$load_state_dict(torch::torch_load(un$model_file))
    model$to(device = device)

    f_full <- .importance_forward(model, inputs, batch_size)
    base <- tibble::as_tibble(.importance_scores(
      f_full, as.numeric(meta_u$target_native), as.numeric(meta_u$target_transform),
      transform, clamp))
    .importance_check_baseline(base, un, pred_t = f_full, sample_id = meta_u$sample_id,
                               obs = as.numeric(meta_u$target_native))
    sub <- if (length(sel) == n_all) inputs else {
      st <- torch::torch_tensor(as.integer(sel), dtype = torch::torch_long())
      lapply(inputs, function(t) torch::torch_index_select(t, 1L, st))
    }
    res <- .importance_sage_unit(model, sub, as.numeric(meta_u$target_transform)[sel], G,
                                 lapply(keys, function(k) bg[[k]]), loss = method$loss,
                                 permutations = method$permutations, seed = method$seed,
                                 batch_size = max(batch_size, 512L))
    # EXACT BY CONSTRUCTION, and checked: off by more than rounding is wiring.
    if (abs(res$gap) > 1e-8 * max(1, abs(res$loss_empty))) {
      stop(sprintf("Model %s: its SAGE values do not add up to the loss explained (off by %.3g).",
                   un$unit, res$gap), call. = FALSE)
    }
    baseline[[u]] <- dplyr::mutate(base, unit = un$unit, .before = 1)
    per_model[[u]] <- tibble::tibble(unit = un$unit, variable = names(vars), importance = res$phi)
    losses[[u]] <- tibble::tibble(unit = un$unit, loss_model = res$loss_model,
                                  loss_mean_prediction = res$loss_empty)
    rm(model, sub); invisible(gc(verbose = FALSE))
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    say(sprintf("  [%d/%d] %s: CCC %.4f, as its run wrote | %s %.4g explained of %.4g | %.1f min%s",
                u, nrow(units), un$unit, base$ccc, method$loss, res$loss_empty - res$loss_model,
                res$loss_empty, el, .importance_eta(el / u * (nrow(units) - u))))
  }
  rm(cache, bg); invisible(gc(verbose = FALSE))

  by_model <- dplyr::bind_rows(per_model)
  tab <- by_model %>%
    dplyr::rename(imp_model = "importance") %>%
    dplyr::group_by(.data$variable) %>%
    dplyr::summarise(importance = mean(.data$imp_model),
                     sd_models = if (dplyr::n() > 1L) stats::sd(.data$imp_model) else NA_real_,
                     n_models = dplyr::n(),
                     n_models_positive = sum(.data$imp_model > 0), .groups = "drop")
  tab$share <- tab$importance / sum(tab$importance)
  tab$n_channels <- as.integer(lengths(vars)[tab$variable])
  tab <- tab[order(-tab$importance, tab$variable), , drop = FALSE]
  tab$rank <- seq_len(nrow(tab))
  tab$target <- tab$variable
  tab <- dplyr::relocate(tab, "rank", "target", "variable", "n_channels")
  out <- structure(list(
    table = tab, by_model = by_model, baseline = dplyr::bind_rows(baseline),
    losses = dplyr::bind_rows(losses), groups = grp, method = method, rows = rows,
    units = units[, c("unit", "kind", "fold", "seed", "role", "n_rows", "model_file")],
    run_dir = fr$run_dir, config_id = fr$config_id, window_sizes = ws_model,
    transform_name = data$transform$name %||% "none", rows_alone_share = 0, label = NULL),
    class = "dsm_importance")
  out$label <- .importance_object_label(out)
  out
}

# THE CHECK THAT STOPS, for attributions: they must add up. Each point's values
# sum to its prediction less the reference -- exactly, for integrated
# gradients, up to how finely the path is cut; on average, for expected
# gradients, whose sum is right in expectation and noisy point by point. A
# sum that is off on average is not noise: an input, a window or a scaling
# is not the model's, and every value would be wrong in a way no plot shows.
.importance_check_completeness <- function(res, estimator, unit) {
  d <- res$delta
  gap <- res$gap
  scale <- mean(abs(d))
  out <- tibble::tibble(unit = unit, mean_gap = mean(gap), sd_gap = stats::sd(gap),
                        rel_noise = mean(abs(gap)) / scale, rel_bias = abs(mean(gap)) / scale)
  if (!is.finite(scale) || scale <= 0) return(out)       # one prediction for all: nothing to share
  if (identical(estimator, "kernel")) {
    # Exact by construction: the coalition of all is the prediction and the
    # empty one the background's mean. Off by more than rounding is wiring.
    if (out$rel_noise > 1e-5) {
      stop(sprintf(paste0("Model %s: its Shapley values do not add up to the prediction less the ",
                          "background's mean (off by %.2g%%): the coalitions are not the model's."),
                   unit, 100 * out$rel_noise), call. = FALSE)
    }
    return(out)
  }
  if (identical(estimator, "integrated_gradients")) {
    if (out$rel_noise > 0.05) {
      stop(sprintf(paste0("Model %s: its integrated gradients add up to the prediction less the ",
                          "baseline's only to within %.1f%%. The path is cut too coarsely for this ",
                          "network: give shap_importance(steps =) more."), unit, 100 * out$rel_noise),
           call. = FALSE)
    }
  } else {
    se <- stats::sd(gap) / sqrt(length(gap))
    if (abs(mean(gap)) > 4 * se + 0.01 * scale) {
      stop(sprintf(paste0("Model %s: its attributions do not add up, on average, to the prediction ",
                          "less the background's mean prediction (off by %.3g, against a sampling ",
                          "noise of %.3g). The gradients are not the model's: an input, a window or ",
                          "a scaling is wired wrongly."), unit, mean(gap), se), call. = FALSE)
    }
  }
  out
}

# ── SHAP at points of the map ─────────────────────────────────────────────────

# The models of a call kept to the seeds asked for.
.importance_pick_seeds <- function(units, seeds) {
  if (is.null(seeds)) return(units)
  keep <- units$seed %in% as.integer(seeds)
  if (!any(keep)) {
    stop("None of the seeds asked for (", paste(seeds, collapse = ", "), ") is the run's: ",
         paste(unique(units$seed), collapse = ", "), ".", call. = FALSE)
  }
  units[keep, , drop = FALSE]
}

#' Points of the map to explain: a regular grid of the model's raster cells.
#'
#' The cells of the rasters the model was fitted on, every `every` rows and
#' columns, inside `extent`, where the first channel has a value. With
#' `every = 1` every cell -- the map at full resolution, for a tile; with 40 at
#' 250 m, one point every 10 km, for a region. The points carry their grid, so
#' [importance_map()] can lay the values back on it.
#'
#' @param final   A `dsm_final`, or the directory of a final run.
#' @param data    The `dsm_data` it was fitted on.
#' @param extent  c(xmin, xmax, ymin, ymax) in the rasters' coordinates; NULL
#'   for the whole raster.
#' @param every   Take every `every`-th cell, in rows and in columns.
#' @param rasters Where the rasters are, as in [dsm_predict()]; NULL for the
#'   store's own raster table.
#' @return A tibble of `x`, `y` (cell centres), with the grid as an attribute.
#' @export
importance_points <- function(final, data, extent = NULL, every = 1L, rasters = NULL) {
  if (!is.numeric(every) || length(every) != 1L || every < 1 || every != round(every)) {
    stop("every must be one whole number of 1 or more.", call. = FALSE)
  }
  fr  <- .predict_final(final)
  inp <- .predict_inputs(data, rasters, NULL, fr$scaling)
  ref <- terra::rast(inp$files[1])
  if (!is.null(extent)) {
    if (!is.numeric(extent) || length(extent) != 4L || extent[1] >= extent[2] ||
        extent[3] >= extent[4]) {
      stop("extent must be c(xmin, xmax, ymin, ymax).", call. = FALSE)
    }
    ref <- terra::crop(ref, terra::ext(extent))
  }
  every <- as.integer(every)
  rr <- seq.int(1L, terra::nrow(ref), by = every)
  cc <- seq.int(1L, terra::ncol(ref), by = every)
  cells <- as.vector(outer(cc, (rr - 1L) * terra::ncol(ref), "+"))
  xy  <- terra::xyFromCell(ref, cells)
  val <- terra::extract(ref, xy)[, 1]
  keep <- is.finite(val)
  if (!any(keep)) stop("Every cell taken is empty in the first channel.", call. = FALSE)
  out <- tibble::tibble(x = xy[keep, 1], y = xy[keep, 2])
  attr(out, "grid") <- list(every = every, res = terra::res(ref) * every,
                            crs = terra::crs(ref), extent = as.vector(terra::ext(ref)))
  out
}

# shap_importance() at points of the map: their patches cut from the rasters in
# chunks, checked first against the store's own patches, then explained by the
# final run's seeds.
#
# THE CHECK THAT STOPS, where nothing was observed. A map point has no score to
# reproduce, so the reading itself is checked: patches cut by this very path at
# some of the store's own profiles must be the patches the store holds. That
# covers the raster files, their order, the QC and the windows' geometry --
# every way a map point's patch could be another place's -- and stops if any
# differs.
.importance_shap_map_run <- function(fr, data, at, grp, vars, method, transform, clamp,
                                     batch_size, say, rasters, seeds, chunk_points) {
  if (!is.data.frame(at) || !all(c("x", "y") %in% names(at))) {
    stop("at must be a data frame with columns x and y, e.g. from importance_points().",
         call. = FALSE)
  }
  if (!is.numeric(chunk_points) || length(chunk_points) != 1L || chunk_points < 1) {
    stop("chunk_points must be one positive whole number.", call. = FALSE)
  }
  chunk_points <- as.integer(chunk_points)
  grid <- attr(at, "grid")               # before the reordering below drops it
  inp  <- .predict_inputs(data, rasters, NULL, fr$scaling)
  ch   <- as.character(data$store$predictors)
  n_ch <- length(ch)
  ws   <- as.integer(fr$cfg$window_sizes[[1]])
  keys <- patch_window_key(ws)
  G <- matrix(0, n_ch, length(vars), dimnames = list(ch, names(vars)))
  for (v in seq_along(vars)) G[vars[[v]], v] <- 1
  # The final run's seeds, straight from it: a map needs no test set.
  units <- .importance_pick_seeds(tibble::tibble(
    unit = sprintf("seed%04d", fr$seeds), kind = "final seed", fold = NA_integer_,
    seed = fr$seeds, role = "map", n_rows = NA_integer_, cfg = rep(list(fr$cfg), length(fr$seeds)),
    model_file = fr$model_files), seeds)
  # The variables of one channel, whose value at a point can be read: kept at
  # every point, for the direction and for plots of SHAP against the value.
  single <- names(vars)[lengths(vars) == 1L]
  w_small <- min(ws)
  c0 <- (w_small + 1L) %/% 2L
  est <- method$estimator
  is_eg <- identical(est, "expected_gradients")
  needs_bg <- est %in% c("expected_gradients", "kernel")
  if (identical(est, "kernel")) .importance_kernel_size(ncol(G))
  K <- switch(est, expected_gradients = method$samples, integrated_gradients = method$steps,
              kernel = NA_integer_)
  b_size <- min(batch_size, 128L)
  quiet <- function(...) invisible(NULL)
  extract <- function(xy) {
    .prep_extract_patches(files = inp$files, qc_table = inp$qc_table, xy = as.matrix(xy),
                          windows = sort(unique(ws)), chunk_nrows = 1000L, n_cores = 1L,
                          max_ram_gb = NULL, read_gap = 256L, read_max_cols = 4096L,
                          cell_bytes = 8, say = quiet)
  }

  # ── the probe: the store's own profiles, cut again by this path
  n_probe <- min(12L, nrow(data$store$meta))
  probe_rows <- with_local_seed(method$seed + 4243L, sort(sample.int(nrow(data$store$meta), n_probe)))
  ex_p <- extract(cbind(data$store$meta$x[probe_rows], data$store$meta$y[probe_rows]))
  worst <- 0
  for (w in sort(unique(ws))) {
    k  <- patch_window_key(w)
    st <- data$store$windows[[k]]
    if (is.null(st)) {
      st <- .read_patch_array(data$store$patch_dir, w, expect_points = nrow(data$store$meta),
                              expect_channels = n_ch)
    }
    stored <- as.array(.rows_to_float(st, probe_rows))
    cut    <- ex_p$patch_list[[k]]
    worst  <- max(worst, max(abs(cut - stored) / pmax(1, abs(stored)), na.rm = TRUE))
    if (anyNA(cut) != anyNA(stored) || worst > 1e-5) {
      stop(sprintf(paste0("The map's patches are not the store's: cut again at %d of the store's ",
                          "own profiles, window %s differs by %.3g (relative). Another raster, ",
                          "another channel order, other QC rules or another grid -- every map ",
                          "point would be read wrongly. Pass the rasters the store was cut from."),
                   n_probe, k, worst), call. = FALSE)
    }
  }

  # ── the background, for expected gradients: the final run's training rows
  split <- readRDS(file.path(fr$run_dir, "run_spec.rds"))$split
  train <- as.integer(split$train)
  bg <- NULL; fill <- NULL
  if (needs_bg) {
    bg_rows <- with_local_seed(method$seed + 7919L, {
      if (length(train) > method$background) sort(sample(train, method$background)) else train
    })
    bg <- build_fold_cache(data$store, data$points, data$type_table, list(background = bg_rows),
                           sort(unique(ws)), scaling = fr$scaling, verbose = FALSE)$cache$background
  } else {
    fill <- .importance_fill_values(data, fr$scaling, train)
  }
  models <- lapply(seq_len(nrow(units)), function(u) {
    m <- build_cnn_from_config(units$cfg[[u]], n_ch)
    m$load_state_dict(torch::torch_load(units$model_file[[u]]))
    m$to(device = torch::torch_device("cpu"))
    m
  })

  # Read in rows, so a chunk's points share the rows their patches span.
  at <- at[order(-at$y, at$x), c("x", "y"), drop = FALSE]
  n_at <- nrow(at)
  say(sprintf("Importance -- %s | %s map point(s) of final run '%s' (%s)",
              .importance_label(method), format(n_at, big.mark = ","), basename(fr$run_dir),
              fr$config_id))
  say(sprintf("  the probe: %d of the store's profiles cut again by this path, as the store holds them (worst %.1e)",
              n_probe, worst))
  say("  ", .importance_groups_line(grp))
  say("  ", .importance_shap_cost_line(method, ncol(G), nrow(units)),
      sprintf(", in chunks of %s point(s)", format(chunk_points, big.mark = ",")))

  pts <- list(); vals <- list(); comp <- list(); ring_sum <- 0; pix_sum <- NULL
  n_done <- 0L; dropped <- 0L
  t0 <- Sys.time()
  for (cs in seq.int(1L, n_at, by = chunk_points)) {
    ce <- min(n_at, cs + chunk_points - 1L)
    ex <- extract(at[cs:ce, , drop = FALSE])
    arrs <- ex$patch_list[keys]
    # A point whose patch is not whole -- the edge of the rasters, a coast, a
    # gap in one channel -- is not predicted by the map either; nor here.
    ok <- ex$edge_ok
    for (a in arrs) ok <- ok & rowSums(!is.finite(matrix(a, nrow = dim(a)[1]))) == 0L
    dropped <- dropped + sum(!ok)
    if (!any(ok)) next
    inputs <- lapply(arrs, function(a) {
      t <- torch::torch_tensor(a[ok, , , , drop = FALSE], dtype = torch::torch_float())
      scale_patches(t, fr$scaling, inplace = TRUE)
    })
    n_ok <- sum(ok)
    draws <- if (is_eg) with_local_seed(method$seed + cs, list(
      j = matrix(sample.int(as.integer(bg[[keys[1]]]$shape[1]), n_ok * K, replace = TRUE), n_ok, K),
      a = matrix(stats::runif(n_ok * K), n_ok, K))) else NULL
    phi <- 0; fx <- 0; fx_nat <- 0; ref <- 0
    for (u in seq_along(models)) {
      res <- .importance_attribute(models[[u]], inputs, G, method, K = K,
                                   bg_inputs = if (needs_bg) lapply(keys, function(k) bg[[k]]) else NULL,
                                   draws = draws, fill = fill, transform = transform,
                                   clamp = clamp, batch_size = b_size)
      comp[[length(comp) + 1L]] <- .importance_check_completeness(
        res, method$estimator, sprintf("%s, points %d-%d", units$unit[u], cs, ce))
      phi <- phi + res$phi
      fx  <- fx + res$f_x
      ref <- ref + (res$f_x - res$delta)
      pn  <- transform(res$f_x)
      if (is.finite(clamp[1])) pn <- pmax(pn, clamp[1])
      if (is.finite(clamp[2])) pn <- pmin(pn, clamp[2])
      fx_nat <- fx_nat + pn
      if (!is.null(res$ring_abs)) {
        ring_sum <- ring_sum + res$ring_abs * n_ok
        pix_sum  <- if (is.null(pix_sum)) lapply(res$pix_abs, function(m) m * n_ok) else
          Map(function(a, b) a + b * n_ok, pix_sum, res$pix_abs)
      }
    }
    nm <- length(models)
    colnames(phi) <- names(vars)
    # The value at the point: the centre of the smallest window, QC'd, unscaled.
    cv <- ex$patch_list[[patch_window_key(w_small)]][ok, , c0, c0, drop = FALSE]
    cv <- matrix(cv, nrow = n_ok)[, unlist(vars[single]), drop = FALSE]
    colnames(cv) <- single
    vals[[length(vals) + 1L]] <- cv
    pts[[length(pts) + 1L]] <- dplyr::bind_cols(
      tibble::tibble(x = at$x[cs:ce][ok], y = at$y[cs:ce][ok], prediction = fx / nm,
                     prediction_native = fx_nat / nm, reference = ref / nm),
      tibble::as_tibble(phi / nm, .name_repair = "minimal"))
    n_done <- n_done + n_ok
    rm(inputs, arrs, ex); invisible(gc(verbose = FALSE))
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    say(sprintf("  points %s-%s: %d explained | %.1f min%s", format(cs, big.mark = ","),
                format(ce, big.mark = ","), n_ok, el, .importance_eta(el / ce * (n_at - ce))))
  }
  if (n_done == 0L) stop("No map point had a whole patch to explain.", call. = FALSE)
  points <- dplyr::bind_rows(pts)
  values <- do.call(rbind, vals)
  # From the matrices, not from `points`: a variable named like one of its
  # columns (x, y) would be read as the coordinate there.
  phi_all <- do.call(rbind, lapply(pts, function(p) as.matrix(p[, -(1:5), drop = FALSE])))
  colnames(phi_all) <- names(vars)

  by_model <- NULL
  imp <- colMeans(abs(phi_all))
  tab <- tibble::tibble(variable = names(vars), importance = imp,
                        sd_models = NA_real_, mean_signed = colMeans(phi_all),
                        n_models = length(models))
  tab$share <- tab$importance / sum(tab$importance)
  # The direction at the map points: Spearman between the value at the point
  # and its SHAP value, as at the profiles.
  tab$direction <- vapply(tab$variable, function(v) {
    if (!v %in% single) return(NA_real_)
    a <- values[, v]; s <- phi_all[, v]
    if (!is.finite(stats::sd(a)) || stats::sd(a) == 0 || stats::sd(s) == 0) return(NA_real_)
    suppressWarnings(stats::cor(a, s, method = "spearman"))
  }, numeric(1))
  tab$n_channels <- as.integer(lengths(vars))
  tab <- tab[order(-tab$importance, tab$variable), , drop = FALSE]
  tab$rank <- seq_len(nrow(tab))
  tab$target <- tab$variable
  tab <- dplyr::relocate(tab, "rank", "target", "variable", "n_channels")
  patch <- NULL
  if (!is.null(pix_sum)) {
    ring_mean <- ring_sum / n_done
    patch <- dplyr::bind_rows(lapply(seq_along(ws), function(i) {
      r_max <- (ws[i] - 1L) %/% 2L
      tibble::tibble(variable = rep(names(vars), times = r_max + 1L), window = keys[i],
                     ring = rep(0:r_max, each = length(vars)),
                     mean_abs = as.vector(ring_mean[, i, seq_len(r_max + 1L)]))
    }))
  }
  units_out <- units[, c("unit", "kind", "fold", "seed", "role", "n_rows", "model_file")]
  units_out$role <- "map"
  units_out$n_rows <- n_done
  out <- structure(list(
    table = tab, by_model = by_model, baseline = tibble::tibble(unit = units$unit, ccc = NA_real_),
    points = points, values = values, patch = patch,
    pixels = if (is.null(pix_sum)) NULL else stats::setNames(lapply(pix_sum, function(m) m / n_done), keys),
    completeness = dplyr::bind_rows(comp), groups = grp, method = method, rows = "map",
    units = units_out, run_dir = fr$run_dir, config_id = fr$config_id, window_sizes = ws,
    transform_name = data$transform$name %||% "none", rows_alone_share = 0,
    grid = grid, n_dropped = dropped, probe_worst = worst, label = NULL),
    class = "dsm_importance")
  out$label <- .importance_object_label(out)
  out
}

#' Maps of SHAP values, from an importance computed at points of the map.
#'
#' Lays the values of [dsm_importance()]`(at = )` back on a grid: each cell
#' the mean of the points in it. With the points' own grid (from
#' [importance_points()]) every point is its own cell; with a coarser
#' `resolution`, each cell averages the points it holds.
#'
#' @param x A `dsm_importance` from [shap_importance()] at map points.
#' @param resolution Cell size of the maps, in the rasters' units; NULL for the
#'   points' own grid.
#' @param output_dir Where to write the GeoTIFFs (shap.tif, one layer per
#'   variable; shap_dominant.tif and its legend, shap_dominant.csv;
#'   prediction.tif); NULL to only return them.
#' @return An `importance_map`, a list: `shap` (one layer per variable: the mean
#'   SHAP value of the points in each cell, in the network's units), `dominant`
#'   (in each cell, the variable with the largest mean |SHAP|, as the number in
#'   `legend`), `legend`, `prediction` (the mean prediction, native units)
#'   and `files`. `plot()` draws it.
#' @export
importance_map <- function(x, resolution = NULL, output_dir = NULL) {
  if (!inherits(x, "dsm_importance") || !identical(x$rows, "map")) {
    stop("importance_map() needs SHAP values at points of the map: ",
         "dsm_importance(..., shap_importance(), at = importance_points(...)).", call. = FALSE)
  }
  res <- resolution %||% x$grid$res
  if (is.null(res)) {
    stop("The points carry no grid -- they were not made by importance_points() -- so ",
         "give resolution.", call. = FALSE)
  }
  res <- rep_len(as.numeric(res), 2L)
  crs <- x$grid$crs %||% ""
  p <- as.data.frame(x$points)
  v_names <- x$table$variable[order(match(x$table$variable, names(p)))]
  # Each point at a cell's centre on its own grid: the grid's corner is half a
  # cell from the first point.
  tmpl <- terra::rast(xmin = min(p$x) - res[1] / 2, xmax = max(p$x) + res[1] / 2,
                      ymin = min(p$y) - res[2] / 2, ymax = max(p$y) + res[2] / 2,
                      resolution = res, crs = crs)
  abs_names <- paste0(".abs_", seq_along(v_names))
  for (i in seq_along(v_names)) p[[abs_names[i]]] <- abs(p[[v_names[i]]])
  pv <- terra::vect(p, geom = c("x", "y"), crs = crs)
  lay <- function(f) terra::rasterize(pv, tmpl, field = f, fun = "mean")
  shap <- terra::rast(lapply(v_names, lay))
  names(shap) <- v_names
  dominant <- terra::which.max(terra::rast(lapply(abs_names, lay)))
  names(dominant) <- "dominant"
  legend <- tibble::tibble(value = seq_along(v_names), variable = v_names)
  prediction <- lay("prediction_native")
  names(prediction) <- "prediction"
  files <- character(0)
  if (!is.null(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    files <- file.path(output_dir, c("shap.tif", "shap_dominant.tif", "prediction.tif",
                                     "shap_dominant.csv"))
    terra::writeRaster(shap, files[1], overwrite = TRUE)
    terra::writeRaster(dominant, files[2], overwrite = TRUE)
    terra::writeRaster(prediction, files[3], overwrite = TRUE)
    safe_write_csv2(legend, files[4])
  }
  structure(list(shap = shap, dominant = dominant, legend = legend, prediction = prediction,
                 files = files),
            class = "importance_map")
}

# ── ALE: the bins, the moves, the curves ──────────────────────────────────────

# ale_effect() through every model of the call.
#
# ONE SET OF BINS FOR THE WHOLE CALL, cut from every row any model scores.
# Each fold's model sees its own rows, and bins cut from each fold's rows would
# give each fold's curve other edges: they could not be averaged. A bin a
# fold's rows do not reach has no local effect there -- flat in that fold's
# curve, and no weight in its centring.
.importance_ale_run <- function(fr, data, units, groups, method, transform, clamp,
                                batch_size, say, rows) {
  ch  <- as.character(data$store$predictors)
  grp <- importance_groups(data, "auto")
  vars <- lapply(split(seq_along(ch), factor(grp$variable, levels = unique(grp$variable))),
                 as.integer)
  if (!is.null(method$variables)) {
    unknown <- setdiff(method$variables, names(vars))
    if (length(unknown) > 0L) {
      stop("ALE: variable(s) ", paste(utils::head(unknown, 10L), collapse = ", "),
           " are not among the store's. The names are importance_groups(data)'s: a ",
           "channel, or a one-hot set.", call. = FALSE)
    }
    vars <- vars[method$variables]
  }
  rows_all <- sort(unique(unlist(units$rows)))
  plan <- .importance_ale_plan(data, vars, rows_all, method$bins)
  skipped <- names(plan)[vapply(plan, is.null, logical(1))]
  plan <- plan[!vapply(plan, is.null, logical(1))]
  if (length(plan) == 0L) {
    stop("ALE: no variable has two or more distinct values at these points.", call. = FALSE)
  }
  kinds <- vapply(plan, `[[`, character(1), "kind")
  ws_model <- as.integer(units$cfg[[1]]$window_sizes[[1]])
  keys <- patch_window_key(ws_model)

  say(sprintf("Importance -- %s | %s of final run '%s' (%s)", .importance_label(method),
              if (rows == "test") "test set" else "fold validation rows",
              basename(fr$run_dir), fr$config_id))
  if (!identical(groups, "auto")) {
    say("  ALE is computed per channel and per one-hot set; the grouping given is not used.")
  }
  say(sprintf("  %d variable(s): %d continuous, %d categorical%s", length(plan),
              sum(kinds == "continuous"), sum(kinds == "categorical"),
              if (length(skipped)) sprintf(" | %d with one value at these points, skipped",
                                           length(skipped)) else ""))

  device <- torch::torch_device("cpu")
  baseline <- list(); per <- list()
  cache <- NULL; cache_key <- NULL
  t0 <- Sys.time()
  for (u in seq_len(nrow(units))) {
    un <- units[u, , drop = FALSE]
    rows_u <- un$rows[[1]]
    if (!identical(cache_key, un$cache_key)) {
      cache <- NULL; invisible(gc(verbose = FALSE))
      ws_all <- sort(unique(unlist(un$cfg[[1]]$window_sizes)))
      idx <- stats::setNames(list(rows_u), un$role)
      cache <- build_fold_cache(data$store, data$points, data$type_table, idx, ws_all,
                                scaling = un$scaling[[1]], verbose = FALSE)$cache[[un$role]]
      cache_key <- un$cache_key
    }
    inputs <- lapply(keys, function(k) cache[[k]])
    meta_u <- data$store$meta[rows_u, , drop = FALSE]
    model <- build_cnn_from_config(un$cfg[[1]], data$store$n_channels)
    model$load_state_dict(torch::torch_load(un$model_file))
    model$to(device = device)

    res <- .importance_ale_unit(model, inputs, plan, data, rows_u, un$scaling[[1]],
                                as.numeric(meta_u$target_native),
                                as.numeric(meta_u$target_transform), transform, clamp,
                                batch_size)
    .importance_check_baseline(res$baseline, un, pred_t = res$pred_t,
                               sample_id = meta_u$sample_id,
                               obs = as.numeric(meta_u$target_native))
    baseline[[u]] <- dplyr::mutate(res$baseline, unit = un$unit, .before = 1)
    per[[u]] <- res$effects
    rm(model); invisible(gc(verbose = FALSE))
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    say(sprintf("  [%d/%d] %s: CCC %.4f, as its run wrote | %.1f min%s", u, nrow(units),
                un$unit, res$baseline$ccc, el, .importance_eta(el / u * (nrow(units) - u))))
  }
  rm(cache); invisible(gc(verbose = FALSE))

  # ── models to curves, classes and one table
  unit_ids <- units$unit
  curves <- list(); classes <- list(); by_model <- list(); tab <- list()
  for (v in names(plan)) {
    p <- plan[[v]]
    imp <- vapply(per, function(e) e[[v]]$importance, numeric(1))
    by_model[[v]] <- tibble::tibble(unit = unit_ids, variable = v, importance = imp)
    if (identical(p$kind, "continuous")) {
      A <- do.call(rbind, lapply(per, function(e) e[[v]]$ale))
      z_all <- data$points[[ch[p$channel]]][rows_all]
      n_bin <- tabulate(cut(z_all, breaks = p$edges, include.lowest = TRUE, labels = FALSE),
                        length(p$edges) - 1L)
      ale <- colMeans(A)
      curves[[v]] <- tibble::tibble(variable = v, value = p$edges, ale = ale,
                                    ale_sd = if (nrow(A) > 1L) apply(A, 2, stats::sd) else NA_real_,
                                    n_points = c(0L, n_bin))
      rho <- if (length(ale) > 2L) suppressWarnings(stats::cor(p$edges, ale, method = "spearman")) else
        sign(ale[length(ale)] - ale[1])
      trend <- if (!is.finite(rho)) NA_character_ else if (rho > 0.9) "rises" else
        if (rho < -0.9) "falls" else "bends"
      tab[[v]] <- tibble::tibble(variable = v, kind = "continuous", n_values = length(p$edges) - 1L,
                                 trend = trend, low_at = p$edges[which.min(ale)],
                                 high_at = p$edges[which.max(ale)])
    } else {
      E  <- do.call(rbind, lapply(per, function(e) e[[v]]$effect))
      Ch <- do.call(rbind, lapply(per, function(e) e[[v]]$change))
      n_cls <- .importance_ale_class_counts(p, data, rows_all)
      classes[[v]] <- tibble::tibble(variable = v, class = p$classes, effect = colMeans(E),
                                     effect_sd = if (nrow(E) > 1L) apply(E, 2, stats::sd) else NA_real_,
                                     change = colMeans(Ch), n_points = as.integer(n_cls))
      tab[[v]] <- tibble::tibble(variable = v, kind = "categorical", n_values = length(p$classes),
                                 trend = NA_character_, low_at = NA_real_, high_at = NA_real_)
    }
  }
  by_model <- dplyr::bind_rows(by_model)
  imp_tab <- by_model %>%
    dplyr::rename(imp_model = "importance") %>%
    dplyr::group_by(.data$variable) %>%
    dplyr::summarise(importance = mean(.data$imp_model),
                     sd_models = if (dplyr::n() > 1L) stats::sd(.data$imp_model) else NA_real_,
                     n_models = dplyr::n(), .groups = "drop")
  tab <- dplyr::left_join(imp_tab, dplyr::bind_rows(tab), by = "variable")
  # Classes named in the class column rather than one per variable's channels.
  tab$class_low  <- NA_character_
  tab$class_high <- NA_character_
  for (v in names(classes)) {
    cl <- classes[[v]]
    i <- which(tab$variable == v)
    tab$class_low[i]  <- cl$class[which.min(cl$effect)]
    tab$class_high[i] <- cl$class[which.max(cl$effect)]
  }
  tab <- tab[order(-tab$importance, tab$variable), , drop = FALSE]
  tab$rank <- seq_len(nrow(tab))
  tab$target <- tab$variable
  tab$n_channels <- as.integer(lengths(vars)[tab$variable])
  tab <- dplyr::relocate(tab, "rank", "target", "variable", "kind", "n_channels")

  out <- structure(list(
    table = tab, by_model = by_model, baseline = dplyr::bind_rows(baseline),
    curves = dplyr::bind_rows(curves), classes = dplyr::bind_rows(classes),
    skipped = skipped, groups = grp, method = method, rows = rows,
    units = units[, c("unit", "kind", "fold", "seed", "role", "n_rows", "model_file")],
    run_dir = fr$run_dir, config_id = fr$config_id, window_sizes = ws_model,
    transform_name = data$transform$name %||% "none", rows_alone_share = 0, label = NULL),
    class = "dsm_importance")
  out$label <- .importance_object_label(out)
  out
}

# What each variable is, for ALE: a continuous channel with the edges of its
# bins, or a categorical with its classes. NULL for a variable with one value.
.importance_ale_plan <- function(data, vars, rows_all, bins) {
  ch <- as.character(data$store$predictors)
  is_dummy <- as.logical(data$type_table$is_dummy)
  stats::setNames(lapply(names(vars), function(v) {
    idx <- vars[[v]]
    if (length(idx) > 1L) {
      return(list(kind = "categorical", channels = idx, classes = ch[idx], binary = FALSE))
    }
    if (isTRUE(is_dummy[idx])) {
      return(list(kind = "categorical", channels = idx, classes = c("absent", "present"),
                  binary = TRUE))
    }
    z <- data$points[[ch[idx]]][rows_all]
    z <- z[is.finite(z)]
    if (length(unique(z)) < 2L) return(NULL)
    # Quantiles that are values the points hold (type 1), as Apley & Zhu's own
    # code cuts them; ties merge bins, so a variable with few values gets few.
    edges <- unique(stats::quantile(z, probs = seq(0, 1, length.out = bins + 1L), type = 1,
                                    names = FALSE))
    if (length(edges) < 2L) return(NULL)
    list(kind = "continuous", channel = idx, edges = edges)
  }), names(vars))
}

# How many of the rows are in each class, at the point.
.importance_ale_class_counts <- function(p, data, rows) {
  ch <- as.character(data$store$predictors)
  if (isTRUE(p$binary)) {
    z <- data$points[[ch[p$channels]]][rows]
    return(c(sum(z == 0, na.rm = TRUE), sum(z == 1, na.rm = TRUE)))
  }
  M <- as.matrix(data$points[rows, ch[p$channels], drop = FALSE])
  colSums(M == 1, na.rm = TRUE)
}

# One model's effects, its rows already cut and scaled. Kept apart from the
# files so the tests can hand it a model whose effects are known.
.importance_ale_unit <- function(model, inputs, plan, data, rows, scaling, obs_native,
                                 obs_transform, transform = identity, clamp = c(0, Inf),
                                 batch_size = 512L) {
  model$eval()
  n_ch <- as.integer(inputs[[1]]$shape[2])
  ch   <- as.character(data$store$predictors)
  f_x  <- .importance_forward(model, inputs, batch_size)
  base <- .importance_scores(f_x, obs_native, obs_transform, transform, clamp)
  effects <- lapply(plan, function(p) {
    if (identical(p$kind, "continuous")) {
      z <- data$points[[ch[p$channel]]][rows]
      K <- length(p$edges) - 1L
      k <- cut(z, breaks = p$edges, include.lowest = TRUE, labels = FALSE)
      inb <- !is.na(k)
      kk <- ifelse(inb, k, 1L)
      sc <- scaling$scale[p$channel]
      # In the units the model reads: a raw move divided by the channel's scale.
      lo <- ifelse(inb, (p$edges[kk] - z) / sc, 0)
      hi <- ifelse(inb, (p$edges[kk + 1L] - z) / sc, 0)
      f_lo <- .importance_forward(model, inputs, batch_size, shift = list(channel = p$channel, values = lo))
      f_hi <- .importance_forward(model, inputs, batch_size, shift = list(channel = p$channel, values = hi))
      d <- (f_hi - f_lo)[inb]
      kk <- kk[inb]
      n_k <- tabulate(kk, K)
      delta <- vapply(seq_len(K), function(j) if (n_k[j] > 0L) mean(d[kk == j]) else 0, numeric(1))
      fJ <- c(0, cumsum(delta))
      mid <- (fJ[-1] + fJ[-(K + 1L)]) / 2
      fJ <- fJ - sum(mid * n_k) / sum(n_k)
      mid <- (fJ[-1] + fJ[-(K + 1L)]) / 2
      return(list(ale = fJ, n = n_k, importance = sqrt(sum(n_k * mid^2) / sum(n_k))))
    }
    mask <- torch::torch_tensor(seq_len(n_ch) %in% p$channels,
                                dtype = torch::torch_bool())$view(c(1L, n_ch, 1L, 1L))
    eff <- vapply(seq_along(p$classes), function(j) {
      fillv <- numeric(n_ch)
      if (isTRUE(p$binary)) fillv[p$channels] <- j - 1 else fillv[p$channels[j]] <- 1
      mean(.importance_forward(model, inputs, batch_size, mask = mask, fill = fillv) - f_x)
    }, numeric(1))
    n_c <- .importance_ale_class_counts(p, data, rows)
    w <- if (sum(n_c) > 0) n_c / sum(n_c) else rep(1 / length(n_c), length(n_c))
    # CENTRED ON THE CLASSES' AVERAGE AT THESE POINTS. Every class's patch is
    # made uniform, and real patches are mixed: on the smoke all 33 FAO classes
    # came out negative, the uniform patch moving every prediction the same
    # way. That shared shift is not any class's effect; the differences between
    # classes are, and centring keeps only them. The raw move stays in `change`.
    list(effect = eff - sum(w * eff), change = eff, n = n_c,
         importance = sqrt(sum(w * (eff - sum(w * eff))^2)))
  })
  list(baseline = tibble::as_tibble(base), effects = effects, pred_t = f_x)
}

# ── the draws ─────────────────────────────────────────────────────────────────

# Where a donor may come from: NULL (anywhere), or one label per row scored.
.importance_strata <- function(within, data, rows) {
  if (is.null(within)) return(NULL)
  meta <- data$store$meta
  if (is.numeric(within) && length(within) == 1L) {
    if (!all(c("x", "y") %in% names(meta))) {
      stop("Blocks need the points' coordinates, x and y, in the store's table.", call. = FALSE)
    }
    lab <- paste(floor(meta$x[rows] / within), floor(meta$y[rows] / within), sep = "_")
    lab[!is.finite(meta$x[rows]) | !is.finite(meta$y[rows])] <- NA_character_
  } else if (is.character(within) && length(within) == 1L) {
    src <- if (within %in% names(data$points)) {
      data$points[[within]]
    } else if (within %in% names(meta)) {
      meta[[within]]
    } else {
      stop("within = \"", within, "\" is not a column of the point table or of the store's ",
           "table. Add it to the points given to dsm_load(), or pass one label per point.",
           call. = FALSE)
    }
    lab <- as.character(src[rows])
  } else {
    if (length(within) != nrow(meta)) {
      stop("within as labels needs one per point of the store (", nrow(meta), "); got ",
           length(within), ".", call. = FALSE)
    }
    lab <- as.character(within[rows])
  }
  # A POINT WITH NO LABEL HAS NO PLACE TO DRAW FROM. Pooling every unlabelled
  # point into one class would mix them wherever they are; alone, each keeps its
  # own values and is counted as such.
  na <- is.na(lab)
  lab[na] <- paste0(".none_", which(na))
  lab
}

# One permutation of the rows: within each stratum, no row its own donor.
#
# A DERANGEMENT, NOT A SHUFFLE. A row that draws itself is not perturbed, and in
# a small block that is a large share of the rows. Each fixed point is swapped
# with its neighbour in the block, which keeps the draw a permutation -- every
# value still there, once -- and leaves no fixed point behind.
.importance_donors <- function(n, strata = NULL, seed) {
  with_local_seed(seed, {
    groups <- if (is.null(strata)) list(seq_len(n)) else unname(split(seq_len(n), strata))
    donor <- seq_len(n)
    for (g in groups) {
      m <- length(g)
      if (m < 2L) next
      p <- g[sample.int(m)]
      for (i in seq_len(m)) {
        if (p[i] == g[i]) {
          j <- if (i < m) i + 1L else 1L
          tmp <- p[i]; p[i] <- p[j]; p[j] <- tmp
        }
      }
      donor[g] <- p
    }
    donor
  })
}

# The share of rows alone in their stratum: they have no donor.
.importance_alone_share <- function(strata) {
  size <- table(strata)
  sum(size[size == 1L]) / length(strata)
}

# Each channel's training mean, scaled as the model's inputs are: 0 for a
# z-scored channel, the class frequency for a dummy, the mean fraction for a
# percentage.
.importance_fill_values <- function(data, scaling, train_rows) {
  ch <- as.character(scaling$predictor)
  m  <- vapply(ch, function(p) mean(data$points[[p]][train_rows], na.rm = TRUE), numeric(1))
  (m - scaling$center) / scaling$scale
}

# ── from draws and models to the table ───────────────────────────────────────

# Larger is more important in every column: the CCC lost, the RMSE gained.
.importance_aggregate <- function(raw, baseline, metric) {
  b <- baseline[, c("unit", "ccc", "rmse", "rmse_transform")]
  names(b)[-1] <- paste0("base_", names(b)[-1])
  r <- dplyr::left_join(raw, b, by = "unit")
  r$imp_ccc            <- r$base_ccc - r$ccc
  r$imp_rmse           <- r$rmse - r$base_rmse
  r$imp_rmse_transform <- r$rmse_transform - r$base_rmse_transform
  r$imp <- r[[paste0("imp_", metric)]]

  by_model <- r %>%
    dplyr::group_by(.data$unit, .data$variable) %>%
    dplyr::summarise(importance = mean(.data$imp),
                     sd_draws = if (dplyr::n() > 1L) stats::sd(.data$imp) else NA_real_,
                     importance_ccc = mean(.data$imp_ccc),
                     importance_rmse = mean(.data$imp_rmse),
                     importance_rmse_transform = mean(.data$imp_rmse_transform),
                     .groups = "drop")

  base_mean <- mean(baseline[[metric]])
  # THE PER-MODEL VALUES UNDER A NAME OF THEIR OWN. summarise() lets each
  # expression see the columns made before it, so a summary named `importance`
  # over a column named `importance` left the spread and the count below to
  # read the one mean just made: sd_models came out NA and every variable
  # "mattered" in one model or none (tests/test_importance.R, section 5).
  tab <- by_model %>%
    dplyr::rename(imp_model = "importance") %>%
    dplyr::group_by(.data$variable) %>%
    dplyr::summarise(importance = mean(.data$imp_model),
                     sd_models = if (dplyr::n() > 1L) stats::sd(.data$imp_model) else NA_real_,
                     sd_draws = if (all(is.na(.data$sd_draws))) NA_real_ else
                       sqrt(mean(.data$sd_draws^2, na.rm = TRUE)),
                     n_models = dplyr::n(),
                     n_models_positive = sum(.data$imp_model > 0),
                     importance_ccc = mean(.data$importance_ccc),
                     importance_rmse = mean(.data$importance_rmse),
                     importance_rmse_transform = mean(.data$importance_rmse_transform),
                     .groups = "drop")
  tab$pct_of_baseline <- 100 * tab$importance / base_mean
  tab <- tab[order(-tab$importance, tab$variable), , drop = FALSE]
  tab$rank <- seq_len(nrow(tab))
  tab <- dplyr::relocate(tab, "rank", "variable")
  list(table = tab, by_model = by_model)
}

.importance_object_label <- function(x) {
  sprintf("%s | %s | %s", basename(x$run_dir),
          switch(x$rows, test = "test", map = "map points", "folds"),
          .importance_label(x$method))
}

#' Print a variable importance.
#'
#' @param x   A `dsm_importance`, from [dsm_importance()].
#' @param n   How many variables to show.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.dsm_importance <- function(x, n = 20L, ...) {
  m <- x$method
  what <- c(ccc = "the drop in CCC", rmse = "the rise in RMSE",
            rmse_transform = "the rise in RMSE in the trained space")[[m$metric]]
  shown <- c(ccc = "CCC", rmse = "RMSE", rmse_transform = "RMSE (trained space)")[[m$metric]]
  b <- x$baseline[[m$metric]]
  # The label names the method; a title naming it again read "SHAP importance
  # -- SHAP by expected gradients".
  cat("\nImportance -- ", .importance_label(m), "\n", sep = "")
  cat(sprintf("  %s of '%s' (%s) | %d model(s) | %s point(s) each\n",
              switch(x$rows, test = "test set", map = "map points", "fold validation rows"),
              basename(x$run_dir), x$config_id, nrow(x$units),
              paste(unique(range(x$units$n_rows)), collapse = " to ")))
  if (identical(m$kind, "shap")) {
    .importance_print_shap(x, n)
    return(invisible(x))
  }
  if (identical(m$kind, "ale")) {
    .importance_print_ale(x, n)
    return(invisible(x))
  }
  if (identical(m$kind, "sage")) {
    .importance_print_sage(x, n)
    return(invisible(x))
  }
  cat(sprintf("  ranked by %s; the models' own %s: %.4f (%.4f to %.4f)\n", what, shown,
              mean(b), min(b), max(b)))
  if (identical(m$kind, "context")) {
    .importance_print_context(x, n)
    return(invisible(x))
  }
  if (identical(m$kind, "refit")) {
    .importance_print_refit(x, n)
    return(invisible(x))
  }
  cat("  ", .importance_groups_line(x$groups), "\n", sep = "")
  cat(strrep("-", 72), "\n")
  show <- utils::head(x$table, n)
  print(tibble::tibble(rank = show$rank, variable = show$variable, channels = show$n_channels,
                       importance = signif(show$importance, 4),
                       sd_models = signif(show$sd_models, 3),
                       in_models = sprintf("%d/%d", show$n_models_positive, show$n_models),
                       pct = round(show$pct_of_baseline, 1)), n = Inf)
  if (nrow(x$table) > n) cat(sprintf("  ... %d more in $table\n", nrow(x$table) - n))
  share <- mean(x$rows_alone_share)
  if (is.finite(share) && share > 0) {
    cat(sprintf("  %.1f%% of the rows were alone in their block or class and kept their values.\n",
                100 * share))
  }
  # WHAT THE NUMBER IS. Said every time, because it is the misreading every
  # reader of an importance table makes: a variable the model does not lean on
  # is not therefore uninformative.
  cat("  Reliance of these models on each variable, not its necessity: a variable the\n")
  cat("  model finds elsewhere too can score low and still carry the signal.\n")
  invisible(x)
}

# The space SHAP values and ALE curves are in: the network's own output.
.importance_units_text <- function(name) {
  if (identical(name, "log1p")) {
    "log1p, as the network predicts: +0.1 raises the predicted value by about 10%"
  } else if (is.null(name) || name %in% c("none", "identity")) {
    "the target's own units"
  } else {
    paste0(name, ", as the network predicts")
  }
}

# The body of a SAGE print.
.importance_print_sage <- function(x, n) {
  b <- x$baseline$ccc
  l <- x$losses
  tn <- x$transform_name %||% "none"
  space <- if (tn %in% c("none", "identity")) "the target's own units" else
    paste0(tn, ", the space the models were trained in")
  cat(sprintf("  the models' own CCC: %.4f (%.4f to %.4f), each as its run wrote\n",
              mean(b), min(b), max(b)))
  cat(sprintf("  loss (%s, %s): the models' %.4g; the mean prediction's, no variable known, %.4g\n",
              x$method$loss, space, mean(l$loss_model), mean(l$loss_mean_prediction)))
  cat(sprintf("  SAGE shares the %.4g between: %s\n",
              mean(l$loss_mean_prediction - l$loss_model), .importance_groups_line(x$groups)))
  cat(strrep("-", 72), "\n")
  show <- utils::head(x$table, n)
  print(tibble::tibble(rank = show$rank, variable = show$variable, channels = show$n_channels,
                       sage = signif(show$importance, 3), sd_models = signif(show$sd_models, 3),
                       in_models = sprintf("%d/%d", show$n_models_positive, show$n_models),
                       share = round(100 * show$share, 1)), n = Inf)
  if (nrow(x$table) > n) cat(sprintf("  ... %d more in $table\n", nrow(x$table) - n))
  cat("  sage: the loss a variable takes away, shared fairly among variables that carry\n")
  cat("  the same information. Below zero: it costs skill. share: of all explained, in %.\n")
  invisible(NULL)
}

# The body of an ALE's print.
.importance_print_ale <- function(x, n) {
  b <- x$baseline$ccc
  cat(sprintf("  the models' own CCC: %.4f (%.4f to %.4f), each as its run wrote\n",
              mean(b), min(b), max(b)))
  cat("  effects in ", .importance_units_text(x$transform_name), "\n", sep = "")
  k <- x$table$kind
  cat(sprintf("  %d variable(s): %d continuous (curves in $curves), %d categorical (classes in $classes)%s\n",
              length(k), sum(k == "continuous"), sum(k == "categorical"),
              if (length(x$skipped)) sprintf("; %d with one value, skipped", length(x$skipped)) else ""))
  cat(strrep("-", 72), "\n")
  show <- utils::head(x$table, n)
  cont <- show$kind == "continuous"
  # Each value on its own, three figures, never in e-notation: format() over the
  # column gave 13.6 beside 1840 as 1.36e+01, and "%.3g" still gave 1.91e+03 --
  # both cut to "1.36..." by a tibble narrowing the column. A plain data frame
  # prints them whole. A class without its set's prefix: "ferralsols".
  num <- function(v) ifelse(is.na(v), NA_character_,
                            formatC(v, digits = 3, format = "fg", flag = ""))
  cls <- function(c, v) vapply(seq_along(c), function(i) {
    if (is.na(c[i])) return(NA_character_)
    pre <- paste0(v[i], "_")
    if (startsWith(c[i], pre)) substring(c[i], nchar(pre) + 1L) else c[i]
  }, character(1))
  print(data.frame(rank = show$rank, variable = show$variable,
                   kind = ifelse(cont, "cont", "class"),
                   importance = signif(show$importance, 3), sd_models = signif(show$sd_models, 3),
                   trend = ifelse(is.na(show$trend), "", show$trend),
                   low = ifelse(cont, num(show$low_at), cls(show$class_low, show$variable)),
                   high = ifelse(cont, num(show$high_at), cls(show$class_high, show$variable)),
                   stringsAsFactors = FALSE), row.names = FALSE)
  if (nrow(x$table) > n) cat(sprintf("  ... %d more in $table\n", nrow(x$table) - n))
  cat("  importance: the spread of the effect over the points -- flat, none. trend: the\n")
  cat("  curve rises or falls along the range, or bends; low and high: where the effect\n")
  cat("  is least and most -- a value of the variable, or a class.\n")
  if (any(k == "categorical")) {
    cat("  A class effect: the whole patch made that class, against the classes' average\n")
    cat("  at these points ($classes$change keeps the move before centring). It asks also\n")
    cat("  where the class does not occur: $classes$n_points says where it does.\n")
  }
  invisible(NULL)
}

# The body of a SHAP importance's print.
.importance_print_shap <- function(x, n) {
  b  <- x$baseline$ccc
  cp <- x$completeness
  is_eg <- identical(x$method$estimator, "expected_gradients")
  if (identical(x$rows, "map")) {
    # A map point has no score to reproduce: what was checked is the reading.
    cat(sprintf("  %s point(s) explained, %s without a whole patch | the store's patches cut again identically (worst %.1e)\n",
                format(nrow(x$points), big.mark = ","), format(x$n_dropped, big.mark = ","),
                x$probe_worst))
  } else {
    cat(sprintf("  the models' own CCC: %.4f (%.4f to %.4f), each as its run wrote\n",
                mean(b), min(b), max(b)))
  }
  cat("  values in ", .importance_units_text(x$transform_name), "\n", sep = "")
  is_kernel <- identical(x$method$estimator, "kernel")
  cat(sprintf("  they add up: each point's values sum to its prediction less the %s, %s\n",
              if (is_eg || is_kernel) "background's mean" else "baseline's",
              if (is_eg) sprintf("on average (sampling noise %.1f%% per point)", 100 * mean(cp$rel_noise)) else
                if (is_kernel) "exactly (the Shapley values of the coalitions)" else
                  sprintf("to within %.2f%%", 100 * max(cp$rel_noise))))
  cat("  ", .importance_groups_line(x$groups), "\n", sep = "")
  cat(strrep("-", 72), "\n")
  show <- utils::head(x$table, n)
  print(tibble::tibble(rank = show$rank, variable = show$variable, channels = show$n_channels,
                       mean_abs = signif(show$importance, 3), sd_models = signif(show$sd_models, 3),
                       share = round(100 * show$share, 1), direction = round(show$direction, 2)),
        n = Inf)
  if (nrow(x$table) > n) cat(sprintf("  ... %d more in $table\n", nrow(x$table) - n))
  cat("  mean_abs: how much a variable MOVES the predictions -- not how much it improves\n")
  cat("  them, which permutation_importance() measures. share: of the sum, in %.\n")
  cat("  direction: Spearman between the variable's value at the point and its SHAP\n")
  cat("  value -- +1, higher values raise the prediction; NA for a group of channels.\n")
  cat("  $points: every point's values, the map of what drives the prediction where;\n")
  if (!is.null(x$patch)) cat("  $patch and $pixels: where in the patch the attribution sits.\n")
  # THE NOISE IS PER POINT. Averaged over the points it washes out of the table
  # above; a map of single points keeps it, and only more samples lower it.
  if (is_eg && mean(cp$rel_noise) > 0.05) {
    cat(sprintf(paste0("  Per point, the sampling noise is %.0f%% of each prediction's distance from the\n",
                       "  reference: read $points as a map with that in mind. It falls as 1/sqrt(samples):\n",
                       "  %d samples would bring it near %.0f%%.\n"),
                100 * mean(cp$rel_noise), 4L * x$method$samples, 50 * mean(cp$rel_noise)))
  }
  invisible(NULL)
}

# The body of a context importance's print: by ring, per variable, by window.
.importance_print_context <- function(x, n) {
  m <- x$method
  t <- x$table
  in_models <- function(r) sprintf("%d/%d", r$n_models_positive, r$n_models)
  cat(sprintf("  windows read: %s\n", paste(patch_window_key(x$window_sizes), collapse = " + ")))
  cat(strrep("-", 72), "\n")

  if (identical(m$by, "window")) {
    t <- t[order(t$window), , drop = FALSE]
    print(tibble::tibble(window = t$target, importance = signif(t$importance, 4),
                         sd_models = signif(t$sd_models, 3), in_models = in_models(t),
                         pct = round(t$pct_of_baseline, 1)), n = Inf)
    if (is.null(x$gate)) {
      cat("  No gate:", if (length(x$window_sizes) < 2L) "one window, one branch." else
        "the two branches are concatenated, not weighed.", "\n")
    } else {
      gm <- vapply(split(x$gate$gate, x$gate$unit), mean, numeric(1))
      keys <- patch_window_key(x$window_sizes)
      cat(sprintf("  The gate's weight on the %s branch: %.2f on average (%.2f to %.2f between models; %.2f to %.2f for 90%% of the points)\n",
                  keys[1], mean(gm), min(gm), max(gm), stats::quantile(x$gate$gate, 0.05),
                  stats::quantile(x$gate$gate, 0.95)))
      cat(sprintf("  Embedding length, %s branch over %s: %.2f (median of the points)\n",
                  keys[1], keys[2], stats::median(x$gate$norm_1 / x$gate$norm_2)))
      cat("  The gate is how the model mixes its branches, not how much each carries: read\n")
      cat("  it beside the cost of permuting each window. $gate holds it point by point.\n")
    }
    return(invisible(NULL))
  }

  if (isTRUE(m$per_variable) && all(c("centre", "context") %in% x$targets$band)) {
    cols <- c("variable", "importance", "per_pixel", "n_models_positive", "n_models")
    j <- merge(t[t$band == "centre", cols], t[t$band == "context", cols], by = "variable",
               suffixes = c("_centre", "_context"))
    tot <- j$importance_centre + j$importance_context
    # PER PIXEL, NOT AS A SHARE OF THE TOTAL. The first version printed the
    # context's share of the two costs, and on the smoke's 3 + 9 windows every
    # variable came out near 0.9: the context is 80 pixels against the centre's
    # one, and a model that weighs every pixel alike would show 80/81 -- the
    # area, not the reading. What one pixel of the context costs against the
    # centre pixel is the comparison the area does not decide.
    j$px_ratio <- ifelse(j$per_pixel_centre > 0 & j$per_pixel_context >= 0,
                         j$per_pixel_context / j$per_pixel_centre, NA_real_)
    j <- utils::head(j[order(-tot), , drop = FALSE], n)
    print(tibble::tibble(variable = j$variable, centre = signif(j$importance_centre, 3),
                         context = signif(j$importance_context, 3),
                         px_ratio = round(j$px_ratio, 2),
                         in_models = sprintf("%d/%d", j$n_models_positive_context, j$n_models_context)),
          n = Inf)
    cat(sprintf("  px_ratio: what one pixel of the context (%d pixels) costs against the centre\n",
                as.integer(x$targets$pixels[x$targets$band == "context"][1])))
    cat("  pixel -- 0, the variable is read at the point alone; near 1, every pixel alike.\n")
    cat("  A smooth variable can score low at the centre because its rim repeats it.\n")
    if (length(x$window_sizes) > 1L) {
      # TWO WINDOWS READ THE CENTRE TWICE. The 3x3 and the 9x9 both hold the
      # point and its first ring, the 9x9 alone the rings beyond: part of the
      # centre's higher cost per pixel is the architecture's, not the variable's.
      cat(sprintf("  With %s, the inner %dx%d is read by every branch and the rest by fewer:\n",
                  paste(patch_window_key(x$window_sizes), collapse = " + "),
                  min(x$window_sizes), min(x$window_sizes)))
      cat("  part of the centre's weight per pixel is the architecture's own emphasis.\n")
    }
    cat("  in_models: of the context. $table has every variable and band.\n")
    return(invisible(NULL))
  }

  if (isTRUE(m$per_variable)) {
    show <- utils::head(t, n)
    print(tibble::tibble(target = show$target, importance = signif(show$importance, 4),
                         per_pixel = signif(show$per_pixel, 3), in_models = in_models(show)), n = Inf)
    return(invisible(NULL))
  }

  t <- t[match(x$targets$target, t$target), , drop = FALSE]
  print(tibble::tibble(band = t$band, pixels = t$pixels, importance = signif(t$importance, 4),
                       per_pixel = signif(t$per_pixel, 3), sd_models = signif(t$sd_models, 3),
                       in_models = in_models(t)), n = Inf)
  # AREA IS A CONFOUND, as R/occlusion.R learned: ring d holds 8d pixels, and a
  # wide ring costs more for its area alone. And a costly context means the
  # network READS the neighbourhood, which smooth covariates make a near-copy of
  # the centre -- not that the neighbourhood adds what the centre lacks.
  cat("  Ring d holds 8d pixels: compare rings by per_pixel, not by their totals.\n")
  cat("  A costly context says this network reads the neighbourhood, not that the\n")
  cat("  neighbourhood adds what the centre lacks: at 250 m the rim is close to a copy.\n")
  invisible(NULL)
}

# ── the importance as the AOA's weights ───────────────────────────────────────

#' One weight per channel, for the area of applicability, from an importance.
#'
#' The dissimilarity index measures how far a pixel is from the training data,
#' one axis per channel. Unweighted, a channel the model ignores counts as much
#' as the one it leans on, and a pixel unlike the training data in an ignored
#' channel falls outside the area of applicability for nothing. Meyer & Pebesma
#' (2021) weight each axis by the predictor's importance; this gives each
#' channel its variable's importance -- every channel of a one-hot set, or of a
#' group of yours, its variable's -- and a negative importance, noise around
#' zero, weight zero.
#'
#' @param x A `dsm_importance` with one value per variable: by
#'   [permutation_importance()], [shap_importance()], [sage_importance()],
#'   [refit_importance()] or [ale_effect()].
#' @return A named numeric vector, one weight per channel in the store's order,
#'   for `dsm_predict(aoa_weights = )` or `aoa_reference(weights = )`.
#' @export
importance_weights <- function(x) {
  if (!inherits(x, "dsm_importance")) {
    stop("x must be a dsm_importance, from dsm_importance().", call. = FALSE)
  }
  if (identical(x$method$kind, "context")) {
    stop("A context importance has one value per ring or window, not per variable: ",
         "take the weights from a permutation, SHAP, SAGE, refit or ALE importance.",
         call. = FALSE)
  }
  g <- x$groups
  w <- x$table$importance[match(g$variable, x$table$variable)]
  # A variable the importance left out (ALE skips one with a single value)
  # carries no weight, as one it found worthless.
  w[is.na(w)] <- 0
  w <- pmax(w, 0)
  if (sum(w) == 0) {
    stop("Every importance is zero or below: there is no weighting to take from it.",
         call. = FALSE)
  }
  stats::setNames(w, g$channel)
}

# ── several importances side by side ─────────────────────────────────────────

#' Several importances, side by side.
#'
#' Aligns importances computed by different methods, variants, rows or runs --
#' permutation over all rows against within blocks, the test set against the
#' folds, the spatial design against the random one -- by what each perturbed
#' (the variable, for a permutation), and says how far their rankings agree. Where they disagree is the reading: a
#' variable that keeps its importance within blocks discriminates locally; one
#' that loses it carries a regional gradient.
#'
#' @param ...    Two or more `dsm_importance`, named or not. Names become the
#'   labels; unnamed ones are labelled by run, rows and method.
#' @param n      How many variables the print shows.
#' @return An `importance_comparison`: `table` (one row per target: each
#'   importance, its rank, and its share of that importance's largest),
#'   `agreement` (Spearman's correlation between the rankings, over the
#'   variables they share), `labels`, and for two SHAP importances of the same
#'   points `points_agreement` (per variable, the correlation of their values
#'   point by point and their mean difference against the second's size).
#' @export
compare_importance <- function(..., n = 20L) {
  xs <- list(...)
  if (length(xs) < 2L) stop("compare_importance() needs two importances or more.", call. = FALSE)
  if (!all(vapply(xs, inherits, logical(1), "dsm_importance"))) {
    stop("Every argument must be a dsm_importance, from dsm_importance().", call. = FALSE)
  }
  labs <- names(xs) %||% rep("", length(xs))
  auto <- vapply(xs, function(x) x$label %||% "", character(1))
  labs[!nzchar(labs)] <- auto[!nzchar(labs)]
  if (anyDuplicated(labs)) labs <- make.unique(labs, sep = " #")
  cols <- paste0("i", seq_along(xs))

  tab <- NULL
  for (i in seq_along(xs)) {
    t <- xs[[i]]$table
    top <- max(t$importance)
    # By what was perturbed: a variable for a permutation, a band or window or
    # "variable | band" for a context -- rows of two methods meet only where
    # they perturbed the same thing.
    key <- if ("target" %in% names(t)) t$target else t$variable
    one <- tibble::tibble(target = key, importance = t$importance, rank = t$rank,
                          share = if (is.finite(top) && top > 0) t$importance / top else NA_real_)
    names(one)[-1] <- paste0(names(one)[-1], "_", cols[i])
    tab <- if (is.null(tab)) one else dplyr::full_join(tab, one, by = "target")
  }
  rk <- as.matrix(tab[, paste0("rank_", cols), drop = FALSE])
  tab$mean_rank <- rowMeans(rk, na.rm = TRUE)
  tab <- tab[order(tab$mean_rank), , drop = FALSE]

  imp <- as.matrix(tab[, paste0("importance_", cols), drop = FALSE])
  agree <- suppressWarnings(stats::cor(imp, method = "spearman", use = "pairwise.complete.obs"))
  dimnames(agree) <- list(labs, labs)
  # What each column is ranked by: a SHAP importance by mean |SHAP|, whatever
  # the score its frame carries for the models' own check.
  metrics <- vapply(xs, function(x) {
    switch(x$method$kind %||% "permutation", shap = "mean |SHAP|", ale = "ALE spread",
           sage = "SAGE (loss explained)", refit = paste("refit,", x$method$metric),
           x$method$metric)
  }, character(1))
  # POINT BY POINT, for two SHAP importances of the same points: per variable,
  # the correlation of their values over the points both explain, and how far
  # apart they are against the second's size. Two estimators share each
  # point's total; where they part, they split it differently -- interactions
  # between the variables, or the sampling noise of expected gradients.
  pa <- NULL
  shp <- which(vapply(xs, function(x) identical(x$method$kind, "shap") && !is.null(x$points),
                      logical(1)))
  if (length(shp) >= 2L) {
    a <- xs[[shp[1]]]$points
    b <- xs[[shp[2]]]$points
    key <- function(p) if ("sample_id" %in% names(p)) as.character(p$sample_id) else
      sprintf("%.10g_%.10g", p$x, p$y)
    ka <- key(a); kb <- key(b)
    common <- intersect(ka, kb)
    if (length(common) >= 3L) {
      vs <- intersect(xs[[shp[1]]]$table$variable, xs[[shp[2]]]$table$variable)
      pa <- dplyr::bind_rows(lapply(vs, function(v) {
        va <- a[[v]][match(common, ka)]
        vb <- b[[v]][match(common, kb)]
        sd_ok <- stats::sd(va) > 0 && stats::sd(vb) > 0
        tibble::tibble(variable = v, r = if (sd_ok) stats::cor(va, vb) else NA_real_,
                       mean_abs_diff = mean(abs(va - vb)),
                       rel_diff = mean(abs(va - vb)) / max(mean(abs(vb)), .Machine$double.eps))
      }))
      attr(pa, "pair") <- cols[shp[1:2]]
      attr(pa, "n_points") <- length(common)
    }
  }
  structure(list(table = tab, agreement = agree, labels = stats::setNames(labs, cols),
                 metrics = stats::setNames(metrics, cols), points_agreement = pa,
                 n = as.integer(n)),
            class = "importance_comparison")
}

#' Print a comparison of importances.
#'
#' @param x   An `importance_comparison`, from [compare_importance()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.importance_comparison <- function(x, ...) {
  cat("\nImportance compared --", length(x$labels), "importance(s)\n")
  for (i in seq_along(x$labels)) {
    cat(sprintf("  %s  %s  (ranked by %s)\n", names(x$labels)[i], x$labels[[i]], x$metrics[[i]]))
  }
  cat("\n  Agreement between the rankings (Spearman):\n")
  ag <- x$agreement
  dimnames(ag) <- list(names(x$labels), names(x$labels))
  print(round(ag, 2))
  cat("\n")
  show <- utils::head(x$table, x$n)
  out <- tibble::tibble(target = show$target)
  for (col in names(x$labels)) out[[paste0("rank_", col)]] <- show[[paste0("rank_", col)]]
  for (col in names(x$labels)) out[[paste0("share_", col)]] <- round(show[[paste0("share_", col)]], 2)
  print(out, n = Inf)
  if (nrow(x$table) > x$n) cat(sprintf("  ... %d more in $table\n", nrow(x$table) - x$n))
  cat("  share: each importance over the largest of the same column.\n")
  pa <- x$points_agreement
  if (!is.null(pa)) {
    pr <- attr(pa, "pair")
    cat(sprintf("\n  SHAP point by point, %s against %s, over %d point(s) both explain:\n",
                pr[1], pr[2], attr(pa, "n_points")))
    print(data.frame(variable = pa$variable, r = round(pa$r, 3), rel_diff = round(pa$rel_diff, 3)),
          row.names = FALSE)
    cat("  Both split each point's same total. Where r is low or rel_diff high they split\n")
    cat("  it differently: the variables interact (or expected gradients' noise shows).\n")
  }
  invisible(x)
}
