# ── The model registry ────────────────────────────────────────────────────────
#
# WHY THIS EXISTS.
#
# The project's central claim is that a convolution over a neighbourhood of
# predictors beats the same predictors read at a point. Until now that claim
# had nothing to be measured against: the pipeline could only train one kind of
# model, so "the CNN reached CCC 0.62" was a number with no scale on it.
#
# A baseline fixes that, and a Random Forest is the useful baseline precisely
# because it is NOT a torch model. It has no epochs, no early stopping, no
# learning rate, no device; it consumes a matrix, not a 4-D tensor. Building
# the registry around a model that shares nothing with the CNN is what keeps
# the abstraction honest -- an interface designed against one implementation
# only ever describes that implementation.
#
# WHAT A MODEL DECLARES.
#
#   name          how it is asked for
#   input         "patches" (N x C x H x W tensors) or "table" (N x P matrix)
#   fit           function(x, y, cfg, ...) -> fitted object
#   predict       function(object, x, ...) -> numeric vector, length nrow(x)
#   default_grid  function(tune_length, seed, x, y) -> tibble of configs,
#                 or NULL
#   count_params  function(object) -> integer, for one_se()'s complexity rule
#
# `input` is the field that matters. It says which VIEW of the fold cache the
# model consumes, and the runner uses it to decide what to build -- so adding a
# model never means editing the runner.
#
# WHAT THIS IS NOT.
#
# It is not caret's `modelInfo`. caret carries `library`, `type`, `sort`,
# `loop`, `levels`, `oob`, `varImp` and more, because it supports 200+ models
# for classification and regression across resampling schemes it also owns.
# This carries five fields because a sixth would be a field nothing reads.
# Fields get added when a model needs one, not in anticipation.

# ── the registry itself ───────────────────────────────────────────────────────
#
# An environment, not a list in the global workspace: registering is a side
# effect, and a side effect on an object the user can overwrite by assigning a
# variable of the same name is a bug waiting for an unlucky script.
.model_registry <- new.env(parent = emptyenv())

#' Describe a model the framework can fit.
#'
#' @param name         Unique identifier, e.g. "rf", "mlp", "cnn".
#' @param input        "table" or "patches" -- which view of the fold cache the
#'   model consumes.
#' @param fit          function(x, y, cfg, ...) returning a fitted object.
#' @param predict      function(object, x, ...) returning a numeric vector.
#' @param default_grid function(tune_length, seed, x, y) returning a tibble
#'   with a config_id column, or NULL if the model has nothing to tune.
#'
#'   x AND y ARE THE REAL TRAINING DATA OF THE FIRST FOLD, and they are in the
#'   signature because some grid generators need them. mtry is a fraction of
#'   ncol(x); glmnet's lambda path is computed from the VALUES. A grid built
#'   against a synthetic matrix of the right width is correct for the first
#'   kind and quietly wrong for the second, so the framework hands over the
#'   real thing and a generator takes what it needs.
#'
#'   A generator that needs neither still has to accept them -- an argument it
#'   ignores costs nothing, and a signature that varies per model is a
#'   signature the runner cannot call.
#' @param count_params function(object) returning the number of free
#'   parameters, used as the complexity axis in one_se(). NULL means the model
#'   cannot report it and one_se() falls back to its other rule.
#' @param description  One line, printed by list_models().
#' @return A model_spec.
model_spec <- function(name, input, fit, predict,
                       default_grid = NULL, count_params = NULL,
                       description = "") {
  input <- match.arg(input, c("table", "patches"))

  # Checked here rather than at fit time: a typo in an argument name is
  # otherwise found after the fold cache has been built, minutes in.
  if (!is.function(fit) || !is.function(predict)) {
    stop("fit and predict must both be functions.", call. = FALSE)
  }
  need_fit <- c("x", "y", "cfg")
  if (!all(need_fit %in% names(formals(fit)))) {
    stop("A model's fit() must accept (", paste(need_fit, collapse = ", "),
         "); '", name, "' accepts (",
         paste(names(formals(fit)), collapse = ", "), ").", call. = FALSE)
  }
  if (!all(c("object", "x") %in% names(formals(predict)))) {
    stop("A model's predict() must accept (object, x); '", name,
         "' accepts (", paste(names(formals(predict)), collapse = ", "), ").",
         call. = FALSE)
  }

  structure(
    list(name = name, input = input, fit = fit, predict = predict,
         default_grid = default_grid, count_params = count_params,
         description = description),
    class = "model_spec"
  )
}

#' Add a model to the registry.
#'
#' @param spec From model_spec().
#' @param overwrite FALSE (default) refuses to replace an existing name. A
#'   silent replacement is how two different models come to answer to the same
#'   name in the same session, and the results carry no mark of which ran.
register_model <- function(spec, overwrite = FALSE) {
  stopifnot(inherits(spec, "model_spec"))
  if (!overwrite && exists(spec$name, envir = .model_registry, inherits = FALSE)) {
    stop("A model named '", spec$name, "' is already registered. Pass ",
         "overwrite = TRUE if replacing it is what you mean.", call. = FALSE)
  }
  assign(spec$name, spec, envir = .model_registry)
  invisible(spec)
}

#' Fetch a registered model.
#'
#' @param name Registered name.
get_model <- function(name) {
  if (!exists(name, envir = .model_registry, inherits = FALSE)) {
    stop("No model named '", name, "'. Registered: ",
         paste(sort(ls(.model_registry)), collapse = ", "), call. = FALSE)
  }
  get(name, envir = .model_registry, inherits = FALSE)
}

#' Every registered model, as a table.
list_models <- function() {
  nms <- sort(ls(.model_registry))
  if (length(nms) == 0L) {
    return(tibble::tibble(name = character(0), input = character(0),
                          tunable = logical(0), description = character(0)))
  }
  specs <- lapply(nms, get_model)
  tibble::tibble(
    name        = nms,
    input       = vapply(specs, function(s) s$input, character(1)),
    tunable     = vapply(specs, function(s) !is.null(s$default_grid), logical(1)),
    description = vapply(specs, function(s) s$description, character(1))
  )
}

#' @export
print.model_spec <- function(x, ...) {
  cat("<model_spec> ", x$name, "\n", sep = "")
  cat("  input       : ", x$input, "\n", sep = "")
  cat("  tunable     : ", if (is.null(x$default_grid)) "no" else "yes", "\n", sep = "")
  cat("  reports size: ", if (is.null(x$count_params)) "no" else "yes", "\n", sep = "")
  if (nzchar(x$description)) cat("  ", x$description, "\n", sep = "")
  invisible(x)
}
