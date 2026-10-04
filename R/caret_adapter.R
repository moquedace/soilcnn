# ── Borrowing caret's model library ───────────────────────────────────────────
#
# THE ARGUMENT FOR DOING THIS.
#
# caret's `modelInfo` objects are the expensive part of supporting many models:
# for ~230 methods they already carry the parameter NAMES, a grid generator
# that knows sensible ranges, the fit and predict closures, and which package
# to load. Hand-writing a model_spec per family means re-deriving all of that,
# once per family, and getting the ranges slightly wrong each time.
#
# So a caret method becomes a model_spec here, and everything downstream --
# folds, seeds, scaling, metrics, noise floor, one_se() -- is unchanged.
#
# WHAT caret IS NOT ALLOWED TO DO: RESAMPLE.
#
# The fold plan carves the test set and the folds by ONE criterion, with block
# structure and a Chebyshev buffer. caret's trainControl can express arbitrary
# folds through `index`/`indexOut`, so this is not impossible -- it is
# undesirable. It would mean translating our plan into caret's index lists,
# reading our metrics back out through a summaryFunction, and having two
# objects that both believe they own the resampling. The seed noise floor and
# one_se() are already ours and already tested.
#
# Hence `trainControl(method = "none")` with a ONE-ROW tuneGrid: caret is used
# as a fit/predict adapter and nothing else. Our loop stays in charge.
#
# THE THREE SETTINGS THAT ARE NOT NEGOTIABLE, AND WHY.
#
#   returnData = FALSE   train() otherwise stores the training data inside the
#                        fitted object. At 30k rows x 360 features that is
#                        ~85 MB per unit, and a run holds many units.
#   preProcess unset     the fold cache is ALREADY scaled, on this fold's
#                        training rows. A second scaling inside caret would be
#                        fitted on the same rows and be harmless -- until
#                        someone changes one of the two, and then the network
#                        and the baseline are standardised differently while
#                        every table still says they were compared.
#   allowParallel FALSE  the parallelism belongs to the model (ranger's
#                        threads, xgboost's nthread). Nesting a foreach backend
#                        inside it oversubscribes the cores and runs slower.
#
# COST TO BE HONEST ABOUT: caret Depends on ggplot2 and lattice and Imports
# recipes, plyr, pROC, ModelMetrics, reshape2, foreach. That is a heavy tree
# for a package someone installs to fit a CNN. It is therefore a SUGGESTS: this
# file loads with the package like any other, and nothing fails until
# caret_spec() is actually called.

.need_caret <- function() {
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop("caret_spec() needs the caret package.\n",
         "  install.packages(\"caret\")\n",
         "The rest of the framework does not: caret is optional, and only ",
         "models borrowed from it require it.", call. = FALSE)
  }
}

#' What caret knows about one of its methods.
#'
#' @param method A caret method name, e.g. "ranger", "xgbTree", "glmnet".
#' @return The modelInfo list.
#' @noRd
caret_model_info <- function(method) {
  .need_caret()
  mi <- caret::getModelInfo(method, regex = FALSE)
  if (length(mi) == 0L) {
    stop("caret has no method called '", method, "'. ",
         "Names are case-sensitive; see names(caret::getModelInfo()).",
         call. = FALSE)
  }
  info <- mi[[1]]
  if (!"Regression" %in% info$type) {
    stop("caret method '", method, "' does not do regression (it declares: ",
         paste(info$type, collapse = ", "), "). This framework predicts a ",
         "continuous target.", call. = FALSE)
  }
  info
}

#' A grid for a caret method, drawn by caret's own generator.
#'
#' @param method      caret method name.
#' @param tune_length How many values per parameter caret should propose.
#' @param seed        Draw seed -- caret's random search is random.
#' @param x,y         The REAL training data. caret's generators use it: mtry
#'   from ncol(x), and glmnet's lambda path from the values themselves.
#' @param search      "grid" (caret's regular lattice, deduplicated) or
#'   "random". Random is usually the better spend at equal budget once more
#'   than two parameters are in play -- the same reason make_tune_grid() draws
#'   rather than crosses.
#' @noRd
caret_grid <- function(method, tune_length = 6L, seed = 42L, x, y,
                       search = c("grid", "random")) {
  search <- match.arg(search)
  info   <- caret_model_info(method)

  g <- with_local_seed(seed, {
    info$grid(x = as.data.frame(x), y = y, len = as.integer(tune_length),
              search = search)
  })
  g <- unique(as.data.frame(g, stringsAsFactors = FALSE))

  # caret's grid generator is free to return fewer rows than asked for -- a
  # model with one parameter and three sensible values returns three whatever
  # tune_length says. Reporting that is better than silently running a smaller
  # grid than the script asked for.
  if (nrow(g) < tune_length) {
    message("caret's generator for '", method, "' returned ", nrow(g),
            " config(s) for tune_length = ", tune_length,
            " -- that is the number of distinct settings it has.")
  }

  tibble::as_tibble(g) %>%
    dplyr::mutate(config_id = sprintf("%s_%03d", method, dplyr::row_number())) %>%
    dplyr::relocate(config_id)
}

#' Turn a caret method into a model_spec.
#'
#' @param method  caret method name, e.g. "ranger", "xgbTree", "cubist".
#' @param name    Name to register it under. Defaults to the method name.
#' @param search  Passed to caret_grid() for the default grid.
#' @param ...     Extra arguments forwarded to every caret::train() call --
#'   this is where a model's own arguments go (num.threads, nthread, ...).
#' @return A model_spec with input == "table".
#' @examplesIf requireNamespace("caret", quietly = TRUE)
#' caret_spec("rf")
#' @export
caret_spec <- function(method, name = method, search = c("grid", "random"),
                       ...) {
  .need_caret()
  search    <- match.arg(search)
  info      <- caret_model_info(method)
  # The parameter names caret expects in the tuneGrid. Our config rows carry
  # config_id and whatever else a script attached; passing those through would
  # make train() reject the grid.
  par_names <- as.character(info$parameters$parameter)
  extra     <- list(...)

  model_spec(
    name  = name,
    input = "table",
    description = paste0("caret::train(method = \"", method, "\") -- ",
                         info$label),

    fit = function(x, y, cfg, ...) {
      tg <- as.data.frame(cfg[, intersect(names(cfg), par_names), drop = FALSE])
      if (length(par_names) > 0L && ncol(tg) != length(par_names)) {
        stop("The config row is missing caret parameter(s) for '", method,
             "': ", paste(setdiff(par_names, names(tg)), collapse = ", "),
             "\nBuild the grid with caret_grid(\"", method, "\", ...), which ",
             "produces exactly the columns the method declares.",
             call. = FALSE)
      }

      args <- c(list(
        x = as.data.frame(x), y = y, method = method,
        # See the header: caret fits ONE model here. The folds are ours.
        trControl = caret::trainControl(method = "none", returnData = FALSE,
                                        allowParallel = FALSE),
        tuneGrid  = tg
      ), extra)

      fit <- do.call(caret::train, args)
      structure(list(fit = fit, method = method, features = colnames(x)),
                class = "caret_fitted")
    },

    predict = function(object, x, ...) {
      # The column ORDER is the contract. caret's predict matches by NAME for
      # data frames, so a reordered table would in fact work -- and would then
      # be the one place in this framework where order silently does not
      # matter, which is worse than a consistent rule.
      if (!identical(colnames(x), object$features)) {
        stop("The prediction table's columns differ from the training ",
             "table's.", call. = FALSE)
      }
      as.numeric(stats::predict(object$fit, newdata = as.data.frame(x)))
    },

    default_grid = function(tune_length, seed, x = NULL, y = NULL) {
      if (is.null(x)) {
        stop("caret's grid generator for '", method, "' needs the training ",
             "data. Let run_table_resample() generate the grid (pass ",
             "tune_grid = NULL), or call caret_grid() with x and y.",
             call. = FALSE)
      }
      caret_grid(method, tune_length, seed, x = x, y = y, search = search)
    },

    # caret's fitted objects have no common notion of size -- a forest counts
    # nodes, a glmnet counts non-zero coefficients, an SVM counts support
    # vectors. NULL says so, and one_se() falls back to its other rule rather
    # than comparing quantities that are not the same quantity.
    count_params = NULL
  )
}

#' Every caret method that could be borrowed here.
#'
#' @param pattern Optional regular expression on the method name.
#' @return A tibble of method, label and the parameters it tunes.
#' @examplesIf requireNamespace("caret", quietly = TRUE)
#' caret_available("^rf$|^ranger$")
#' @export
caret_available <- function(pattern = NULL) {
  .need_caret()
  all <- caret::getModelInfo()
  nms <- names(all)
  keep <- vapply(all, function(m) "Regression" %in% m$type, logical(1))
  nms  <- nms[keep]
  if (!is.null(pattern)) nms <- grep(pattern, nms, value = TRUE)
  tibble::tibble(
    method     = nms,
    label      = vapply(all[nms], function(m) m$label %||% "", character(1)),
    parameters = vapply(all[nms], function(m)
      paste(m$parameters$parameter, collapse = ", "), character(1))
  )
}

#' Print a `caret_fitted`
#'
#' @param x   A `caret_fitted`, fitted through [caret_spec()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.caret_fitted <- function(x, ...) {
  cat("<caret_fitted> method \"", x$method, "\" | ", length(x$features),
      " feature(s)\n", sep = "")
  bt <- tryCatch(x$fit$bestTune, error = function(e) NULL)
  if (!is.null(bt) && ncol(bt) > 0L) {
    cat("  tuned at: ", paste(names(bt), unlist(bt), sep = " = ", collapse = ", "),
        "\n", sep = "")
  }
  invisible(x)
}

