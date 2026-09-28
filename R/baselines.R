# ── Baselines ─────────────────────────────────────────────────────────────────
#
# WHAT THESE ARE FOR.
#
# The project's claim is that a convolution over a neighbourhood of predictors
# beats the same predictors read at a point. Four models, under folds that are
# identical by construction, turn that claim into a measurement:
#
#   rf   on centre                the classic DSM baseline. Is the CNN even
#                                 competitive with what everyone already does?
#   rf   on centre + window means context WITHOUT spatial structure. The same
#                                 neighbourhood the convolution sees, with the
#                                 arrangement thrown away.
#   mlp  on centre                is it the architecture, or just the
#                                 covariates? Same optimiser, same loss, same
#                                 early stopping as the CNN, no convolution.
#   cnn  on the whole patch       context WITH spatial structure.
#
# THE GAP BETWEEN THE SECOND AND THE FOURTH IS WHAT THE CONVOLUTION IS WORTH.
# If they match, the convolution is doing averaging -- a falsifiable claim,
# measured cheaply, and worth far more than another point of CCC.
#
# All four run under the same fold plan, the same seeds and the same noise
# floor, because a baseline measured on another split is not a baseline. It is
# a second experiment, and the difference between it and the first is
# unattributable.

# ── Random Forest ─────────────────────────────────────────────────────────────
#
# ranger IF AVAILABLE, randomForest OTHERWISE.
#
# They are not interchangeable in cost: on 30k rows and 360 features
# randomForest is single-threaded and formula-free but slow enough to dominate
# a run that trains neural networks, while ranger is multi-threaded and
# typically an order of magnitude faster. The fallback exists so the baseline
# RUNS for someone who has not installed ranger -- not because the two are
# equivalent, and the run says which one it used.
#
# Neither needs the scaling, and both get it: the table comes from the fold
# cache, which is scaled because the neural models need it to be. A monotone
# per-channel affine transform cannot change a tree's splits, so this costs
# nothing and keeps one code path instead of two views of the same fold.

.rf_backend <- function() {
  if (requireNamespace("ranger", quietly = TRUE)) return("ranger")
  if (requireNamespace("randomForest", quietly = TRUE)) return("randomForest")
  stop("A Random Forest baseline needs either the 'ranger' package ",
       "(recommended: multi-threaded, far faster here) or 'randomForest'.\n",
       "  install.packages(\"ranger\")", call. = FALSE)
}

#' Grid for the RF baseline.
#'
#' @param tune_length How many configs to draw.
#' @param seed        Draw seed.
#' @param n_features  Number of columns in the table, so mtry can be expressed
#'   as a FRACTION. An mtry written as a count is wrong the moment the feature
#'   set changes -- and adding the window means triples it.
rf_grid <- function(tune_length = 4L, seed = 42L, n_features = NULL) {
  tune_length <- max(1L, as.integer(tune_length))

  # A SPREAD, NOT A DRAW -- and this is a correction, not a preference.
  #
  # The first version sampled mtry_frac and min_node_size from four values
  # each. With tune_length = 4 it drew 0.1 FOUR TIMES, so the "four-config
  # grid" tested two distinct settings, both at the smallest mtry, and three
  # of the four units trained the same model. make_tune_grid() de-duplicates
  # for exactly this reason; this did not.
  #
  # A random draw earns its place when the space is large and the axes
  # interact -- that is the CNN's grid. A forest has two parameters with a
  # handful of sensible values each, so the space can simply be COVERED, and
  # covering it is both cheaper and reproducible.
  #
  # The order is outward from the defaults, so tune_length = 1 gives the
  # textbook forest and every increment adds the next most informative
  # setting. mtry = p/3 is the regression default; min.node.size = 5 is
  # ranger's and randomForest's.
  frac_cand <- c(1/3, 0.10, 0.50, 0.05, 0.75, 0.20, 1.00)
  node_cand <- c(5L, 20L, 1L, 10L)

  g <- expand.grid(mtry_frac = frac_cand, min_node_size = node_cand,
                   KEEP.OUT.ATTRS = FALSE)
  # Rank by how far the pair is from the default pair, so the budget is spent
  # near what is known to work before it is spent at the extremes.
  g <- g[order(match(g$mtry_frac, frac_cand) + match(g$min_node_size, node_cand)), ]
  g <- unique(g)[seq_len(min(tune_length, nrow(g))), , drop = FALSE]

  out <- tibble::tibble(
    config_id     = sprintf("rf_%03d", seq_len(nrow(g))),
    mtry_frac     = g$mtry_frac,
    min_node_size = as.integer(g$min_node_size),
    # Fixed, not tuned. More trees never overfit a forest; they only cost time,
    # and the curve is flat long before 500. Tuning it would spend the budget
    # on the one parameter with a known answer.
    n_trees       = 500L
  )
  if (!is.null(n_features)) {
    out$mtry <- pmax(1L, as.integer(round(out$mtry_frac * n_features)))
    # Two fractions can round to the same mtry on a narrow feature set, and two
    # rows that fit the identical forest are one row that wastes a fold.
    out <- out[!duplicated(out[, c("mtry", "min_node_size")]), , drop = FALSE]
    out$config_id <- sprintf("rf_%03d", seq_len(nrow(out)))
  }
  out
}

#' The Random Forest baseline.
rf_spec <- function() {
  model_spec(
    name  = "rf",
    input = "table",
    description = "Random Forest on the tabular view (ranger, or randomForest).",

    fit = function(x, y, cfg, n_cores = NULL, ...) {
      backend <- .rf_backend()
      # [[ ]] behind a names() check, never cfg$mtry: on a tibble, $ on a
      # missing column returns NULL *and* warns ("Unknown or uninitialised
      # column"). The path is deliberately optional -- mtry is resolved from
      # the real feature count when the grid did not carry it -- so a warning
      # here is noise that trains people to ignore warnings.
      mtry_col <- if ("mtry" %in% names(cfg)) cfg[["mtry"]] else NULL
      mtry <- if (!is.null(mtry_col) && length(mtry_col) && !is.na(mtry_col[1])) {
        as.integer(mtry_col[1])
      } else {
        max(1L, as.integer(round(cfg$mtry_frac * ncol(x))))
      }

      if (backend == "ranger") {
        fit <- ranger::ranger(
          x = as.data.frame(x), y = y,
          num.trees     = as.integer(cfg$n_trees),
          mtry          = mtry,
          min.node.size = as.integer(cfg$min_node_size),
          # n_cores has one meaning across the framework (resolve_cores()):
          # NULL is the physical cores minus one. This was 0 -- every logical
          # core ranger can see, 32 threads on this machine's 16 cores -- so
          # the forest and the network answered the same question two ways.
          # A caller who wants every logical core asks for it.
          num.threads   = resolve_cores(n_cores, what = "ranger"),
          # The forest is refit per fold and per seed and never reloaded, so
          # keeping the training data inside it would multiply the run's peak
          # memory by the number of units for no gain.
          write.forest  = TRUE,
          verbose       = FALSE
        )
      } else {
        fit <- randomForest::randomForest(
          x = as.data.frame(x), y = y,
          ntree    = as.integer(cfg$n_trees),
          mtry     = mtry,
          nodesize = as.integer(cfg$min_node_size)
        )
      }
      structure(list(fit = fit, backend = backend, mtry = mtry,
                     features = colnames(x)),
                class = "rf_fitted")
    },

    # (print methods for rf_fitted and mlp_fitted are at the end of this file)

    predict = function(object, x, ...) {
      # The column ORDER is the contract: a forest indexes features by
      # position, and a table rebuilt in another order predicts confidently
      # from the wrong columns without any error at all.
      if (!identical(colnames(x), object$features)) {
        stop("The prediction table's columns differ from the training ",
             "table's. A forest reads features by position.", call. = FALSE)
      }
      if (object$backend == "ranger") {
        as.numeric(stats::predict(object$fit, data = as.data.frame(x))$predictions)
      } else {
        as.numeric(stats::predict(object$fit, newdata = as.data.frame(x)))
      }
    },

    # n_features comes from the real table, so mtry is resolved from the
    # feature set that actually exists rather than from a fraction carried
    # around and multiplied later.
    default_grid = function(tune_length, seed, x = NULL, y = NULL) {
      rf_grid(tune_length, seed, n_features = if (!is.null(x)) ncol(x) else NULL)
    },

    # A forest's size is its total node count -- the honest complexity axis for
    # one_se(), and comparable across configs of the same family. It is not
    # comparable with a network's parameter count, and one_se() is only ever
    # applied within a family.
    count_params = function(object) {
      if (object$backend == "ranger") {
        as.integer(sum(vapply(object$fit$forest$child.nodeIDs,
                              function(z) length(z[[1]]), integer(1))))
      } else {
        as.integer(sum(object$fit$forest$ndbigtree))
      }
    }
  )
}

# ── MLP ───────────────────────────────────────────────────────────────────────
#
# The control that separates the architecture from the covariates.
#
# It is trained the same way the CNN is -- Adam, the same loss, early stopping
# on validation, the same seed -- so the only difference left between them is
# the convolution. An MLP trained by some other recipe would confound the two
# and answer nothing.

#' Grid for the MLP baseline.
mlp_grid <- function(tune_length = 6L, seed = 42L) {
  with_local_seed(seed, {
    hidden  <- sample(c("256_128", "512_256", "512_256_128", "128_64"),
                      tune_length, replace = TRUE)
    dropout <- round(stats::runif(tune_length, 0.0, 0.4), 2)
    base_lr <- 10^stats::runif(tune_length, -4, -2.5)
    batch   <- sample(c(64L, 128L, 256L), tune_length, replace = TRUE)
  })
  out <- tibble::tibble(
    hidden      = hidden,
    dropout     = dropout,
    base_lr     = signif(base_lr, 4),
    batch_size  = batch,
    loss_fn     = "smooth_l1"
  )
  # De-duplicated, for the reason rf_grid() spells out: two identical rows are
  # one row that trains twice and reports twice. The draw stays a draw here --
  # four interacting axes are what random search is for -- but a repeat is
  # waste, not coverage.
  out <- unique(out)
  out$config_id <- sprintf("mlp_%03d", seq_len(nrow(out)))
  dplyr::relocate(out, config_id)
}

.mlp_module <- torch::nn_module(
  "mlp_baseline",
  initialize = function(n_in, hidden, dropout) {
    layers <- list()
    prev <- n_in
    for (h in hidden) {
      layers[[length(layers) + 1L]] <- torch::nn_linear(prev, h)
      # BatchNorm before the activation, as in the CNN's conv blocks: the same
      # recipe on both sides is the whole point of this control.
      layers[[length(layers) + 1L]] <- torch::nn_batch_norm1d(h)
      layers[[length(layers) + 1L]] <- torch::nn_relu()
      if (dropout > 0) layers[[length(layers) + 1L]] <- torch::nn_dropout(dropout)
      prev <- h
    }
    layers[[length(layers) + 1L]] <- torch::nn_linear(prev, 1L)
    self$net <- do.call(torch::nn_sequential, layers)
  },
  forward = function(x) self$net(x)
)

#' The MLP baseline.
#'
#' @param n_epochs Maximum epochs.
#' @param patience Early-stopping patience, in epochs.
mlp_spec <- function(n_epochs = 300L, patience = 40L) {
  model_spec(
    name  = "mlp",
    input = "table",
    description = "Fully connected network on the tabular view (torch).",

    fit = function(x, y, cfg, x_val = NULL, y_val = NULL, device = NULL, ...) {
      if (is.null(device)) device <- torch::torch_device("cpu")
      hidden <- as.integer(strsplit(as.character(cfg$hidden), "_")[[1]])

      model <- .mlp_module(ncol(x), hidden, cfg$dropout)$to(device = device)
      opt   <- torch::optim_adam(model$parameters, lr = cfg$base_lr)
      lossf <- switch(as.character(cfg$loss_fn),
                      smooth_l1 = torch::nn_smooth_l1_loss(),
                      mse       = torch::nn_mse_loss(),
                      mae       = torch::nn_l1_loss(),
                      stop("Unknown loss_fn: ", cfg$loss_fn, call. = FALSE))

      xt <- torch::torch_tensor(x, dtype = torch::torch_float())
      yt <- torch::torch_tensor(y, dtype = torch::torch_float())$view(c(-1L, 1L))
      ds <- torch::tensor_dataset(xt, yt)
      # drop_last: a final batch of one makes BatchNorm fail on the variance of
      # a single sample -- the same reason the CNN's training loader drops it.
      dl <- torch::dataloader(ds, batch_size = as.integer(cfg$batch_size),
                              shuffle = TRUE, drop_last = TRUE)

      has_val <- !is.null(x_val) && !is.null(y_val)
      if (has_val) {
        xv <- torch::torch_tensor(x_val, dtype = torch::torch_float())$to(device = device)
        yv <- torch::torch_tensor(y_val, dtype = torch::torch_float())$view(c(-1L, 1L))$to(device = device)
      }

      best_loss  <- Inf
      best_state <- NULL
      best_epoch <- 0L
      since      <- 0L
      history    <- vector("list", n_epochs)

      for (ep in seq_len(n_epochs)) {
        model$train()
        tot <- 0; nb <- 0
        coro::loop(for (b in dl) {
          opt$zero_grad()
          out <- model(b[[1]]$to(device = device))
          l   <- lossf(out, b[[2]]$to(device = device))
          l$backward(); opt$step()
          tot <- tot + as.numeric(l$item()); nb <- nb + 1L
        })
        train_loss <- if (nb > 0L) tot / nb else NA_real_

        val_loss <- NA_real_
        if (has_val) {
          model$eval()
          torch::with_no_grad({
            val_loss <- as.numeric(lossf(model(xv), yv)$item())
          })
        }
        history[[ep]] <- tibble::tibble(epoch = ep, train_loss = train_loss,
                                        val_loss = val_loss)

        # Selection on VALIDATION, never on training: the epoch with the lowest
        # training loss is simply the last one.
        gate <- if (has_val) val_loss else train_loss
        if (is.finite(gate) && gate < best_loss - 1e-8) {
          best_loss  <- gate
          best_epoch <- ep
          # lapply(..., clone) and not the state dict itself: the dict holds
          # references to the live parameters, so keeping it would "restore"
          # whatever the last epoch left behind.
          best_state <- lapply(model$state_dict(), function(t) t$clone())
          since <- 0L
        } else {
          since <- since + 1L
          if (since >= patience) break
        }
      }

      if (!is.null(best_state)) model$load_state_dict(best_state)
      structure(list(model = model, device = device,
                     features = colnames(x), best_epoch = best_epoch,
                     best_val_loss = best_loss,
                     history = dplyr::bind_rows(history)),
                class = "mlp_fitted")
    },

    predict = function(object, x, ...) {
      if (!identical(colnames(x), object$features)) {
        stop("The prediction table's columns differ from the training ",
             "table's.", call. = FALSE)
      }
      object$model$eval()
      xt <- torch::torch_tensor(x, dtype = torch::torch_float())$to(device = object$device)
      torch::with_no_grad({
        as.numeric(object$model(xt)$to(device = "cpu"))
      })
    },

    # x and y are accepted and ignored: nothing in this grid depends on the
    # data, but a signature that varies per model is one the runner cannot call.
    default_grid = function(tune_length, seed, x = NULL, y = NULL) {
      mlp_grid(tune_length, seed)
    },

    count_params = function(object) {
      as.integer(sum(vapply(object$model$parameters,
                            function(p) prod(dim(p)), numeric(1))))
    }
  )
}

# ── the CNN, in the same registry ─────────────────────────────────────────────
#
# It is registered so that list_models() describes everything the framework can
# fit rather than everything EXCEPT the model the framework is about -- a
# registry that omits the main case teaches people to look elsewhere.
#
# Its fit() genuinely wraps train_one_cnn(): the adaptation is that `x` is a
# fold cache rather than a matrix, which is exactly what input = "patches"
# declares. What it is NOT is the path run_cnn_resample() takes; that runner
# calls train_one_cnn() directly, for the reasons set out at the top of
# R/train_table.R. Registering the CNN here describes it; it does not reroute
# the trained path days before a definitive run.
cnn_spec <- function() {
  model_spec(
    name  = "cnn",
    input = "patches",
    description = "Dual-branch CNN over the patch tensors (the model under test).",

    fit = function(x, y, cfg, points_valid, transform = identity,
                   device = NULL, model_name = "cnn", ...) {
      if (is.null(device)) device <- torch::torch_device("cpu")
      loaders <- .make_loaders_from_cache(x, cfg)
      # Channel count read from the cache rather than passed: the tensor knows,
      # and a second source for a fact that already exists is a second source
      # that can disagree with the first.
      k  <- setdiff(names(x$train), "y")[1]
      nc <- as.integer(x$train[[k]]$shape[2])
      train_one_cnn(cfg = cfg, n_channels = nc, loaders = loaders,
                    points_valid = points_valid, transform = transform,
                    device = device, model_name = model_name, ...)
    },

    predict = function(object, x, ...) {
      stop("The CNN's predictions are produced by predict_loader() over a ",
           "DataLoader, not from a matrix. Use run_cnn_resample().",
           call. = FALSE)
    },

    # `windows` is what the loaded store holds and `n_train` the smallest
    # fold's training set; dsm_train() passes both, so a default grid never
    # asks for a window the store cannot serve or a batch no fold can fill.
    default_grid = function(tune_length, seed, x = NULL, y = NULL,
                            windows = NULL, n_train = NULL) {
      make_tune_grid(tune_length = tune_length, seed = seed, windows = windows,
                     n_train = n_train)
    },

    count_params = function(object) {
      if (!is.null(object$n_params)) as.integer(object$n_params) else NA_integer_
    }
  )
}

# ── register them ─────────────────────────────────────────────────────────────
#
# When the package loads (.onLoad(), R/zzz.R), not as this file is read -- a
# package reads its files at install time, alphabetically, before
# register_model() exists. overwrite = TRUE so that calling it twice in a
# session is harmless.
.register_builtin_models <- function() {
  register_model(rf_spec(),  overwrite = TRUE)
  register_model(mlp_spec(), overwrite = TRUE)
  register_model(cnn_spec(), overwrite = TRUE)
  invisible(NULL)
}

# ── print methods ─────────────────────────────────────────────────────────────
#
# Typing the object used to dump the backend's own print -- a randomForest
# call, or an nn_module with every layer -- and the two things a reader
# wants first (what was it fitted on, how big is it) were not on the screen.
#' @export
print.rf_fitted <- function(x, ...) {
  n_tree <- tryCatch(
    if (identical(x$backend, "ranger")) x$fit$num.trees else x$fit$ntree,
    error = function(e) NA_integer_)
  cat("<rf_fitted> ", x$backend, " | ", n_tree, " trees | mtry ", x$mtry,
      " | ", length(x$features), " feature(s)\n", sep = "")
  cat("  features: ", paste(utils::head(x$features, 6L), collapse = ", "),
      if (length(x$features) > 6L) ", ..." else "", "\n", sep = "")
  invisible(x)
}

#' @export
print.mlp_fitted <- function(x, ...) {
  cat("<mlp_fitted> ", length(x$features), " feature(s) | best epoch ",
      x$best_epoch, " | best validation loss ",
      format(x$best_val_loss, digits = 4), " | on ", x$device$type, "\n",
      sep = "")
  cat("  features: ", paste(utils::head(x$features, 6L), collapse = ", "),
      if (length(x$features) > 6L) ", ..." else "", "\n", sep = "")
  invisible(x)
}

