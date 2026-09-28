# Performance metrics for regression models
#
# Metrics used throughout the framework:
#   CCC  – Lin's Concordance Correlation Coefficient: measures agreement
#           between predictions and observations (accuracy + precision).
#           Range: [-1, 1]. 1 = perfect agreement.
#   R²   – Coefficient of determination: proportion of variance explained.
#           Unlike Pearson r², CCC also penalises systematic bias.
#   MAE  – Mean Absolute Error: interpretable in the original units.
#   NSE  – Nash-Sutcliffe Efficiency: 1 = perfect model, 0 = mean-only model,
#           negative = worse than predicting the mean.
#   RMSE – Root Mean Squared Error: penalises large errors more than MAE.
#   RPD  – Ratio of Performance to Deviation = sd(obs) / RMSE. Standard
#           pedometric metric. RPD < 1.4 poor, 1.4–2.0 fair, > 2.0 good.
#   MQI  – Model Quality Index: (CCC × NSE) / (MAE / mean(obs)).
#           Composite that balances agreement, efficiency and relative error.
#           No direct literature precedent — treated as a project innovation.
#           Higher is better; use alongside CCC and RMSE for paper reporting.

# ── Lin's Concordance Correlation Coefficient ─────────────────────────────────
#
# Four lines of arithmetic, written out rather than imported.
#
# It used to come from DescTools::CCC(), which is a heavy dependency for one
# formula -- and which computes a confidence interval nobody reads, on every
# validation set, EVERY EPOCH. Removing it drops a dependency and takes work
# out of the training loop.
#
# The moments are POPULATION moments (divide by n, not n-1), which is how Lin
# (1989) defines it and what DescTools computes; the test asserts equality with
# DescTools to 1e-12 so this stays a reimplementation and not a variant.
#
# Note the behaviour at zero variance: a model predicting a constant gives
# s_pred = 0, so ccc = 0 -- no agreement, which is the honest reading. Only an
# empty or all-NA input gives NA.

#' Lin's Concordance Correlation Coefficient.
#'
#' @param obs,pred Numeric vectors of the same length.
#' @return A single numeric, or NA when fewer than two finite pairs remain.
#' @export
ccc <- function(obs, pred) {
  keep <- is.finite(obs) & is.finite(pred)
  obs  <- obs[keep]
  pred <- pred[keep]
  n <- length(obs)
  if (n < 2L) return(NA_real_)

  mo <- mean(obs)
  mp <- mean(pred)
  vo <- stats::var(obs)  * (n - 1) / n
  vp <- stats::var(pred) * (n - 1) / n
  cv <- sum((obs - mo) * (pred - mp)) / n

  den <- vo + vp + (mo - mp)^2
  if (den == 0) return(NA_real_)     # both constant AND equal: undefined
  2 * cv / den
}

# ── Core metric function ──────────────────────────────────────────────────────

#' Compute all regression metrics for one obs/pred pair.
#'
#' @param obs  Numeric vector of observed values.
#' @param pred Numeric vector of predicted values (same length as obs).
#' @return A one-row tibble with columns: n, ccc, r2, mae, nse, rmse, rpd,
#'   mqi, bias (signed, native units) and bias_pct (relative to mean(obs)).
#' @export
calc_metrics <- function(obs, pred) {
  obs  <- as.numeric(obs)
  pred <- as.numeric(pred)

  keep <- !is.na(obs) & !is.na(pred) & is.finite(obs) & is.finite(pred)
  obs  <- obs[keep]
  pred <- pred[keep]

  if (length(obs) < 2) {
    return(tibble::tibble(n = length(obs), ccc = NA_real_, r2 = NA_real_,
                          mae = NA_real_, nse = NA_real_, rmse = NA_real_,
                          rpd = NA_real_, mqi = NA_real_,
                          bias = NA_real_, bias_pct = NA_real_))
  }

  ccc_val <- ccc(obs, pred)

  # A constant prediction has no correlation to report, and cor() says so with
  # a warning. NA is the answer; a warning from a reporting function only
  # teaches people to ignore warnings.
  r2_val <- if (stats::sd(pred) == 0 || stats::sd(obs) == 0) {
    NA_real_
  } else {
    tryCatch(as.numeric(cor(pred, obs, use = "pairwise.complete.obs"))^2,
             error = function(e) NA_real_)
  }

  mae_val  <- mean(abs(pred - obs), na.rm = TRUE)
  nse_val  <- 1 - sum((obs - pred)^2, na.rm = TRUE) / sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE)
  rmse_val <- sqrt(mean((pred - obs)^2, na.rm = TRUE))
  mean_obs <- mean(obs, na.rm = TRUE)

  sd_obs  <- stats::sd(obs, na.rm = TRUE)
  rpd_val <- if (rmse_val == 0) NA_real_ else sd_obs / rmse_val

  mqi_val <- if (any(is.na(c(ccc_val, nse_val, mae_val))) ||
                 mean_obs == 0 || mae_val == 0) {
    NA_real_
  } else {
    (ccc_val * nse_val) / (mae_val / mean_obs)
  }

  # ── THE SIGNED BIAS, AND WHY IT TOOK THIS LONG ────────────────────────────
  #
  # Every metric above is blind to the SIGN of the error. MAE and RMSE are
  # unsigned by construction; R2, NSE and RPD are unchanged by a constant
  # offset in the right circumstances; and CCC penalises bias but mixes it with
  # scatter, so a low CCC never says which one it is.
  #
  # The consequence was measured on this project's own final model, by a script
  # (06_avaliacao_grafica.R) that had not run in months and computed the bias
  # itself:
  #
  #   observed on test   mean 39.45   median 31.18
  #   predicted          mean 29.84   median 25.60
  #   bias              -9.61 t/ha, -24.4%, and 53% OF THE MAE
  #
  # More than half the average error was a systematic shortfall, and nothing in
  # the framework could see it. For a stock map that is the number that matters:
  # summing the map for a total gives a quarter less carbon than the data say.
  #
  # The cause is not a coding error but a modelling choice that nobody had
  # written down: the target is trained on log1p with a SmoothL1 loss, so the
  # network estimates a conditional MEDIAN in log space, and expm1() of that is
  # the conditional median of the stock -- not its mean. The target is
  # right-skewed (mean/median = 1.27 here), so the median is systematically
  # below the mean, and the extremes are compressed besides.
  #
  # bias_pct is relative to mean(obs) so it is comparable across targets and
  # depths; bias stays in native units because that is what a user subtracts.
  bias_val <- mean(pred - obs, na.rm = TRUE)
  bias_pct_val <- if (is.finite(mean_obs) && mean_obs != 0) {
    100 * bias_val / mean_obs
  } else NA_real_

  tibble::tibble(n = length(obs), ccc = ccc_val, r2 = r2_val, mae = mae_val,
                 nse = nse_val, rmse = rmse_val, rpd = rpd_val, mqi = mqi_val,
                 bias = bias_val, bias_pct = bias_pct_val)
}

# ── Aggregated tables ─────────────────────────────────────────────────────────

#' Compute metrics per dataset role (train / validation / test).
#'
#' @param pred_obs_data A tibble with columns: model, target_version,
#'   dataset_role, obs, pred.
make_performance_table <- function(pred_obs_data) {
  pred_obs_data %>%
    dplyr::group_by(model, target_version, dataset_role) %>%
    dplyr::group_modify(~ calc_metrics(obs = .x$obs, pred = .x$pred)) %>%
    dplyr::ungroup()
}

#' Compute metrics broken down by quantile bin of the observed values.
#'
#' Helps identify where the model struggles (e.g., extreme high SOC values).
#'
#' Note on bin edges: the quantile thresholds are computed on the POOLED `obs`
#' across every (model, target_version, dataset_role) present in the input,
#' because the mutate() runs before group_by(). This is deliberate — it gives
#' the SAME bin edges to every split, so a "Q95_Q99" row means the same range of
#' observed values in train, validation and test and the splits stay comparable.
#' Call this per model if you instead want split-specific edges.
make_quantile_performance <- function(pred_obs_data) {
  pred_obs_data %>%
    dplyr::mutate(
      obs_q_group = dplyr::case_when(
        obs <= stats::quantile(obs, 0.25, na.rm = TRUE) ~ "Q00_Q25",
        obs <= stats::quantile(obs, 0.50, na.rm = TRUE) ~ "Q25_Q50",
        obs <= stats::quantile(obs, 0.75, na.rm = TRUE) ~ "Q50_Q75",
        obs <= stats::quantile(obs, 0.90, na.rm = TRUE) ~ "Q75_Q90",
        obs <= stats::quantile(obs, 0.95, na.rm = TRUE) ~ "Q90_Q95",
        obs <= stats::quantile(obs, 0.99, na.rm = TRUE) ~ "Q95_Q99",
        TRUE                                             ~ "Q99_Q100"
      )
    ) %>%
    dplyr::group_by(model, target_version, dataset_role, obs_q_group) %>%
    dplyr::group_modify(~ calc_metrics(obs = .x$obs, pred = .x$pred)) %>%
    dplyr::ungroup()
}

# ── Loader-level loss (standalone utility) ────────────────────────────────────
# NOTE: the per-epoch validation loss is computed by transform_space_loss() in
# train_cnn.R, derived from the predictions that predict_loader() already
# collects (one forward pass). This function remains a standalone helper for
# ad-hoc loss evaluation over an arbitrary loader; it is NOT in the training
# hot path.

#' Compute weighted-average loss over all batches of a DataLoader.
compute_loader_loss <- function(model, data_loader, loss_fn, device) {
  model$eval()
  loss_sum <- 0
  n_sum    <- 0L
  torch::with_no_grad({
    coro::loop(for (batch in data_loader) {
      inputs <- lapply(batch[-length(batch)], function(t) t$to(device = device))
      y      <- batch[[length(batch)]]$to(device = device)
      pred   <- do.call(model, inputs)
      loss   <- loss_fn(pred, y)
      bn     <- as.integer(inputs[[1]]$shape[[1]])
      loss_sum <- loss_sum + as.numeric(loss$item()) * bn
      n_sum    <- n_sum + bn
    })
  })
  loss_sum / n_sum
}
