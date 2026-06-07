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

# ── Core metric function ──────────────────────────────────────────────────────

#' Compute all regression metrics for one obs/pred pair.
#'
#' @param obs  Numeric vector of observed values.
#' @param pred Numeric vector of predicted values (same length as obs).
#' @return A one-row tibble with columns: n, ccc, r2, mae, nse, rmse, rpd, mqi.
calc_metrics <- function(obs, pred) {
  obs  <- as.numeric(obs)
  pred <- as.numeric(pred)

  keep <- !is.na(obs) & !is.na(pred) & is.finite(obs) & is.finite(pred)
  obs  <- obs[keep]
  pred <- pred[keep]

  if (length(obs) < 2) {
    return(tibble::tibble(n = length(obs), ccc = NA_real_, r2 = NA_real_,
                          mae = NA_real_, nse = NA_real_, rmse = NA_real_,
                          rpd = NA_real_, mqi = NA_real_))
  }

  ccc_val <- tryCatch(
    as.numeric(DescTools::CCC(obs, pred, conf.level = 0.95)$rho.c$est)[1],
    error = function(e) NA_real_
  )

  r2_val <- tryCatch(
    as.numeric(cor(pred, obs, use = "pairwise.complete.obs"))^2,
    error = function(e) NA_real_
  )

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

  tibble::tibble(n = length(obs), ccc = ccc_val, r2 = r2_val, mae = mae_val,
                 nse = nse_val, rmse = rmse_val, rpd = rpd_val, mqi = mqi_val)
}

# ── Aggregated tables ─────────────────────────────────────────────────────────

#' Compute metrics per dataset role (train / validation / test).
#'
#' @param pred_obs_data A tibble with columns: model, target_version,
#'   dataset_role, obs, pred.
make_performance_table <- function(pred_obs_data) {
  pred_obs_data |>
    dplyr::group_by(model, target_version, dataset_role) |>
    dplyr::group_modify(~ calc_metrics(obs = .x$obs, pred = .x$pred)) |>
    dplyr::ungroup()
}

#' Compute metrics broken down by quantile bin of the observed values.
#'
#' Helps identify where the model struggles (e.g., extreme high SOC values).
make_quantile_performance <- function(pred_obs_data) {
  pred_obs_data |>
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
    ) |>
    dplyr::group_by(model, target_version, dataset_role, obs_q_group) |>
    dplyr::group_modify(~ calc_metrics(obs = .x$obs, pred = .x$pred)) |>
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
