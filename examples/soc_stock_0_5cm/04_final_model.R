source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/install_load_pkg.R"
)

pkg <- c(
  "torch",
  "coro",
  "dplyr",
  "readr",
  "tibble",
  "purrr",
  "DescTools"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))
source(file.path(project_root, "R", "metrics.R"))
source(file.path(project_root, "R", "cnn_architecture.R"))
source(file.path(project_root, "R", "tune_grid.R"))
source(file.path(project_root, "R", "train_cnn.R"))

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

# Run ID do tuning que gerou o ranking — ajuste para o seu run_id do round 2
tuning_run_id <- "soc_0_5cm_20260606_174341"   # <- substitua pelo run_id do round 2

# Sementes a usar. Cada seed é um treino independente com o mesmo config.
# A variância entre seeds estima a instabilidade do treinamento — se for alta,
# o config é frágil; se for baixa, o resultado é robusto e publicável.
seeds <- c(42L, 123L, 456L, 789L, 2025L)

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# ── Hiperparâmetros de treinamento final ──────────────────────────────────────
# Mais épocas e mais patience do que no tuning: agora que sabemos o melhor
# config, deixamos o modelo convergir completamente sem pressa.

training_args <- list(
  n_epochs             = 700L,
  patience             = 100L,
  es_min_delta         = 0.0003,
  warmup_start_lr      = 1e-5,
  lr_plateau_factor    = 0.5,
  lr_plateau_patience  = 30L,
  lr_plateau_min_delta = 0.0003,
  min_lr               = 1e-6,
  gradient_clip        = 1.0,
  print_every          = 10L,
  augment              = TRUE
)

# ── Paths ─────────────────────────────────────────────────────────────────────

patch_file    <- file.path(project_root, "outputs", "patches",
                            "soc_stock_modeling", target_label,
                            "patches_all_splits.rds")

tuning_dir    <- file.path(project_root, "outputs", "tuning",
                            "soc_stock_modeling", target_label,
                            tuning_run_id)

ranking_file  <- file.path(tuning_dir, "comparison", "comparison_ranked.csv")

run_id        <- paste0("final_", format(Sys.time(), "%Y%m%d_%H%M%S"))

output_dir    <- file.path(project_root, "outputs", "final_model",
                            "soc_stock_modeling", target_label, run_id)

output_dirs <- file.path(output_dir,
                          c("models", "history", "predictions",
                            "metrics", "gates", "seed_summary"))

create_output_dirs(output_dirs)

# ── Validações ────────────────────────────────────────────────────────────────

if (!file.exists(patch_file))   stop("Patches não encontrados: ", patch_file)
if (!file.exists(ranking_file)) stop("Ranking não encontrado: ",  ranking_file)

# ── Carregar patches ──────────────────────────────────────────────────────────

message("Carregando patches...")
patches <- readRDS(patch_file)

n_channels <- dim(patches$train$x_3x3_array)[2]

points_valid <- list(
  train      = patches$train$meta,
  validation = patches$validation$meta,
  test       = patches$test$meta
)

message("Canais: ", n_channels)
message("Train: ", nrow(patches$train$meta),
        " | Val: ", nrow(patches$validation$meta),
        " | Test: ", nrow(patches$test$meta))

# ── Carregar melhor config do ranking ─────────────────────────────────────────

ranking <- readr::read_csv2(ranking_file, show_col_types = FALSE)

best_row <- dplyr::filter(ranking, rank == 1L)

message("\n── Melhor config selecionado ────────────────────────────────────")
message("  config_id    : ", best_row$config_id)
message("  window_sizes : ", best_row$window_sizes)
message("  conv_channels: ", best_row$conv_channels)
message("  embedding_dim: ", best_row$embedding_dim)
message("  gate_type    : ", best_row$gate_type)
message("  use_se_block : ", best_row$use_se_block)
message("  base_lr      : ", best_row$base_lr)
message("  dropout      : ", best_row$dropout)
message("  tuning CCC   : ", round(best_row$test_ccc, 4))
message("  tuning MAE   : ", round(best_row$test_mae, 3))

# Reconstruir o config como tibble de uma linha (mesmo formato do tune_grid)
# lendo o tune_grid original salvo pelo run_cnn_tuning
tune_grid_file <- file.path(tuning_dir, "tune_grid.rds")

if (!file.exists(tune_grid_file)) {
  stop("tune_grid.rds não encontrado em: ", tuning_dir,
       "\nEsse arquivo é salvo automaticamente pelo run_cnn_tuning().")
}

tune_grid_full <- readRDS(tune_grid_file)
best_cfg <- dplyr::filter(tune_grid_full, config_id == best_row$config_id)

if (nrow(best_cfg) == 0) {
  stop("config_id '", best_row$config_id, "' não encontrado no tune_grid.rds.")
}

# ── Treinar com múltiplas seeds ───────────────────────────────────────────────
# Cada seed é um treino independente do zero com o mesmo config.
# A variância entre seeds indica estabilidade — resultados publicáveis devem
# ter baixo desvio padrão entre seeds (< ~5% do CCC médio).

message("\n── Iniciando treino com ", length(seeds), " seeds ──────────────────")

# Cache de tensores construído UMA vez (reusado por todas as seeds)
tensor_cache <- .build_tensor_cache(patches, best_cfg$window_sizes[[1]])

seed_results <- vector("list", length(seeds))

for (s_idx in seq_along(seeds)) {

  seed_val <- seeds[s_idx]
  message("\n── Seed ", s_idx, "/", length(seeds), ": ", seed_val, " ──")

  set.seed(seed_val)
  torch::torch_manual_seed(seed_val)

  loaders <- .make_loaders_from_cache(tensor_cache, best_cfg)

  result <- tryCatch(
    do.call(
      train_one_cnn,
      c(
        list(
          cfg          = best_cfg,
          n_channels   = n_channels,
          loaders      = loaders,
          points_valid = points_valid,
          transform    = expm1,
          device       = device,
          model_name   = paste0("final_seed", seed_val)
        ),
        training_args
      )
    ),
    error = function(e) {
      message("  ERRO na seed ", seed_val, ": ", e$message)
      NULL
    }
  )

  if (is.null(result)) next

  seed_label <- sprintf("seed%04d", seed_val)

  # Salvar pesos do melhor epoch
  safe_torch_save(
    result$best_state,
    file.path(output_dir, "models", paste0(seed_label, "_best.pt"))
  )

  # Salvar histórico
  safe_write_csv2(
    result$history,
    file.path(output_dir, "history", paste0(seed_label, "_history.csv"))
  )

  # Salvar predições (train + val + test)
  safe_write_csv2(
    result$pred_all,
    file.path(output_dir, "predictions", paste0(seed_label, "_pred_all.csv"))
  )

  # Salvar métricas
  safe_write_csv2(
    result$perf_all,
    file.path(output_dir, "metrics", paste0(seed_label, "_perf.csv"))
  )

  safe_write_csv2(
    result$perf_quantile,
    file.path(output_dir, "metrics", paste0(seed_label, "_perf_quantile.csv"))
  )

  # Salvar gate analysis (se disponível)
  if (!is.null(result$gate)) {
    safe_write_csv2(
      result$gate$summary,
      file.path(output_dir, "gates", paste0(seed_label, "_gate_summary.csv"))
    )
    safe_write_csv2(
      result$gate$by_profile,
      file.path(output_dir, "gates", paste0(seed_label, "_gate_profiles.csv"))
    )
  }

  # Extrair métricas de teste para resumo entre seeds
  test_perf <- dplyr::filter(result$perf_all, dataset_role == "test") |>
    dplyr::mutate(seed = seed_val, best_epoch = result$best_epoch,
                  runtime_min = result$runtime_min)

  seed_results[[s_idx]] <- test_perf

  message("  seed ", seed_val, " | best_epoch: ", result$best_epoch,
          " | CCC: ",  round(test_perf$ccc,  4),
          " | MAE: ",  round(test_perf$mae,  3),
          " | RMSE: ", round(test_perf$rmse, 3),
          " | RPD: ",  round(test_perf$rpd,  3))

  gc()
}

# ── Resumo entre seeds ────────────────────────────────────────────────────────

seed_table <- dplyr::bind_rows(purrr::compact(seed_results))

if (nrow(seed_table) == 0) stop("Nenhuma seed completou com sucesso.")

seed_summary <- seed_table |>
  dplyr::summarise(
    n_seeds       = dplyr::n(),
    config_id     = best_row$config_id,
    window_sizes  = best_row$window_sizes,
    conv_channels = best_row$conv_channels,
    # métricas: média ± desvio entre seeds
    ccc_mean  = mean(ccc,  na.rm = TRUE),  ccc_sd  = sd(ccc,  na.rm = TRUE),
    r2_mean   = mean(r2,   na.rm = TRUE),  r2_sd   = sd(r2,   na.rm = TRUE),
    mae_mean  = mean(mae,  na.rm = TRUE),  mae_sd  = sd(mae,  na.rm = TRUE),
    nse_mean  = mean(nse,  na.rm = TRUE),  nse_sd  = sd(nse,  na.rm = TRUE),
    rmse_mean = mean(rmse, na.rm = TRUE),  rmse_sd = sd(rmse, na.rm = TRUE),
    rpd_mean  = mean(rpd,  na.rm = TRUE),  rpd_sd  = sd(rpd,  na.rm = TRUE),
    mqi_mean  = mean(mqi,  na.rm = TRUE),  mqi_sd  = sd(mqi,  na.rm = TRUE),
    best_epoch_mean = mean(best_epoch, na.rm = TRUE),
    runtime_min_total = sum(runtime_min, na.rm = TRUE)
  )

message("\n── Resumo final (", nrow(seed_table), " seeds) ──────────────────────")
message(sprintf("  CCC  : %.4f ± %.4f", seed_summary$ccc_mean,  seed_summary$ccc_sd))
message(sprintf("  MAE  : %.3f ± %.3f", seed_summary$mae_mean,  seed_summary$mae_sd))
message(sprintf("  RMSE : %.3f ± %.3f", seed_summary$rmse_mean, seed_summary$rmse_sd))
message(sprintf("  R²   : %.4f ± %.4f", seed_summary$r2_mean,   seed_summary$r2_sd))
message(sprintf("  NSE  : %.4f ± %.4f", seed_summary$nse_mean,  seed_summary$nse_sd))
message(sprintf("  RPD  : %.3f ± %.3f", seed_summary$rpd_mean,  seed_summary$rpd_sd))
message(sprintf("  MQI  : %.4f ± %.4f", seed_summary$mqi_mean,  seed_summary$mqi_sd))

safe_write_csv2(
  seed_table,
  file.path(output_dir, "seed_summary", "seed_results_test.csv")
)

safe_write_csv2(
  seed_summary,
  file.path(output_dir, "seed_summary", "seed_summary.csv")
)

safe_save_rds(
  list(
    config      = best_cfg,
    seed_table  = seed_table,
    seed_summary = seed_summary,
    run_id      = run_id,
    tuning_run_id = tuning_run_id
  ),
  file.path(output_dir, "seed_summary", "final_run_summary.rds"),
  compress = FALSE
)

message("\nResultados salvos em: ", output_dir)
