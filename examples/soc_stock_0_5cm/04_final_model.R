source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/install_load_pkg.R"
)

pkg <- c(
  "torch",
  "coro",
  "dplyr",
  "tidyr",
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

# Run de tuning a usar. "latest" pega automaticamente o run mais recente
# (ordenado pelo timestamp no nome da pasta). Ou informe um run_id explícito.
tuning_run_id <- "latest"

# Configs a treinar no modelo final. Cada um é treinado com TODAS as seeds.
# NULL = usa automaticamente o rank 1 do ranking de validação — recomendado para
# a primeira corrida a 250 m (ainda não sabemos quem vence).
# Para comparar o top-N pareado por seed: rode o 03, abra
#   outputs/.../tuning/<run>/comparison/comparison_ranked.csv
# e liste aqui os config_ids vencedores, ex.: c("cfg_007", "cfg_013").
# ATENÇÃO: NÃO reutilize IDs de runs antigos — o mesmo "cfg_004" é uma
# arquitetura DIFERENTE em cada run de tuning (a numeração é por run).
# As seeds são as mesmas para todos os configs → comparação pareada por seed.
selected_config_ids <- NULL

# Sementes. Cada seed é um treino independente do zero. A variância entre seeds
# estima a estabilidade do treinamento — resultado publicável deve ter baixo
# desvio (idealmente < ~5% do CCC médio).
seeds <- c(42L, 123L, 456L, 789L, 2025L)

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# ── Hiperparâmetros de treinamento final ──────────────────────────────────────
# Mais épocas e mais patience do que no tuning: agora que sabemos os configs,
# deixamos o modelo convergir completamente sem pressa.

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

patch_file        <- file.path(project_root, "outputs", "patches",
                                "soc_stock_modeling", target_label,
                                "patches_all_splits.rds")

output_tuning_dir <- file.path(project_root, "outputs", "tuning",
                                "soc_stock_modeling", target_label)

# Resolver "latest" para o run de tuning mais recente
if (identical(tuning_run_id, "latest")) {
  run_dirs <- list.dirs(output_tuning_dir, recursive = FALSE, full.names = FALSE)
  run_dirs <- run_dirs[grepl("^soc_", run_dirs)]
  if (length(run_dirs) == 0) stop("Nenhum run de tuning encontrado em: ", output_tuning_dir)
  tuning_run_id <- sort(run_dirs, decreasing = TRUE)[1]
  message("tuning_run_id resolvido para: ", tuning_run_id)
}

tuning_dir   <- file.path(output_tuning_dir, tuning_run_id)
ranking_file <- file.path(tuning_dir, "comparison", "comparison_ranked.csv")
tune_grid_file <- file.path(tuning_dir, "tune_grid.rds")

run_id     <- paste0("final_", format(Sys.time(), "%Y%m%d_%H%M%S"))
output_dir <- file.path(project_root, "outputs", "final_model",
                        "soc_stock_modeling", target_label, run_id)

create_output_dirs(c(output_dir, file.path(output_dir, "comparison")))

# ── Validações ────────────────────────────────────────────────────────────────

if (!file.exists(patch_file))    stop("Patches não encontrados: ", patch_file)
if (!file.exists(ranking_file))  stop("Ranking não encontrado: ",  ranking_file)
if (!file.exists(tune_grid_file)) stop("tune_grid.rds não encontrado: ", tune_grid_file)

# ── Carregar patches ──────────────────────────────────────────────────────────

message("Carregando patches...")
patches <- readRDS(patch_file)

n_channels <- dim(patches$train$x_3x3_array)[2]

points_valid <- list(
  train      = patches$train$meta,
  validation = patches$validation$meta,
  test       = patches$test$meta
)

message("Canais: ", n_channels,
        " | Train: ", nrow(patches$train$meta),
        " | Val: ", nrow(patches$validation$meta),
        " | Test: ", nrow(patches$test$meta))

# ── Selecionar configs ────────────────────────────────────────────────────────

ranking        <- readr::read_csv2(ranking_file, show_col_types = FALSE)
tune_grid_full <- readRDS(tune_grid_file)

if (is.null(selected_config_ids)) {
  selected_config_ids <- dplyr::filter(ranking, rank == 1L)$config_id
}

missing_cfgs <- setdiff(selected_config_ids, tune_grid_full$config_id)
if (length(missing_cfgs) > 0) {
  stop("config_ids não encontrados no tune_grid: ", paste(missing_cfgs, collapse = ", "))
}

selected_cfgs <- dplyr::filter(tune_grid_full, config_id %in% selected_config_ids)

message("\n── Configs selecionados para o modelo final ──────────────────────")
for (cid in selected_config_ids) {
  r <- dplyr::filter(ranking, config_id == cid)
  message("  ", cid,
          " | janela ", r$window_sizes,
          " | ", r$conv_channels,
          " | embed ", r$embedding_dim,
          " | ", r$gate_type,
          " | val_CCC ", round(r$val_ccc, 4),
          " | test_CCC ", round(r$test_ccc, 4), " (diag)")
}

# ── Cache de tensores (cobre a união das janelas de todos os configs) ─────────

windows_needed <- sort(unique(unlist(selected_cfgs$window_sizes)))
message("\nConstruindo cache de tensores para janelas: ",
        paste(windows_needed, collapse = ", "))
tensor_cache <- .build_tensor_cache(patches, windows_needed)

# ── Função: treinar um config com todas as seeds ──────────────────────────────

train_config_all_seeds <- function(cfg, config_id) {

  cfg_out_dir <- file.path(output_dir, config_id)
  create_output_dirs(file.path(cfg_out_dir,
                               c("models", "history", "predictions",
                                 "metrics", "gates")))

  seed_rows <- vector("list", length(seeds))

  for (s_idx in seq_along(seeds)) {
    seed_val <- seeds[s_idx]
    message("\n── [", config_id, "] seed ", s_idx, "/", length(seeds),
            ": ", seed_val, " ──")

    set.seed(seed_val)
    torch::torch_manual_seed(seed_val)

    loaders <- .make_loaders_from_cache(tensor_cache, cfg)

    result <- tryCatch(
      do.call(
        train_one_cnn,
        c(list(cfg = cfg, n_channels = n_channels, loaders = loaders,
               points_valid = points_valid, transform = expm1, device = device,
               model_name = paste0(config_id, "_seed", seed_val)),
          training_args)
      ),
      error = function(e) {
        message("  ERRO [", config_id, "] seed ", seed_val, ": ", e$message)
        NULL
      }
    )
    if (is.null(result)) next

    sl <- sprintf("seed%04d", seed_val)
    safe_torch_save(result$best_state,  file.path(cfg_out_dir, "models",      paste0(sl, "_best.pt")))
    safe_write_csv2(result$history,     file.path(cfg_out_dir, "history",     paste0(sl, "_history.csv")))
    safe_write_csv2(result$pred_all,    file.path(cfg_out_dir, "predictions", paste0(sl, "_pred_all.csv")))
    safe_write_csv2(result$perf_all,    file.path(cfg_out_dir, "metrics",     paste0(sl, "_perf.csv")))
    safe_write_csv2(result$perf_quantile, file.path(cfg_out_dir, "metrics",   paste0(sl, "_perf_quantile.csv")))
    if (!is.null(result$gate)) {
      safe_write_csv2(result$gate$summary,    file.path(cfg_out_dir, "gates", paste0(sl, "_gate_summary.csv")))
      safe_write_csv2(result$gate$by_profile, file.path(cfg_out_dir, "gates", paste0(sl, "_gate_profiles.csv")))
    }

    seed_rows[[s_idx]] <- dplyr::filter(result$perf_all, dataset_role == "test") %>%
      dplyr::mutate(config_id = config_id, seed = seed_val,
                    best_epoch = result$best_epoch,
                    runtime_min = result$runtime_min)

    message("  [", config_id, "] seed ", seed_val,
            " | best_ep ", result$best_epoch,
            " | CCC ",  round(seed_rows[[s_idx]]$ccc,  4),
            " | MAE ",  round(seed_rows[[s_idx]]$mae,  3),
            " | RMSE ", round(seed_rows[[s_idx]]$rmse, 3))
    gc()
  }

  dplyr::bind_rows(purrr::compact(seed_rows))
}

# ── Rodar todos os configs ────────────────────────────────────────────────────

all_seed_results <- tibble::tibble()
for (i in seq_len(nrow(selected_cfgs))) {
  cfg <- selected_cfgs[i, ]
  res <- train_config_all_seeds(cfg, cfg$config_id)
  all_seed_results <- dplyr::bind_rows(all_seed_results, res)
}

if (nrow(all_seed_results) == 0) stop("Nenhuma seed completou com sucesso.")

# ── Resumo por config: média ± desvio entre seeds ─────────────────────────────

config_summary <- all_seed_results %>%
  dplyr::group_by(config_id) %>%
  dplyr::summarise(
    n_seeds   = dplyr::n(),
    ccc_mean  = mean(ccc),  ccc_sd  = sd(ccc),
    r2_mean   = mean(r2),   r2_sd   = sd(r2),
    mae_mean  = mean(mae),  mae_sd  = sd(mae),
    nse_mean  = mean(nse),  nse_sd  = sd(nse),
    rmse_mean = mean(rmse), rmse_sd = sd(rmse),
    rpd_mean  = mean(rpd),  rpd_sd  = sd(rpd),
    mqi_mean  = mean(mqi),  mqi_sd  = sd(mqi),
    best_epoch_mean   = mean(best_epoch),
    runtime_min_total = sum(runtime_min),
    .groups = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(ccc_mean))

safe_write_csv2(all_seed_results, file.path(output_dir, "comparison", "all_seed_results_test.csv"))
safe_write_csv2(config_summary,   file.path(output_dir, "comparison", "config_summary_test.csv"))

safe_save_rds(
  list(selected_cfgs = selected_cfgs, seeds = seeds,
       all_seed_results = all_seed_results, config_summary = config_summary,
       run_id = run_id, tuning_run_id = tuning_run_id),
  file.path(output_dir, "comparison", "final_run_summary.rds"),
  compress = FALSE
)

# ── Comparação pareada por seed (cfg_004 vs cfg_012) ──────────────────────────
# Mesma seed = mesmo estado de RNG inicial, então a diferença por seed isola o
# efeito da arquitetura. Reporta a diferença média e se ela é consistente.

if (length(selected_config_ids) == 2) {
  paired <- all_seed_results %>%
    dplyr::select(config_id, seed, ccc, mae, rmse, mqi) %>%
    tidyr::pivot_wider(names_from = config_id, values_from = c(ccc, mae, rmse, mqi))

  c1 <- selected_config_ids[1]; c2 <- selected_config_ids[2]
  paired <- paired %>%
    dplyr::mutate(
      d_ccc  = .data[[paste0("ccc_",  c1)]] - .data[[paste0("ccc_",  c2)]],
      d_mae  = .data[[paste0("mae_",  c1)]] - .data[[paste0("mae_",  c2)]],
      d_rmse = .data[[paste0("rmse_", c1)]] - .data[[paste0("rmse_", c2)]],
      d_mqi  = .data[[paste0("mqi_",  c1)]] - .data[[paste0("mqi_",  c2)]]
    )

  safe_write_csv2(paired, file.path(output_dir, "comparison", "paired_by_seed.csv"))

  message("\n── Diferença pareada (", c1, " − ", c2, "), média entre seeds ──")
  message(sprintf("  ΔCCC : %+.4f", mean(paired$d_ccc,  na.rm = TRUE)))
  message(sprintf("  ΔMAE : %+.3f", mean(paired$d_mae,  na.rm = TRUE)))
  message(sprintf("  ΔRMSE: %+.3f", mean(paired$d_rmse, na.rm = TRUE)))
  message(sprintf("  ΔMQI : %+.4f", mean(paired$d_mqi,  na.rm = TRUE)))
  message("  (ΔCCC > 0 favorece ", c1, "; |ΔCCC| menor que o desvio entre seeds = empate técnico)")
}

# ── Relatório final ───────────────────────────────────────────────────────────

message("\n── Resumo por config (média ± desvio entre ", length(seeds), " seeds) ──")
for (i in seq_len(nrow(config_summary))) {
  s <- config_summary[i, ]
  message("\n  ", s$config_id, " (n=", s$n_seeds, "):")
  message(sprintf("    CCC  : %.4f ± %.4f", s$ccc_mean,  s$ccc_sd))
  message(sprintf("    MAE  : %.3f ± %.3f", s$mae_mean,  s$mae_sd))
  message(sprintf("    RMSE : %.3f ± %.3f", s$rmse_mean, s$rmse_sd))
  message(sprintf("    R²   : %.4f ± %.4f", s$r2_mean,   s$r2_sd))
  message(sprintf("    NSE  : %.4f ± %.4f", s$nse_mean,  s$nse_sd))
  message(sprintf("    RPD  : %.3f ± %.3f", s$rpd_mean,  s$rpd_sd))
  message(sprintf("    MQI  : %.4f ± %.4f", s$mqi_mean,  s$mqi_sd))
}

message("\nResultados salvos em: ", output_dir)
