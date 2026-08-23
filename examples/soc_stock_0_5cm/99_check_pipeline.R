project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("dplyr", "readr", "tibble", "purrr", "stringr")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

# ══════════════════════════════════════════════════════════════════════════════
# 99 — Checkpoint de qualidade do pipeline (check & recheck)
#
# Roda checagens automáticas sobre o que já foi executado (01, 02, ...),
# comparando contra limiares conhecidos e contra a consistência interna dos
# próprios arquivos. Objetivo: pegar problemas ESTRUTURAIS (tipo o bug do
# clip PNV, que silenciosamente descartou ~55% dos dados por ~2 meses de
# processamento) no minuto em que a etapa termina, não semanas depois.
#
# Como crescer isto: cada etapa (01, 02, 03, ...) tem sua própria seção
# `check_0X_*()`. Ao terminar de rodar uma nova etapa do pipeline, adicione
# uma seção nova aqui seguindo o mesmo padrão (ver `add_check()` abaixo) e
# rode o script inteiro de novo — ele re-verifica tudo que já rodou, não só
# o novo.
#
# Uso: source() direto, sem parâmetros. Roda em segundos (só lê CSVs
# pequenos e metadados — NUNCA carrega os arrays de patches inteiros, que
# têm dezenas de GB; usa só o manifest, que já tem os números agregados).
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"

data_dir     <- file.path(project_root, "data", "processed", "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata", "soc_stock_modeling", target_label)
patch_dir    <- file.path(project_root, "outputs", "patches", "soc_stock_modeling", target_label)
patch_meta_dir <- file.path(metadata_dir, "patches")

# ── Infraestrutura de checagem ──────────────────────────────────────────────────

.results <- tibble::tibble(
  stage = character(), check = character(), status = character(), detail = character()
)

add_check <- function(stage, check, status, detail = "") {
  .results <<- dplyr::bind_rows(
    .results,
    tibble::tibble(stage = stage, check = check, status = status, detail = detail)
  )
  icon <- switch(status, PASS = "  [OK]", WARN = "[WARN]", FAIL = "[FAIL]", "  [?]")
  message(sprintf("%s %-6s | %-45s | %s", icon, stage, check, detail))
}

check_exists <- function(stage, label, path) {
  ok <- file.exists(path)
  add_check(stage, label, if (ok) "PASS" else "FAIL",
            if (ok) path else paste("NAO ENCONTRADO:", path))
  ok
}

check_threshold <- function(stage, label, value, warn_above, fail_above, unit = "%") {
  status <- if (value > fail_above) "FAIL" else if (value > warn_above) "WARN" else "PASS"
  add_check(stage, label, status,
            sprintf("valor=%.2f%s (warn>%.1f%s, fail>%.1f%s)",
                    value, unit, warn_above, unit, fail_above, unit))
  status
}

check_equal <- function(stage, label, a, b, name_a = "a", name_b = "b") {
  ok <- isTRUE(all.equal(a, b, tolerance = 1e-6))
  add_check(stage, label, if (ok) "PASS" else "FAIL",
            sprintf("%s=%s | %s=%s", name_a, a, name_b, b))
  ok
}

message("\n", strrep("=", 90))
message("99 — Checkpoint de qualidade do pipeline: ", target_label)
message(strrep("=", 90), "\n")

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 01 — Preparo do dataset tabular
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Etapa 01: preparo do dataset --\n")

f_qc      <- file.path(metadata_dir, "qc_summary.csv")
f_dscheck <- file.path(metadata_dir, "dataset_check.csv")
f_split   <- file.path(metadata_dir, "split_check.csv")
f_ptype   <- file.path(metadata_dir, "predictor_type_table.csv")
f_scaling <- file.path(metadata_dir, "predictor_scaling.csv")
f_rtable  <- file.path(metadata_dir, "raster_table_used.csv")
f_tconfig <- file.path(metadata_dir, "target_config.csv")

files_01 <- c(f_qc, f_dscheck, f_split, f_ptype, f_scaling, f_rtable, f_tconfig,
             file.path(data_dir, c("train_raw.csv", "train_scaled.csv",
                                    "validation_raw.csv", "validation_scaled.csv",
                                    "test_raw.csv", "test_scaled.csv")))
all_01_exist <- all(purrr::map_lgl(files_01, ~ check_exists("01", basename(.x), .x)))

if (all_01_exist) {

  qc      <- readr::read_csv2(f_qc, show_col_types = FALSE)
  dscheck <- readr::read_csv2(f_dscheck, show_col_types = FALSE)
  split   <- readr::read_csv2(f_split, show_col_types = FALSE)
  ptype   <- readr::read_csv2(f_ptype, show_col_types = FALSE)
  scaling <- readr::read_csv2(f_scaling, show_col_types = FALSE)
  rtable  <- readr::read_csv2(f_rtable, show_col_types = FALSE)
  tconfig <- readr::read_csv2(f_tconfig, show_col_types = FALSE)

  # A checagem mais importante: fracao de linhas descartadas por problema de
  # PREDITOR (nao de alvo). Alvo com problema (spline ruim, NA, <=0) e normal
  # e esperado; preditor com problema em excesso e a assinatura do tipo de
  # bug que causou a perda de ~55% dos dados por 2 meses (clip vs NA).
  pct_pred_problem <- 100 * qc$n_predictor_problem / qc$n_rows_extracted
  check_threshold("01", "pct linhas com problema de PREDITOR (nao-alvo)",
                  pct_pred_problem, warn_above = 2, fail_above = 10)

  # Contagem de preditores consistente entre TODOS os arquivos que deveriam
  # concordar.
  n_pred_rtable  <- nrow(rtable)
  n_pred_ptype   <- nrow(ptype)
  n_pred_scaling <- nrow(scaling)
  n_pred_dscheck <- dscheck$n_predictors[1]
  n_pred_tconfig <- tconfig$n_predictors_final[1]

  check_equal("01", "n_preditores: raster_table vs predictor_type_table",
              n_pred_rtable, n_pred_ptype, "raster_table", "predictor_type")
  check_equal("01", "n_preditores: raster_table vs predictor_scaling",
              n_pred_rtable, n_pred_scaling, "raster_table", "scaling")
  check_equal("01", "n_preditores: raster_table vs dataset_check",
              n_pred_rtable, n_pred_dscheck, "raster_table", "dataset_check")
  check_equal("01", "n_preditores: raster_table vs target_config",
              n_pred_rtable, n_pred_tconfig, "raster_table", "target_config")

  # dummy + percentage + continuous deve somar o total de preditores
  n_type_sum <- dscheck$n_dummy_predictors[1] + dscheck$n_percentage_predictors[1] +
    dscheck$n_continuous_predictors[1]
  check_equal("01", "soma dummy+percentage+continuous == total preditores",
              n_type_sum, n_pred_rtable, "soma_tipos", "total")

  # Proporcao do split perto de 70/15/15 (tolerancia 2 p.p.)
  total_n <- sum(split$n)
  for (i in seq_len(nrow(split))) {
    role <- split$dataset_role[i]
    pct  <- 100 * split$n[i] / total_n
    expected <- c(train = 70, validation = 15, test = 15)[[role]]
    dev <- abs(pct - expected)
    status <- if (dev > 3) "FAIL" else if (dev > 1) "WARN" else "PASS"
    add_check("01", paste0("split % ", role), status,
              sprintf("%.1f%% (esperado ~%d%%)", pct, expected))
  }

  # target_native e target_log1p sao consistentes (log1p(native) == log1p)
  # -- checagem indireta via mediana ja calculada em dataset_check.csv
  implied_log1p <- log1p(dscheck$median_target[1])
  check_equal("01", "median_target_log1p == log1p(median_target)",
              round(dscheck$median_target_log1p[1], 4), round(implied_log1p, 4),
              "salvo", "recalculado")

  # scaling de zscore nao pode ter sd degenerado (checagem redundante ao que
  # o proprio 01 ja faz, mas re-verifica no arquivo final salvo em disco)
  zscore_rows <- dplyr::filter(scaling, scaling_method == "zscore_train")
  n_bad_sd <- sum(is.na(zscore_rows$train_sd) | zscore_rows$train_sd <= 0, na.rm = TRUE)
  add_check("01", "scaling zscore: nenhum train_sd degenerado",
            if (n_bad_sd == 0) "PASS" else "FAIL",
            sprintf("%d preditores com sd invalido", n_bad_sd))

  # row counts iguais entre raw.csv e scaled.csv por split (mesma linhagem).
  # col_select=1 mantem a leitura rapida mesmo com 187+ colunas nos CSVs.
  for (role in c("train", "validation", "test")) {
    n_raw    <- nrow(readr::read_csv2(file.path(data_dir, paste0(role, "_raw.csv")),
                                      show_col_types = FALSE, col_select = 1))
    n_scaled <- nrow(readr::read_csv2(file.path(data_dir, paste0(role, "_scaled.csv")),
                                      show_col_types = FALSE, col_select = 1))
    check_equal("01", paste0("n_linhas raw vs scaled (", role, ")"),
                n_raw, n_scaled, "raw", "scaled")
  }

} else {
  message("Etapa 01 incompleta -- pulando checagens de conteudo.")
}

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 02 — Extração de patches
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Etapa 02: extracao de patches --\n")

f_manifest     <- file.path(patch_meta_dir, "patch_manifest.csv")
f_meta_train   <- file.path(patch_meta_dir, "meta_train.csv")
f_meta_val     <- file.path(patch_meta_dir, "meta_validation.csv")
f_meta_test    <- file.path(patch_meta_dir, "meta_test.csv")
f_patches_rds  <- file.path(patch_dir, "patches_all_splits.rds")
f_manifest_rds <- file.path(patch_dir, "patch_manifest.rds")

files_02 <- c(f_manifest, f_meta_train, f_meta_val, f_meta_test,
             f_patches_rds, f_manifest_rds)
all_02_exist <- all(purrr::map_lgl(files_02, ~ check_exists("02", basename(.x), .x)))

if (all_02_exist) {

  manifest <- readr::read_csv2(f_manifest, show_col_types = FALSE)

  # A CHECAGEM MAIS IMPORTANTE DESTE SCRIPT INTEIRO: fracao de perfis
  # descartados na extracao de patches. Antes da correcao do clip PNV, isso
  # rodava consistentemente em ~55%. Depois da correcao, esperado < 2%.
  # Se isso voltar pra cima de 10%, ALGO REGREDIU -- pare e investigue antes
  # de gastar dias/semanas de tuning/predicao em cima de dado quebrado.
  for (role in c("train", "validation", "test")) {
    col <- paste0("pct_", role, "_removed")
    val <- manifest[[col]][1]
    check_threshold("02", paste0("pct_", role, "_removed (janela completa)"),
                    val, warn_above = 2, fail_above = 10)
  }

  # n_channels e window_sizes batem com o que o 01 preparou / o esperado
  check_equal("02", "n_channels == 187", manifest$n_channels[1], 187L,
              "manifest", "esperado")

  expected_windows <- "3, 9, 15"
  ok_windows <- identical(trimws(manifest$window_sizes_extracted[1]), expected_windows)
  add_check("02", "window_sizes_extracted == '3, 9, 15'",
            if (ok_windows) "PASS" else "FAIL",
            paste("valor:", manifest$window_sizes_extracted[1]))

  # n_*_valid do manifest bate com o numero real de linhas em meta_*.csv
  n_meta_train <- nrow(readr::read_csv2(f_meta_train, show_col_types = FALSE))
  n_meta_val   <- nrow(readr::read_csv2(f_meta_val,   show_col_types = FALSE))
  n_meta_test  <- nrow(readr::read_csv2(f_meta_test,  show_col_types = FALSE))

  check_equal("02", "n_train_valid: manifest vs meta_train.csv",
              manifest$n_train_valid[1], n_meta_train, "manifest", "meta_csv")
  check_equal("02", "n_validation_valid: manifest vs meta_validation.csv",
              manifest$n_validation_valid[1], n_meta_val, "manifest", "meta_csv")
  check_equal("02", "n_test_valid: manifest vs meta_test.csv",
              manifest$n_test_valid[1], n_meta_test, "manifest", "meta_csv")

  # tamanho do arquivo de patches nao deveria ser suspeitosamente pequeno
  # (ex.: um crash no meio da escrita deixaria um arquivo truncado)
  patches_size_gb <- file.info(f_patches_rds)$size / 1e9
  add_check("02", "patches_all_splits.rds tem tamanho plausivel",
            if (patches_size_gb > 1) "PASS" else "WARN",
            sprintf("%.1f GB", patches_size_gb))

} else {
  message("Etapa 02 incompleta -- pulando checagens de conteudo.")
}

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 03 — Busca de hiperparâmetros (tuning)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Etapa 03: tuning de hiperparametros --\n")

tuning_dir <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling", target_label)

if (!dir.exists(tuning_dir)) {
  message("Etapa 03 nao iniciada -- pasta nao encontrada: ", tuning_dir)
} else {
  tuning_runs <- list.dirs(tuning_dir, recursive = FALSE, full.names = FALSE)
  if (length(tuning_runs) == 0) {
    message("Etapa 03 incompleta -- nenhum run encontrado em: ", tuning_dir)
  } else {
    # Mais recente por ordenacao do nome (run_id e timestamped) -- mesmo
    # criterio usado para resolver "latest" no 04/05/06.
    tuning_run_id <- sort(tuning_runs, decreasing = TRUE)[1]
    run_dir <- file.path(tuning_dir, tuning_run_id)
    message("Run mais recente: ", tuning_run_id)

    f_grid_csv <- file.path(run_dir, "tune_grid.csv")
    f_grid_rds <- file.path(run_dir, "tune_grid.rds")
    f_cmp_all  <- file.path(run_dir, "comparison", "comparison_all.csv")
    f_cmp_rank <- file.path(run_dir, "comparison", "comparison_ranked.csv")

    files_03 <- c(f_grid_csv, f_grid_rds, f_cmp_all, f_cmp_rank)
    all_03_exist <- all(purrr::map_lgl(files_03, ~ check_exists("03", basename(.x), .x)))

    if (all_03_exist) {

      tune_grid  <- readr::read_csv2(f_grid_csv, show_col_types = FALSE)
      comparison <- readr::read_csv2(f_cmp_rank, show_col_types = FALSE)

      n_grid <- nrow(tune_grid)
      n_cmp  <- nrow(comparison)

      # A checagem mais importante desta etapa: nenhuma config do grid ficou
      # pra tras (crash silencioso, config pulada por engano no resume, etc.)
      missing_ids <- setdiff(tune_grid$config_id, comparison$config_id)
      add_check("03", "todas as configs do grid tem linha na comparacao",
                if (length(missing_ids) == 0) "PASS" else "FAIL",
                if (length(missing_ids) == 0) sprintf("%d/%d configs", n_cmp, n_grid)
                else paste("faltando:", paste(missing_ids, collapse = ", ")))

      # Nenhuma linha extra na comparacao que nao esteja no grid atual --
      # indicaria mistura de runs diferentes (ex.: resume com tune_grid trocado
      # sem passar por um run_id novo).
      extra_ids <- setdiff(comparison$config_id, tune_grid$config_id)
      add_check("03", "nenhuma config na comparacao fora do grid atual",
                if (length(extra_ids) == 0) "PASS" else "FAIL",
                if (length(extra_ids) == 0) "" else paste("extras:", paste(extra_ids, collapse = ", ")))

      # status == success para todas -- configs com erro nunca escrevem linha
      # em comparison_all.csv (ver run_cnn_tuning), entao qualquer coisa
      # != success aqui seria corrupcao inesperada do CSV, nao uma falha normal
      # de treino (essas simplesmente nao aparecem, ja coberto pelo check acima).
      n_not_success <- sum(comparison$status != "success", na.rm = TRUE)
      add_check("03", "todas as linhas tem status == success",
                if (n_not_success == 0) "PASS" else "FAIL",
                sprintf("%d linha(s) com status != success", n_not_success))

      # Cada config com linha na comparacao precisa ter o checkpoint .pt -- e
      # o sinal de "realmente terminou o treino" usado pelo resume (ver
      # run_cnn_tuning em R/train_cnn.R). Uma linha sem checkpoint deixaria um
      # resume futuro confuso sobre se aquela config precisa ser retreinada.
      ckpt_files <- file.path(run_dir, "models", paste0(comparison$config_id, "_best.pt"))
      n_missing_ckpt <- sum(!file.exists(ckpt_files))
      add_check("03", "todo config_id da comparacao tem checkpoint .pt",
                if (n_missing_ckpt == 0) "PASS" else "FAIL",
                sprintf("%d checkpoint(s) faltando", n_missing_ckpt))

      # best_epoch nao pode ser NA nem <= 0 (indicaria que o treino nunca
      # passou no criterio de melhora do early stopping -- treino quebrado).
      n_bad_epoch <- sum(is.na(comparison$best_epoch) | comparison$best_epoch <= 0)
      add_check("03", "best_epoch valido (nao-NA, > 0) em todas as configs",
                if (n_bad_epoch == 0) "PASS" else "FAIL",
                sprintf("%d config(s) com best_epoch invalido", n_bad_epoch))

      # Metricas de validacao dentro de faixa FISICAMENTE plausivel (nao NA,
      # CCC em [-1,1], MAE/RMSE > 0). Nao julga "quao bom" o modelo e -- isso
      # e decisao de modelagem, nao bug estrutural -- so descarta valores
      # impossiveis (sinal de erro no calculo, nao de modelo ruim).
      n_na_metrics <- sum(is.na(comparison$val_ccc) | is.na(comparison$val_mae) |
                          is.na(comparison$val_rmse))
      add_check("03", "val_ccc/val_mae/val_rmse sem NA",
                if (n_na_metrics == 0) "PASS" else "FAIL",
                sprintf("%d config(s) com metrica NA", n_na_metrics))

      n_ccc_out_of_range <- sum(comparison$val_ccc < -1 | comparison$val_ccc > 1, na.rm = TRUE)
      add_check("03", "val_ccc dentro de [-1, 1]",
                if (n_ccc_out_of_range == 0) "PASS" else "FAIL",
                sprintf("%d config(s) fora do range", n_ccc_out_of_range))

      n_nonpos_error <- sum(comparison$val_mae <= 0 | comparison$val_rmse <= 0, na.rm = TRUE)
      add_check("03", "val_mae e val_rmse > 0",
                if (n_nonpos_error == 0) "PASS" else "FAIL",
                sprintf("%d config(s) com erro <= 0", n_nonpos_error))

      # rank 1 realmente e o maior val_ccc -- confirma que a ordenacao nao
      # inverteu nem ficou desalinhada apos um resume que reordenou linhas.
      top_by_rank <- comparison$config_id[comparison$rank == 1][1]
      top_by_ccc  <- comparison$config_id[which.max(comparison$val_ccc)]
      check_equal("03", "rank==1 corresponde ao maior val_ccc",
                  top_by_rank, top_by_ccc, "rank_1", "max_ccc")

      # gate_summary.csv so deveria existir para configs dual-branch (janela
      # com "x" no nome, ex. "9x15") com gate != no_gate_concat -- confirma
      # que a logica condicional de extract_gate_analysis() nao esta gerando
      # (ou deixando de gerar) arquivo para o tipo de config errado.
      gated_ids <- comparison$config_id[
        grepl("x", comparison$window_sizes) & comparison$gate_type != "no_gate_concat"
      ]
      gate_files <- file.path(run_dir, "gates", paste0(gated_ids, "_gate_summary.csv"))
      n_missing_gate <- sum(!file.exists(gate_files))
      add_check("03", "gate_summary.csv existe p/ toda config dual-branch com gate",
                if (n_missing_gate == 0) "PASS" else "WARN",
                sprintf("%d/%d faltando", n_missing_gate, length(gated_ids)))

      message(sprintf(
        "\n  Melhor config (val_ccc): %s | CCC=%.3f | MAE=%.2f | RMSE=%.2f | janela=%s | gate=%s",
        top_by_ccc,
        comparison$val_ccc[comparison$config_id == top_by_ccc][1],
        comparison$val_mae[comparison$config_id == top_by_ccc][1],
        comparison$val_rmse[comparison$config_id == top_by_ccc][1],
        comparison$window_sizes[comparison$config_id == top_by_ccc][1],
        comparison$gate_type[comparison$config_id == top_by_ccc][1]))

    } else {
      message("Etapa 03 incompleta -- pulando checagens de conteudo.")
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 04 — Modelo final (ensemble multi-seed)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Etapa 04: modelo final (ensemble multi-seed) --\n")

final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

if (!dir.exists(final_model_base)) {
  message("Etapa 04 nao iniciada -- pasta nao encontrada: ", final_model_base)
} else {
  final_runs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  final_runs <- final_runs[grepl("^final_", final_runs)]
  if (length(final_runs) == 0) {
    message("Etapa 04 incompleta -- nenhum run encontrado em: ", final_model_base)
  } else {
    final_run_id <- sort(final_runs, decreasing = TRUE)[1]
    run_dir <- file.path(final_model_base, final_run_id)
    message("Run mais recente: ", final_run_id)

    f_summary_rds <- file.path(run_dir, "comparison", "final_run_summary.rds")
    f_all_seeds   <- file.path(run_dir, "comparison", "all_seed_results_test.csv")
    f_cfg_summary <- file.path(run_dir, "comparison", "config_summary_test.csv")

    files_04 <- c(f_summary_rds, f_all_seeds, f_cfg_summary)
    all_04_exist <- all(purrr::map_lgl(files_04, ~ check_exists("04", basename(.x), .x)))

    if (all_04_exist) {

      summary_rds      <- readRDS(f_summary_rds)
      all_seed_results  <- readr::read_csv2(f_all_seeds,   show_col_types = FALSE)
      config_summary    <- readr::read_csv2(f_cfg_summary, show_col_types = FALSE)

      selected_cfgs    <- summary_rds$selected_cfgs
      seeds_expected    <- summary_rds$seeds
      n_seeds_expected  <- length(seeds_expected)

      # O 04 grava qual run de tuning (etapa 03) usou -- confirma que essa
      # pasta ainda existe (nao foi apagada/renomeada depois) e, quando a
      # etapa 03 tambem rodou nesta mesma checagem, que e exatamente o run
      # mais recente resolvido la em cima (evita ficar preso num run antigo
      # por engano, ex.: tuning_run_id fixo esquecido no script).
      linked_tuning_dir <- file.path(tuning_dir, summary_rds$tuning_run_id)
      add_check("04", "tuning_run_id referenciado pelo 04 ainda existe",
                if (dir.exists(linked_tuning_dir)) "PASS" else "FAIL",
                summary_rds$tuning_run_id)
      if (exists("tuning_run_id") && all_03_exist) {
        check_equal("04", "tuning_run_id do 04 == run mais recente da etapa 03",
                    summary_rds$tuning_run_id, tuning_run_id, "usado_pelo_04", "mais_recente_03")
      }

      # Quando selected_config_ids foi deixado NULL (comportamento padrao,
      # recomendado no cabecalho do 04), o config escolhido tem que ser
      # exatamente o rank==1 do ranking de validacao daquele run de tuning --
      # senao o modelo final estaria sendo treinado numa arquitetura que nao
      # e a melhor encontrada na etapa 03. Selecao manual de top-N e valida,
      # entao isso e so um alerta (WARN), nao falha.
      if (exists("tuning_run_id") && all_03_exist &&
          identical(summary_rds$tuning_run_id, tuning_run_id)) {
        rank1_id <- comparison$config_id[comparison$rank == 1L]
        add_check("04", "config(s) selecionado(s) inclui o rank==1 da etapa 03",
                  if (rank1_id %in% selected_cfgs$config_id) "PASS" else "WARN",
                  paste0("rank1=", rank1_id, " | selecionados=",
                        paste(selected_cfgs$config_id, collapse = ", ")))
      }

      # Cada config selecionado precisa ter exatamente n_seeds_expected linhas
      # de resultado -- nem seed faltando (crash/erro silencioso), nem seed a
      # mais (resquicio de outro run com seeds diferentes).
      seed_counts <- dplyr::count(all_seed_results, config_id, name = "n_seeds_found")
      for (cid in selected_cfgs$config_id) {
        found <- seed_counts$n_seeds_found[seed_counts$config_id == cid]
        found <- if (length(found) == 0) 0L else found
        add_check("04", paste0("n_seeds completas (", cid, ")"),
                  if (found == n_seeds_expected) "PASS" else "FAIL",
                  sprintf("%d/%d seeds", found, n_seeds_expected))
      }

      # Cada (config, seed) esperado tem checkpoint .pt salvo -- mesmo
      # principio do check em "03": resultado sem modelo salvo por tras
      # deixaria o run inutilizavel para inferencia futura mesmo aparecendo
      # como sucesso na tabela de metricas.
      ckpt_paths <- character(0)
      for (cid in selected_cfgs$config_id) {
        ckpt_paths <- c(ckpt_paths, file.path(run_dir, cid, "models",
                                              sprintf("seed%04d_best.pt", seeds_expected)))
      }
      n_missing_ckpt <- sum(!file.exists(ckpt_paths))
      add_check("04", "todo (config, seed) esperado tem checkpoint .pt",
                if (n_missing_ckpt == 0) "PASS" else "FAIL",
                sprintf("%d/%d checkpoint(s) faltando", n_missing_ckpt, length(ckpt_paths)))

      # Metricas de teste sem NA e em faixa fisicamente plausivel (mesma
      # logica do 03: nao julga "quao bom", so descarta valores impossiveis).
      n_na_metrics <- sum(is.na(all_seed_results$ccc) | is.na(all_seed_results$mae) |
                          is.na(all_seed_results$rmse))
      add_check("04", "ccc/mae/rmse sem NA (todas as seeds)",
                if (n_na_metrics == 0) "PASS" else "FAIL",
                sprintf("%d linha(s) com metrica NA", n_na_metrics))

      n_ccc_out <- sum(all_seed_results$ccc < -1 | all_seed_results$ccc > 1, na.rm = TRUE)
      add_check("04", "ccc dentro de [-1, 1] (todas as seeds)",
                if (n_ccc_out == 0) "PASS" else "FAIL",
                sprintf("%d linha(s) fora do range", n_ccc_out))

      n_nonpos <- sum(all_seed_results$mae <= 0 | all_seed_results$rmse <= 0, na.rm = TRUE)
      add_check("04", "mae e rmse > 0 (todas as seeds)",
                if (n_nonpos == 0) "PASS" else "FAIL",
                sprintf("%d linha(s) com erro <= 0", n_nonpos))

      # Estabilidade entre seeds: SD do CCC como % da media. Um desvio grande
      # (ver docs/design_decisions.md secao 11) indica treino instavel/pouco
      # reprodutivel, nao so "sorte" de inicializacao -- resultado publicavel
      # deveria ter baixo desvio.
      for (i in seq_len(nrow(config_summary))) {
        cs <- config_summary[i, ]
        pct_sd <- 100 * cs$ccc_sd / cs$ccc_mean
        check_threshold("04", paste0("CCC SD relativo (", cs$config_id, ")"),
                        pct_sd, warn_above = 10, fail_above = 20, unit = "% da media")
      }

      # config_summary bate com a agregacao recalculada a partir de
      # all_seed_results -- redundancia contra corrupcao/desalinhamento do CSV.
      recalc <- all_seed_results %>%
        dplyr::group_by(config_id) %>%
        dplyr::summarise(ccc_mean_recalc = mean(ccc), .groups = "drop")
      merged <- dplyr::left_join(config_summary, recalc, by = "config_id")
      for (i in seq_len(nrow(merged))) {
        check_equal("04", paste0("ccc_mean salvo == recalculado (", merged$config_id[i], ")"),
                    round(merged$ccc_mean[i], 6), round(merged$ccc_mean_recalc[i], 6),
                    "salvo", "recalculado")
      }

      # gate_summary.csv por seed so deveria existir para configs dual-branch
      # (2 janelas) com gate_type != no_gate_concat -- mesma logica do 03.
      for (i in seq_len(nrow(selected_cfgs))) {
        cid <- selected_cfgs$config_id[i]
        ws  <- selected_cfgs$window_sizes[[i]]
        gt  <- selected_cfgs$gate_type[i]
        if (length(ws) == 2L && gt != "no_gate_concat") {
          gate_files <- file.path(run_dir, cid, "gates",
                                  sprintf("seed%04d_gate_summary.csv", seeds_expected))
          n_missing_gate <- sum(!file.exists(gate_files))
          add_check("04", paste0("gate_summary.csv por seed existe (", cid, ")"),
                    if (n_missing_gate == 0) "PASS" else "WARN",
                    sprintf("%d/%d faltando", n_missing_gate, length(seeds_expected)))
        }
      }

      for (i in seq_len(nrow(config_summary))) {
        cs <- config_summary[i, ]
        message(sprintf(
          "\n  Modelo final [%s]: CCC=%.4f +/- %.4f | MAE=%.3f | RMSE=%.3f | %d seeds",
          cs$config_id, cs$ccc_mean, cs$ccc_sd, cs$mae_mean, cs$rmse_mean, cs$n_seeds))
      }

    } else {
      message("Etapa 04 incompleta -- pulando checagens de conteudo.")
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# [Placeholder para etapas futuras]
#
# Etapa 05/06 (predicao espacial): valid_fraction dos tiles nao caiu de novo
# pra perto de zero -- mesma logica do check do 02, adaptada para os logs
# de shard/merge. Ver tambem 05a_test.R / 05c_estimate_eta.R, que ja cobrem
# parte disso para o pipeline 2D.
# ══════════════════════════════════════════════════════════════════════════════

# ── Resumo final ─────────────────────────────────────────────────────────────

message("\n", strrep("=", 90))
message("RESUMO")
message(strrep("=", 90))

n_pass <- sum(.results$status == "PASS")
n_warn <- sum(.results$status == "WARN")
n_fail <- sum(.results$status == "FAIL")

message(sprintf("\n  PASS: %d   WARN: %d   FAIL: %d   (total: %d checagens)\n",
                n_pass, n_warn, n_fail, nrow(.results)))

if (n_fail > 0) {
  message("Checagens que FALHARAM:")
  print(dplyr::filter(.results, status == "FAIL"), n = Inf, width = Inf)
}
if (n_warn > 0) {
  message("\nCheckagens com AVISO:")
  print(dplyr::filter(.results, status == "WARN"), n = Inf, width = Inf)
}

report_dir <- file.path(project_root, "outputs", "qc")
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
report_file <- file.path(report_dir, paste0("pipeline_check_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
readr::write_csv2(.results, report_file)
message("\nRelatorio completo salvo em: ", report_file)

if (n_fail > 0) {
  warning(n_fail, " checagem(ns) FALHOU. Revise antes de seguir para a proxima etapa.")
} else if (n_warn > 0) {
  message("\nNenhuma falha critica, mas ha avisos -- revise antes de investir tempo de GPU/CPU na proxima etapa.")
} else {
  message("\nTudo OK. Pode seguir para a proxima etapa do pipeline.")
}
