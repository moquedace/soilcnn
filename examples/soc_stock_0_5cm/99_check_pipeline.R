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
# [Placeholder para etapas futuras]
#
# Ao terminar de rodar o 03, adicione uma secao "ETAPA 03" aqui, seguindo o
# mesmo padrao: checar arquivos existem -> checar metricas dentro de faixas
# plausiveis -> checar consistencia entre arquivos relacionados. Ideias para
# quando chegar la:
#   - todas as configs do grid terminaram (nenhum config faltando no ranking)
#   - CCC/MAE/RMSE de validacao dentro de faixa plausivel (nao NA, nao
#     negativo absurdo, RMSE > 0)
#   - comparison_ranked.csv tem exatamente tune_length linhas (ou menos, se
#     algum config falhou -- nesse caso, WARN com quantos falharam)
#
# Etapa 04 (modelo final): todos os seeds terminaram, desvio entre seeds nao
# e absurdamente alto (>20% do CCC medio, por ex.), arquivos .pt existem e
# tem tamanho > 0 para cada seed.
#
# Etapa 05 (predicao espacial): valid_fraction dos tiles nao caiu de novo
# pra perto de zero -- mesma logica do check do 02, adaptada para os logs
# de shard/merge.
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
