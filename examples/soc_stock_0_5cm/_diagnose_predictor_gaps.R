project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("terra", "dplyr", "readr", "tibble", "purrr", "stringr", "ggplot2")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)
source(file.path(project_root, "R", "utils.R"))

# ══════════════════════════════════════════════════════════════════════════════
# Diagnóstico: por que o mapa final tem falhas (speckle) espalhadas pelo mundo?
#
# Hipótese testada: o pipeline exige que TODOS os 187 preditores estejam
# finitos em TODA a janela (até 15x15 = 225 células) para prever um pixel.
# Se qualquer preditor tiver NA esparso e espalhado (comum em produtos raster
# globais — costuras de mosaico, falhas de sensor, lacunas de dado), um único
# pixel NA nesse preditor invalida uma vizinhança de até 15x15 ao redor dele.
# Isso amplifica NA esparso (pouco visível olhando 1 preditor sozinho) em
# "buracos" muito maiores e visíveis no mapa final.
#
# Este script:
#   1. Amostra pontos regulares sobre TERRA (usa uma camada de referência com
#      cobertura completa em terra — ex. elevação — como máscara de terra/água,
#      já que rodar NA-check nos 187 preditores completos, na resolução nativa,
#      é caro).
#   2. Para cada um dos 187 preditores, calcula a fração de NA SOBRE TERRA.
#   3. Ranqueia os preditores por fração de NA — aponta o(s) culpado(s).
#   4. Quantifica o efeito de amplificação da janela: compara
#        - fração válida "ingênua" (só o pixel central, sem considerar janela)
#        - fração válida real do pipeline (lida do valid_mask.tif já gerado)
#   5. Salva um CSV ranqueado + um gráfico de barras dos piores preditores.
#
# Uso: ajuste `n_samples` e `land_reference_predictor` abaixo se necessário.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"
config_id    <- "cfg_022"

# Quantos pontos amostrar (grade regular sobre o raster inteiro). 1-2 milhões
# é rápido (segundos a poucos minutos) e estatisticamente representativo.
n_samples <- 2000000L

# Preditor usado como máscara de "isto é terra" — precisa ter cobertura
# completa sobre toda a terra firme (sem gaps conhecidos). O DEM é a escolha
# mais segura (produtos DEM globais são tipicamente void-filled).
land_reference_pattern <- "^ensemble_digital_terrain_model"

metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)
raster_table_file <- file.path(metadata_dir, "raster_table_used.csv")

output_dir <- file.path(project_root, "outputs", "spatial_prediction",
                        "soc_stock_modeling", target_label, config_id)
diag_dir <- file.path(output_dir, "diagnostics")
create_output_dirs(diag_dir)

# ── Carregar tabela de preditores ──────────────────────────────────────────────

raster_table <- readr::read_csv2(raster_table_file, show_col_types = FALSE)
predictor_cols <- raster_table$predictor
n_predictors <- length(predictor_cols)

message("Preditores: ", n_predictors)

missing_files <- raster_table$raster_file[!file.exists(raster_table$raster_file)]
if (length(missing_files) > 0) {
  print(missing_files)
  stop("Alguns rasters de preditores nao existem mais.")
}

rast_stack <- terra::rast(raster_table$raster_file)
names(rast_stack) <- predictor_cols

r_nrow <- terra::nrow(rast_stack)
r_ncol <- terra::ncol(rast_stack)
message("Grade: ", r_nrow, " x ", r_ncol, " x ", n_predictors, " bandas")

# ── Identificar preditor de referência (máscara de terra) ─────────────────────

land_ref_idx <- grep(land_reference_pattern, predictor_cols, ignore.case = TRUE)

if (length(land_ref_idx) == 0) {
  message("\nWARNING: nenhum preditor bateu com o padrao '", land_reference_pattern,
          "'. Preditores disponiveis (primeiros 30):")
  print(head(predictor_cols, 30))
  stop("Ajuste 'land_reference_pattern' para um preditor com cobertura completa em terra.")
}

land_ref_name <- predictor_cols[land_ref_idx[1]]
message("Preditor de referencia (mascara de terra): ", land_ref_name)

# ── Amostragem regular sobre TODO o raster (inclui oceano, filtramos depois) ──

message("\nAmostrando ", format(n_samples, big.mark = ","), " pontos regulares...")
t0 <- Sys.time()

samp <- terra::spatSample(rast_stack, size = n_samples, method = "regular",
                          na.rm = FALSE, values = TRUE, xy = FALSE)

message("Amostragem concluida em ",
        round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s | ",
        nrow(samp), " pontos amostrados.")

# ── Máscara de terra: pontos onde o preditor de referência é finito ───────────

is_land <- is.finite(samp[[land_ref_name]])
n_land  <- sum(is_land)
message(sprintf("Pontos classificados como terra: %s / %s (%.1f%%)",
                format(n_land, big.mark = ","), format(nrow(samp), big.mark = ","),
                100 * n_land / nrow(samp)))

samp_land <- samp[is_land, , drop = FALSE]

# ── Fração de NA por preditor, SOBRE TERRA ─────────────────────────────────────

na_frac <- purrr::map_dbl(predictor_cols, function(p) {
  mean(!is.finite(samp_land[[p]]))
})

na_summary <- tibble::tibble(
  predictor = predictor_cols,
  na_fraction_over_land = na_frac,
  na_pct_over_land = round(100 * na_frac, 4)
) %>%
  dplyr::arrange(dplyr::desc(na_fraction_over_land))

message("\n── Top 20 preditores por fração de NA sobre terra ──────────────────")
print(head(na_summary, 20), n = 20, width = Inf)

n_offenders <- sum(na_summary$na_fraction_over_land > 0)
message(sprintf("\n%d de %d preditores têm pelo menos 1 NA sobre terra na amostra.",
                n_offenders, n_predictors))

# ── Amplificação pela janela: ingênuo (1 pixel) vs pipeline (janela completa) ──
# "Ingênuo": fração de pontos-terra onde TODOS os 187 preditores são finitos
#            NESSE ÚNICO PIXEL (sem considerar vizinhança).
# Isso é o piso teórico de cobertura SE a janela não amplificasse nada.
# Compare com valid_fraction real do pipeline (raster_summary / valid_mask).

all_finite_center <- rowSums(!sapply(predictor_cols, function(p) {
  is.finite(samp_land[[p]])
})) == 0
naive_valid_fraction <- mean(all_finite_center)

message(sprintf(
  "\nCobertura ingênua (1 pixel, sem janela): %.2f%% da terra amostrada",
  100 * naive_valid_fraction))
message("(compare esse número com o valid_fraction real do pipeline, salvo em")
message(" outputs/spatial_prediction/.../cfg_022/log/prediction_config*.csv")
message(" ou no log de merge: se o valor final for MUITO menor que este, a")
message(" amplificação pela janela é a causa dominante das falhas.)")

# ── Salvar resultados ──────────────────────────────────────────────────────────

safe_write_csv2(na_summary, file.path(diag_dir, "predictor_na_fraction_over_land.csv"))

diag_summary <- tibble::tibble(
  n_samples_total = nrow(samp),
  n_samples_land = n_land,
  land_reference_predictor = land_ref_name,
  n_predictors = n_predictors,
  n_predictors_with_any_na_over_land = n_offenders,
  naive_valid_fraction_single_pixel = naive_valid_fraction,
  computed_at = as.character(Sys.time())
)
safe_write_csv2(diag_summary, file.path(diag_dir, "diagnostic_summary.csv"))

# ── Gráfico dos piores ofensores ────────────────────────────────────────────────

top_n <- 25
plot_data <- na_summary %>%
  dplyr::slice_head(n = top_n) %>%
  dplyr::filter(na_fraction_over_land > 0) %>%
  dplyr::mutate(predictor = factor(predictor, levels = rev(predictor)))

if (nrow(plot_data) > 0) {
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = na_pct_over_land, y = predictor)) +
    ggplot2::geom_col() +
    ggplot2::labs(
      title = paste0(target_label, " — preditores com mais NA sobre terra"),
      subtitle = paste0("Amostra: ", format(n_land, big.mark = ","), " pontos de terra"),
      x = "% de NA sobre terra (amostrado)", y = NULL
    ) +
    ggplot2::theme_bw()

  print(p)
  ggplot2::ggsave(
    filename = file.path(diag_dir, "predictor_na_fraction_top25.png"),
    plot = p, width = 9, height = 7, dpi = 300
  )
}

message("\n── Diagnóstico concluído ─────────────────────────────────────────")
message("Resultados salvos em: ", diag_dir)
message("  predictor_na_fraction_over_land.csv — ranking completo dos 187 preditores")
message("  diagnostic_summary.csv — resumo + cobertura ingênua")
message("  predictor_na_fraction_top25.png — gráfico dos piores ofensores")
