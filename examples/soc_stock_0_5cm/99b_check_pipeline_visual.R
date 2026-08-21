project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("dplyr", "readr", "tibble", "purrr", "ggplot2", "tidyr")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

# ══════════════════════════════════════════════════════════════════════════════
# 99b — Checkpoint VISUAL do pipeline (companheiro do 99_check_pipeline.R)
#
# O 99 confere números (contagens, percentuais, consistência entre arquivos).
# Este aqui mostra COISA DE VERDADE na tela: onde os perfis estão no mapa, e
# como são os patches que a CNN vai realmente receber — recorte de elevação,
# vegetação e (de propósito, é o vilão da história) uma camada PNV, extraídos
# ao redor de perfis reais.
#
# Duas partes, custo bem diferente:
#   PARTE 1 (leve, segundos): mapa-múndi dos ~37 mil perfis por split. So le
#     split_metadata.csv.
#   PARTE 2 (pesada, minutos): carrega patches_all_splits.rds INTEIRO (~17 GB)
#     na memoria so pra tirar uns 6 patches de exemplo. E o unico jeito de
#     acessar o arquivo (RDS nao suporta leitura parcial) — rode quando nao
#     estiver com pouca RAM sobrando. Comente a PARTE 2 se so quiser o mapa.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"

metadata_dir   <- file.path(project_root, "outputs", "metadata", "soc_stock_modeling", target_label)
patch_dir      <- file.path(project_root, "outputs", "patches", "soc_stock_modeling", target_label)
patch_meta_dir <- file.path(metadata_dir, "patches")
fig_dir        <- file.path(project_root, "outputs", "qc", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 1 — Mapa-múndi dos perfis (leve)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Parte 1: mapa dos perfis por split --\n")

split_meta_file <- file.path(metadata_dir, "split_metadata.csv")
stopifnot(file.exists(split_meta_file))

split_meta <- readr::read_csv2(split_meta_file, show_col_types = FALSE)

message("Perfis totais: ", nrow(split_meta))
print(dplyr::count(split_meta, dataset_role))

p_map <- ggplot2::ggplot(split_meta, ggplot2::aes(x = x, y = y, color = dataset_role)) +
  ggplot2::geom_point(size = 2, alpha = 0.5) +
  ggplot2::coord_fixed() +
  ggplot2::scale_color_manual(values = c(train = "#1b9e77", validation = "#d95f02", test = "#7570b3")) +
  ggplot2::labs(
    title = paste0(target_label, " — ", format(nrow(split_meta), big.mark = ","),
                  " perfis WOSIS, por split"),
    subtitle = "Sem mapa-base de propósito: o formato dos continentes deveria aparecer sozinho pela densidade de pontos",
    x = "Longitude", y = "Latitude", color = "Split"
  ) +
  ggplot2::theme_bw() +
  ggplot2::guides(color = ggplot2::guide_legend(override.aes = list(size = 3, alpha = 1)))

print(p_map)
ggplot2::ggsave(file.path(fig_dir, "99b_world_map_profiles.png"), p_map,
                width = 12, height = 6, dpi = 150)
message("Salvo: ", file.path(fig_dir, "99b_world_map_profiles.png"))

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 2 — Patches reais ao redor de perfis (pesada — carrega ~17 GB)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Parte 2: patches reais (carregando patches_all_splits.rds, pode demorar) --\n")

manifest <- readRDS(file.path(patch_dir, "patch_manifest.rds"))
# predictor_cols_final e salvo como string unica separada por ";" (mesmo
# formato do patch_manifest.csv), nao como vetor -- precisa split.
predictor_cols <- strsplit(manifest$predictor_cols_final, ";")[[1]]

# Canais escolhidos por serem visualmente reconhecíveis (mesmo escalados em
# z-score, a FORMA espacial se preserva — só muda a unidade da escala de cor):
#   - elevação: relevo deveria aparecer nitidamente (cristas/vales)
#   - NDVI: padrão de vegetação
#   - pnv_open_forest_evergreen_broadleaf: a própria camada PNV que
#     investigamos a fundo (~33% de perda no interior antes da correção) —
#     ver essa camada saindo limpa aqui fecha o ciclo da investigação
channels_to_show <- c(
  elevacao = "ensemble_digital_terrain_model_v1_1",
  ndvi     = "landsat_2020_2025_ndvi",
  pnv      = "pnv_open_forest_evergreen_broadleaf"
)

missing_ch <- setdiff(channels_to_show, predictor_cols)
if (length(missing_ch) > 0) {
  stop("Canal(is) nao encontrado(s) em predictor_cols_final: ", paste(missing_ch, collapse = ", "))
}
channel_idx <- match(channels_to_show, predictor_cols)
names(channel_idx) <- names(channels_to_show)

message("Canais escolhidos e seus índices: ")
print(tibble::tibble(nome = names(channel_idx), predictor = channels_to_show, indice = channel_idx))

t0 <- Sys.time()
patches <- readRDS(file.path(patch_dir, "patches_all_splits.rds"))
message("patches_all_splits.rds carregado em ",
        round(difftime(Sys.time(), t0, units = "mins"), 1), " min")

set.seed(42)
n_examples <- 6
n_train <- nrow(patches$train$meta)
example_idx <- sample.int(n_train, n_examples)

window_arrays <- list(
  `3x3`   = patches$train$x_3x3_array,
  `9x9`   = patches$train$x_9x9_array,
  `15x15` = patches$train$x_15x15_array
)

meta_examples <- patches$train$meta[example_idx, ] %>%
  dplyr::mutate(example_id = paste0("perfil ", dplyr::row_number(),
                                    "\nSOC=", round(target_native, 1), " t/ha"))

# Monta um data.frame longo: uma linha por (exemplo, janela, canal, pixel)
patch_long <- purrr::map_dfr(seq_along(window_arrays), function(w_i) {
  w_name <- names(window_arrays)[w_i]
  arr    <- window_arrays[[w_i]]  # [N, C, w, w]
  w_size <- dim(arr)[3]

  purrr::map_dfr(seq_along(example_idx), function(e_i) {
    idx <- example_idx[e_i]
    purrr::map_dfr(names(channel_idx), function(ch_name) {
      ch <- channel_idx[[ch_name]]
      mat <- arr[idx, ch, , ]
      grid <- expand.grid(row = seq_len(w_size), col = seq_len(w_size))
      tibble::tibble(
        example    = meta_examples$example_id[e_i],
        window     = factor(w_name, levels = c("3x3", "9x9", "15x15")),
        channel    = ch_name,
        row        = grid$row,
        col        = grid$col,
        value      = as.vector(mat)
      )
    })
  })
})

message("\nValores extraídos (checagem rápida de sanidade — não deveria haver NA/Inf):")
print(dplyr::summarise(patch_long,
                       n = dplyr::n(),
                       n_na = sum(is.na(value)),
                       n_inf = sum(!is.finite(value) & !is.na(value)),
                       min = min(value, na.rm = TRUE),
                       max = max(value, na.rm = TRUE)))

for (ch_name in names(channel_idx)) {
  df_ch <- dplyr::filter(patch_long, channel == ch_name)

  p <- ggplot2::ggplot(df_ch, ggplot2::aes(x = col, y = row, fill = value)) +
    ggplot2::geom_raster() +
    ggplot2::scale_fill_viridis_c() +
    ggplot2::coord_fixed() +
    ggplot2::scale_y_reverse() +
    ggplot2::facet_grid(example ~ window, switch = "y") +
    ggplot2::labs(
      title = paste0(target_label, " — patches reais: ", ch_name,
                    " (", channels_to_show[[ch_name]], ")"),
      subtitle = "Valores em z-score (a forma espacial se preserva mesmo escalado)",
      x = NULL, y = NULL, fill = "z-score"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      strip.text.y.left = ggplot2::element_text(angle = 0, hjust = 1)
    )

  print(p)
  out_png <- file.path(fig_dir, paste0("99b_patches_", ch_name, ".png"))
  ggplot2::ggsave(out_png, p, width = 9, height = 11, dpi = 150)
  message("Salvo: ", out_png)
}

message("\nFeito. Figuras em: ", fig_dir)
message("Olhe principalmente se elevacao/ndvi mostram um padrao espacial coerente")
message("(nao ruido aleatorio) e se a janela maior (15x15) mostra mais contexto")
message("ao redor do mesmo centro que a menor (3x3) — devem estar alinhadas.")
