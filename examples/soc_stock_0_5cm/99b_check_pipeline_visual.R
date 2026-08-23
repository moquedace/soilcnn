project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("dplyr", "readr", "tibble", "purrr", "ggplot2", "tidyr", "DescTools")
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
# Este aqui mostra COISA DE VERDADE na tela: onde os perfis estão no mapa, se
# o tuning realmente melhorou, se o modelo final acerta o valor de SOC de
# verdade (não só a métrica agregada), e como são os patches que a CNN
# recebe — recorte de elevação, vegetação e (de propósito, é o vilão da
# história) uma camada PNV, extraídos ao redor de perfis reais.
#
# Quatro partes, custo crescente:
#   PARTE 1 (leve, segundos): mapa-múndi dos ~37 mil perfis por split. So le
#     split_metadata.csv.
#   PARTE 2 (leve, segundos): etapa 03 (tuning) — leaderboard dos configs e
#     "será que o vencedor acerta o valor real?" (obs x predito, validação).
#   PARTE 3 (leve, segundos): etapa 04 (modelo final) — curvas de treino por
#     seed, estabilidade entre seeds, e obs x predito no TEST (o ensemble
#     final, o número que de fato vai pro mapa).
#   PARTE 4 (pesada, minutos): carrega patches_all_splits.rds INTEIRO (~17 GB)
#     na memoria so pra tirar uns 6 patches de exemplo. E o unico jeito de
#     acessar o arquivo (RDS nao suporta leitura parcial) — rode quando nao
#     estiver com pouca RAM sobrando. Comente a PARTE 4 se nao quiser esperar.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"

metadata_dir   <- file.path(project_root, "outputs", "metadata", "soc_stock_modeling", target_label)
patch_dir      <- file.path(project_root, "outputs", "patches", "soc_stock_modeling", target_label)
patch_meta_dir <- file.path(metadata_dir, "patches")
tuning_base    <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling", target_label)
final_base     <- file.path(project_root, "outputs", "final_model", "soc_stock_modeling", target_label)
fig_dir        <- file.path(project_root, "outputs", "qc", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# Resolve o run mais recente numa pasta de runs timestamped (mesmo criterio
# usado no 99_check_pipeline.R e nos proprios 03/04/05): ordena os nomes das
# subpastas e pega o primeiro em ordem decrescente.
.latest_run <- function(base_dir, prefix) {
  runs <- list.dirs(base_dir, recursive = FALSE, full.names = FALSE)
  runs <- runs[grepl(paste0("^", prefix), runs)]
  if (length(runs) == 0) return(NULL)
  sort(runs, decreasing = TRUE)[1]
}

# CCC de Lin, calculado exatamente como em R/metrics.R (DescTools::CCC) --
# so pra anotar os graficos obs-x-predito com o mesmo numero que o resto do
# pipeline reportaria, sem precisar fazer source() do metrics.R inteiro.
.lin_ccc <- function(obs, pred) {
  as.numeric(DescTools::CCC(obs, pred, conf.level = 0.95)$rho.c$est)[1]
}

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
# PARTE 2 — Etapa 03: tuning (leaderboard + o vencedor acerta o valor real?)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Parte 2: etapa 03 (tuning) --\n")

tuning_run_id <- .latest_run(tuning_base, "soc_")

if (is.null(tuning_run_id)) {
  message("Etapa 03 nao encontrada em ", tuning_base, " -- pulando Parte 2.")
} else {

  tuning_run_dir <- file.path(tuning_base, tuning_run_id)
  message("Run de tuning: ", tuning_run_id)

  ranking <- readr::read_csv2(
    file.path(tuning_run_dir, "comparison", "comparison_ranked.csv"),
    show_col_types = FALSE
  )

  # ── Leaderboard: todos os configs, ordenados por val_ccc ──────────────────
  # Lollipop em vez de barra pura: a régua horizontal deixa mais fácil ver a
  # DIFERENÇA entre vizinhos no ranking (é isso que decide qual config vira o
  # modelo final), não só o valor absoluto de cada um.
  ranking_plot <- ranking %>%
    dplyr::mutate(
      config_id = factor(config_id, levels = config_id[order(val_ccc)]),
      is_winner = rank == 1L
    )

  p_leaderboard <- ggplot2::ggplot(ranking_plot,
                                   ggplot2::aes(x = val_ccc, y = config_id)) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, xend = val_ccc, yend = config_id,
                                       color = window_sizes), linewidth = 1) +
    ggplot2::geom_point(ggplot2::aes(color = window_sizes, size = is_winner)) +
    ggplot2::geom_point(data = dplyr::filter(ranking_plot, is_winner),
                        shape = 21, size = 5, stroke = 1.2, color = "black", fill = NA) +
    ggplot2::scale_size_manual(values = c(`TRUE` = 4, `FALSE` = 2.5), guide = "none") +
    ggplot2::coord_cartesian(xlim = c(0, max(ranking_plot$val_ccc) * 1.05)) +
    ggplot2::labs(
      title = paste0(target_label, " — leaderboard do tuning (", tuning_run_id, ")"),
      subtitle = paste0(nrow(ranking), " configs testados | círculo com contorno preto = rank 1 (vencedor)"),
      x = "CCC de validação", y = NULL, color = "Janela(s)"
    ) +
    ggplot2::theme_bw()

  print(p_leaderboard)
  ggplot2::ggsave(file.path(fig_dir, "99b_tuning_leaderboard.png"), p_leaderboard,
                  width = 9, height = max(4, 0.35 * nrow(ranking)), dpi = 150)
  message("Salvo: ", file.path(fig_dir, "99b_tuning_leaderboard.png"))

  # ── O vencedor acerta o valor real? Obs x Predito no split de VALIDAÇÃO ───
  # A métrica CCC=0.xx da tabela é abstrata; ver os pontos ao redor da reta
  # 1:1 é o que de fato convence que o "melhor config" aprendeu algo físico,
  # não só um número que ficou bom por acaso na agregação.
  best_id   <- ranking$config_id[ranking$rank == 1L]
  pred_file <- file.path(tuning_run_dir, "predictions", paste0(best_id, "_pred_all.csv"))

  if (file.exists(pred_file)) {
    pred_best <- readr::read_csv2(pred_file, show_col_types = FALSE) %>%
      dplyr::filter(dataset_role == "validation")

    ccc_val <- .lin_ccc(pred_best$obs, pred_best$pred)
    lims    <- range(c(pred_best$obs, pred_best$pred))

    p_best_val <- ggplot2::ggplot(pred_best, ggplot2::aes(x = obs, y = pred)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
      ggplot2::geom_point(alpha = 0.4, color = "#1b9e77") +
      ggplot2::coord_fixed(xlim = lims, ylim = lims) +
      ggplot2::annotate("text", x = lims[1], y = lims[2], hjust = 0, vjust = 1,
                        label = sprintf("CCC = %.3f\nn = %d", ccc_val, nrow(pred_best))) +
      ggplot2::labs(
        title = paste0(target_label, " — config vencedor (", best_id, "): observado x predito"),
        subtitle = "Split de validação | linha tracejada = 1:1 (acerto perfeito)",
        x = "SOC observado (t/ha)", y = "SOC predito (t/ha)"
      ) +
      ggplot2::theme_bw()

    print(p_best_val)
    ggplot2::ggsave(file.path(fig_dir, "99b_tuning_best_obs_vs_pred.png"), p_best_val,
                    width = 7, height = 7, dpi = 150)
    message("Salvo: ", file.path(fig_dir, "99b_tuning_best_obs_vs_pred.png"))
  } else {
    message("Predicoes do config vencedor nao encontradas: ", pred_file)
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 3 — Etapa 04: modelo final (convergência, estabilidade, acerto no test)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Parte 3: etapa 04 (modelo final, ensemble multi-seed) --\n")

final_run_id <- .latest_run(final_base, "final_")

if (is.null(final_run_id)) {
  message("Etapa 04 nao encontrada em ", final_base, " -- pulando Parte 3.")
} else {

  final_run_dir <- file.path(final_base, final_run_id)
  message("Run do modelo final: ", final_run_id)

  summary_rds <- readRDS(file.path(final_run_dir, "comparison", "final_run_summary.rds"))
  selected_cfgs <- summary_rds$selected_cfgs
  seeds         <- summary_rds$seeds

  # ── Curvas de treino, uma linha por seed ───────────────────────────────────
  # Se as sementes convergem para curvas parecidas, o treino é estável (não é
  # sorte de inicialização). Uma curva muito fora do feixe das outras é o
  # tipo de coisa que fica invisível numa média ± desvio, mas salta aos olhos
  # aqui -- é a versão "fisica" do check numérico de SD relativo do CCC.
  history_long <- purrr::map_dfr(selected_cfgs$config_id, function(cid) {
    purrr::map_dfr(seeds, function(sd_val) {
      f <- file.path(final_run_dir, cid, "history", sprintf("seed%04d_history.csv", sd_val))
      if (!file.exists(f)) return(NULL)
      readr::read_csv2(f, show_col_types = FALSE) %>%
        dplyr::mutate(config_id = cid, seed = factor(sd_val))
    })
  })

  if (nrow(history_long) > 0) {
    p_curves <- ggplot2::ggplot(history_long,
                                ggplot2::aes(x = epoch, y = validation_loss, color = seed)) +
      ggplot2::geom_line(alpha = 0.8) +
      ggplot2::facet_wrap(~ config_id, scales = "free_y") +
      ggplot2::labs(
        title = paste0(target_label, " — curvas de treino por seed (", final_run_id, ")"),
        subtitle = "Perda de validação por época | feixe apertado = treino estável entre seeds",
        x = "Época", y = "Perda de validação (loss)", color = "Seed"
      ) +
      ggplot2::theme_bw()

    print(p_curves)
    ggplot2::ggsave(file.path(fig_dir, "99b_final_training_curves.png"), p_curves,
                    width = 9, height = 6, dpi = 150)
    message("Salvo: ", file.path(fig_dir, "99b_final_training_curves.png"))
  }

  # ── Estabilidade entre seeds: CCC de cada seed ao redor da média ──────────
  # Companheiro visual direto do check "CCC SD relativo" do 99_check_pipeline:
  # aqui dá pra ver a nuvem de pontos (cada seed é um treino independente do
  # zero), não só o número do desvio.
  all_seed_results <- readr::read_csv2(
    file.path(final_run_dir, "comparison", "all_seed_results_test.csv"),
    show_col_types = FALSE
  )
  config_summary <- readr::read_csv2(
    file.path(final_run_dir, "comparison", "config_summary_test.csv"),
    show_col_types = FALSE
  )

  p_stability <- ggplot2::ggplot(all_seed_results, ggplot2::aes(x = config_id, y = ccc)) +
    ggplot2::geom_jitter(width = 0.08, height = 0, size = 2.5, alpha = 0.6, color = "#7570b3") +
    ggplot2::geom_pointrange(
      data = config_summary,
      ggplot2::aes(x = config_id, y = ccc_mean,
                  ymin = ccc_mean - ccc_sd, ymax = ccc_mean + ccc_sd),
      color = "black", linewidth = 0.6, size = 0.4
    ) +
    ggplot2::labs(
      title = paste0(target_label, " — estabilidade entre seeds (test, ", final_run_id, ")"),
      subtitle = "Pontos roxos = cada seed individual | barra preta = média ± desvio-padrão",
      x = NULL, y = "CCC (test)"
    ) +
    ggplot2::theme_bw()

  print(p_stability)
  ggplot2::ggsave(file.path(fig_dir, "99b_final_seed_stability.png"), p_stability,
                  width = 7, height = 5, dpi = 150)
  message("Salvo: ", file.path(fig_dir, "99b_final_seed_stability.png"))

  # ── O ensemble final acerta o valor real? Obs x Predito no TEST ──────────
  # Este é o número que efetivamente vira mapa no 05: a mediana das previsões
  # das N seeds por perfil (mesma lógica de agregação usada na predição
  # espacial). O leque cinza atrás de cada ponto mostra a dispersão entre
  # seeds daquele perfil específico -- um jeito tangível de ver a incerteza
  # do ensemble, perfil a perfil, não só como um número médio de erro.
  pred_test_all <- purrr::map_dfr(selected_cfgs$config_id, function(cid) {
    purrr::map_dfr(seeds, function(sd_val) {
      f <- file.path(final_run_dir, cid, "predictions", sprintf("seed%04d_pred_all.csv", sd_val))
      if (!file.exists(f)) return(NULL)
      readr::read_csv2(f, show_col_types = FALSE) %>%
        dplyr::filter(dataset_role == "test") %>%
        dplyr::mutate(config_id = cid, seed = sd_val)
    })
  })

  if (nrow(pred_test_all) > 0) {
    pred_test_ens <- pred_test_all %>%
      dplyr::group_by(config_id, profile_id) %>%
      dplyr::summarise(
        obs        = obs[1],
        pred_median = median(pred),
        pred_min    = min(pred),
        pred_max    = max(pred),
        .groups = "drop"
      )

    for (cid in unique(pred_test_ens$config_id)) {
      df_cid  <- dplyr::filter(pred_test_ens, config_id == cid)
      ccc_ens <- .lin_ccc(df_cid$obs, df_cid$pred_median)
      lims    <- range(c(df_cid$obs, df_cid$pred_min, df_cid$pred_max))

      p_test <- ggplot2::ggplot(df_cid, ggplot2::aes(x = obs, y = pred_median)) +
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
        ggplot2::geom_linerange(ggplot2::aes(ymin = pred_min, ymax = pred_max),
                                color = "grey70", alpha = 0.6) +
        ggplot2::geom_point(color = "#d95f02", alpha = 0.6, size = 1.8) +
        ggplot2::coord_fixed(xlim = lims, ylim = lims) +
        ggplot2::annotate("text", x = lims[1], y = lims[2], hjust = 0, vjust = 1,
                          label = sprintf("CCC (mediana ensemble) = %.3f\nn = %d perfis | %d seeds",
                                          ccc_ens, nrow(df_cid), length(seeds))) +
        ggplot2::labs(
          title = paste0(target_label, " — ensemble final (", cid, "): observado x predito"),
          subtitle = "Split de TESTE (nunca visto em treino) | traço cinza = min-max entre seeds daquele perfil",
          x = "SOC observado (t/ha)", y = "SOC predito (mediana das seeds, t/ha)"
        ) +
        ggplot2::theme_bw()

      print(p_test)
      out_png <- file.path(fig_dir, paste0("99b_final_test_obs_vs_pred_", cid, ".png"))
      ggplot2::ggsave(out_png, p_test, width = 7, height = 7, dpi = 150)
      message("Salvo: ", out_png)
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 4 — Patches reais ao redor de perfis (pesada — carrega ~17 GB)
# ══════════════════════════════════════════════════════════════════════════════

message("\n-- Parte 4: patches reais (carregando patches_all_splits.rds, pode demorar) --\n")

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
message("Patches (Parte 4): olhe se elevacao/ndvi mostram um padrao espacial")
message("coerente (nao ruido aleatorio) e se a janela maior (15x15) mostra mais")
message("contexto ao redor do mesmo centro que a menor (3x3) — devem estar alinhadas.")
message("Tuning/modelo final (Partes 2-3): olhe se o vencedor do leaderboard fica")
message("perto da reta 1:1 em validacao E em teste, se o feixe de curvas de treino")
message("e apertado entre seeds, e se a nuvem de estabilidade nao tem outlier isolado.")
