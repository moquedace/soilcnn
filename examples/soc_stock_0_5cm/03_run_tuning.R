source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c(
  "torch",
  "coro",
  "dplyr",
  "readr",
  "tibble",
  "purrr",
  "DescTools",
  "terra"       # so para ler a resolucao do raster no relatorio de vazamento
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))
source(file.path(project_root, "R", "patches.R"))
source(file.path(project_root, "R", "preprocess.R"))
source(file.path(project_root, "R", "dataset.R"))
source(file.path(project_root, "R", "diagnostics.R"))
source(file.path(project_root, "R", "resample.R"))
source(file.path(project_root, "R", "metrics.R"))
source(file.path(project_root, "R", "cnn_architecture.R"))
source(file.path(project_root, "R", "tune_grid.R"))
source(file.path(project_root, "R", "train_cnn.R"))

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

set.seed(42)
torch::torch_manual_seed(42)

# ── Torch device ──────────────────────────────────────────────────────────────

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# ── Training hyperparameters ──────────────────────────────────────────────────
# These are fixed across all configs in the grid.
# Only architecture and optimiser params vary per config (see tune_grid.R).

training_args <- list(
  n_epochs            = 500L,
  patience            = 60L,
  es_min_delta        = 0.0005,
  warmup_start_lr     = 1e-5,
  lr_plateau_factor   = 0.5,
  lr_plateau_patience = 25L,   # raised from 15: the exploratory run cut LR too
                               # early, before models could settle on the plateau
  lr_plateau_min_delta = 0.0005,
  min_lr              = 1e-6,
  gradient_clip       = 1.0,
  print_every         = 10L,
  augment             = TRUE   # D4 rotation/flip augmentation (regulariser)
)

# ── Paths ─────────────────────────────────────────────────────────────────────

patch_dir   <- file.path(project_root, "outputs", "patches",
                          "soc_stock_modeling", target_label)
data_dir    <- file.path(project_root, "data", "processed",
                          "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)

output_tuning_dir <- file.path(project_root, "outputs", "tuning",
                                "soc_stock_modeling", target_label)

# ── Tuning grid ───────────────────────────────────────────────────────────────
# See R/tune_grid.R and docs/tuning_guide.md for full parameter descriptions.
#
# RESOLUTION RESET. The earlier exploratory run (where 7×7 dominated, CCC ~0.61)
# was at 20 km, where 7×7 covers ~140 km of context. THIS run is at 250 m, where
# the same window spans only ~1.75 km — so that finding does NOT transfer. We
# therefore make WINDOW and LEARNING RATE the two primary search axes and let the
# data tell us which spatial scale matters at 250 m, instead of pre-committing:
#   • window_sizes: all six 3 / 9 / 15 options (single + dual) → ~0.75–3.75 km.
#   • base_lr: a spread (1e-4 … 1e-3); don't assume 20 km's 1e-4 still wins.
# Secondary knobs (depth, SE, gate, embedding, dropout, weight_decay) vary lightly
# around a lean baseline. D4 augmentation is on.
#
# `fixed` fixes single values AND restricts multi-value pools (see R/tune_grid.R);
# duplicate draws are dropped automatically. tune_length = 24 gives every window a
# few LR/architecture samples — raise it for denser coverage, lower for a quick
# first pass. make_manual_tune_grid() is the alternative for a full factorial.

tune_grid <- make_tune_grid(
  tune_length = 3,
  seed        = 666,
  fixed = list(
    loss_fn       = "smooth_l1",                # robust to outlier SOC values
    batch_size    = 256L,                       # 15×15 patches are larger than 7×7;
                                                # 256 is safe — raise to 512 if GPU allows
    # PRIMARY axis #1 — spatial scale at 250 m (single branch and dual branch)
    window_sizes  = list(c(3L), c(9L), c(15L),
                         c(3L, 9L), c(3L, 15L), c(9L, 15L)),
    # PRIMARY axis #2 — learning rate
    base_lr       = c(1e-4, 3e-4, 1e-3),
    # Secondary knobs — lean baseline
    conv_channels = list(c(64L, 128L),
                         c(64L, 128L, 128L)),
    use_residual  = TRUE,                        # always on for ≥2 blocks
    use_se_block  = c(TRUE, FALSE),
    gate_type     = c("vector_featurewise", "no_gate_concat"),
    embedding_dim = c(256L, 384L),
    # flatten keeps full spatial detail (params grow with window²); gap pools to
    # C and keeps large windows light. Let tuning decide which wins at 250 m —
    # especially relevant for the 15×15 branch (~11 M params under flatten).
    embed_pool    = c("flatten", "gap"),
    dropout       = c(0.1, 0.2),                 # mild–moderate regularisation
    weight_decay  = c(0.0, 1e-4)
  )
)

# Preview the grid before running (check it makes sense)
message("\n── Tune grid (", nrow(tune_grid), " configs) ──────────────────────")
print(
  dplyr::mutate(
    tune_grid,
    window_sizes  = purrr::map_chr(window_sizes,  paste, collapse = "x"),
    conv_channels = purrr::map_chr(conv_channels, paste, collapse = "_")
  ),
  width = Inf
)

# ── Load only the windows this grid actually needs ────────────────────────────
# The patch store keeps one file per window, so a grid that never uses 15x15
# does not pay ~6 GB of RAM for it. That is why the grid is built first.

windows_needed <- sort(unique(unlist(tune_grid$window_sizes)))
message("\nWindows required by this grid: ",
        paste(windows_needed, collapse = ", "))

store <- load_patch_store(patch_dir, windows_needed)
n_channels <- store$n_channels

# Point values feed the SCALING only -- the patches themselves are already
# extracted. 02 drops points whose window was not fully valid, so the dataset
# written by 01 has more rows than the store: align on sample_id rather than
# trusting row order.
points <- readr::read_csv2(file.path(data_dir, "full_modeling_dataset_raw.csv"),
                           show_col_types = FALSE)
type_table <- readr::read_csv2(file.path(metadata_dir, "predictor_type_table.csv"),
                               show_col_types = FALSE)
points <- align_points_to_meta(points, store$meta)

# ── Resolucao e unidades das coordenadas ──────────────────────────────────────
#
# Lida do PROPRIO raster, nunca escrita a mao: block_size e buffer sao dados
# nas MESMAS unidades de x/y, e aqui elas sao GRAUS (lon/lat, extensao global
# do WOSIS), nao metros. Um buffer escrito como "3750" pensando em metros
# viraria 3750 graus e o plano abortaria -- e um block_size errado na direcao
# oposta produziria um split que so PARECE espacial, sem abortar nada.
r_ref     <- terra::rast(readr::read_csv2(
  file.path(metadata_dir, "raster_table_used.csv"),
  show_col_types = FALSE)$raster_file[1])
cell_size <- terra::res(r_ref)[1]
rm(r_ref)

message(sprintf("\nResolucao do raster: %.8f por pixel (unidades de x/y)",
                cell_size))
message(sprintf("Piso do buffer (janela %d x resolucao): %.6f",
                max(windows_needed), max(windows_needed) * cell_size))

# ── PLANO DE REAMOSTRAGEM ─────────────────────────────────────────────────────
#
# Troque UMA linha aqui para mudar como o modelo e validado. Nada abaixo muda.
#
#   holdout(meta)     o split fixo que o 01 escreveu. DEFAULT.
#   random_folds(meta, k = 5)
#   spatial_folds(meta, k = 5, block_size = ..., buffer = ...)
#   region_folds(meta, group = meta$<coluna>)
#
# POR QUE O DEFAULT E O HOLDOUT: ele reproduz exatamente o que este pipeline
# fazia antes de existir reamostragem, entao ligar CV e uma escolha sua. E e o
# unico plano cuja validacao e a que o 01 escreveu, o que e o que torna os
# numeros comparaveis com os runs anteriores.
#
# O CUSTO DE LIGAR: k folds custam k vezes o tempo do grid. Com 24 configs x 3
# sementes x 5 folds sao 360 treinos, nao 24. Comece pelo holdout com 3
# sementes para medir o piso de ruido -- ele ja diz se o grid tem sinal --
# antes de multiplicar por k.
#
# O BUFFER, e por que ele nao e opcional no espacial.
#
# Dois patches de largura w a resolucao res dividem pelo menos um pixel quando
# os centros estao a menos de w * res um do outro. Por isso o buffer e escrito
# como `max(windows_needed) * cell_size` e nao como um numero: ele acompanha a
# janela e a resolucao sozinho, e nas unidades certas.
#
# Nesta escala a garantia e EXATA, nao aproximada: o raster tambem esta em
# graus, entao "dividir pixel" e uma pergunta em graus. (Como medida de
# DISTANCIA um grau de longitude encolhe com a latitude -- o que significa que
# o buffer e conservador longe do equador, que e o lado seguro do erro.)
#
# Blocar e bufferizar nao sao alternativas. Sem bloco o buffer nao deixa ponto
# de treino nenhum de pe: todo cluster tem ponto de treino e de validacao,
# entao todo ponto de treino tem um vizinho de validacao (veja apply_buffer()
# em R/resample.R -- isso e medido, nao suposto).
#
# BLOCK_SIZE: medido sobre estes 31.179 pontos do pool.
#   0,25 graus (~27 km)  -> 9.541 blocos, maior = 0,7% do pool
#   1,0  graus (~111 km) -> 2.788 blocos, maior = 2,5%
#   2,0  graus (~222 km) -> 1.279 blocos, maior = 4,4%   <- escolhido
#   3,0  graus (~333 km) ->   778 blocos, maior = 8,9%   (comeca a desbalancear)
# Blocos maiores separam mais; blocos grandes demais fazem um unico bloco
# dominar um fold. 2 graus e o ponto onde os dois ainda cabem.

plan <- spatial_folds(store$meta,
                      k          = 3,
                      block_size = 2,                                # ~222 km
                      buffer     = max(windows_needed) * cell_size)  # 15 px

# Para voltar ao split fixo do 01 (mais barato, sem folds):
# plan <- holdout(store$meta)

message("\n-- Plano de reamostragem --")
print(plan)

leak <- fold_leakage_report(plan, store$meta, cell_size = cell_size,
                            windows = windows_needed)
message("\n-- Vazamento por fold (mesma celula / patches sobrepostos) --")
print(dplyr::filter(leak, criterion == "same raster cell"), n = Inf, width = Inf)

# O escalonamento e ajustado no treino de CADA fold, dentro do
# run_cnn_resample() -- e por isso que os patches sao guardados CRUS. Um fold
# cujo escalonamento veio do treino de outro fold ja viu dado que nao devia.
#
# Nota: isso e ajustado sobre os pontos que de fato TREINAM (pos-QC de janela),
# enquanto o predictor_scaling.csv do 01 usou todo ponto de treino, incluindo
# os ~0,8% depois descartados pela regra de janela completa. A diferenca e
# pequena, mas esta e a versao honesta -- e a unica que generaliza para um fold.

message("Channels: ", n_channels, " | Points: ", nrow(store$meta))

# ── Run tuning ────────────────────────────────────────────────────────────────
# For each config in tune_grid:
#   1. Builds a dual_branch_cnn with that config's architecture
#   2. Trains with early stopping on validation SmoothL1 loss
#   3. Evaluates on train / validation / test (all metrics)
#   4. Saves weights, history, predictions, metrics, gate analysis
#   5. Appends to comparison table (ranked by VALIDATION CCC then validation MAE;
#      test metrics are recorded for diagnostics only, never used for selection)
#
# transform = expm1: back-transform from log1p space to native ton/ha.
#   Applied to predictions before computing CCC, MAE, etc.
#   The model trains in log1p space; metrics are always in native units.

# Para RETOMAR um run interrompido (crash, queda de luz, etc.): preencha
# resume_run_id com o run_id exato da pasta em outputs/tuning/ que parou no
# meio (ex: "soc_0_5cm_20260715_093000") e rode o script de novo. run_cnn_tuning
# detecta sozinho quais config_id já têm checkpoint (models/{id}_best.pt) e
# pula direto para os que faltam -- NÃO retreina do zero. Deixe NULL para
# sempre começar um run novo (comportamento padrão, gera timestamp novo).
resume_run_id <- NULL

run_id <- if (is.null(resume_run_id)) {
  paste0("soc_0_5cm_", format(Sys.time(), "%Y%m%d_%H%M%S"))
} else {
  resume_run_id
}

# QUANTAS SEMENTES POR CONFIG
#
# 3, nao 1. Com uma semente cada, "a config A ganhou da B" e uma afirmacao sem
# barra de erro: se retreinar a MESMA config com outra semente move o CCC mais
# do que a distancia entre A e B, o ranking e ranking de sorte. Tres repeticoes
# sao o minimo que estima esse piso, e o run imprime ele no fim.
#
# Custo: 3x o tempo do grid. Aumentar depois e RETOMAVEL: as repeticoes que ja
# estao em disco sao reconhecidas e so as novas treinam -- da para comecar com
# 1, ver o grid de pe, e subir para 3 sem perder nada.
n_seeds <- 3L

results <- do.call(
  run_cnn_resample,
  c(
    list(
      tune_grid  = tune_grid,
      store      = store,
      points     = points,
      type_table = type_table,
      plan       = plan,
      transform  = expm1,
      output_dir = output_tuning_dir,
      device     = device,
      run_id     = run_id,
      n_seeds    = n_seeds,
      resume     = TRUE
    ),
    training_args
  )
)

# ── Results summary ───────────────────────────────────────────────────────────

message("\n── Tuning complete ────────────────────────────────────────────────")
message("Run ID: ", run_id)
message("Results saved to: ", file.path(output_tuning_dir, run_id))

# A decisao se le na tabela POR CONFIG (media +/- sd sobre as repeticoes), nao
# na tabela por unidade: uma semente sortuda de uma config mediocre passa na
# frente da media firme de uma boa se o ranking for por linha.
if (nrow(results$by_config) > 0) {
  message("\nTop 5 configs por CCC de VALIDACAO medio (metrica de selecao; ",
          "test_* so diagnostico):")
  print(
    dplyr::select(
      results$by_config,
      rank, config_id, n_units, n_folds, n_seeds,
      dplyr::starts_with("val_ccc"), dplyr::starts_with("val_mae"),
      dplyr::starts_with("test_ccc"), n_failed
    ),
    n = 5, width = Inf
  )

  message("\nPiso de ruido -- so a semente muda:")
  print_noise_floor(seed_noise_floor(results$comparison))
}

if (nrow(results$comparison) > 0) {
  message("\nTrilha de auditoria (toda unidade, como foi medida): ",
          file.path(results$run_dir, "comparison", "comparison_ranked.csv"))
}

