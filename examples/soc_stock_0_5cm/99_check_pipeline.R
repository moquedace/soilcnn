project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("dplyr", "readr", "tibble", "purrr", "stringr", "terra")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

# As checagens genericas vivem em R/ (framework); este script e so o
# orquestrador do exemplo -- quem usar o framework com outro dado ganha as
# mesmas checagens sem copiar nada daqui.
source(file.path(project_root, "R", "utils.R"))
source(file.path(project_root, "R", "patches.R"))
source(file.path(project_root, "R", "preprocess.R"))
source(file.path(project_root, "R", "dataset.R"))
source(file.path(project_root, "R", "diagnostics.R"))
source(file.path(project_root, "R", "resample.R"))

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

# Todo o relatorio sai por UM canal.
#
# message() escreve em stderr e print()/tibble em stdout; no console do RStudio
# os dois se intercalam e linhas se colam ("channel_risk.csv  [OK] 01 | ..."),
# porque cada canal e esvaziado na sua propria hora. Como este script alterna
# linha de texto com print() de tibble o tempo todo, a ordem so e garantida se
# tudo for pelo mesmo lugar -- e print() nao tem como ir para stderr, entao o
# texto e que vai para stdout.
#
# .say() imita message(): cola os argumentos e acrescenta a quebra de linha.
.say <- function(...) cat(paste0(...), "
", sep = "")

add_check <- function(stage, check, status, detail = "") {
  .results <<- dplyr::bind_rows(
    .results,
    tibble::tibble(stage = stage, check = check, status = status, detail = detail)
  )
  icon <- switch(status, PASS = "  [OK]", WARN = "[WARN]", FAIL = "[FAIL]", "  [?]")
  .say(sprintf("%s %-6s | %-45s | %s", icon, stage, check, detail))
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

.say("\n", strrep("=", 90))
.say("99 — Checkpoint de qualidade do pipeline: ", target_label)
.say(strrep("=", 90), "\n")

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 01 — Preparo do dataset tabular
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Etapa 01: preparo do dataset --\n")

f_qc      <- file.path(metadata_dir, "qc_summary.csv")
f_dscheck <- file.path(metadata_dir, "dataset_check.csv")
f_split   <- file.path(metadata_dir, "split_check.csv")
f_ptype   <- file.path(metadata_dir, "predictor_type_table.csv")
f_scaling <- file.path(metadata_dir, "predictor_scaling.csv")
f_rtable  <- file.path(metadata_dir, "raster_table_used.csv")
f_smeta   <- file.path(metadata_dir, "split_metadata.csv")   # traz x, y
f_tconfig <- file.path(metadata_dir, "target_config.csv")

# Os 6 CSVs por split (train/validation/test x raw/scaled) sairam: os *_scaled
# nenhum codigo lia, e os *_raw eram um filter() do dataset unico. Com
# escalonamento por fold, "o conjunto escalonado" deixou de existir como
# objeto unico. Entraram no lugar: qc_table.csv (as regras que o 02 obedece)
# e channel_risk.csv (os canais que historicamente quebraram o MAPA).
f_dataset <- file.path(data_dir, "full_modeling_dataset_raw.csv")
f_qctable <- file.path(metadata_dir, "qc_table.csv")
f_crisk   <- file.path(metadata_dir, "channel_risk.csv")

files_01 <- c(f_qc, f_dscheck, f_split, f_ptype, f_scaling, f_rtable, f_tconfig,
             f_dataset, f_qctable, f_crisk)
all_01_exist <- all(purrr::map_lgl(files_01, ~ check_exists("01", basename(.x), .x)))

if (all_01_exist) {

  qc      <- safe_read_csv2(f_qc)
  dscheck <- safe_read_csv2(f_dscheck)
  split   <- safe_read_csv2(f_split)
  ptype   <- safe_read_csv2(f_ptype)
  scaling <- safe_read_csv2(f_scaling)
  rtable  <- safe_read_csv2(f_rtable)
  tconfig <- safe_read_csv2(f_tconfig)

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

  # dataset unico bate com a soma dos splits (mesma linhagem).
  # col_select=1 mantem a leitura rapida mesmo com 180+ colunas.
  n_dataset <- nrow(safe_read_csv2(f_dataset,
                                     col_select = 1))
  check_equal("01", "n_linhas dataset vs soma dos splits",
              n_dataset, sum(split$n), "dataset", "split_check")

  # qc_table tem uma regra por preditor, na mesma ordem do type_table --
  # e essa ordem que o 02 usa pra saber qual regra aplicar em qual banda.
  qctable <- safe_read_csv2(f_qctable)
  add_check("01", "qc_table.csv na mesma ordem de predictor_type_table.csv",
            if (identical(qctable$predictor, ptype$predictor)) "PASS" else "FAIL",
            sprintf("%d regras / %d preditores", nrow(qctable), nrow(ptype)))

  # PROTECAO: nenhum canal constante deve ter sobrevivido ao drop.
  # Canal constante nos pontos nao e constante no MAPA -- acende sobre
  # geleira, ilha, oceano -- e como seu gradiente e sempre zero, os pesos
  # ficam na inicializacao aleatoria e aplicam vies exatamente onde a rede
  # esta extrapolando. Se isso falhar, some o canal em manual_predictor_drop.
  crisk    <- safe_read_csv2(f_crisk)
  n_const  <- sum(crisk$risk == "constant", na.rm = TRUE)
  n_withna <- sum(crisk$risk == "has_na",   na.rm = TRUE)
  add_check("01", "nenhum canal constante sobreviveu ao drop",
            if (n_const == 0L) "PASS" else "FAIL",
            sprintf("%d constante(s)", n_const))
  add_check("01", "canais com NA nos pontos",
            if (n_withna == 0L) "PASS" else "WARN",
            sprintf("%d canal(is) com NA -- ver channel_risk.csv", n_withna))

  # ── Sobreposicao espacial entre splits ─────────────────────────────────
  # Torna visivel, a cada run, a metrica que invalidou a primeira rodada de
  # resultados e que nada reportava: num split aleatorio sobre perfis
  # espacialmente agrupados, 33% do teste cai no MESMO pixel de 250 m que um
  # perfil de treino -- mesmo patch de entrada, bit a bit.
  #
  # WARN, nunca FAIL: split aleatorio e escolha legitima. O que nao e
  # aceitavel e nao saber.
  if (file.exists(f_smeta) && file.exists(f_rtable)) {
    sp <- safe_read_csv2(f_smeta)
    rt <- safe_read_csv2(f_rtable)

    if (all(c("x", "y", "dataset_role") %in% names(sp)) &&
        file.exists(rt$raster_file[1])) {
      r1 <- try(terra::rast(rt$raster_file[1]), silent = TRUE)
      if (!inherits(r1, "try-error")) {
        cells <- terra::cellFromXY(r1, as.matrix(sp[, c("x", "y")]))
        ov <- spatial_overlap_report(
          terra::rowFromCell(r1, cells),
          terra::colFromCell(r1, cells),
          sp$dataset_role)

        .say("\n-- Sobreposicao espacial entre splits --")
        print(ov, n = Inf, width = Inf)

        same <- dplyr::filter(ov, criterion == "same raster cell")
        for (i in seq_len(nrow(same))) {
          add_check("01", paste0("'", same$split[i], "' no mesmo pixel do treino"),
                    if (same$pct[i] < 5) "PASS" else "WARN",
                    sprintf("%.2f%% (%s de %s)", same$pct[i],
                            format(same$n[i], big.mark = ","),
                            format(same$n_split[i], big.mark = ",")))
        }
        rm(r1)
      }
    }
  }

} else {
  .say("Etapa 01 incompleta -- pulando checagens de conteudo.")
}

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 02 — Extração de patches
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Etapa 02: extracao de patches --\n")

# O 02 agora escreve UM arquivo por janela (float32) em vez de um .rds unico
# com os tres splits, e um patch_meta.csv unico em vez de tres meta_*.csv:
# o split virou indice, nao propriedade do dado armazenado.
f_manifest     <- file.path(patch_meta_dir, "patch_manifest.csv")
f_pfiles       <- file.path(patch_meta_dir, "patch_files.csv")
f_blame        <- file.path(patch_meta_dir, "channel_invalidation.csv")
f_patch_meta   <- file.path(patch_dir, "patch_meta.csv")
f_manifest_rds <- file.path(patch_dir, "patch_manifest.rds")

files_02 <- c(f_manifest, f_pfiles, f_blame, f_patch_meta, f_manifest_rds)

# NAO INICIADA vs FALHOU: sao coisas diferentes e so uma merece FAIL.
#
# O fluxo aqui e "codigo 1 -> teste, codigo 2 -> teste", entao este script roda
# varias vezes com as etapas seguintes ainda por fazer. Se etapa que nunca
# rodou contasse como falha, todo run intermediario terminaria com FAIL e o
# sinal perderia valor justamente no momento em que ele mais serve.
#
# Regra: nenhum arquivo presente = ainda nao rodou (informativo). ALGUNS
# presentes = rodou pela metade, e isso sim e problema.
n_02_presentes <- sum(file.exists(files_02))

if (n_02_presentes == 0L) {
  .say("Etapa 02 nao iniciada -- nenhum arquivo em: ", patch_dir)
  all_02_exist <- FALSE
} else {
  all_02_exist <- all(purrr::map_lgl(files_02, ~ check_exists("02", basename(.x), .x)))
  if (!all_02_exist) {
    .say("  ATENCAO: a etapa 02 rodou PARCIALMENTE (", n_02_presentes, " de ",
            length(files_02), " arquivos). Um store incompleto e pior que ",
            "nenhum -- rode o 02 de novo.")
  }
}

if (all_02_exist) {

  manifest <- safe_read_csv2(f_manifest)

  # A CHECAGEM MAIS IMPORTANTE DESTE SCRIPT INTEIRO: fracao de perfis
  # descartados na extracao de patches. Antes da correcao do clip PNV, isso
  # rodava consistentemente em ~55%. Depois da correcao, esperado < 2%.
  # Se isso voltar pra cima de 10%, ALGO REGREDIU -- pare e investigue antes
  # de gastar dias/semanas de tuning/predicao em cima de dado quebrado.
  # Agora e um numero so: o 02 extrai todos os pontos de uma vez.
  check_threshold("02", "pct_removed (janela completa)",
                  manifest$pct_removed[1], warn_above = 2, fail_above = 10)

  # COMPLEMENTO: o numero acima diz QUANTO se perdeu; este diz QUAL canal
  # perdeu. valid_common e um AND sobre todos os canais, entao sozinho nunca
  # pode apontar o culpado -- e sem culpado, um canal com NA esparso so se
  # revela como buraco no mapa, semanas depois.
  blame <- safe_read_csv2(f_blame)
  worst <- if (nrow(blame) > 0) max(blame$pct_invalidated, na.rm = TRUE) else 0
  check_threshold("02", "pior canal, % de pontos invalidados",
                  worst, warn_above = 1, fail_above = 5)
  if (worst > 0) {
    top <- blame[which.max(blame$pct_invalidated), ]
    add_check("02", "canal com maior invalidacao", "PASS",
              sprintf("%s (%s): %.3f%%", top$predictor[1], top$type[1],
                      top$pct_invalidated[1]))
  }

  # patches sem escalonamento: o 03 aplica o escalonamento do fold. Um store
  # pre-escalonado esta amarrado a UM split e nao serve pra reamostragem.
  add_check("02", "patches gravados SEM escalonamento",
            if (isFALSE(manifest$scaling_applied[1])) "PASS" else "FAIL",
            paste("scaling_applied =", manifest$scaling_applied[1]))

  # n_channels bate com o que o 01 preparou (nao um numero fixo: dropar um
  # preditor problematico e uma acao legitima e nao deve quebrar o check).
  # `ptype` vem do bloco da etapa 01 -- guardado porque aquele bloco so roda se
  # os arquivos do 01 existirem, e a etapa 02 precisa sobreviver sem ele.
  if (exists("ptype")) {
    check_equal("02", "n_channels: manifest vs predictor_type_table",
                manifest$n_channels[1], nrow(ptype), "manifest", "01")
  } else {
    add_check("02", "n_channels: manifest vs predictor_type_table", "WARN",
              "etapa 01 incompleta -- sem referencia para comparar")
  }

  expected_windows <- "3, 9, 15"
  ok_windows <- identical(trimws(manifest$windows_extracted[1]), expected_windows)
  add_check("02", "windows_extracted == '3, 9, 15'",
            if (ok_windows) "PASS" else "FAIL",
            paste("valor:", manifest$windows_extracted[1]))

  # n_points_valid do manifest bate com as linhas reais de patch_meta.csv
  n_patch_meta <- nrow(safe_read_csv2(f_patch_meta,
                                        col_select = 1))
  check_equal("02", "n_points_valid: manifest vs patch_meta.csv",
              manifest$n_points_valid[1], n_patch_meta, "manifest", "patch_meta")

  # todo arquivo de janela existe, foi verificado na escrita, e nao esta
  # suspeitosamente pequeno (um crash no meio da escrita deixa truncado)
  pfiles <- safe_read_csv2(f_pfiles)
  missing_pt <- pfiles$file[!file.exists(file.path(patch_dir, pfiles$file))]
  add_check("02", "todo arquivo de janela existe em disco",
            if (length(missing_pt) == 0L) "PASS" else "FAIL",
            if (length(missing_pt) == 0L) sprintf("%d arquivo(s)", nrow(pfiles))
            else paste("faltando:", paste(missing_pt, collapse = ", ")))
  # A coluna e `status` ("written" / "kept" / "size_mismatch"). Era `verified`
  # na versao que gravava tensores; o 99 ficou lendo o nome antigo e reportava
  # "0/3 verificados" num store perfeito. Um check que le a coluna errada e
  # pior que check nenhum: gasta atencao num alarme falso.
  n_ok_files <- sum(pfiles$status %in% c("written", "kept"))
  add_check("02", "todo arquivo gravado sem erro de tamanho",
            if (n_ok_files == nrow(pfiles)) "PASS" else "WARN",
            sprintf("%d/%d (%s)", n_ok_files, nrow(pfiles),
                    paste(unique(pfiles$status), collapse = ", ")))
  add_check("02", "tamanho total dos patches plausivel",
            if (sum(pfiles$gb, na.rm = TRUE) > 0.5) "PASS" else "WARN",
            sprintf("%.1f GB em %d arquivo(s)", sum(pfiles$gb, na.rm = TRUE),
                    nrow(pfiles)))

  # ── A checagem mais forte deste arquivo ───────────────────────────────
  # O centro de cada patch TEM que ser igual ao valor que a tabela de pontos
  # guarda para aquele preditor naquele ponto -- e a mesma celula do mesmo
  # raster, alcancada por dois caminhos totalmente independentes:
  #
  #   tabela de pontos : terra::extract() sobre um SpatVector (script 01)
  #   patch store      : cellFromXY -> row/col -> patch_cell_index (script 02)
  #
  # Erro de CRS, troca de linha/coluna, off-by-one, reordenacao de canal ou
  # diretorio de raster desatualizado quebram essa igualdade. Os testes
  # sinteticos provam que a algebra esta certa; isto prova que ela foi
  # aplicada no lugar certo do raster REAL.
  if (file.exists(f_dataset) && file.exists(f_patch_meta) &&
      file.exists(f_manifest_rds)) {

    pts_all <- safe_read_csv2(f_dataset)
    pmeta   <- safe_read_csv2(f_patch_meta)
    preds   <- strsplit(readRDS(f_manifest_rds)$predictor_cols_final[1], ";")[[1]]

    cc <- try(check_patch_centres(patch_dir,
                                  align_points_to_meta(pts_all, pmeta),
                                  preds), silent = TRUE)

    if (inherits(cc, "try-error")) {
      add_check("02", "centro do patch == valor pontual", "FAIL",
                paste("erro:", conditionMessage(attr(cc, "condition"))))
    } else {
      add_check("02", "centro do patch == valor pontual",
                if (cc$ok) "PASS" else "FAIL",
                sprintf("%s celulas (%s pts x %d canais, janela %dx%d) | %d divergentes",
                        format(cc$n_cells, big.mark = ","),
                        format(cc$n_points, big.mark = ","),
                        cc$n_channels, cc$window, cc$window, cc$n_mismatch))
      if (!cc$ok) {
        .say("\n  DIVERGENCIA no centro dos patches -- canais afetados:")
        print(dplyr::slice_head(cc$by_channel, n = 10), width = Inf)
        .say("  Maior diferenca absoluta: ", signif(cc$worst, 6))
      }
    }
    rm(pts_all, pmeta); invisible(gc(verbose = FALSE))
  }

  add_check("02", "amostra de patches para inspecao visual",
            if (file.exists(file.path(patch_dir, "patch_sample.rds"))) "PASS" else "WARN",
            "patch_sample.rds -- gravado pelo 02 de graca")

} else {
  .say("Etapa 02 incompleta -- pulando checagens de conteudo.")
}

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 03 — Busca de hiperparâmetros (tuning)
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Etapa 03: tuning de hiperparametros --\n")

tuning_dir <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling", target_label)

if (!dir.exists(tuning_dir)) {
  .say("Etapa 03 nao iniciada -- pasta nao encontrada: ", tuning_dir)
} else {
  tuning_runs <- list.dirs(tuning_dir, recursive = FALSE, full.names = FALSE)
  if (length(tuning_runs) == 0) {
    .say("Etapa 03 incompleta -- nenhum run encontrado em: ", tuning_dir)
  } else {
    # Mais recente por ordenacao do nome (run_id e timestamped) -- mesmo
    # criterio usado para resolver "latest" no 04/05/06.
    tuning_run_id <- sort(tuning_runs, decreasing = TRUE)[1]
    run_dir <- file.path(tuning_dir, tuning_run_id)
    .say("Run mais recente: ", tuning_run_id)

    f_grid_csv <- file.path(run_dir, "tune_grid.csv")
    f_grid_rds <- file.path(run_dir, "tune_grid.rds")
    f_cmp_all  <- file.path(run_dir, "comparison", "comparison_all.csv")
    f_cmp_rank <- file.path(run_dir, "comparison", "comparison_ranked.csv")

    files_03 <- c(f_grid_csv, f_grid_rds, f_cmp_all, f_cmp_rank)
    all_03_exist <- all(purrr::map_lgl(files_03, ~ check_exists("03", basename(.x), .x)))

    if (all_03_exist) {

      tune_grid  <- safe_read_csv2(f_grid_csv)
      comparison <- safe_read_csv2(f_cmp_rank)

      # Runs anteriores a reamostragem tem uma linha por config e nenhuma
      # coluna unit_id/fold/seed. Preenche-las com o que aquelas linhas de
      # fato eram deixa todo o resto deste bloco com um caminho so.
      if (!"unit_id" %in% names(comparison)) comparison$unit_id <- comparison$config_id
      if (!"fold"    %in% names(comparison)) comparison$fold    <- 1L
      if (!"seed"    %in% names(comparison)) comparison$seed    <- NA_integer_

      n_grid <- nrow(tune_grid)
      n_cmp  <- nrow(comparison)

      # A checagem mais importante desta etapa: nenhuma config do grid ficou
      # pra tras (crash silencioso, config pulada por engano no resume, etc.)
      missing_ids <- setdiff(tune_grid$config_id, comparison$config_id)
      # Contar configs no numerador e configs no denominador. Com unidades,
      # `n_cmp` sao LINHAS -- imprimir "27/3 configs" mistura as duas escalas
      # e le como se sobrassem configs.
      n_cfg_seen <- dplyr::n_distinct(comparison$config_id)
      add_check("03", "todas as configs do grid tem linha na comparacao",
                if (length(missing_ids) == 0) "PASS" else "FAIL",
                if (length(missing_ids) == 0)
                  sprintf("%d/%d configs (%d unidades)", n_cfg_seen, n_grid, n_cmp)
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
      # Por UNIDADE: com repeticoes, duas sementes da mesma config sao dois
      # modelos. Procurar por config_id acharia um arquivo que nao existe e
      # deixaria de checar os que existem.
      ckpt_files <- file.path(run_dir, "models", paste0(comparison$unit_id, "_best.pt"))
      n_missing_ckpt <- sum(!file.exists(ckpt_files))
      add_check("03", "toda unidade da comparacao tem checkpoint .pt",
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

      # ── Reamostragem: a contabilidade das repeticoes ────────────────────────
      f_plan     <- file.path(run_dir, "fold_plan.rds")
      f_byconfig <- file.path(run_dir, "comparison", "comparison_by_config.csv")
      has_plan   <- file.exists(f_plan)

      n_folds_run <- dplyr::n_distinct(comparison$fold)
      n_seeds_run <- dplyr::n_distinct(comparison$seed)

      add_check("03", "plano de reamostragem gravado com o run",
                if (has_plan) "PASS" else "WARN",
                if (has_plan) {
                  pl <- readRDS(f_plan)
                  sprintf("%s | %d fold(s)%s", pl$method, pl$n_folds,
                          if (!is.null(pl$params$buffer))
                            sprintf(" | buffer %s", format(pl$params$buffer))
                          else "")
                } else "fold_plan.rds ausente -- run anterior a reamostragem")

      # O grid tem que ter sido treinado por INTEIRO em cada fold e cada
      # semente. Um buraco aqui nao aparece em lugar nenhum: a media da config
      # sai de menos repeticoes que as outras e a comparacao fica torta.
      n_expected <- n_grid * n_folds_run * n_seeds_run
      add_check("03", "unidades = configs x folds x sementes",
                if (n_cmp == n_expected) "PASS" else "WARN",
                sprintf("%d linhas | %d configs x %d fold(s) x %d semente(s) = %d",
                        n_cmp, n_grid, n_folds_run, n_seeds_run, n_expected))

      # A semente TEM que ser a mesma em todas as configs de uma repeticao --
      # e isso que faz duas configs serem comparadas sob o mesmo sorteio. Se
      # cada config tiver sua propria semente, parte de toda diferenca medida
      # e sorte, e nada no resultado avisa.
      if (!all(is.na(comparison$seed))) {
        seeds_per_cfg <- comparison %>%
          dplyr::group_by(config_id) %>%
          dplyr::summarise(s = paste(sort(unique(seed)), collapse = ","),
                           .groups = "drop")
        add_check("03", "mesmo conjunto de sementes em todas as configs",
                  if (dplyr::n_distinct(seeds_per_cfg$s) == 1L) "PASS" else "FAIL",
                  sprintf("%d conjunto(s) distinto(s) | sementes: %s",
                          dplyr::n_distinct(seeds_per_cfg$s),
                          seeds_per_cfg$s[1]))
      }

      # rank 1 da tabela POR CONFIG e realmente a maior media -- e nao a linha
      # que teve o melhor run isolado.
      if (file.exists(f_byconfig)) {
        by_config <- safe_read_csv2(f_byconfig)
        top_by_rank <- by_config$config_id[by_config$rank == 1][1]
        top_by_mean <- by_config$config_id[which.max(by_config$val_ccc_mean)]
        check_equal("03", "rank==1 corresponde a maior media de val_ccc",
                    top_by_rank, top_by_mean, "rank_1", "max_mean")

        # A pergunta que decide se este tuning significa alguma coisa: a
        # distancia entre o 1o e o 2o e maior que o ruido de semente?
        if (nrow(by_config) > 1L && "val_ccc_sd" %in% names(by_config)) {
          ord <- dplyr::arrange(by_config, rank)
          gap <- ord$val_ccc_mean[1] - ord$val_ccc_mean[2]
          # O piso de ruido e o sd entre SEMENTES dentro do mesmo fold, medido
          # por seed_noise_floor(). O sd da tabela por config mistura fold e
          # semente -- usa-lo aqui com o rotulo "entre sementes" reportava uma
          # quantidade diferente da que o 03 imprime, com o mesmo nome.
          nf    <- seed_noise_floor(comparison)
          noise <- nf$median_sd
          add_check("03", "vencedor se destaca acima do ruido de semente",
                    if (!is.finite(noise)) "WARN"
                    else if (gap >= noise) "PASS" else "WARN",
                    if (!is.finite(noise))
                      "sd entre sementes indisponivel (1 repeticao por config)"
                    else sprintf("1o-2o = %.4f | sd tipico entre sementes = %.4f",
                                 gap, noise))
        }
        top_by_ccc <- top_by_mean
      } else {
        top_by_rank <- comparison$config_id[comparison$rank == 1][1]
        top_by_ccc  <- comparison$config_id[which.max(comparison$val_ccc)]
        check_equal("03", "rank==1 corresponde ao maior val_ccc",
                    top_by_rank, top_by_ccc, "rank_1", "max_ccc")
      }

      # gate_summary.csv so deveria existir para configs dual-branch (janela
      # com "x" no nome, ex. "9x15") com gate != no_gate_concat -- confirma
      # que a logica condicional de extract_gate_analysis() nao esta gerando
      # (ou deixando de gerar) arquivo para o tipo de config errado.
      gated_ids <- comparison$unit_id[
        grepl("x", comparison$window_sizes) & comparison$gate_type != "no_gate_concat"
      ]
      gate_files <- file.path(run_dir, "gates", paste0(gated_ids, "_gate_summary.csv"))
      n_missing_gate <- sum(!file.exists(gate_files))
      add_check("03", "gate_summary.csv existe p/ toda config dual-branch com gate",
                if (n_missing_gate == 0) "PASS" else "WARN",
                sprintf("%d/%d faltando", n_missing_gate, length(gated_ids)))

      # Metrica media quando ha repeticoes; a da unica linha quando nao ha.
      top_rows <- dplyr::filter(comparison, config_id == top_by_ccc)
      .say(sprintf(
        "\n  Melhor config: %s | CCC=%.3f | MAE=%.2f | RMSE=%.2f | janela=%s | gate=%s | %d repeticao(oes)",
        top_by_ccc,
        mean(top_rows$val_ccc,  na.rm = TRUE),
        mean(top_rows$val_mae,  na.rm = TRUE),
        mean(top_rows$val_rmse, na.rm = TRUE),
        top_rows$window_sizes[1],
        top_rows$gate_type[1],
        nrow(top_rows)))

    } else {
      .say("Etapa 03 incompleta -- pulando checagens de conteudo.")
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 04 — Modelo final (ensemble multi-seed)
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Etapa 04: modelo final (ensemble multi-seed) --\n")

final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

if (!dir.exists(final_model_base)) {
  .say("Etapa 04 nao iniciada -- pasta nao encontrada: ", final_model_base)
} else {
  final_runs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  final_runs <- final_runs[grepl("^final_", final_runs)]
  if (length(final_runs) == 0) {
    .say("Etapa 04 incompleta -- nenhum run encontrado em: ", final_model_base)
  } else {
    final_run_id <- sort(final_runs, decreasing = TRUE)[1]
    run_dir <- file.path(final_model_base, final_run_id)
    .say("Run mais recente: ", final_run_id)

    f_summary_rds <- file.path(run_dir, "comparison", "final_run_summary.rds")
    f_all_seeds   <- file.path(run_dir, "comparison", "all_seed_results_test.csv")
    f_cfg_summary <- file.path(run_dir, "comparison", "config_summary_test.csv")

    files_04 <- c(f_summary_rds, f_all_seeds, f_cfg_summary)
    all_04_exist <- all(purrr::map_lgl(files_04, ~ check_exists("04", basename(.x), .x)))

    if (all_04_exist) {

      summary_rds      <- readRDS(f_summary_rds)
      all_seed_results  <- safe_read_csv2(f_all_seeds)
      config_summary    <- safe_read_csv2(f_cfg_summary)

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
        .say(sprintf(
          "\n  Modelo final [%s]: CCC=%.4f +/- %.4f | MAE=%.3f | RMSE=%.3f | %d seeds",
          cs$config_id, cs$ccc_mean, cs$ccc_sd, cs$mae_mean, cs$rmse_mean, cs$n_seeds))
      }

    } else {
      .say("Etapa 04 incompleta -- pulando checagens de conteudo.")
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

# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT — o que mudou desde o run anterior?
#
# Num ciclo de refatoracao, a pergunta feita apos cada execucao e "mexeu em
# alguma coisa?", e responde-la significava rolar a tela atras de saida
# antiga. "Tudo identico" e o resultado que se quer ver, e e justamente o
# mais dificil de confirmar de olho.
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Snapshot: comparacao com o run anterior --\n")

snap_vals <- list()
add_snap <- function(k, v) {
  if (length(v) == 1L && !is.null(v) && !is.na(v)) snap_vals[[k]] <<- v
}

if (exists("dscheck")) {
  add_snap("01_n_linhas",     dscheck$n_rows[1])
  add_snap("01_n_preditores", dscheck$n_predictors[1])
  add_snap("01_n_dummy",      dscheck$n_dummy_predictors[1])
  add_snap("01_n_percentage", dscheck$n_percentage_predictors[1])
  add_snap("01_n_continuous", dscheck$n_continuous_predictors[1])
  add_snap("01_mediana_alvo", round(dscheck$median_target[1], 6))
}
if (exists("split")) {
  for (i in seq_len(nrow(split))) {
    add_snap(paste0("01_split_", split$dataset_role[i]), split$n[i])
  }
}
if (exists("qc")) add_snap("01_pct_problema", qc$pct_any_problem[1])
if (exists("crisk")) {
  add_snap("01_canais_constantes", sum(crisk$risk == "constant", na.rm = TRUE))
  add_snap("01_canais_com_na",     sum(crisk$risk == "has_na",   na.rm = TRUE))
}
if (exists("ov")) {
  so <- dplyr::filter(ov, criterion == "same raster cell")
  for (i in seq_len(nrow(so))) {
    add_snap(paste0("01_overlap_pixel_", so$split[i]), so$pct[i])
  }
}
if (exists("manifest") && "n_points_valid" %in% names(manifest)) {
  add_snap("02_n_pontos_validos", manifest$n_points_valid[1])
  add_snap("02_pct_removido",     manifest$pct_removed[1])
}
if (exists("blame") && nrow(blame) > 0L) {
  add_snap("02_pior_canal",     blame$predictor[which.max(blame$pct_invalidated)])
  add_snap("02_pior_canal_pct", max(blame$pct_invalidated, na.rm = TRUE))
}
if (exists("cc") && !inherits(cc, "try-error")) {
  add_snap("02_centros_divergentes", cc$n_mismatch)
}
add_snap("99_n_pass", sum(.results$status == "PASS"))
add_snap("99_n_warn", sum(.results$status == "WARN"))
add_snap("99_n_fail", sum(.results$status == "FAIL"))

snap_dir <- file.path(project_root, "outputs", "qc", "snapshots")
# Os contadores do proprio 99 (99_n_pass/warn/fail) ficam GRAVADOS no
# snapshot -- sao o resumo do run e valem para o historico -- mas nao entram
# no diff. Sao derivados de todas as outras chaves: se uma metrica real
# mudar, ela aparece no diff sozinha. Compara-los cria uma alca fechada em
# que CONSERTAR um WARN gera um WARN ("valores alterados"), que foi
# exatamente o que aconteceu no run que corrigiu os dois checks quebrados.
cmp <- compare_run_snapshot(snap_vals, snap_dir,
                            exclude = c("99_n_pass", "99_n_warn", "99_n_fail"))
print_snapshot_diff(cmp)
write_run_snapshot(snap_vals, snap_dir)

if (cmp$has_previous) {
  n_changed <- sum(cmp$diff$status != "=")
  add_check("99", "valores alterados desde o run anterior",
            if (n_changed == 0L) "PASS" else "WARN",
            sprintf("%d de %d", n_changed, nrow(cmp$diff)))
}

# ── Resumo final ─────────────────────────────────────────────────────────────

.say("\n", strrep("=", 90))
.say("RESUMO")
.say(strrep("=", 90))

n_pass <- sum(.results$status == "PASS")
n_warn <- sum(.results$status == "WARN")
n_fail <- sum(.results$status == "FAIL")

.say(sprintf("\n  PASS: %d   WARN: %d   FAIL: %d   (total: %d checagens)\n",
                n_pass, n_warn, n_fail, nrow(.results)))

# So conta como problema o que realmente falhou -- etapa nao iniciada nem
# chega a registrar checagem.
if (n_fail > 0) {
  .say("Checagens que FALHARAM:")
  print(dplyr::filter(.results, status == "FAIL"), n = Inf, width = Inf)
}
if (n_warn > 0) {
  .say("\nCheckagens com AVISO:")
  print(dplyr::filter(.results, status == "WARN"), n = Inf, width = Inf)
}

report_dir <- file.path(project_root, "outputs", "qc")
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
report_file <- file.path(report_dir, paste0("pipeline_check_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
readr::write_csv2(.results, report_file)
.say("\nRelatorio completo salvo em: ", report_file)

if (n_fail > 0) {
  warning(n_fail, " checagem(ns) FALHOU. Revise antes de seguir para a proxima etapa.")
} else if (n_warn > 0) {
  .say("\nNenhuma falha critica, mas ha avisos -- revise antes de investir tempo de GPU/CPU na proxima etapa.")
} else {
  .say("\nTudo OK. Pode seguir para a proxima etapa do pipeline.")
}
