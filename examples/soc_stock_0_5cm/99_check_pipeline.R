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
# 99 — Checkpoint of qualidade do pipeline (check & recheck)
#
# Roda checagens automáticas sobre o que já foi executado (01, 02, ...),
# comparando contra limiares conhecidos e contra a consistência interna dos
# próprios arquivos. Objetivo: pegar problemas ESTRUTURAIS (tipo o bug do
# clip PNV, que silenciosamente descartou ~55% dos dados por ~2 meses de
# processamento) no minuto em que a etapa termina, não semanas depois.
#
# Como crescer isto: cada etapa (01, 02, 03, ...) tem sua própria seção
# `check_0X_*()`. Ao terminar of rodar uma nova etapa do pipeline, adicione
# uma seção nova aqui seguindo o mesmo padrão (ver `add_check()` abaixo) e
# rode o script inteiro of novo — ele re-verifica tudo que já rodou, não só
# o novo.
#
# Uso: source() direto, sem parâmetros. Roda em segundos (só lê CSVs
# pequenos e metadados — NUNCA carrega os arrays of patches inteiros, que
# têm dezenas of GB; usa só o manifest, que já tem os números agregados).
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"

data_dir     <- file.path(project_root, "data", "processed", "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata", "soc_stock_modeling", target_label)
patch_dir    <- file.path(project_root, "outputs", "patches", "soc_stock_modeling", target_label)
patch_meta_dir <- file.path(metadata_dir, "patches")

# ── Infraestrutura of checagem ──────────────────────────────────────────────────

.results <- tibble::tibble(
  stage = character(), check = character(), status = character(), detail = character()
)

# Todo o relatorio sai por UM canal.
#
# message() escreve em stderr e print()/tibble em stdout; no console do RStudio
# os dois se intercalam e linhas se colam ("channel_risk.csv  [OK] 01 | ..."),
# porque cada canal e esvaziado na sua propria hora. Como este script alterna
# linha of texto com print() of tibble o tempo todo, a ordem so e garantida se
# tudo for pelo mesmo lugar -- e print() nao tem como ir para stderr, entao o
# texto e que vai para stdout.
#
# .say() imita message(): cola os argumentos e acrescenta a quebra of linha.
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
            sprintf("value=%.2f%s (warn>%.1f%s, fail>%.1f%s)",
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
.say("99 - pipeline quality checkpoint: ", target_label)
.say(strrep("=", 90), "\n")

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 01 — Preparo do dataset tabular
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Stage 01: dataset preparation --\n")

f_qc      <- file.path(metadata_dir, "qc_summary.csv")
f_dscheck <- file.path(metadata_dir, "dataset_check.csv")

f_ptype   <- file.path(metadata_dir, "predictor_type_table.csv")

f_rtable  <- file.path(metadata_dir, "raster_table_used.csv")
f_pmeta   <- file.path(metadata_dir, "point_metadata.csv")   # sample_id, x, y
f_tconfig <- file.path(metadata_dir, "target_config.csv")

# Os 6 CSVs por split (train/validation/test x raw/scaled) sairam: os *_scaled
# nenhum codigo lia, e os *_raw eram um filter() do dataset unico. Com
# escalonamento por fold, "o conjunto escalonado" deixou of existir como
# objeto unico. Entraram no lugar: qc_table.csv (as regras que o 02 obedece)
# e channel_risk.csv (os canais que historicamente quebraram o MAPA).
f_dataset <- file.path(data_dir, "full_modeling_dataset_raw.csv")
f_qctable <- file.path(metadata_dir, "qc_table.csv")
f_crisk   <- file.path(metadata_dir, "channel_risk.csv")

# ── Run profile ───────────────────────────────────────────────────────────────
# A development run writes the SAME files, with the SAME names, on a tenth of
# the data. Without this in plain sight, a month from now a subsample CCC is
# indistinguishable from a result.
if (file.exists(f_tconfig)) {
  .tc <- safe_read_csv2(f_tconfig)
  if ("run_profile" %in% names(.tc) && !identical(.tc$run_profile[1], "full")) {
    .say(strrep("!", 90))
    .say("RUN PROFILE: ", toupper(.tc$run_profile[1]),
         "  --  THIS IS NOT A RESULT RUN")
    if ("subsample" %in% names(.tc)) .say("  ", .tc$subsample[1])
    .say("  Comparable only with another run of the same profile.",
         "  See docs/reference_performance.md")
    .say(strrep("!", 90), "
")
  }
}

# Gone with the split: split_check.csv, split_bin_check.csv,
# scaled_extreme_check.csv and predictor_scaling.csv. Stage 01 no longer
# decides who trains, and the scaling is now part of the fitted model (stage
# 04 writes it next to the weights), not a property of the dataset.
files_01 <- c(f_qc, f_dscheck, f_ptype, f_rtable, f_tconfig,
             f_dataset, f_qctable, f_crisk, f_pmeta)
all_01_exist <- all(purrr::map_lgl(files_01, ~ check_exists("01", basename(.x), .x)))

if (all_01_exist) {

  qc      <- safe_read_csv2(f_qc)
  dscheck <- safe_read_csv2(f_dscheck)
  ptype   <- safe_read_csv2(f_ptype)
  rtable  <- safe_read_csv2(f_rtable)
  pmeta   <- safe_read_csv2(f_pmeta)
  tconfig <- safe_read_csv2(f_tconfig)

  # A checagem mais importante: fracao of linhas descartadas por problema de
  # PREDITOR (nao of alvo). Alvo com problema (spline ruim, NA, <=0) e normal
  # e esperado; preditor com problema em excesso e a assinatura do tipo de
  # bug que causou a perda of ~55% dos dados por 2 meses (clip vs NA).
  pct_pred_problem <- 100 * qc$n_predictor_problem / qc$n_rows_extracted
  check_threshold("01", "% of rows with a PREDICTOR problem (not the target)",
                  pct_pred_problem, warn_above = 2, fail_above = 10)

  # Contagem of predictors consistente entre TODOS os arquivos que deveriam
  # concordar.
  n_pred_rtable  <- nrow(rtable)
  n_pred_ptype   <- nrow(ptype)
  n_pred_scaling <- nrow(scaling)
  n_pred_dscheck <- dscheck$n_predictors[1]
  n_pred_tconfig <- tconfig$n_predictors_final[1]

  check_equal("01", "n_predictors: raster_table vs predictor_type_table",
              n_pred_rtable, n_pred_ptype, "raster_table", "predictor_type")
  check_equal("01", "n_predictors: raster_table vs predictor_scaling",
              n_pred_rtable, n_pred_scaling, "raster_table", "scaling")
  check_equal("01", "n_predictors: raster_table vs dataset_check",
              n_pred_rtable, n_pred_dscheck, "raster_table", "dataset_check")
  check_equal("01", "n_predictors: raster_table vs target_config",
              n_pred_rtable, n_pred_tconfig, "raster_table", "target_config")

  # dummy + percentage + continuous deve somar o total of predictors
  n_type_sum <- dscheck$n_dummy_predictors[1] + dscheck$n_percentage_predictors[1] +
    dscheck$n_continuous_predictors[1]
  check_equal("01", "dummy + percentage + continuous == total predictors",
              n_type_sum, n_pred_rtable, "soma_tipos", "total")

  # Proporcao do split perto of 70/15/15 (tolerancia 2 p.p.)
  total_n <- sum(split$n)
  
  # target_native e target_log1p sao consistentes (log1p(native) == log1p)
  # -- checagem indireta via mediana ja calculada em dataset_check.csv
  implied_log1p <- log1p(dscheck$median_target[1])
  check_equal("01", "median_target_log1p == log1p(median_target)",
              round(dscheck$median_target_log1p[1], 4), round(implied_log1p, 4),
              "salvo", "recalculado")

  # dataset unico bate com a soma dos splits (mesma linhagem).
  # col_select=1 mantem a leitura rapida mesmo com 180+ colunas.
  n_dataset <- nrow(safe_read_csv2(f_dataset,
                                     col_select = 1))
  check_equal("01", "dataset rows vs the sum of the splits",
              n_dataset, sum(split$n), "dataset", "split_check")

  # qc_table tem uma regra por preditor, na mesma ordem do type_table --
  # e essa ordem que o 02 usa pra saber qual regra aplicar em qual banda.
  qctable <- safe_read_csv2(f_qctable)
  add_check("01", "qc_table.csv in the same order as predictor_type_table.csv",
            if (identical(qctable$predictor, ptype$predictor)) "PASS" else "FAIL",
            sprintf("%d rules / %d predictors", nrow(qctable), nrow(ptype)))

  # PROTECAO: nenhum canal constante deve ter sobrevivido ao drop.
  # Canal constante nos pontos nao e constante no MAPA -- acende sobre
  # geleira, ilha, oceano -- e como seu gradiente e sempre zero, os pesos
  # ficam na inicializacao aleatoria e aplicam vies exatamente onde a rede
  # esta extrapolando. Se isso falhar, some o canal em manual_predictor_drop.
  crisk    <- safe_read_csv2(f_crisk)
  n_const  <- sum(crisk$risk == "constant", na.rm = TRUE)
  n_withna <- sum(crisk$risk == "has_na",   na.rm = TRUE)
  add_check("01", "no constant channel survived the drop",
            if (n_const == 0L) "PASS" else "FAIL",
            sprintf("%d constant(s)", n_const))
  add_check("01", "channels with NA at the points",
            if (n_withna == 0L) "PASS" else "WARN",
            sprintf("%d channel(s) with NA -- see channel_risk.csv", n_withna))

  # The spatial-overlap report used to live here, comparing the fixed split
  # stage 01 wrote. There is no split here any more: it is carved by the fold
  # plan in stage 03, and that is where the leakage is measured -- per fold, on
  # the plan that will actually be used. See fold_leakage_report().
  add_check("01", "point table carries no role column",
            if (!any(c("dataset_role", "split_bin") %in% names(pmeta)))
              "PASS" else "FAIL",
            paste("columns:", paste(names(pmeta), collapse = ", ")))

} else {
  .say("Stage 01 incomplete -- skipping content checks.")
}

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 02 — Extração of patches
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Stage 02: patch extraction --\n")

# O 02 agora escreve UM arquivo por janela (float32) em vez of um .rds unico
# com os tres splits, e um patch_meta.csv unico em vez of tres meta_*.csv:
# o split virou indice, nao propriedade do dado armazenado.
f_manifest     <- file.path(patch_meta_dir, "patch_manifest.csv")
f_pfiles       <- file.path(patch_meta_dir, "patch_files.csv")
f_blame        <- file.path(patch_meta_dir, "channel_invalidation.csv")
f_patch_meta   <- file.path(patch_dir, "patch_meta.csv")
f_manifest_rds <- file.path(patch_dir, "patch_manifest.rds")

files_02 <- c(f_manifest, f_pfiles, f_blame, f_patch_meta, f_manifest_rds)

# NAO INICIADA vs FALHOU: sao coisas diferentes e so uma merece FAIL.
#
# O fluxo aqui e "script 1 -> check, script 2 -> check", entao este script roda
# varias vezes com as etapas seguintes ainda por fazer. Se etapa que nunca
# rodou contasse como falha, todo run intermediario terminaria com FAIL e o
# sinal perderia valor justamente no momento em que ele mais serve.
#
# Regra: nenhum arquivo presente = ainda nao rodou (informativo). ALGUNS
# presentes = rodou pela metade, e isso sim e problema.
n_02_presentes <- sum(file.exists(files_02))

if (n_02_presentes == 0L) {
  .say("Stage 02 not started -- no files in: ", patch_dir)
  all_02_exist <- FALSE
} else {
  all_02_exist <- all(purrr::map_lgl(files_02, ~ check_exists("02", basename(.x), .x)))
  if (!all_02_exist) {
    .say("  WARNING: stage 02 ran PARTIALLY (", n_02_presentes, " of ",
            length(files_02), " files). An incomplete store is worse than ",
            "nenhum -- rode o 02 of novo.")
  }
}

if (all_02_exist) {

  manifest <- safe_read_csv2(f_manifest)

  # A CHECAGEM MAIS IMPORTANTE DESTE SCRIPT INTEIRO: fracao of perfis
  # descartados na extracao of patches. Antes da correcao do clip PNV, isso
  # rodava consistentemente em ~55%. Depois da correcao, esperado < 2%.
  # Se isso voltar pra cima of 10%, ALGO REGREDIU -- pare e investigue antes
  # of gastar dias/semanas of tuning/predicao em cima of dado quebrado.
  # Agora e um numero so: o 02 extrai todos os pontos of uma vez.
  check_threshold("02", "pct_removed (full window)",
                  manifest$pct_removed[1], warn_above = 2, fail_above = 10)

  # COMPLEMENTO: o numero acima diz QUANTO se perdeu; este diz QUAL canal
  # perdeu. valid_common e um AND sobre todos os canais, entao sozinho nunca
  # pode apontar o culpado -- e sem culpado, um canal com NA esparso so se
  # revela como buraco no mapa, semanas depois.
  blame <- safe_read_csv2(f_blame)
  worst <- if (nrow(blame) > 0) max(blame$pct_invalidated, na.rm = TRUE) else 0
  check_threshold("02", "worst channel, % of points invalidated",
                  worst, warn_above = 1, fail_above = 5)
  if (worst > 0) {
    top <- blame[which.max(blame$pct_invalidated), ]
    add_check("02", "channel invalidating the most points", "PASS",
              sprintf("%s (%s): %.3f%%", top$predictor[1], top$type[1],
                      top$pct_invalidated[1]))
  }

  # patches sem escalonamento: o 03 aplica o escalonamento do fold. Um store
  # pre-escalonado esta amarrado a UM split e nao serve pra reamostragem.
  add_check("02", "patches stored WITHOUT scaling",
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
              "stage 01 incomplete -- nothing to compare against")
  }

  expected_windows <- "3, 9, 15"
  ok_windows <- identical(trimws(manifest$windows_extracted[1]), expected_windows)
  add_check("02", "windows_extracted == '3, 9, 15'",
            if (ok_windows) "PASS" else "FAIL",
            paste("valor:", manifest$windows_extracted[1]))

  # n_points_valid do manifest bate com as linhas reais of patch_meta.csv
  n_patch_meta <- nrow(safe_read_csv2(f_patch_meta,
                                        col_select = 1))
  check_equal("02", "n_points_valid: manifest vs patch_meta.csv",
              manifest$n_points_valid[1], n_patch_meta, "manifest", "patch_meta")

  # todo arquivo of janela existe, foi verificado na escrita, e nao esta
  # suspeitosamente pequeno (um crash no meio da escrita deixa truncado)
  pfiles <- safe_read_csv2(f_pfiles)
  missing_pt <- pfiles$file[!file.exists(file.path(patch_dir, pfiles$file))]
  add_check("02", "every window file exists on disk",
            if (length(missing_pt) == 0L) "PASS" else "FAIL",
            if (length(missing_pt) == 0L) sprintf("%d file(s)", nrow(pfiles))
            else paste("missing:", paste(missing_pt, collapse = ", ")))
  # A coluna e `status` ("written" / "kept" / "size_mismatch"). Era `verified`
  # na versao que gravava tensores; o 99 ficou lendo o nome antigo e reportava
  # "0/3 verificados" num store perfeito. Um check que le a coluna errada e
  # pior que check nenhum: gasta atencao num alarme falso.
  n_ok_files <- sum(pfiles$status %in% c("written", "kept"))
  add_check("02", "every file written without a size error",
            if (n_ok_files == nrow(pfiles)) "PASS" else "WARN",
            sprintf("%d/%d (%s)", n_ok_files, nrow(pfiles),
                    paste(unique(pfiles$status), collapse = ", ")))
  add_check("02", "total patch size is plausible",
            if (sum(pfiles$gb, na.rm = TRUE) > 0.5) "PASS" else "WARN",
            sprintf("%.1f GB em %d file(s)", sum(pfiles$gb, na.rm = TRUE),
                    nrow(pfiles)))

  # ── A checagem mais forte deste arquivo ───────────────────────────────
  # O centro of cada patch TEM que ser igual ao valor que a tabela of pontos
  # guarda para aquele preditor naquele ponto -- e a mesma celula do mesmo
  # raster, alcancada por dois caminhos totalmente independentes:
  #
  #   tabela of pontos : terra::extract() sobre um SpatVector (script 01)
  #   patch store      : cellFromXY -> row/col -> patch_cell_index (script 02)
  #
  # Erro of CRS, troca of linha/coluna, off-by-one, reordenacao of canal ou
  # diretorio of raster desatualizado quebram essa igualdade. Os testes
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
      add_check("02", "patch centre == point-table value", "FAIL",
                paste("error:", conditionMessage(attr(cc, "condition"))))
    } else {
      add_check("02", "patch centre == point-table value",
                if (cc$ok) "PASS" else "FAIL",
                sprintf("%s cells (%s pts x %d channels, window %dx%d) | %d divergent",
                        format(cc$n_cells, big.mark = ","),
                        format(cc$n_points, big.mark = ","),
                        cc$n_channels, cc$window, cc$window, cc$n_mismatch))
      if (!cc$ok) {
        .say("\n  DIVERGENCE at the patch centres -- channels affected:")
        print(dplyr::slice_head(cc$by_channel, n = 10), width = Inf)
        .say("  Largest absolute difference: ", signif(cc$worst, 6))
      }
    }
    rm(pts_all, pmeta); invisible(gc(verbose = FALSE))
  }

  add_check("02", "patch sample saved for visual inspection",
            if (file.exists(file.path(patch_dir, "patch_sample.rds"))) "PASS" else "WARN",
            "patch_sample.rds -- written by stage 02 at no extra cost")

} else {
  .say("Stage 02 incomplete -- skipping content checks.")
}

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 03 — Busca of hiperparâmetros (tuning)
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Stage 03: hyperparameter tuning --\n")

tuning_dir <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling", target_label)

if (!dir.exists(tuning_dir)) {
  .say("Stage 03 not started -- directory not found: ", tuning_dir)
} else {
  tuning_runs <- list.dirs(tuning_dir, recursive = FALSE, full.names = FALSE)
  if (length(tuning_runs) == 0) {
    .say("Stage 03 incomplete -- no run found in: ", tuning_dir)
  } else {
    # Mais recente por ordenacao do nome (run_id e timestamped) -- mesmo
    # criterio usado para resolver "latest" no 04/05/06.
    tuning_run_id <- sort(tuning_runs, decreasing = TRUE)[1]
    run_dir <- file.path(tuning_dir, tuning_run_id)
    .say("Most recent run: ", tuning_run_id)

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
      # Contar configs no numerador e configs no denominador. Com units,
      # `n_cmp` sao LINHAS -- imprimir "27/3 configs" mistura as duas escalas
      # e le como se sobrassem configs.
      n_cfg_seen <- dplyr::n_distinct(comparison$config_id)
      add_check("03", "every config in the grid has a comparison row",
                if (length(missing_ids) == 0) "PASS" else "FAIL",
                if (length(missing_ids) == 0)
                  sprintf("%d/%d configs (%d units)", n_cfg_seen, n_grid, n_cmp)
                else paste("missing:", paste(missing_ids, collapse = ", ")))

      # Nenhuma linha extra na comparacao que nao esteja no grid atual --
      # indicaria mistura of runs diferentes (ex.: resume com tune_grid trocado
      # sem passar por um run_id novo).
      extra_ids <- setdiff(comparison$config_id, tune_grid$config_id)
      add_check("03", "no config in the comparison outside the current grid",
                if (length(extra_ids) == 0) "PASS" else "FAIL",
                if (length(extra_ids) == 0) "" else paste("extra:", paste(extra_ids, collapse = ", ")))

      # status == success para todas -- configs com erro nunca escrevem linha
      # em comparison_all.csv (ver run_cnn_tuning), entao qualquer coisa
      # != success aqui seria corrupcao inesperada do CSV, nao uma falha normal
      # of treino (essas simplesmente nao aparecem, ja coberto pelo check acima).
      n_not_success <- sum(comparison$status != "success", na.rm = TRUE)
      add_check("03", "every row has status == success",
                if (n_not_success == 0) "PASS" else "FAIL",
                sprintf("%d row(s) with status != success", n_not_success))

      # Cada config com linha na comparacao precisa ter o checkpoint .pt -- e
      # o sinal of "realmente terminou o treino" usado pelo resume (ver
      # run_cnn_tuning em R/train_cnn.R). Uma linha sem checkpoint deixaria um
      # resume futuro confuso sobre se aquela config precisa ser retreinada.
      # Por UNIDADE: com repeticoes, duas sementes da mesma config sao dois
      # modelos. Procurar por config_id acharia um arquivo que nao existe e
      # deixaria of checar os que existem.
      ckpt_files <- file.path(run_dir, "models", paste0(comparison$unit_id, "_best.pt"))
      n_missing_ckpt <- sum(!file.exists(ckpt_files))
      add_check("03", "every unit in the comparison has a .pt checkpoint",
                if (n_missing_ckpt == 0) "PASS" else "FAIL",
                sprintf("%d checkpoint(s) missing", n_missing_ckpt))

      # best_epoch nao pode ser NA nem <= 0 (indicaria que o treino nunca
      # passou no criterio of melhora do early stopping -- treino quebrado).
      n_bad_epoch <- sum(is.na(comparison$best_epoch) | comparison$best_epoch <= 0)
      add_check("03", "best_epoch valid (non-NA, > 0) in every config",
                if (n_bad_epoch == 0) "PASS" else "FAIL",
                sprintf("%d config(s) with an invalid best_epoch", n_bad_epoch))

      # Metricas of validacao dentro of faixa FISICAMENTE plausivel (nao NA,
      # CCC em [-1,1], MAE/RMSE > 0). Nao julga "quao bom" o modelo e -- isso
      # e decisao of modelagem, nao bug estrutural -- so descarta valores
      # impossiveis (sinal of erro no calculo, nao of modelo ruim).
      n_na_metrics <- sum(is.na(comparison$val_ccc) | is.na(comparison$val_mae) |
                          is.na(comparison$val_rmse))
      add_check("03", "val_ccc/val_mae/val_rmse free of NA",
                if (n_na_metrics == 0) "PASS" else "FAIL",
                sprintf("%d config(s) with an NA metric", n_na_metrics))

      n_ccc_out_of_range <- sum(comparison$val_ccc < -1 | comparison$val_ccc > 1, na.rm = TRUE)
      add_check("03", "val_ccc within [-1, 1]",
                if (n_ccc_out_of_range == 0) "PASS" else "FAIL",
                sprintf("%d config(s) out of range", n_ccc_out_of_range))

      n_nonpos_error <- sum(comparison$val_mae <= 0 | comparison$val_rmse <= 0, na.rm = TRUE)
      add_check("03", "val_mae and val_rmse > 0",
                if (n_nonpos_error == 0) "PASS" else "FAIL",
                sprintf("%d config(s) with error <= 0", n_nonpos_error))

      # ── Reamostragem: a contabilidade das repeticoes ────────────────────────
      f_plan     <- file.path(run_dir, "fold_plan.rds")
      f_byconfig <- file.path(run_dir, "comparison", "comparison_by_config.csv")
      has_plan   <- file.exists(f_plan)

      n_folds_run <- dplyr::n_distinct(comparison$fold)
      n_seeds_run <- dplyr::n_distinct(comparison$seed)

      add_check("03", "resampling plan saved with the run",
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
      # sai of menos repeticoes que as outras e a comparacao fica torta.
      n_expected <- n_grid * n_folds_run * n_seeds_run
      add_check("03", "units = configs x folds x seeds",
                if (n_cmp == n_expected) "PASS" else "WARN",
                sprintf("%d linhas | %d configs x %d fold(s) x %d semente(s) = %d",
                        n_cmp, n_grid, n_folds_run, n_seeds_run, n_expected))

      # A semente TEM que ser a mesma em todas as configs of uma repeticao --
      # e isso que faz duas configs serem comparadas sob o mesmo sorteio. Se
      # cada config tiver sua propria semente, parte of toda diferenca medida
      # e sorte, e nada no resultado avisa.
      if (!all(is.na(comparison$seed))) {
        seeds_per_cfg <- comparison %>%
          dplyr::group_by(config_id) %>%
          dplyr::summarise(s = paste(sort(unique(seed)), collapse = ","),
                           .groups = "drop")
        add_check("03", "same set of seeds across every config",
                  if (dplyr::n_distinct(seeds_per_cfg$s) == 1L) "PASS" else "FAIL",
                  sprintf("%d distinct set(s) | seeds: %s",
                          dplyr::n_distinct(seeds_per_cfg$s),
                          seeds_per_cfg$s[1]))
      }

      # rank 1 da tabela POR CONFIG e realmente a maior media -- e nao a linha
      # que teve o melhor run isolado.
      if (file.exists(f_byconfig)) {
        by_config <- safe_read_csv2(f_byconfig)
        top_by_rank <- by_config$config_id[by_config$rank == 1][1]
        top_by_mean <- by_config$config_id[which.max(by_config$val_ccc_mean)]
        check_equal("03", "rank 1 is the highest mean val_ccc",
                    top_by_rank, top_by_mean, "rank_1", "max_mean")

        # A pergunta que decide se este tuning significa alguma coisa: a
        # distancia entre o 1o e o 2o e maior que o ruido of semente?
        if (nrow(by_config) > 1L && "val_ccc_sd" %in% names(by_config)) {
          ord <- dplyr::arrange(by_config, rank)
          gap <- ord$val_ccc_mean[1] - ord$val_ccc_mean[2]
          # O piso of ruido e o sd entre SEMENTES dentro do mesmo fold, medido
          # por seed_noise_floor(). O sd da tabela por config mistura fold e
          # semente -- usa-lo aqui com o rotulo "entre sementes" reportava uma
          # quantidade diferente da que o 03 imprime, com o mesmo nome.
          nf    <- seed_noise_floor(comparison)
          noise <- nf$median_sd
          add_check("03", "the winner stands above the seed noise",
                    if (!is.finite(noise)) "WARN"
                    else if (gap >= noise) "PASS" else "WARN",
                    if (!is.finite(noise))
                      "sd entre sementes indisponivel (1 repeticao por config)"
                    else sprintf("1o-2o = %.4f | typical sd between seeds = %.4f",
                                 gap, noise))
        }
        top_by_ccc <- top_by_mean
      } else {
        top_by_rank <- comparison$config_id[comparison$rank == 1][1]
        top_by_ccc  <- comparison$config_id[which.max(comparison$val_ccc)]
        check_equal("03", "rank 1 is the highest val_ccc",
                    top_by_rank, top_by_ccc, "rank_1", "max_ccc")
      }

      # gate_summary.csv so deveria existir para configs dual-branch (janela
      # com "x" no nome, ex. "9x15") com gate != no_gate_concat -- confirma
      # que a logica condicional of extract_gate_analysis() nao esta gerando
      # (ou deixando of gerar) arquivo para o tipo of config errado.
      gated_ids <- comparison$unit_id[
        grepl("x", comparison$window_sizes) & comparison$gate_type != "no_gate_concat"
      ]
      gate_files <- file.path(run_dir, "gates", paste0(gated_ids, "_gate_summary.csv"))
      n_missing_gate <- sum(!file.exists(gate_files))
      add_check("03", "gate_summary.csv exists for every gated dual-branch config",
                if (n_missing_gate == 0) "PASS" else "WARN",
                sprintf("%d/%d faltando", n_missing_gate, length(gated_ids)))

      # Metrica media quando ha repeticoes; a da unica linha quando nao ha.
      top_rows <- dplyr::filter(comparison, config_id == top_by_ccc)
      .say(sprintf(
        "\n  Best config: %s | CCC=%.3f | MAE=%.2f | RMSE=%.2f | janela=%s | gate=%s | %d repetition(s)",
        top_by_ccc,
        mean(top_rows$val_ccc,  na.rm = TRUE),
        mean(top_rows$val_mae,  na.rm = TRUE),
        mean(top_rows$val_rmse, na.rm = TRUE),
        top_rows$window_sizes[1],
        top_rows$gate_type[1],
        nrow(top_rows)))

    } else {
      .say("Stage 03 incomplete -- skipping content checks.")
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 04 — Modelo final (ensemble multi-seed)
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Stage 04: final model (multi-seed ensemble) --\n")

final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

if (!dir.exists(final_model_base)) {
  .say("Stage 04 not started -- directory not found: ", final_model_base)
} else {
  final_runs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  final_runs <- final_runs[grepl("^final_", final_runs)]
  if (length(final_runs) == 0) {
    .say("Stage 04 incomplete -- no run found in: ", final_model_base)
  } else {
    final_run_id <- sort(final_runs, decreasing = TRUE)[1]
    run_dir <- file.path(final_model_base, final_run_id)
    .say("Most recent run: ", final_run_id)

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

      # O 04 grava qual run of tuning (etapa 03) usou -- confirma que essa
      # pasta ainda existe (nao foi apagada/renomeada depois) e, quando a
      # etapa 03 tambem rodou nesta mesma checagem, que e exatamente o run
      # mais recente resolvido la em cima (evita ficar preso num run antigo
      # por engano, ex.: tuning_run_id fixo esquecido no script).
      linked_tuning_dir <- file.path(tuning_dir, summary_rds$tuning_run_id)
      add_check("04", "the tuning_run_id stage 04 refers to still exists",
                if (dir.exists(linked_tuning_dir)) "PASS" else "FAIL",
                summary_rds$tuning_run_id)
      if (exists("tuning_run_id") && all_03_exist) {
        check_equal("04", "tuning_run_id do 04 == run mais recente da etapa 03",
                    summary_rds$tuning_run_id, tuning_run_id, "used_by_04", "mais_recente_03")
      }

      # Quando selected_config_ids foi deixado NULL (comportamento padrao,
      # recomendado no cabecalho do 04), o config escolhido tem que ser
      # exatamente o rank==1 do ranking of validacao daquele run of tuning --
      # senao o modelo final estaria sendo treinado numa arquitetura que nao
      # e a melhor encontrada na etapa 03. Selecao manual of top-N e valida,
      # entao isso e so um alerta (WARN), nao falha.
      if (exists("tuning_run_id") && all_03_exist &&
          identical(summary_rds$tuning_run_id, tuning_run_id)) {
        rank1_id <- comparison$config_id[comparison$rank == 1L]
        add_check("04", "selected config(s) include stage 03's rank 1",
                  if (rank1_id %in% selected_cfgs$config_id) "PASS" else "WARN",
                  paste0("rank1=", rank1_id, " | selecionados=",
                        paste(selected_cfgs$config_id, collapse = ", ")))
      }

      # Cada config selecionado precisa ter exatamente n_seeds_expected linhas
      # of resultado -- nem seed faltando (crash/erro silencioso), nem seed a
      # mais (resquicio of outro run com seeds diferentes).
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
      # como sucesso na tabela of metricas.
      ckpt_paths <- character(0)
      for (cid in selected_cfgs$config_id) {
        ckpt_paths <- c(ckpt_paths, file.path(run_dir, cid, "models",
                                              sprintf("seed%04d_best.pt", seeds_expected)))
      }
      n_missing_ckpt <- sum(!file.exists(ckpt_paths))
      add_check("04", "every expected (config, seed) has a .pt checkpoint",
                if (n_missing_ckpt == 0) "PASS" else "FAIL",
                sprintf("%d/%d checkpoint(s) missing", n_missing_ckpt, length(ckpt_paths)))

      # Metricas of teste sem NA e em faixa fisicamente plausivel (mesma
      # logica do 03: nao julga "quao bom", so descarta valores impossiveis).
      n_na_metrics <- sum(is.na(all_seed_results$ccc) | is.na(all_seed_results$mae) |
                          is.na(all_seed_results$rmse))
      add_check("04", "ccc/mae/rmse free of NA (every seed)",
                if (n_na_metrics == 0) "PASS" else "FAIL",
                sprintf("%d row(s) with an NA metric", n_na_metrics))

      n_ccc_out <- sum(all_seed_results$ccc < -1 | all_seed_results$ccc > 1, na.rm = TRUE)
      add_check("04", "ccc within [-1, 1] (every seed)",
                if (n_ccc_out == 0) "PASS" else "FAIL",
                sprintf("%d linha(s) fora do range", n_ccc_out))

      n_nonpos <- sum(all_seed_results$mae <= 0 | all_seed_results$rmse <= 0, na.rm = TRUE)
      add_check("04", "mae and rmse > 0 (every seed)",
                if (n_nonpos == 0) "PASS" else "FAIL",
                sprintf("%d row(s) with error <= 0", n_nonpos))

      # Estabilidade entre seeds: SD do CCC como % da media. Um desvio grande
      # (ver docs/design_decisions.md secao 11) indica treino instavel/pouco
      # reprodutivel, nao so "sorte" of inicializacao -- resultado publicavel
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
      .say("Stage 04 incomplete -- skipping content checks.")
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# [Placeholder para etapas futuras]
#
# Etapa 05/06 (predicao espacial): valid_fraction dos tiles nao caiu of novo
# pra perto of zero -- mesma logica do check do 02, adaptada para os logs
# of shard/merge. Ver tambem 05a_test.R / 05c_estimate_eta.R, que ja cobrem
# parte disso para o pipeline 2D.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT — o que mudou desde o run anterior?
#
# Num ciclo of refatoracao, a pergunta feita apos cada execucao e "mexeu em
# alguma coisa?", e responde-la significava rolar a tela atras of saida
# antiga. "Tudo identico" e o resultado que se quer ver, e e justamente o
# mais dificil of confirmar of olho.
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Snapshot: comparison with the previous run --\n")

snap_vals <- list()
add_snap <- function(k, v) {
  if (length(v) == 1L && !is.null(v) && !is.na(v)) snap_vals[[k]] <<- v
}

if (exists("dscheck")) {
  add_snap("01_n_linhas",     dscheck$n_rows[1])
  add_snap("01_n_predictors", dscheck$n_predictors[1])
  add_snap("01_n_dummy",      dscheck$n_dummy_predictors[1])
  add_snap("01_n_percentage", dscheck$n_percentage_predictors[1])
  add_snap("01_n_continuous", dscheck$n_continuous_predictors[1])
  add_snap("01_mediana_alvo", round(dscheck$median_target[1], 6))
}
if (exists("qc")) add_snap("01_pct_problema", qc$pct_any_problem[1])
if (exists("crisk")) {
  add_snap("01_canais_constantes", sum(crisk$risk == "constant", na.rm = TRUE))
  add_snap("01_canais_com_na",     sum(crisk$risk == "has_na",   na.rm = TRUE))
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
# no diff. Sao derivados of todas as outras chaves: se uma metrica real
# mudar, ela aparece no diff sozinha. Compara-los cria uma alca fechada em
# que CONSERTAR um WARN gera um WARN ("valores alterados"), que foi
# exatamente o que aconteceu no run que corrigiu os dois checks quebrados.
cmp <- compare_run_snapshot(snap_vals, snap_dir,
                            exclude = c("99_n_pass", "99_n_warn", "99_n_fail"))
print_snapshot_diff(cmp)
write_run_snapshot(snap_vals, snap_dir)

if (cmp$has_previous) {
  n_changed <- sum(cmp$diff$status != "=")
  add_check("99", "values changed since the previous run",
            if (n_changed == 0L) "PASS" else "WARN",
            sprintf("%d of %d", n_changed, nrow(cmp$diff)))
}

# ── Resumo final ─────────────────────────────────────────────────────────────

.say("\n", strrep("=", 90))
.say("SUMMARY")
.say(strrep("=", 90))

n_pass <- sum(.results$status == "PASS")
n_warn <- sum(.results$status == "WARN")
n_fail <- sum(.results$status == "FAIL")

.say(sprintf("\n  PASS: %d   WARN: %d   FAIL: %d   (total: %d checks)\n",
                n_pass, n_warn, n_fail, nrow(.results)))

# So conta como problema o que realmente falhou -- etapa nao iniciada nem
# chega a registrar checagem.
if (n_fail > 0) {
  .say("Checks that FAILED:")
  print(dplyr::filter(.results, status == "FAIL"), n = Inf, width = Inf)
}
if (n_warn > 0) {
  .say("\nChecks with a WARNING:")
  print(dplyr::filter(.results, status == "WARN"), n = Inf, width = Inf)
}

report_dir <- file.path(project_root, "outputs", "qc")
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
report_file <- file.path(report_dir, paste0("pipeline_check_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
readr::write_csv2(.results, report_file)
.say("\nFull report saved to: ", report_file)

if (n_fail > 0) {
  warning(n_fail, " check(s) FAILED. Review before moving on to the next stage.")
} else if (n_warn > 0) {
  .say("\nNo critical failure, but there are warnings -- review before spending CPU on the next stage.")
} else {
  .say("\nAll clear. Safe to move on to the next stage.")
}
