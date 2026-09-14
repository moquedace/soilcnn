project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("processx", "terra", "dplyr", "readr", "tibble", "purrr")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))

# ══════════════════════════════════════════════════════════════════════════════
# SCRATCH — Orquestrador: reamostra os 187 preditores de 250 m para 1 km.
#
# NAO faz parte do pipeline principal, NAO sera comitado (pasta _scratch_1km_test/
# nunca recebe git add; *.tif/*.csv tambem ja caem nos filtros amplos do
# .gitignore do projeto).
#
# Cada preditor roda em um processo Rscript SEPARADO (resample_one_predictor.R),
# nao num loop dentro desta sessao -- a primeira tentativa em loop unico deu
# "std::bad_alloc" ja no primeiro raster (essas camadas globais a 250 m tem
# varios GB cada). Isolamento por processo = memoria devolvida ao SO de forma
# garantida entre preditores, e da pra rodar alguns em paralelo com seguranca.
# Mesmo padrao ja usado no 05a_run_parallel.R para a predicao espacial.
#
# Objetivo do exercicio: validar o 05_predict_spatial.R (05a/05b) numa grade
# 1 km (16x menos celulas que 250 m) antes de comprometer o job real a 250 m.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"

# Comece conservador -- o crash em loop unico ja mostrou que essas camadas sao
# pesadas. 2-3 e um ponto de partida razoavel numa maquina de 32 nucleos/64 GB;
# suba so depois de ver o RSS real nos primeiros preditores concluidos.
max_concurrent  <- 2
poll_interval_s <- 10

src_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_250m"
dst_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_1km"
dir.create(dst_dir, recursive = TRUE, showWarnings = FALSE)

metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)
raster_table_file      <- file.path(metadata_dir, "raster_table_used.csv")
predictor_scaling_file <- file.path(metadata_dir, "predictor_scaling.csv")

scratch_dir   <- file.path(project_root, "_scratch_1km_test")
worker_script <- file.path(scratch_dir, "resample_one_predictor.R")

if (!file.exists(raster_table_file))      stop("Nao encontrado: ", raster_table_file)
if (!file.exists(predictor_scaling_file)) stop("Nao encontrado: ", predictor_scaling_file)
if (!file.exists(worker_script))          stop("Worker nao encontrado: ", worker_script)

rscript_bin <- file.path(R.home("bin"), "Rscript.exe")
if (!file.exists(rscript_bin)) stop("Rscript.exe not found: ", rscript_bin)

raster_table <- readr::read_csv2(raster_table_file, show_col_types = FALSE)
predictor_scaling <- readr::read_csv2(predictor_scaling_file, show_col_types = FALSE) %>%
  dplyr::select(predictor, is_dummy)
raster_table <- dplyr::left_join(raster_table, predictor_scaling, by = "predictor")

missing_type <- sum(is.na(raster_table$is_dummy))
if (missing_type > 0) {
  stop(missing_type, " preditor(es) sem is_dummy resolvido -- checar predictor_scaling.csv")
}

n_total <- nrow(raster_table)
message(sprintf("Preditores a reamostrar: %d (%d dummy -> modal | %d continuo/percentual -> mean)",
                n_total, sum(raster_table$is_dummy), sum(!raster_table$is_dummy)))

log_dir <- file.path(scratch_dir, "_worker_logs", format(Sys.time(), "%Y%m%d_%H%M%S"))
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
message("Logs por preditor: ", log_dir, "\n")

# ── Fila de processos ───────────────────────────────────────────────────────
# Resume NAO e decidido aqui -- o worker (resample_one_predictor.R) abre e
# valida a geometria de qualquer .tif de saida existente antes de decidir
# pular ou reprocessar (so checar file.exists() aqui aceitaria cegamente um
# arquivo corrompido/incompleto, como aconteceu com um leftover da primeira
# tentativa). Lanca todos os N: quem ja esta pronto sai quase instantaneo.

pending    <- seq_len(n_total)
active     <- list()
exit_codes <- integer(n_total)

launch_worker <- function(idx) {
  log_file <- file.path(log_dir, sprintf("predictor_%03d_%s.log", idx, raster_table$predictor[idx]))
  p <- processx::process$new(
    rscript_bin,
    args = c(worker_script, as.character(idx)),
    stdout = log_file, stderr = log_file, cleanup = TRUE
  )
  message(sprintf("  [%3d/%d] %-45s iniciado (PID %d)",
                  idx, n_total, raster_table$predictor[idx], p$get_pid()))
  p
}

while (length(active) < max_concurrent && length(pending) > 0) {
  idx <- pending[1]; pending <- pending[-1]
  active[[as.character(idx)]] <- launch_worker(idx)
}

t0 <- Sys.time()
n_done <- 0L

while (length(active) > 0 || length(pending) > 0) {
  Sys.sleep(poll_interval_s)

  finished_ids <- character(0)
  for (idx_chr in names(active)) {
    p <- active[[idx_chr]]
    if (!p$is_alive()) {
      idx <- as.integer(idx_chr)
      exit_codes[idx] <- p$get_exit_status()
      n_done <- n_done + 1L
      status <- if (exit_codes[idx] == 0L) "OK" else paste0("FALHOU (exit ", exit_codes[idx], ")")
      message(sprintf("  [%3d/%d] %-45s %s", idx, n_total, raster_table$predictor[idx], status))
      finished_ids <- c(finished_ids, idx_chr)
    }
  }
  active[finished_ids] <- NULL

  while (length(active) < max_concurrent && length(pending) > 0) {
    idx <- pending[1]; pending <- pending[-1]
    active[[as.character(idx)]] <- launch_worker(idx)
  }

  el <- Sys.time() - t0
  message(sprintf("  [%.1f %s] %d/%d concluidos, %d rodando, %d na fila",
                  as.numeric(el), units(el), n_done, n_total, length(active), length(pending)))
}

el_total <- Sys.time() - t0
message(sprintf("\nTodos os preditores processados em %.1f %s.", as.numeric(el_total), units(el_total)))

failed <- which(exit_codes != 0L)
if (length(failed) > 0) {
  message(sprintf("\n[ATENCAO] %d preditor(es) falharam: %s",
                  length(failed), paste(raster_table$predictor[failed], collapse = ", ")))
  message("Logs em: ", log_dir)
  message("Rode este script de novo (resume automatico) apos investigar os logs.")
}

# ── QC dos rasters 1 km gerados ────────────────────────────────────────────────

message("\n-- QC dos rasters 1 km gerados --\n")

out_files    <- file.path(dst_dir, basename(raster_table$raster_file))
missing_out  <- out_files[!file.exists(out_files)]
if (length(missing_out) > 0) {
  message(length(missing_out), " arquivo(s) de saida ainda faltando:")
  print(missing_out)
}

present <- out_files[file.exists(out_files)]
if (length(present) > 0) {
  stack_1km <- terra::rast(present)

  geom_ok <- purrr::map_lgl(
    seq_len(terra::nlyr(stack_1km)),
    ~ terra::compareGeom(stack_1km[[1]], stack_1km[[.x]], stopOnError = FALSE)
  )
  message(sprintf("Geometria consistente entre camadas: %d/%d", sum(geom_ok), length(geom_ok)))
  if (!all(geom_ok)) {
    print(tibble::tibble(file = basename(present), geometry_ok = geom_ok) %>%
            dplyr::filter(!geometry_ok))
  }

  frac_na <- purrr::map_dbl(seq_len(terra::nlyr(stack_1km)), function(i) {
    s <- terra::global(stack_1km[[i]], fun = "isNA")
    s[1, 1] / terra::ncell(stack_1km[[i]])
  })
  all_na <- tibble::tibble(file = basename(present), frac_na = frac_na) %>%
    dplyr::filter(frac_na > 0.999)
  if (nrow(all_na) > 0) {
    message("\n[ATENCAO] camada(s) 100% (ou quase) NA apos agregacao:")
    print(all_na)
  } else {
    message("Nenhuma camada 100% NA.")
  }

  message(sprintf("\nGrade 1 km: %d linhas x %d colunas x %d camadas",
                  terra::nrow(stack_1km), terra::ncol(stack_1km), terra::nlyr(stack_1km)))
  message(sprintf("Grade 250 m original: %d linhas x %d colunas",
                  terra::nrow(terra::rast(raster_table$raster_file[1])),
                  terra::ncol(terra::rast(raster_table$raster_file[1]))))
}

# ── raster_table_used_1km.csv: mesma estrutura, apontando para dst_dir ────────
# So troca o diretorio -- mesmos nomes de arquivo, mesma ordem de preditores.
# predictor_scaling.csv NAO muda -- as estatisticas de escala vieram do treino
# em resolucao nativa dos pontos, sao independentes da resolucao do raster de
# predicao (ver apply_predictor_scaling() em 05_predict_spatial.R -- a correcao
# de clamp dos percentuais roda la, nao aqui, e vale igual pra 1km e 250m).

raster_table_1km <- raster_table %>%
  dplyr::mutate(raster_file = file.path(dst_dir, basename(raster_file))) %>%
  dplyr::select(raster_file, raster_name_raw, predictor)

safe_write_csv2(raster_table_1km, file.path(scratch_dir, "raster_table_used_1km.csv"))
message("\nSalvo: ", file.path(scratch_dir, "raster_table_used_1km.csv"))
