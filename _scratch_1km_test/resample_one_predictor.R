project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("terra", "dplyr", "readr")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

# ══════════════════════════════════════════════════════════════════════════════
# SCRATCH — worker: reamostra UM unico preditor de 250 m para 1 km.
#
# v2: chunking MANUAL em vez de confiar no terra::aggregate() de raster
# inteiro. A v1 (terraOptions(memfrac baixo) + aggregate(filename=...)) ainda
# quebrava com "std::bad_alloc" em ~10s, mesmo em processo isolado -- ou seja,
# nao era falta de RAM disponivel, era o proprio aggregate() tentando montar
# um buffer grande demais de uma vez pra essas camadas globais de varios GB,
# antes mesmo de comecar a ler em blocos de verdade.
#
# Solucao: recorta (crop, barato/lazy) uma tira de linhas por vez, agrega SO
# essa tira pequena (aggregate em cima de um pedaco pequeno e seguro), escreve
# no raster de saida via writeStart/writeValues, descarta e segue -- exatamente
# o mesmo principio de streaming por blocos que 05_predict_spatial.R ja usa
# (e comprovadamente funciona) pra ler essas mesmas camadas na predicao
# espacial.
#
# Resume mais rigoroso que a v1: um .tif de saida so conta como "pronto" se
# abrir sem erro E tiver a geometria (nrow/ncol) esperada -- nao so existir.
# A v1 deixou pra tras aboveground_biomass_carbon.tif corrompido (sobra do
# crash da primeira tentativa em loop unico) que a v1 aceitava cegamente como
# "ja feito".
#
# CLI: Rscript resample_one_predictor.R <indice_do_preditor>
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"
src_res_nominal_m <- 250   # resolucao nominal dos rasters de origem (nome do diretorio)
target_res_m      <- 1000  # resolucao nominal desejada

# Orcamento de RAM por chunk (linhas de ENTRADA lidas de uma vez, largura
# inteira do raster). Ajuste pra baixo se ainda ver bad_alloc/RSS alto.
max_chunk_ram_gb <- 1.5

src_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_250m"
dst_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_1km"
dir.create(dst_dir, recursive = TRUE, showWarnings = FALSE)

metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)
raster_table_file      <- file.path(metadata_dir, "raster_table_used.csv")
predictor_scaling_file <- file.path(metadata_dir, "predictor_scaling.csv")

# ── Índice do preditor: CLI (Rscript) ou env var (source() no console) ────────
predictor_idx <- NA_integer_
.cli_args <- commandArgs(trailingOnly = TRUE)
if (length(.cli_args) >= 1L) {
  predictor_idx <- as.integer(.cli_args[1])
} else if (nzchar(Sys.getenv("SOC_PREDICTOR_IDX"))) {
  predictor_idx <- as.integer(Sys.getenv("SOC_PREDICTOR_IDX"))
}
if (is.na(predictor_idx)) {
  stop("Informe o indice do preditor: Rscript resample_one_predictor.R <idx> ",
      "ou Sys.setenv(SOC_PREDICTOR_IDX=<idx>) antes de source().")
}

raster_table <- readr::read_csv2(raster_table_file, show_col_types = FALSE)
predictor_scaling <- readr::read_csv2(predictor_scaling_file, show_col_types = FALSE) %>%
  dplyr::select(predictor, is_dummy)
raster_table <- dplyr::left_join(raster_table, predictor_scaling, by = "predictor")

n_total <- nrow(raster_table)
stopifnot(predictor_idx >= 1L, predictor_idx <= n_total)
if (is.na(raster_table$is_dummy[predictor_idx])) {
  stop(raster_table$predictor[predictor_idx], ": is_dummy nao resolvido (checar predictor_scaling.csv)")
}

row <- raster_table[predictor_idx, ]
out_file <- file.path(dst_dir, basename(row$raster_file))

r <- terra::rast(row$raster_file)
src_res <- terra::res(r)
r_nrow  <- terra::nrow(r)
r_ncol  <- terra::ncol(r)

if (abs(src_res[1] - src_res[2]) > 1e-9) {
  message(sprintf("  [AVISO] resolucao X/Y nao identica em %s: %.6f vs %.6f",
                  row$predictor, src_res[1], src_res[2]))
}

# terra::res() retorna unidades NATIVAS do CRS do raster -- essas camadas
# estao em CRS geografico (graus), nao projetado em metros, entao res()[1] e
# algo como 0.00225 (~250 m no equador), nao 250. Dividir target_res_m por
# isso direto (1000/0.00225) da um fator absurdo (~445 mil) -- foi exatamente
# o bug da primeira tentativa: virou uma grade 1x1 e ainda assim quebrou.
#
# O fator certo nao depende de saber a conversao grau->metro: "250 m" e "1 km"
# sao as resolucoes NOMINAIS que ja nomeiam os diretorios de origem/destino
# neste projeto -- o fator e literalmente target_res_m / src_res_nominal_m.
fact <- round(target_res_m / src_res_nominal_m)
if (fact < 1L) {
  stop("src_res_nominal_m (", src_res_nominal_m, ") ja e mais grosseiro que target_res_m (",
      target_res_m, ") -- fator < 1.")
}

out_nrow <- ceiling(r_nrow / fact)
out_ncol <- ceiling(r_ncol / fact)

# ── Resume: aceita o arquivo existente SO se abrir e tiver a geometria certa ──
existing_ok <- FALSE
if (file.exists(out_file)) {
  existing_ok <- tryCatch({
    r_existing <- terra::rast(out_file)
    terra::nrow(r_existing) == out_nrow && terra::ncol(r_existing) == out_ncol
  }, error = function(e) FALSE)
  if (!existing_ok) {
    message(sprintf("[%3d/%d] %-45s -- saida existente invalida/incompleta, reprocessando",
                    predictor_idx, n_total, row$predictor))
    suppressWarnings(file.remove(out_file))
  }
}
if (existing_ok) {
  message(sprintf("[%3d/%d] %-45s -- ja existe e valido, pulando (resume)",
                  predictor_idx, n_total, row$predictor))
  quit(save = "no", status = 0L)
}

agg_fun <- if (isTRUE(row$is_dummy)) "modal" else "mean"

gdal_opts <- c("COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES",
              "BLOCKXSIZE=512", "BLOCKYSIZE=512")

# ── Template do raster de saida (so geometria -- mesma convencao do terra::
# aggregate: ancorado no canto superior-esquerdo do raster de origem) ─────────

full_ext <- as.vector(terra::ext(r))
xres <- src_res[1]; yres <- src_res[2]

out_template <- terra::rast(
  nrows = out_nrow, ncols = out_ncol,
  xmin  = full_ext[["xmin"]], xmax = full_ext[["xmin"]] + out_ncol * xres * fact,
  ymin  = full_ext[["ymax"]] - out_nrow * yres * fact, ymax = full_ext[["ymax"]],
  crs   = terra::crs(r)
)

# ── Tamanho do chunk (linhas de SAIDA por vez) a partir do orcamento de RAM ───

bytes_per_input_row <- as.numeric(r_ncol) * 8  # 8 bytes/celula (double interno do terra)
chunk_out_rows <- max(1L, floor((max_chunk_ram_gb * 1e9 / bytes_per_input_row) / fact))
n_chunks <- ceiling(out_nrow / chunk_out_rows)

message(sprintf("[%3d/%d] %-45s fact=%d (%s) | grade %dx%d -> %dx%d | %d chunk(s) de %d linha(s) de saida",
                predictor_idx, n_total, row$predictor, fact, agg_fun,
                r_nrow, r_ncol, out_nrow, out_ncol, n_chunks, chunk_out_rows))

t0 <- Sys.time()

result <- tryCatch({
  terra::writeStart(out_template, out_file, overwrite = TRUE,
                    datatype = "FLT4S", gdal = gdal_opts)

  for (ci in seq_len(n_chunks)) {
    out_row_start <- (ci - 1L) * chunk_out_rows + 1L
    out_row_end   <- min(out_nrow, ci * chunk_out_rows)

    in_row_start <- (out_row_start - 1L) * fact + 1L
    in_row_end   <- min(r_nrow, out_row_end * fact)

    chunk_ymax <- full_ext[["ymax"]] - (in_row_start - 1L) * yres
    chunk_ymin <- full_ext[["ymax"]] - in_row_end * yres
    chunk_ext  <- terra::ext(full_ext[["xmin"]], full_ext[["xmax"]], chunk_ymin, chunk_ymax)

    sub_r   <- terra::crop(r, chunk_ext, snap = "near")
    sub_agg <- terra::aggregate(sub_r, fact = fact, fun = agg_fun, na.rm = TRUE)

    vals <- terra::values(sub_agg)
    terra::writeValues(out_template, vals, out_row_start, terra::nrow(sub_agg))

    rm(sub_r, sub_agg, vals)
  }

  terra::writeStop(out_template)
  gc()
  "ok"
}, error = function(e) {
  message(sprintf("[%3d/%d] %-45s -- ERRO: %s", predictor_idx, n_total, row$predictor, conditionMessage(e)))
  try(terra::writeStop(out_template), silent = TRUE)
  suppressWarnings(if (file.exists(out_file)) file.remove(out_file))
  "error"
})

dt <- as.numeric(Sys.time() - t0, units = "secs")

if (identical(result, "ok")) {
  message(sprintf("[%3d/%d] %-45s -- %.1fs OK", predictor_idx, n_total, row$predictor, dt))
  quit(save = "no", status = 0L)
} else {
  quit(save = "no", status = 1L)
}
