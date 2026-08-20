project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("terra", "dplyr", "readr", "tibble", "purrr", "stringr")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))

# ══════════════════════════════════════════════════════════════════════════════
# 06b — Mosaica os tiles 2D produzidos por 05_predict_spatial.R
#
# Encontra todos os arquivos em raster/parts_2d/ com sufixo
# _rXXXofYYY_cXXXofZZZ.tif, agrupa por camada (median, sd, etc.) e monta
# o raster final com terra::merge. Verifica que a extensao do mosaico bate
# com o template original.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"
config_id    <- "auto"
final_run_id <- "latest"

# ── Resolve config_id / final_run_id ──────────────────────────────────────────

metadata_dir     <- file.path(project_root, "outputs", "metadata",
                              "soc_stock_modeling", target_label)
final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

if (identical(final_run_id, "latest")) {
  run_dirs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  run_dirs <- run_dirs[grepl("^final_", run_dirs)]
  if (length(run_dirs) == 0) stop("Nenhum final_* encontrado em: ", final_model_base)
  final_run_id <- sort(run_dirs, decreasing = TRUE)[1]
  message("final_run_id: ", final_run_id)
}

if (identical(config_id, "auto")) {
  summary_path <- file.path(final_model_base, final_run_id, "comparison",
                            "final_run_summary.rds")
  if (!file.exists(summary_path)) stop("final_run_summary.rds nao encontrado.")
  config_id <- readRDS(summary_path)$selected_cfgs$config_id[1]
  message("config_id: ", config_id)
}

# ── Paths ──────────────────────────────────────────────────────────────────────

output_dir        <- file.path(project_root, "outputs", "spatial_prediction",
                               "soc_stock_modeling", target_label, config_id)
parts_dir         <- file.path(output_dir, "raster", "parts_2d")
output_raster_dir <- file.path(output_dir, "raster")
output_log_dir    <- file.path(output_dir, "log")

if (!dir.exists(parts_dir)) stop("Pasta de partes 2D nao encontrada: ", parts_dir)

# ── Listar arquivos por camada ─────────────────────────────────────────────────

all_parts <- list.files(parts_dir, pattern = "_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.tif$",
                        full.names = TRUE)
if (length(all_parts) == 0) stop("Nenhum tile 2D encontrado em: ", parts_dir)

message(sprintf("\n%d tiles 2D encontrados em: %s", length(all_parts), parts_dir))

# Extrai sufixo de camada (parte do nome entre config_id_ e _rXXX)
layer_pattern <- paste0("^", target_label, "_", config_id, "_(.+)_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.tif$")
file_suffixes <- unique(sub(layer_pattern, "\\1", basename(all_parts)))

message("Camadas detectadas: ", paste(file_suffixes, collapse = ", "))

# ── Template do raster completo ────────────────────────────────────────────────

raster_table_file <- file.path(metadata_dir, "raster_table_used.csv")
if (!file.exists(raster_table_file)) stop("raster_table_used.csv nao encontrado.")
raster_table   <- readr::read_csv2(raster_table_file, show_col_types = FALSE)
full_template  <- terra::rast(raster_table$raster_file[1])

# ── Função de merge por camada ─────────────────────────────────────────────────

merge_layer <- function(suffix) {
  pat <- paste0("^", target_label, "_", config_id, "_", suffix,
                "_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.tif$")
  parts <- sort(list.files(parts_dir, pattern = pat, full.names = TRUE))

  if (length(parts) == 0) {
    warning("Nenhum tile para camada: ", suffix)
    return(NULL)
  }

  message(sprintf("\n[%s] Mosaicando %d tiles...", suffix, length(parts)))
  t_start <- Sys.time()

  out_file <- file.path(output_raster_dir,
                        paste0(target_label, "_", config_id, "_", suffix, ".tif"))
  if (file.exists(out_file)) file.remove(out_file)

  datatype <- if (grepl("mask", suffix)) "INT1U" else "FLT4S"
  predictor <- if (grepl("^INT|^UINT|^BYTE", datatype)) 2L else 3L

  rast_list <- lapply(parts, terra::rast)
  merged    <- terra::merge(
    terra::sprc(rast_list),
    filename  = out_file,
    overwrite = TRUE,
    datatype  = datatype,
    gdal      = c("COMPRESS=DEFLATE", paste0("PREDICTOR=", predictor),
                  "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512")
  )

  dt <- Sys.time() - t_start
  message(sprintf("  -> %s  (%.1f %s)", basename(out_file),
                  as.numeric(dt), units(dt)))

  # Verifica extensão e dimensões
  if (!terra::compareGeom(merged, full_template, stopOnError = FALSE)) {
    warning("[", suffix, "] Geometria do mosaico difere do template original — verifique.")
  } else {
    message("  Geometria OK.")
  }

  out_file
}

merged_files <- purrr::map(file_suffixes, merge_layer)

# ── Sanity check no mapa de mediana ───────────────────────────────────────────

median_file <- merged_files[[which(file_suffixes == "ensemble_median_ton_ha")]]
if (!is.null(median_file) && file.exists(median_file)) {
  message("\n── Sanity check (mosaico completo) ─────────────────────────────")
  r_med   <- terra::rast(median_file)
  gstats  <- terra::global(r_med, c("min", "mean", "max"), na.rm = TRUE)
  g_mean  <- gstats[1, "mean"]
  g_max   <- gstats[1, "max"]

  message(sprintf("  Median map: min %.2f | mean %.2f | max %.2f  ton_ha",
                  gstats[1, "min"], g_mean, g_max))

  if (is.na(g_mean) || g_mean < 1 || g_mean > 200) {
    message("  [ATENCAO] Média global fora do range esperado [1, 200] — verifique o mosaico.")
  } else {
    message("  [OK] Média global dentro do range plausivel.")
  }
  if (!is.na(g_max) && g_max > 1000) {
    message(sprintf("  [WARN] Maximo %.0f > 1000 — verifique pixels extremos.", g_max))
  }
}

message("\n── Merge 2D concluido ────────────────────────────────────────────")
message("  Rasters finais: ", output_raster_dir)
message("  Tiles de partes: ", parts_dir, " (podem ser apagados apos verificacao)")
