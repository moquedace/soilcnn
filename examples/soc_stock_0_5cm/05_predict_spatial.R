project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c(
  "torch",
  "terra",
  "dplyr",
  "readr",
  "tibble",
  "purrr",
  "stringr",
  "matrixStats",
  "ps"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))
source(file.path(project_root, "R", "metrics.R"))
source(file.path(project_root, "R", "cnn_architecture.R"))
source(file.path(project_root, "R", "tune_grid.R"))
source(file.path(project_root, "R", "train_cnn.R"))

# ══════════════════════════════════════════════════════════════════════════════
# 05 — Spatial prediction 2D-tiled (block-streaming, seed ensemble)
#
# Each worker receives a RECTANGLE of the raster (a slice of rows AND
# columns) rather than only a row strip. This keeps per-process RAM
# proportional to the tile's column fraction, allowing more concurrent
# workers per machine.
#
# (Historical note: an earlier 1D, row-strip-only version of this script
# was retired after production use showed it needed ~7-8 GB RAM/shard at
# 187 bands x ~160k columns, limiting concurrency. The 2D tiling here
# reduces that to ~2 GB/shard at n_col_shards=4, enough headroom to run
# ~8 workers concurrently on the same RAM budget.)
#
# Each worker writes a tile covering exactly its own rectangle (no NA
# padding). 05b_merge_spatial_parts.R mosaics all tiles at the end.
#
# CLI: Rscript 05_predict_spatial.R <row_shard_id> <col_shard_id>
#                                   <n_row_shards>  <n_col_shards>
#                                   [max_concurrent]
# ══════════════════════════════════════════════════════════════════════════════

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

# ── Argumentos: CLI (Rscript) OU variáveis de ambiente (source() no console) ───
# rm(list=ls()) acima apaga qualquer variável pré-definida antes do source(),
# então a forma de passar parâmetros ao rodar via source() é por env var
# (Sys.setenv() sobrevive ao rm(list=ls()), diferente de objetos no workspace).
row_shard_id   <- 1L
col_shard_id   <- 1L
n_row_shards   <- 1L
n_col_shards   <- 1L
max_concurrent <- 1L   # usado para calcular threads_per_worker

.cli_args <- commandArgs(trailingOnly = TRUE)
if (length(.cli_args) >= 4L) {
  row_shard_id  <- as.integer(.cli_args[1])
  col_shard_id  <- as.integer(.cli_args[2])
  n_row_shards  <- as.integer(.cli_args[3])
  n_col_shards  <- as.integer(.cli_args[4])
  if (length(.cli_args) >= 5L) max_concurrent <- as.integer(.cli_args[5])
} else if (nzchar(Sys.getenv("SOC_ROW_SHARD_ID"))) {
  row_shard_id  <- as.integer(Sys.getenv("SOC_ROW_SHARD_ID"))
  col_shard_id  <- as.integer(Sys.getenv("SOC_COL_SHARD_ID"))
  n_row_shards  <- as.integer(Sys.getenv("SOC_N_ROW_SHARDS"))
  n_col_shards  <- as.integer(Sys.getenv("SOC_N_COL_SHARDS"))
  if (nzchar(Sys.getenv("SOC_MAX_CONCURRENT"))) {
    max_concurrent <- as.integer(Sys.getenv("SOC_MAX_CONCURRENT"))
  }
}

stopifnot(
  n_row_shards >= 1L, n_col_shards >= 1L,
  row_shard_id >= 1L, row_shard_id <= n_row_shards,
  col_shard_id >= 1L, col_shard_id <= n_col_shards,
  max_concurrent >= 1L
)

n_total_shards <- n_row_shards * n_col_shards
is_partitioned <- n_total_shards > 1L

config_id    <- "auto"
final_run_id <- "latest"
seeds        <- c(42L, 123L, 456L, 789L, 2025L)
ensemble_center <- "median"

# Parâmetros de streaming / RAM
# Com tiling 2D, bytes_per_strip_row usa strip_ncol (colunas do tile + margens),
# não r_ncol inteiro -> output_block_rows maior -> menos blocos -> menos overhead.
max_strip_ram_gb  <- 4        # budget de RAM por strip, por shard
output_block_rows <- 64L      # usado somente se max_strip_ram_gb for NULL
# Cap de linhas por bloco independente do strip budget.
# O RSS real por shard é dominado pelo acúmulo de patch arrays no heap do R
# (vals_valid + step1 + step2 para branches 9x9 e 15x15) proporcional ao
# n_valid/bloco. Com output_block_rows=52 e blocos densos (109k válidos/bloco)
# o RSS atingiu 34 GB -- inviável com max_concurrent>1. Limitando em 16 linhas:
# n_valid/bloco cai ~3x -> RSS esperado ~10-15 GB em shards densos.
max_output_block_rows <- 16L  # cap: nunca mais que isso, mesmo com strip barato
# batch_size domina o pico de RAM por chunk em build_patches_multi (a checagem
# de validade da janela indexa strip_values para todo o chunk, criando uma
# matriz temporaria de ~batch_size * n_pos_janela * n_channels * 8 bytes).
# Isso e independente de output_block_rows -- reduzir batch_size e o que
# realmente limita o pico de RSS por chunk. Com batch_size=4096 e janela 15x15
# (225 posicoes) x 187 canais, o pico chegava a ~1.4 GB soh nessa checagem,
# repetido a cada chunk -- e o que gerava os picos de 30-37 GB observados.
batch_size        <- 512L

plausible_median_range <- c(1, 200)
plausible_hard_max     <- 1000

# Threads divididos por max_concurrent (passado via CLI pelo orquestrador)
threads_per_worker <- max(1L, parallel::detectCores() %/% max_concurrent)
device <- setup_torch_device(n_threads = threads_per_worker, use_cuda = TRUE)

# ── Paths ─────────────────────────────────────────────────────────────────────

metadata_dir     <- file.path(project_root, "outputs", "metadata",
                              "soc_stock_modeling", target_label)
patch_dir        <- file.path(project_root, "outputs", "patches",
                              "soc_stock_modeling", target_label)
final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

raster_table_file      <- file.path(metadata_dir, "raster_table_used.csv")
predictor_scaling_file <- file.path(metadata_dir, "predictor_scaling.csv")
patch_manifest_file    <- file.path(patch_dir, "patch_manifest.rds")

if (identical(final_run_id, "latest")) {
  run_dirs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  run_dirs <- run_dirs[grepl("^final_", run_dirs)]
  if (length(run_dirs) == 0) stop("No final model runs found in: ", final_model_base)
  final_run_id <- sort(run_dirs, decreasing = TRUE)[1]
  message("final_run_id resolved to: ", final_run_id)
}

final_run_dir <- file.path(final_model_base, final_run_id)

if (identical(config_id, "auto")) {
  tmp_summary_path <- file.path(final_run_dir, "comparison", "final_run_summary.rds")
  if (!file.exists(tmp_summary_path))
    stop("Nao foi possivel resolver config_id='auto': ", tmp_summary_path)
  config_id <- readRDS(tmp_summary_path)$selected_cfgs$config_id[1]
  message("config_id resolved to: ", config_id)
}

model_dir    <- file.path(final_run_dir, config_id, "models")
summary_file <- file.path(final_run_dir, "comparison", "final_run_summary.rds")

output_dir        <- file.path(project_root, "outputs", "spatial_prediction",
                               "soc_stock_modeling", target_label, config_id)
output_raster_dir <- file.path(output_dir, "raster")
output_log_dir    <- file.path(output_dir, "log")

create_output_dirs(c(output_dir, output_raster_dir, output_log_dir))

# ── Validate inputs ────────────────────────────────────────────────────────────

for (f in c(raster_table_file, predictor_scaling_file, summary_file)) {
  if (!file.exists(f)) stop("Required input not found: ", f)
}
if (!dir.exists(model_dir)) stop("Model directory not found: ", model_dir)

# ── Predictor scaling ─────────────────────────────────────────────────────────

predictor_scaling <- readr::read_csv2(predictor_scaling_file, show_col_types = FALSE)

temperature_min_valid_celsius <- -100

apply_predictor_scaling <- function(mat, pred_names, scale_method,
                                    scale_center, scale_factor,
                                    temp_threshold) {
  for (i in seq_len(ncol(mat))) {
    x <- mat[, i]
    if (grepl("surface_temperature_celsius$", pred_names[i])) {
      x[!is.na(x) & is.finite(x) & x <= temp_threshold] <- NA_real_
    }
    if (scale_method[i] == "zscore_train") {
      x <- (x - scale_center[i]) / scale_factor[i]
    } else if (scale_method[i] == "percentage_0_100_to_0_1") {
      # Clamp into [0, 100] instead of discarding as NA: these are continuous
      # interpolated surfaces (PNV classes, clay mineralogy) that legitimately
      # overshoot slightly past 0/100 near sharp spatial transitions. Treating
      # that as missing data amplifies into large gaps once the CNN's
      # full-window validity rule invalidates the whole patch around each
      # discarded pixel. Genuine NA/Inf pass through untouched.
      finite_idx <- !is.na(x) & is.finite(x)
      x[finite_idx] <- pmin(pmax(x[finite_idx], 0), 100)
      x <- x / 100
    }
    mat[, i] <- x
  }
  mat
}

# ── Model config ──────────────────────────────────────────────────────────────

final_summary <- readRDS(summary_file)

if (identical(config_id, "auto")) {
  if (nrow(final_summary$all_seed_results) > 0) {
    auto_rank <- final_summary$all_seed_results %>%
      dplyr::group_by(config_id) %>%
      dplyr::summarise(mean_ccc = mean(ccc, na.rm = TRUE), .groups = "drop") %>%
      dplyr::arrange(dplyr::desc(mean_ccc))
    config_id <- auto_rank$config_id[1]
    message("config_id resolved to: ", config_id,
            sprintf(" (mean test CCC %.4f)", auto_rank$mean_ccc[1]))
  } else {
    stop("config_id = 'auto' but final_run_summary$all_seed_results is empty.")
  }
}

if (!config_id %in% final_summary$selected_cfgs$config_id) {
  stop("config_id '", config_id, "' not in final run summary.")
}

cfg          <- dplyr::filter(final_summary$selected_cfgs, config_id == !!config_id)
window_sizes <- cfg$window_sizes[[1]]
n_branches   <- length(window_sizes)
if (!n_branches %in% c(1L, 2L)) stop("cfg ", config_id, " has ", n_branches, " windows.")
half_w_max   <- (max(window_sizes) - 1L) %/% 2L

message("\nConfig ", config_id,
        " | window(s) ", paste(window_sizes, collapse = "x"), " (", n_branches, "-branch)",
        " | conv ", paste(cfg$conv_channels[[1]], collapse = "_"),
        " | embed ", cfg$embedding_dim,
        " | gate ", cfg$gate_type)

# ── Raster table ──────────────────────────────────────────────────────────────

raster_table <- readr::read_csv2(raster_table_file, show_col_types = FALSE)
if (!all(c("raster_file", "predictor") %in% names(raster_table))) {
  stop("raster_table_used.csv must contain 'raster_file' and 'predictor'.")
}

predictor_cols <- raster_table$predictor
n_channels     <- length(predictor_cols)
raster_files   <- raster_table$raster_file

missing_rasters <- raster_files[!file.exists(raster_files)]
if (length(missing_rasters) > 0) {
  print(missing_rasters)
  stop("Some predictor raster files no longer exist.")
}

predictor_scaling <- predictor_scaling %>%
  dplyr::filter(predictor %in% predictor_cols) %>%
  dplyr::arrange(match(predictor, predictor_cols))

if (!identical(predictor_scaling$predictor, predictor_cols)) {
  stop("predictor_scaling.csv channel order does not match raster_table_used.csv.")
}

scale_method <- predictor_scaling$scaling_method
scale_center <- predictor_scaling$train_mean
scale_factor <- predictor_scaling$train_sd

message("Predictor scaling loaded and aligned (", n_channels, " channels).")

if (file.exists(patch_manifest_file)) {
  manifest <- readRDS(patch_manifest_file)
  manifest_predictors <- strsplit(manifest$predictor_cols_final, ";")[[1]]
  if (!identical(as.character(manifest_predictors), as.character(predictor_cols))) {
    stop("Predictor order mismatch vs patch manifest.")
  }
  message("Channel alignment verified against patch manifest.")
} else {
  message("WARNING: patch_manifest.rds not found.")
}

# ── Open raster stack ─────────────────────────────────────────────────────────

rast_stack <- terra::rast(raster_files)
names(rast_stack) <- predictor_cols

geom_ok <- purrr::map_lgl(
  seq_len(terra::nlyr(rast_stack)),
  ~ terra::compareGeom(rast_stack[[1]], rast_stack[[.x]], stopOnError = FALSE)
)
if (!all(geom_ok)) {
  print(tibble::tibble(predictor = predictor_cols, geometry_ok = geom_ok) %>%
          dplyr::filter(!geometry_ok))
  stop("Some predictor rasters do not share the same geometry.")
}

raster_template <- rast_stack[[1]]
r_nrow <- terra::nrow(rast_stack)
r_ncol <- terra::ncol(rast_stack)
n_cell <- terra::ncell(rast_stack)

message("Raster grid: ", r_nrow, " rows x ", r_ncol, " cols x ", n_channels, " layers")

# ── Partição 2D ───────────────────────────────────────────────────────────────
# Cada worker cobre um retângulo [my_row_start:my_row_end, my_col_start:my_col_end].
# A LEITURA inclui margens de half_w_max em todas as direções (para os patches
# dos pixels de borda). Apenas o retângulo interno é ESCRITO no tile de saída.

row_bounds   <- floor(seq(1, r_nrow + 1, length.out = n_row_shards + 1L))
my_row_start <- row_bounds[row_shard_id]
my_row_end   <- row_bounds[row_shard_id + 1L] - 1L

col_bounds   <- floor(seq(1, r_ncol + 1, length.out = n_col_shards + 1L))
my_col_start <- col_bounds[col_shard_id]
my_col_end   <- col_bounds[col_shard_id + 1L] - 1L

tile_nrow <- my_row_end   - my_row_start   + 1L
tile_ncol <- my_col_end   - my_col_start   + 1L
tile_ncell <- tile_nrow * tile_ncol

part_suffix <- if (is_partitioned) {
  sprintf("_r%03dof%03d_c%03dof%03d", row_shard_id, n_row_shards, col_shard_id, n_col_shards)
} else {
  ""
}

if (is_partitioned) {
  message(sprintf(
    "Shard [%d/%d row, %d/%d col]: rows %d-%d (%s), cols %d-%d (%s)",
    row_shard_id, n_row_shards, col_shard_id, n_col_shards,
    my_row_start, my_row_end, format(tile_nrow, big.mark = ","),
    my_col_start, my_col_end, format(tile_ncol, big.mark = ",")))
}

# Colunas lidas pela strip: tile + margem (clamped às bordas do raster)
read_col_start <- max(1L,      my_col_start - half_w_max)
read_col_end   <- min(r_ncol,  my_col_end   + half_w_max)
strip_ncol     <- read_col_end - read_col_start + 1L

# ── output_block_rows do budget de RAM ────────────────────────────────────────
# Com 2D: strip_ncol << r_ncol, entao bytes_per_strip_row e muito menor ->
# output_block_rows maior -> menos overhead de I/O vs. um esquema 1D (linha
# inteira por strip).

bytes_per_strip_row <- as.numeric(strip_ncol) * n_channels * 8

if (!is.null(max_strip_ram_gb)) {
  output_block_rows <- max(1L, as.integer(
    floor(max_strip_ram_gb * 1e9 / bytes_per_strip_row) - 2L * half_w_max
  ))
} else {
  # sem auto-cálculo: usa o valor fixo acima
}
# Aplica o cap: independente do strip budget, nunca excede max_output_block_rows.
# Impede que o RSS de predição (patch arrays × n_valid/bloco) exploda em
# shards densos quando o strip seria barato o suficiente para blocos grandes.
output_block_rows <- min(output_block_rows, max_output_block_rows)

message(sprintf(
  "output_block_rows = %d (cap=%d | strip budget %.1f GB | strip_ncol %d vs r_ncol %d -> %.2f GB/strip)",
  output_block_rows, max_output_block_rows,
  if (!is.null(max_strip_ram_gb)) max_strip_ram_gb else NA_real_,
  strip_ncol, r_ncol,
  (output_block_rows + 2L * half_w_max) * bytes_per_strip_row / 1e9
))

strip_gb <- (output_block_rows + 2L * half_w_max) * bytes_per_strip_row / 1e9
if (strip_gb > 8) {
  message(sprintf("  WARNING: strip ~%.1f GB. Considere reduzir max_strip_ram_gb.", strip_gb))
}

# ── Seed models ───────────────────────────────────────────────────────────────

model_files <- file.path(model_dir, sprintf("seed%04d_best.pt", seeds))
have <- file.exists(model_files)
if (!any(have)) stop("No seed model files found in: ", model_dir)
if (!all(have)) {
  message("WARNING: missing seeds, using only ", sum(have), " available.")
}
seeds       <- seeds[have]
model_files <- model_files[have]
n_seeds     <- length(seeds)

message("\nLoading ", n_seeds, " seed model(s) for ", config_id, "...")
models <- purrr::map(seq_len(n_seeds), function(i) {
  m  <- build_cnn_from_config(cfg, n_channels)
  st <- torch::torch_load(model_files[i])
  m$load_state_dict(st)
  m$to(device = device)
  m$eval()
  m
})

# ── Helpers ───────────────────────────────────────────────────────────────────

# build_patches_multi recebe center_col LOCAL (1-based dentro da strip de leitura),
# nao a coluna global. strip_values tem strip_ncol colunas (nao r_ncol).
#
# A versao anterior indexava strip_values[cells, ] DUAS vezes por branch de
# janela: uma vez (para todo o chunk) so para checar is.finite, e de novo
# (para o subconjunto valido) para montar o array final. Essa reindexacao era
# o maior custo de RAM/CPU do chunk. Agora a indexacao acontece uma unica vez
# por branch e o resultado (vals_full) e reaproveitado tanto para a checagem
# de validade quanto para montar o array final (so reshape/slice, sem copiar
# de strip_values de novo).
build_patches_multi <- function(center_row, center_col_local, strip_values,
                                read_row_start, strip_ncol_arg, n_ch, window_sizes) {
  n         <- length(center_row)
  row_local <- center_row - read_row_start + 1L

  per_win      <- vector("list", length(window_sizes))
  valid_common <- rep(TRUE, n)

  for (k in seq_along(window_sizes)) {
    w    <- window_sizes[k]
    half <- (w - 1L) %/% 2L
    offsets <- expand.grid(dr = (-half):half, dc = (-half):half)
    n_pos   <- nrow(offsets)

    cell_mat <- matrix(0L, nrow = n, ncol = n_pos)
    for (j in seq_len(n_pos)) {
      cell_mat[, j] <- (row_local + offsets$dr[j] - 1L) * strip_ncol_arg +
                       (center_col_local + offsets$dc[j])
    }

    vals_full <- strip_values[as.vector(t(cell_mat)), , drop = FALSE]
    row_finite <- rowSums(!is.finite(vals_full)) == 0
    valid_w      <- colSums(matrix(row_finite, nrow = n_pos, ncol = n)) == n_pos
    valid_common <- valid_common & valid_w

    # Reshape para (n_pos, n, n_ch) sem reindexar strip_values -- so muda o
    # atributo dim (barato). Ordem de as.vector(t(cell_mat)) ja e point-major
    # em blocos de n_pos, entao bate exatamente com esse layout.
    dim(vals_full) <- c(n_pos, n, n_ch)
    per_win[[k]] <- list(w = w, n_pos = n_pos, vals_full = vals_full)
  }

  arrays <- vector("list", length(window_sizes))
  if (!any(valid_common)) {
    for (k in seq_along(window_sizes)) {
      arrays[[k]] <- array(0, dim = c(0L, n_ch, per_win[[k]]$w, per_win[[k]]$w))
    }
    return(list(arrays = arrays, valid = valid_common))
  }

  valid_pos <- which(valid_common)
  for (k in seq_along(window_sizes)) {
    w     <- per_win[[k]]$w
    n_pos <- per_win[[k]]$n_pos
    step1 <- per_win[[k]]$vals_full[, valid_pos, , drop = FALSE]
    step2 <- aperm(step1, c(2L, 3L, 1L))
    arrays[[k]] <- array(step2, dim = c(length(valid_pos), n_ch, w, w))
  }

  list(arrays = arrays, valid = valid_common)
}

predict_one_model <- function(model, arr_list, device, batch_size) {
  n <- dim(arr_list[[1]])[1]
  if (n == 0L) return(numeric(0))
  out <- numeric(n)
  idx_groups <- split(seq_len(n), ceiling(seq_len(n) / batch_size))
  torch::with_no_grad({
    for (idx in idx_groups) {
      tensors <- lapply(arr_list, function(a)
        torch::torch_tensor(a[idx, , , , drop = FALSE],
                            dtype = torch::torch_float(), device = device))
      p <- do.call(model, tensors)
      out[idx] <- as.numeric(p$squeeze(2L)$to(device = "cpu"))
    }
  })
  out
}

# ── compute_block ─────────────────────────────────────────────────────────────
# Prediz os pixels do retângulo do tile para um bloco de linhas.
# Diferenças vs 05:
#   - center_cols limitados a my_col_start:my_col_end
#   - terra::values lê somente read_col_start:read_col_end (strip_ncol colunas)
#   - center_col passado para build_patches_multi e quick_ok e LOCAL (strip)
#   - cells_valid sao GLOBAIS (row-1)*r_ncol+col, convertidos para tile-local
#     no loop principal (fora desta funcao)

compute_block <- function(b_start) {
  out_nrows   <- min(output_block_rows, my_row_end - b_start + 1L)
  out_row_end <- b_start + out_nrows - 1L

  center_rows <- intersect(b_start:out_row_end,
                           (half_w_max + 1L):(r_nrow - half_w_max))

  # Colunas do tile clampadas pelas margens globais
  cc_start <- max(my_col_start, half_w_max + 1L)
  cc_end   <- min(my_col_end,   r_ncol - half_w_max)

  empty <- list(out_nrows = out_nrows, n_req = 0L, n_val = 0L,
                cells_all = integer(0), valid = logical(0),
                cells_valid = integer(0),
                median = numeric(0), mean = numeric(0), sd = numeric(0),
                mad = numeric(0), min = numeric(0), max = numeric(0),
                t_read = 0, t_scale = 0, t_predict = 0, n_quick_ok = 0L)
  if (length(center_rows) == 0L || cc_start > cc_end) return(empty)

  read_row_start <- min(center_rows) - half_w_max
  read_row_end   <- max(center_rows) + half_w_max
  read_nrows     <- read_row_end - read_row_start + 1L

  .t_read_start <- Sys.time()
  # Leitura da fatia 2D da strip (so as colunas deste tile + margem)
  strip_values <- terra::values(rast_stack,
                                row   = read_row_start, nrows = read_nrows,
                                col   = read_col_start, ncols = strip_ncol,
                                mat   = TRUE)
  t_read <- as.numeric(Sys.time() - .t_read_start, units = "secs")

  .t_scale_start <- Sys.time()
  strip_values <- apply_predictor_scaling(strip_values, predictor_cols,
                                          scale_method, scale_center, scale_factor,
                                          temperature_min_valid_celsius)
  t_scale <- as.numeric(Sys.time() - .t_scale_start, units = "secs")

  .t_predict_start <- Sys.time()
  center_cols <- cc_start:cc_end
  grid  <- expand.grid(center_row = center_rows, center_col = center_cols)
  n_req <- nrow(grid)

  # Índices GLOBAIS (para cells_all / cells_valid retornados ao caller)
  cells_all <- (grid$center_row - 1L) * r_ncol + grid$center_col
  valid_all <- logical(n_req)

  med <- mean_ <- sd_ <- mad_ <- min_ <- max_ <- numeric(n_req)
  cells_valid <- integer(n_req)
  ptr <- 0L
  n_quick_ok <- 0L

  chunk_list <- split(seq_len(n_req), ceiling(seq_len(n_req) / max(batch_size, 1L)))
  n_chunks <- length(chunk_list)
  .heartbeat_last   <- Sys.time()
  heartbeat_every_s <- 120

  for (chunk_idx in seq_along(chunk_list)) {
    ci <- chunk_list[[chunk_idx]]

    if (as.numeric(Sys.time() - .heartbeat_last, units = "secs") >= heartbeat_every_s) {
      message(sprintf(
        "    [heartbeat] chunk %s/%s | quick_ok so far %s | valid so far %s | %.1f min into this block",
        format(chunk_idx, big.mark = ","), format(n_chunks, big.mark = ","),
        format(n_quick_ok, big.mark = ","), format(ptr, big.mark = ","),
        as.numeric(Sys.time() - .t_predict_start, units = "mins")))
      .heartbeat_last <- Sys.time()
    }

    cr <- grid$center_row[ci]
    cc <- grid$center_col[ci]   # colunas GLOBAIS

    # quick_ok: índices locais dentro da strip (strip_ncol colunas)
    row_local_c <- cr - read_row_start + 1L
    col_local_c <- cc - read_col_start + 1L   # coluna local na strip
    center_idx  <- (row_local_c - 1L) * strip_ncol + col_local_c
    quick_ok    <- rowSums(!is.finite(strip_values[center_idx, , drop = FALSE])) == 0
    n_quick_ok  <- n_quick_ok + sum(quick_ok)

    if (!any(quick_ok)) next

    # build_patches_multi recebe coluna LOCAL na strip
    cc_local <- cc - read_col_start + 1L
    pb <- build_patches_multi(cr[quick_ok], cc_local[quick_ok], strip_values,
                              read_row_start, strip_ncol, n_channels, window_sizes)
    valid_sub      <- logical(length(ci))
    valid_sub[quick_ok] <- pb$valid
    valid_all[ci]  <- valid_sub
    if (dim(pb$arrays[[1]])[1] == 0L) next

    preds_native <- matrix(NA_real_, nrow = dim(pb$arrays[[1]])[1], ncol = n_seeds)
    for (s in seq_len(n_seeds)) {
      preds_native[, s] <- pmax(expm1(predict_one_model(models[[s]], pb$arrays,
                                                         device, batch_size)), 0)
    }

    # cells_valid: índices GLOBAIS
    cv  <- ((cr[quick_ok] - 1L) * r_ncol + cc[quick_ok])[pb$valid]
    nv  <- length(cv)
    rng <- (ptr + 1L):(ptr + nv)

    if (n_seeds >= 2L) {
      med[rng]   <- matrixStats::rowMedians(preds_native)
      mean_[rng] <- rowMeans(preds_native)
      sd_[rng]   <- matrixStats::rowSds(preds_native)
      mad_[rng]  <- matrixStats::rowMads(preds_native)
      min_[rng]  <- matrixStats::rowMins(preds_native)
      max_[rng]  <- matrixStats::rowMaxs(preds_native)
    } else {
      v <- preds_native[, 1]
      med[rng] <- v; mean_[rng] <- v; sd_[rng] <- 0
      mad_[rng] <- 0; min_[rng] <- v; max_[rng] <- v
    }
    cells_valid[rng] <- cv
    ptr <- ptr + nv
  }

  t_predict <- as.numeric(Sys.time() - .t_predict_start, units = "secs")
  rm(strip_values); gc()
  keep <- seq_len(ptr)
  list(out_nrows = out_nrows, n_req = n_req, n_val = ptr,
       cells_all = cells_all, valid = valid_all,
       cells_valid = cells_valid[keep],
       median = med[keep], mean = mean_[keep], sd = sd_[keep],
       mad = mad_[keep], min = min_[keep], max = max_[keep],
       t_read = t_read, t_scale = t_scale, t_predict = t_predict,
       n_quick_ok = n_quick_ok)
}

# ── Template do tile e writers ─────────────────────────────────────────────────

gdal_opts <- function(datatype) {
  predictor <- if (grepl("^INT|^UINT|^BYTE", datatype)) 2L else 3L
  c("COMPRESS=DEFLATE", paste0("PREDICTOR=", predictor),
    "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512")
}

if (is_partitioned) {
  full_ext    <- as.vector(terra::ext(raster_template))
  xres_full   <- terra::xres(raster_template)
  yres_full   <- terra::yres(raster_template)
  worker_ymax <- full_ext[["ymax"]] - (my_row_start - 1L) * yres_full
  worker_ymin <- full_ext[["ymax"]] - my_row_end * yres_full
  worker_xmin <- full_ext[["xmin"]] + (my_col_start - 1L) * xres_full
  worker_xmax <- full_ext[["xmin"]] + my_col_end * xres_full
  worker_ext  <- terra::ext(worker_xmin, worker_xmax, worker_ymin, worker_ymax)
  worker_template <- terra::crop(raster_template, worker_ext, snap = "near")
} else {
  worker_template <- raster_template
}

output_raster_dir_w <- if (is_partitioned)
  file.path(output_raster_dir, "parts_2d") else output_raster_dir
create_output_dirs(output_raster_dir_w)

open_writer <- function(band_name, file_suffix, datatype = "FLT4S") {
  f <- file.path(output_raster_dir_w,
                 paste0(target_label, "_", config_id, "_", file_suffix, part_suffix, ".tif"))
  if (file.exists(f)) file.remove(f)
  r <- terra::rast(worker_template)
  names(r) <- band_name
  terra::writeStart(r, f, overwrite = TRUE, datatype = datatype,
                    gdal = gdal_opts(datatype))
  list(rast = r, file = f)
}

w_median <- open_writer("soc_pred_median_ton_ha", "ensemble_median_ton_ha")
w_mean   <- open_writer("soc_pred_mean_ton_ha",   "ensemble_mean_ton_ha")
w_sd     <- open_writer("soc_uncert_sd_ton_ha",   "ensemble_sd_ton_ha")
w_mad    <- open_writer("soc_uncert_mad_ton_ha",  "ensemble_mad_ton_ha")
w_min    <- open_writer("soc_pred_min_ton_ha",    "ensemble_min_ton_ha")
w_max    <- open_writer("soc_pred_max_ton_ha",    "ensemble_max_ton_ha")
w_mask   <- open_writer("valid_patch_mask",       "valid_mask", datatype = "INT1U")
writers  <- list(w_median, w_mean, w_sd, w_mad, w_min, w_max, w_mask)

abort_and_cleanup <- function(msg) {
  for (w in writers) try(terra::writeStop(w$rast), silent = TRUE)
  files <- vapply(writers, function(w) w$file, character(1))
  suppressWarnings(file.remove(files[file.exists(files)]))
  stop(msg, call. = FALSE)
}

# ── Loop principal ─────────────────────────────────────────────────────────────

diag_log_every <- 1L
.proc_handle   <- ps::ps_handle()

block_starts <- seq(my_row_start, my_row_end, by = output_block_rows)
block_log    <- vector("list", length(block_starts))

run_n <- 0; run_sum <- 0; run_min <- Inf; run_max <- -Inf
run_nonfinite <- 0; probe_done <- FALSE

t0 <- Sys.time()

for (b in seq_along(block_starts)) {
  bs <- block_starts[b]
  cb <- compute_block(bs)

  # blk_len: linhas do bloco x colunas do TILE (nao do raster inteiro)
  blk_len    <- cb$out_nrows * tile_ncol
  blk_median <- rep(NA_real_, blk_len)
  blk_mean   <- rep(NA_real_, blk_len)
  blk_sd     <- rep(NA_real_, blk_len)
  blk_mad    <- rep(NA_real_, blk_len)
  blk_min    <- rep(NA_real_, blk_len)
  blk_max    <- rep(NA_real_, blk_len)
  blk_mask   <- rep(0L, blk_len)

  if (cb$n_req > 0L) {
    # Converte índices GLOBAIS (row-1)*r_ncol+col para índices no bloco do tile
    g2tile <- function(cells) {
      row_g       <- (cells - 1L) %/% r_ncol + 1L
      col_g       <- (cells - 1L) %% r_ncol + 1L
      row_in_blk  <- row_g - bs + 1L
      col_in_tile <- col_g - my_col_start + 1L
      (row_in_blk - 1L) * tile_ncol + col_in_tile
    }

    blk_mask[g2tile(cb$cells_all)] <- as.integer(cb$valid)

    if (cb$n_val > 0L) {
      tv <- g2tile(cb$cells_valid)
      blk_median[tv] <- cb$median
      blk_mean[tv]   <- cb$mean
      blk_sd[tv]     <- cb$sd
      blk_mad[tv]    <- cb$mad
      blk_min[tv]    <- cb$min
      blk_max[tv]    <- cb$max

      run_n         <- run_n + cb$n_val
      run_sum       <- run_sum + sum(cb$median[is.finite(cb$median)])
      run_min       <- min(run_min, min(cb$median, na.rm = TRUE))
      run_max       <- max(run_max, max(cb$median, na.rm = TRUE))
      run_nonfinite <- run_nonfinite + sum(!is.finite(cb$median))

      if (!probe_done) {
        probe_mean <- mean(cb$median[is.finite(cb$median)])
        if (!is.finite(probe_mean) || probe_mean > plausible_hard_max) {
          abort_and_cleanup(sprintf(
            "Fail-fast probe: first-block mean = %.1f %s (> %g).",
            probe_mean, target_unit, plausible_hard_max))
        }
        probe_done <- TRUE
      }
    }
  }

  # Linha local no tile do worker (1-based)
  bs_local <- bs - my_row_start + 1L

  terra::writeValues(w_median$rast, blk_median, bs_local, cb$out_nrows)
  terra::writeValues(w_mean$rast,   blk_mean,   bs_local, cb$out_nrows)
  terra::writeValues(w_sd$rast,     blk_sd,     bs_local, cb$out_nrows)
  terra::writeValues(w_mad$rast,    blk_mad,    bs_local, cb$out_nrows)
  terra::writeValues(w_min$rast,    blk_min,    bs_local, cb$out_nrows)
  terra::writeValues(w_max$rast,    blk_max,    bs_local, cb$out_nrows)
  terra::writeValues(w_mask$rast,   blk_mask,   bs_local, cb$out_nrows)

  rss_mb <- ps::ps_memory_info(.proc_handle)[["rss"]] / 1e6

  block_log[[b]] <- tibble::tibble(
    block = b, row_start = bs, row_end = bs + cb$out_nrows - 1L,
    col_start = my_col_start, col_end = my_col_end,
    n_centers = cb$n_req, n_quick_ok = cb$n_quick_ok, n_valid = cb$n_val,
    n_invalid = cb$n_req - cb$n_val,
    t_read = cb$t_read, t_scale = cb$t_scale, t_predict = cb$t_predict,
    rss_mb = rss_mb
  )

  if (b %% diag_log_every == 0L || b == 1L || b == length(block_starts)) {
    message(sprintf(
      "Block %d/%d | rows %d-%d cols %d-%d | centers %s quick_ok %s valid %s | read %.1fs scale %.1fs predict %.1fs | RSS %.0f MB",
      b, length(block_starts), bs, bs + cb$out_nrows - 1L,
      my_col_start, my_col_end,
      format(cb$n_req, big.mark = ","), format(cb$n_quick_ok, big.mark = ","),
      format(cb$n_val, big.mark = ","), cb$t_read, cb$t_scale, cb$t_predict, rss_mb))
  }

  if (b == 1L || b %% 25L == 0L || b == length(block_starts)) {
    el <- Sys.time() - t0
    message(sprintf("  └─ cumulative elapsed: %.2f %s", as.numeric(el), units(el)))
    safe_write_csv2(dplyr::bind_rows(block_log[seq_len(b)]),
                    file.path(output_log_dir,
                              paste0("prediction_block_summary_partial", part_suffix, ".csv")))
  }
}

for (w in writers) terra::writeStop(w$rast)
rm(models); gc()

total_time    <- Sys.time() - t0
n_valid_total <- run_n

# ── Sanity checks ──────────────────────────────────────────────────────────────

global_mean_med <- if (run_n > 0) run_sum / run_n else NA_real_
global_max      <- run_max

message("\n── Sanity checks ─────────────────────────────────────────────────")
message(sprintf("  Valid pixels         : %s (%.2f%% do tile)",
                format(n_valid_total, big.mark = ","), 100 * n_valid_total / tile_ncell))
message(sprintf("  Median map mean      : %.2f %s", global_mean_med, target_unit))
message(sprintf("  Median map range     : %.2f – %.2f %s", run_min, run_max, target_unit))

f_median <- w_median$file; f_mean <- w_mean$file; f_sd <- w_sd$file
f_mad    <- w_mad$file;    f_min  <- w_min$file;  f_max <- w_max$file
f_mask   <- w_mask$file
written_files <- c(f_median, f_mean, f_sd, f_mad, f_min, f_max, f_mask)

sanity_ok <- TRUE
if (!is_partitioned && n_valid_total == 0L) {
  sanity_ok <- FALSE
  message("  [FAIL] No valid pixels.")
}
if (run_nonfinite > 0L) {
  sanity_ok <- FALSE
  message(sprintf("  [FAIL] %d non-finite values in median map.", run_nonfinite))
}
if (!is_partitioned && is.finite(global_mean_med) &&
    (global_mean_med < plausible_median_range[1] ||
     global_mean_med > plausible_median_range[2])) {
  sanity_ok <- FALSE
  message(sprintf("  [FAIL] Mean %.2f fora do range plausivel.", global_mean_med))
}
if (is.finite(global_max) && global_max > plausible_hard_max) {
  message(sprintf("  [WARN] Max %.0f > %g %s.", global_max, plausible_hard_max, target_unit))
}
if (is_partitioned && n_valid_total == 0L) {
  message("  [INFO] Tile sem pixels validos (provavelmente oceano) — nao e erro.")
}
if (!sanity_ok) {
  suppressWarnings(file.remove(written_files[file.exists(written_files)]))
  stop("Sanity checks failed. Rasters deletados.")
}
message("  All hard checks passed.")

global_med <- tryCatch({
  samp <- terra::spatSample(terra::rast(f_median), size = 2e5,
                            method = "regular", na.rm = TRUE, values = TRUE)
  stats::median(samp[[1]], na.rm = TRUE)
}, error = function(e) NA_real_)
message(sprintf("  Global median (sampled): %.2f %s", global_med, target_unit))

# ── Manifests ─────────────────────────────────────────────────────────────────

block_summary <- dplyr::bind_rows(block_log)
safe_write_csv2(block_summary,
                file.path(output_log_dir, paste0("prediction_block_summary", part_suffix, ".csv")))

prediction_config <- tibble::tibble(
  target_label = target_label, target_unit = target_unit,
  config_id = config_id, final_run_id = final_run_id,
  window_sizes = paste(window_sizes, collapse = "x"), n_branches = n_branches,
  n_channels = n_channels,
  n_seeds = n_seeds, seeds = paste(seeds, collapse = ";"),
  ensemble_center = ensemble_center,
  input_scaling = "predictor_scaling.csv_zscore_percentage_dummy",
  back_transform = "expm1_then_clip0",
  aggregation_space = "per_seed_expm1_then_aggregate_across_seeds",
  output_block_rows = output_block_rows, batch_size = batch_size,
  r_nrow = r_nrow, r_ncol = r_ncol, n_cell = n_cell,
  tile_nrow = tile_nrow, tile_ncol = tile_ncol, tile_ncell = tile_ncell,
  row_shard_id = row_shard_id, col_shard_id = col_shard_id,
  n_row_shards = n_row_shards, n_col_shards = n_col_shards,
  strip_ncol = strip_ncol,
  n_valid = n_valid_total, valid_fraction = n_valid_total / tile_ncell,
  global_median_ton_ha = global_med, global_max_ton_ha = global_max,
  runtime_min = as.numeric(total_time, units = "mins"),
  device = device$type, predicted_at = as.character(Sys.time())
)
safe_write_csv2(prediction_config,
                file.path(output_log_dir, paste0("prediction_config", part_suffix, ".csv")))

raster_summary <- purrr::map_dfr(
  list(median = f_median, mean = f_mean, sd = f_sd, mad = f_mad,
       min = f_min, max = f_max, mask = f_mask),
  function(f) {
    r <- terra::rast(f)
    tibble::tibble(
      file = f, band = names(r),
      gmin  = terra::global(r, "min",  na.rm = TRUE)[1, 1],
      gmean = terra::global(r, "mean", na.rm = TRUE)[1, 1],
      gmax  = terra::global(r, "max",  na.rm = TRUE)[1, 1]
    )
  },
  .id = "layer"
)
safe_write_csv2(raster_summary,
                file.path(output_log_dir, paste0("prediction_raster_summary", part_suffix, ".csv")))

if (row_shard_id == 1L && col_shard_id == 1L) {
  safe_write_csv2(
    tibble::tibble(order = seq_len(n_channels), predictor = predictor_cols),
    file.path(output_log_dir, "predictor_order_used.csv")
  )
}

# ── Report ─────────────────────────────────────────────────────────────────────

message("\n── Spatial prediction complete (2D tile) ────────────────────────")
message(sprintf("  Shard            : row %d/%d, col %d/%d",
                row_shard_id, n_row_shards, col_shard_id, n_col_shards))
message(sprintf("  Config / seeds   : %s / %d", config_id, n_seeds))
message(sprintf("  Valid pixels     : %s", format(n_valid_total, big.mark = ",")))
message(sprintf("  Runtime          : %.2f %s",
                as.numeric(total_time), units(total_time)))
message(sprintf("  strip_ncol       : %d (vs r_ncol %d -> RAM %.1fx menor)",
                strip_ncol, r_ncol, r_ncol / strip_ncol))

rs_get <- function(layer, col) raster_summary[[col]][raster_summary$layer == layer]
message("\n  Central tendency (", target_unit, "):")
message(sprintf("    median map -> median ~%.2f | mean %.2f | max %.2f",
                global_med, rs_get("median", "gmean"), rs_get("median", "gmax")))
message(sprintf("    mean map   ->            mean %.2f | max %.2f",
                rs_get("mean", "gmean"), rs_get("mean", "gmax")))

message("\n  Rasters: ", output_raster_dir_w)
message("  Logs:    ", output_log_dir)
