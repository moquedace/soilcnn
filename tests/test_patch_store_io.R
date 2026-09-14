# Unit test: gravar e reler uma janela devolve exatamente o mesmo dado
#
# Este teste existe por causa de um prejuízo concreto. A versão anterior do
# store gravava tensores float32 com torch_save() para economizar disco.
# torch_save() no R torch 0.17.0 quebra acima de 2^31 bytes -- e não quebra
# de forma honesta: num caso produziu um arquivo do TAMANHO CERTO cujo final
# eram 4,3 GB de zeros. Passou por uma checagem de "nenhum valor não-finito",
# porque zero é finito. Custou ~10 h de extração e duas sessões mortas.
#
# A lição virou regra: nenhuma verificação de escrita pode se contentar com o
# tamanho do arquivo. Tem que ler de volta e comparar CONTEÚDO.
#
# Verifica:
#   1. round-trip é exato para cada tipo de janela
#   2. o tamanho em disco bate com o previsto por patch_window_bytes()
#   3. load_patch_window() devolve float32 com o shape certo
#   4. as asserções de shape pegam um arquivo de outro conjunto de pontos
#   5. safe_torch_save() RECUSA um tensor acima de 2^31 em vez de corromper
#   6. um arquivo truncado é detectado
#
# Rode: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_patch_store_io.R")

suppressMessages({
  library(torch)
  library(tibble)
})

root <- (function() {
  cand <- character(0)
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) cand <- c(cand, dirname(normalizePath(f[1], mustWork = FALSE)))
  for (i in seq_len(sys.nframe())) {
    of <- sys.frame(i)$ofile
    if (!is.null(of) && is.character(of)) {
      cand <- c(cand, dirname(normalizePath(of, mustWork = FALSE)))
    }
  }
  cand <- c(cand, getwd())
  for (d in cand) {
    for (up in c(".", "..")) {
      r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
      if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
    }
  }
  stop("Project root not found.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
source(file.path(root, "R", "utils.R"))
source(file.path(root, "R", "dataset.R"))

set.seed(3)
results <- logical(0)

store <- file.path(tempdir(), "test_store")
unlink(store, recursive = TRUE)
dir.create(store, recursive = TRUE)

n <- 40L; ch <- 7L

# ── 1-3: round-trip por janela ────────────────────────────────────────────────

for (w in c(3L, 9L, 15L)) {
  arr <- array(rnorm(n * ch * w * w), dim = c(n, ch, w, w))

  res <- save_patch_window(arr, store, w)
  results[sprintf("w%02d_size_ok", w)] <- res$ok

  back <- load_patch_window(store, w, expect_points = n, expect_channels = ch)

  # CONTEÚDO, não tamanho. float32 perde precisão do double, então a comparação
  # é contra o mesmo round-trip de precisão -- e tem que bater exatamente.
  esperado <- as.array(torch_tensor(arr, dtype = torch_float()))
  results[sprintf("w%02d_roundtrip_exato", w)] <-
    identical(dim(as.array(back)), dim(esperado)) &&
    max(abs(as.array(back) - esperado)) == 0

  results[sprintf("w%02d_dtype_float32", w)] <-
    identical(as.character(back$dtype), "Float")

  rm(back); gc(verbose = FALSE)
}

# ── 4: shape de outro conjunto de pontos é recusado ──────────────────────────

results["ponto_errado_recusado"] <- inherits(
  try(load_patch_window(store, 3L, expect_points = n + 1L), silent = TRUE),
  "try-error")
results["canal_errado_recusado"] <- inherits(
  try(load_patch_window(store, 3L, expect_channels = ch + 1L), silent = TRUE),
  "try-error")

# ── 5: safe_torch_save recusa acima de 2^31 em vez de corromper ──────────────
# Mede-se o limite, não se confia nele: 2,147,479,648 bytes gravou e
# 2,147,487,648 matou a sessão. A guarda tem que disparar ANTES de tentar.

grande <- torch_empty(floor(2^31 / 4) + 1000L, dtype = torch_float())
results["safe_torch_save_recusa_acima_2_31"] <- inherits(
  try(safe_torch_save(grande, file.path(store, "nao_deve_existir.pt")),
      silent = TRUE), "try-error")
results["safe_torch_save_nao_criou_arquivo"] <-
  !file.exists(file.path(store, "nao_deve_existir.pt"))
rm(grande); gc(verbose = FALSE)

# e um tensor pequeno continua passando normalmente
pequeno <- torch_empty(1000L, dtype = torch_float())
results["safe_torch_save_aceita_pequeno"] <- !inherits(
  try(safe_torch_save(pequeno, file.path(store, "ok.pt")), silent = TRUE),
  "try-error")
rm(pequeno); gc(verbose = FALSE)

# ── 6: arquivo truncado é detectado ──────────────────────────────────────────
# O modo de falha que passou despercebido foi um arquivo do tamanho certo com
# lixo dentro. Aqui o inverso -- tamanho errado -- tem que ser pego na leitura.

f3 <- patch_window_path(store, 3L)
con <- file(f3, "r+b"); truncate(con, 5000L); close(con)
results["arquivo_truncado_detectado"] <- inherits(
  try(load_patch_window(store, 3L), silent = TRUE), "try-error")

unlink(store, recursive = TRUE)

cat(sprintf("  store sintetico     : %d pontos x %d canais, janelas 3/9/15\n", n, ch))
cat(sprintf("  limite do torch_save: %s bytes (2^31)\n",
            format(2^31, big.mark = ",")))
.report(results, "test_patch_store_io")
