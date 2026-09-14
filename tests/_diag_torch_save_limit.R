# Onde exatamente torch_save() quebra?
#
# O diagnostico de escala mostrou: 1,41 GB grava, 2,17 GB mata a sessao.
# 2^31 bytes = 2,147 GB cai exatamente nessa janela, o que sugere estouro de
# inteiro de 32 bits no serializador.
#
# Contra-evidencia que precisa ser explicada: patches_w15.pt tem 5,98 GB, foi
# gravado por torch_save e RELEU perfeitamente (shape certo, zero celulas
# nao-finitas em 1,49 bilhao). Se o limite fosse 2^31, esse arquivo nao
# existiria.
#
# Este teste isola SO o torch_save: os tensores vem de torch_empty(), sem
# nenhum array R por perto, entao nada mais compete por memoria ou atrapalha a
# leitura do resultado. Varre tamanhos em torno de 2^31 e, no fim, tenta um
# tensor GRANDE (na faixa do w15) para testar a contra-evidencia.
#
# Log em disco com flush a cada linha -- a resposta sobrevive ao crash.
#
# Rode: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/_diag_torch_save_limit.R")

suppressMessages(library(torch))

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
log_file <- file.path(project_root, "outputs", "diag_torch_save_limit.log")
dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)

log_linha <- function(...) {
  linha <- paste0(format(Sys.time(), "%H:%M:%S"), "  ", paste0(..., collapse = ""))
  cat(linha, "\n", sep = "", file = log_file, append = TRUE)
  message(linha)
}

cat("", file = log_file)
log_linha("torch ", as.character(utils::packageVersion("torch")))
log_linha("2^31 bytes = ", format(2^31, big.mark = ","),
          "  (", sprintf("%.3f GB", 2^31 / 1e9), ")")
log_linha("2^32 bytes = ", format(2^32, big.mark = ","),
          "  (", sprintf("%.3f GB", 2^32 / 1e9), ")")

tmp <- file.path(tempdir(), "diag_limit")
dir.create(tmp, showWarnings = FALSE, recursive = TRUE)
f <- file.path(tmp, "t.pt")

# ── um tamanho por vez, tensor criado direto pelo torch ──────────────────────
# torch_empty nao passa por nenhum array R: isola o serializador.

testar <- function(n_elem, rotulo) {
  bytes <- n_elem * 4
  log_linha("--- ", rotulo, ": ", format(n_elem, big.mark = ","), " elementos = ",
            format(bytes, big.mark = ","), " bytes (",
            sprintf("%.3f GB", bytes / 1e9), ")",
            if (bytes > 2^31) "   ACIMA de 2^31" else "   abaixo de 2^31")

  x <- torch_empty(n_elem, dtype = torch_float())
  log_linha("    tensor criado, gravando...")

  if (file.exists(f)) file.remove(f)
  torch_save(x, f)

  got <- file.size(f)
  log_linha("    GRAVOU -- ", format(got, big.mark = ","), " bytes",
            if (abs(got - bytes) < 5000) "  [tamanho ok]" else
              sprintf("  [ESPERADO %s]", format(bytes, big.mark = ",")))

  rm(x); invisible(gc(verbose = FALSE))
  file.remove(f)
  invisible(TRUE)
}

# Varredura em torno de 2^31. Cada passo e logado ANTES de tentar, entao a
# ultima linha do log identifica o tamanho que matou a sessao.
n_2_31 <- floor(2^31 / 4)   # elementos que dao exatamente 2^31 bytes

testar(floor(n_2_31 * 0.90), "90% de 2^31")
testar(floor(n_2_31 * 0.99), "99% de 2^31")
testar(n_2_31 - 1000L,       "logo ABAIXO de 2^31")
testar(n_2_31 + 1000L,       "logo ACIMA de 2^31")
testar(floor(n_2_31 * 1.10), "110% de 2^31")

# A contra-evidencia: o w15 tem 1.494.485.325 elementos e gravou. Se chegarmos
# aqui, 2^31 nao e o limite e o w15 deixa de ser contradicao.
log_linha("=== varredura de 2^31 passou inteira ===")
testar(1494485325L, "tamanho do w15 (5,98 GB)")

log_linha("=== TUDO GRAVOU -- torch_save nao tem limite de tamanho aqui ===")
log_linha("Se esta linha aparece, o defeito depende de outra coisa: array R ",
          "vivo junto, estado do heap, ou a forma 4D.")

unlink(tmp, recursive = TRUE)
message("\nLog: ", log_file)
