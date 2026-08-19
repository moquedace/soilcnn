# Rode este script UMA VEZ, interativamente (RStudio ou console R), depois da
# atualizacao do R para 4.6.1 -- os pacotes da instalacao anterior (4.5.3) nao
# sao carregados automaticamente pela nova versao.

options(repos = c(CRAN = "https://cloud.r-project.org"))

pkgs <- c(
  "torch", "terra", "dplyr", "readr", "tibble", "purrr", "stringr",
  "matrixStats", "ps", "processx", "coro", "DescTools", "tidyr",
  "sf", "janitor", "ggplot2"
)

to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install) > 0) {
  message("Instalando: ", paste(to_install, collapse = ", "))
  install.packages(to_install)
} else {
  message("Todos os pacotes ja estao instalados.")
}

# torch precisa baixar o backend libtorch na primeira vez (~ alguns GB).
if (!torch::torch_is_installed()) {
  message("Instalando backend libtorch...")
  torch::install_torch()
}

message("\nVerificando carregamento de cada pacote...")
for (p in pkgs) {
  ok <- requireNamespace(p, quietly = TRUE)
  message(sprintf("  %-12s %s", p, if (ok) "OK" else "FALHOU"))
}
