# Run every test and summarise at the end.
#
#   source("tests/run_all.R")
#
# Each test runs in its own environment, so they cannot leak variables into
# one another, and a failure in one does not stop the rest — you get the full
# picture in a single pass instead of fixing them one at a time.

# -- project root: works under source() in the console AND under Rscript ------

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
  stop("Project root not found. setwd() to the deep_learning_caret root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

# Rapidos: ~3 segundos os oito juntos. Rode a cada edicao, sem pensar.
test_files <- c(
  "test_patch_geometry.R",   # geometry of the patches, both extraction paths
  "test_transform_loss.R",   # early-stopping loss vs the real torch loss
  "test_validation.R",       # clamp contract + option-set validation
  "test_preprocess.R",       # QC vs scaling split, and order equivalence
  "test_patch_store_io.R",   # round-trip do store + guarda do torch_save 2^31
  "test_resample.R",         # planos de fold: particao, vazamento, semente
  "test_architecture.R",     # embed_pool wiring
  "test_augmentation.R"      # D4 symmetries
)

# Lentos: treinam modelos torch de verdade (~3 min). Sao os unicos que provam
# que os modulos estao LIGADOS entre si, entao o default e rodar.
#
# Ponha FALSE quando estiver iterando de minuto em minuto -- mas rode-os antes
# de qualquer script caro, que e onde se pagam: foi assim que apareceram o
# bind_rows com tipo adivinhado e o piso de ruido igual a zero.
run_slow <- TRUE

slow_files <- c(
  "test_resample_run.R"      # fiacao ponta a ponta: store -> folds -> tabelas
)

if (run_slow) test_files <- c(test_files, slow_files)

rule   <- strrep("-", 72)
status <- character(0)
t0     <- Sys.time()

for (tf in test_files) {
  cat("\n", rule, "\n", tf, "\n", sep = "")
  status[tf] <- tryCatch(
    {
      source(file.path(root, "tests", tf), local = new.env())
      "PASS"
    },
    error = function(e) {
      cat("  ", conditionMessage(e), "\n", sep = "")
      "FAIL"
    }
  )
}

el <- Sys.time() - t0

cat("\n", rule, "\n", sep = "")
for (tf in names(status)) {
  cat(sprintf("  [%s] %s\n", status[tf], tf))
}
cat(sprintf("\n  %d/%d passed in %.1f %s\n",
            sum(status == "PASS"), length(status),
            as.numeric(el), units(el)))

if (!run_slow) {
  cat("  (testes lentos PULADOS -- run_slow <- FALSE no topo deste arquivo)
")
}

if (any(status != "PASS")) {
  cat("  -> rerun a failing file on its own for the full output\n")
}
