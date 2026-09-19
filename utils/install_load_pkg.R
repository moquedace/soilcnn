install_load_pkg <- function(packages) {
  missing_packages <- packages[!packages %in% rownames(installed.packages())]

  if (length(missing_packages) > 0) {
    cat("Installing the following packages:", paste(missing_packages, collapse = " | "), "\n")
    cat("-----------------------------------------------------------\n")
    install.packages(missing_packages)
    cat("-----------------------------------------------------------\n")
    cat("Installation completed.\n")
  }

  cat("\n\n-----------------------------------------------------------\n")
  cat("Loading the following packages now:", paste(packages, collapse = " | "), "\n")
  cat("-----------------------------------------------------------\n")

  # library(), not require(): require() returns FALSE on a package that did
  # not load and this banner still printed "completed". The script then died
  # at its first `%>%` or terra call, far from the cause. torch is the usual
  # case -- install.packages() puts the R package in place, but the C++
  # backend needs torch::install_torch() once, and until then library(torch)
  # fails with a message that says exactly that.
  for (pkg in packages) {
    ok <- suppressPackageStartupMessages(
      requireNamespace(pkg, quietly = TRUE))
    if (!ok) {
      stop("Package '", pkg, "' is installed but cannot be loaded.",
           if (pkg == "torch") "\n  Run torch::install_torch() once, then re-run." else "",
           call. = FALSE)
    }
    library(pkg, character.only = TRUE)
  }

  cat("\n-----------------------------------------------------------\n")
  cat("Package loading completed.\n")
  cat("-----------------------------------------------------------\n")
}
