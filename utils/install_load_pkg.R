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

  invisible(lapply(packages, require, character.only = TRUE))

  cat("\n-----------------------------------------------------------\n")
  cat("Package loading completed.\n")
  cat("-----------------------------------------------------------\n")
}
