# Integration test: the package builds, installs, loads, and its workers load
# the same code
#
# WHY THIS FILE EXISTS.
#
# Everything else in tests/ runs against the source tree (pkgload). What
# only an install shows:
#
#   the package's files are READ AT INSTALL TIME, in alphabetical order, and
#     their top-level code runs then -- a registration that needed a later
#     file failed there and nowhere else;
#   NAMESPACE is enforced: an export that does not exist stops the load, and
#     a function not exported is invisible to library() users;
#   a worker of an installed session must load that same installed copy.
#
# And for the source tree, what the other tests only use: a worker opens the
# session's code, and refuses code that changed since the session loaded it.
#
# The install goes to a temporary library, never the user's. It is built
# through a tarball because R CMD INSTALL of the directory copies data/ into
# the library, subdirectories and all -- the user's data lives there.
#
# Verified:
#   1. a worker of this (source tree) session opens the same code
#   2. a worker refuses a fingerprint that is not its session's
#   3. the package builds into a tarball that leaves data/ and outputs/ out
#   4. it installs into a temporary library and loads in a fresh process
#   5. there: the API is exported, the internals are not, print dispatches
#   6. there: the built-in models are registered (.onLoad ran)
#   7. there: the loader says installed, and a worker opens that same copy
#   8. there: citation("soilcnn") is the package's own CITATION, with both
#      authors -- read from the installed copy, which is the only place
#      citation() reads (a session with the source tree loaded cannot)
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_package_install.R")
#      (~2 min: a build, an install, four R processes)

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
.load_framework(root)

for (p in c("callr", "pkgbuild")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop("test_package_install needs ", p, ": install.packages(\"", p, "\").",
         call. = FALSE)
  }
}

ok <- logical(0)
work <- file.path(tempdir(), "dlc_test_install")
unlink(work, recursive = TRUE)
dir.create(work, recursive = TRUE)
lib <- file.path(work, "lib")
dir.create(lib)

# ── 1-2. a worker of this session: the source tree, fingerprint checked ──────
loader <- .pkg_loader()
seen <- callr::r(function(l) {
  ns <- l$open(l)
  list(hash = get(".pkg_state", envir = ns)$code_hash,
       path = getNamespaceInfo(ns, "path"),
       dev  = pkgload::is_dev_package(l$package))
}, args = list(loader))
ok["dev_worker_opens_the_same_code"] <- identical(seen$hash, loader$code_hash)
ok["dev_worker_opens_the_source_tree"] <- isTRUE(seen$dev) &&
  identical(normalizePath(seen$path, winslash = "/"), normalizePath(root, winslash = "/"))

stale <- loader
stale$code_hash <- paste0("not-", loader$code_hash)
refused <- callr::r(function(l) {
  tryCatch({ l$open(l); "opened" }, error = function(e) conditionMessage(e))
}, args = list(stale))
ok["worker_refuses_changed_code"] <- grepl("not the code its session loaded", refused)

# ── 3. the tarball ───────────────────────────────────────────────────────────
# BUILT FROM A COPY OF THE PACKAGE'S OWN FILES. R CMD build lists every file
# under the directory before it applies .Rbuildignore -- outputs/ and data/
# included, tens of thousands of files on one HDD -- and that listing was
# 4m40s of every build here. What it would keep is exactly what is copied, so
# the tarball is the same; that .Rbuildignore keeps the rest out is checked in
# test_package_metadata.R, against the rule R CMD build applies.
pkg   <- unname(read.dcf(file.path(root, "DESCRIPTION"), fields = "Package")[1, 1])
stage <- file.path(work, "stage", pkg)
dir.create(stage, recursive = TRUE)
parts <- c("DESCRIPTION", "NAMESPACE", "LICENSE", ".Rbuildignore", "R", "man", "vignettes",
           "inst")
parts <- parts[file.exists(file.path(root, parts))]
copied <- vapply(parts, function(p) file.copy(file.path(root, p), stage, recursive = TRUE),
                 logical(1))
ok["the_package_was_staged"] <- all(copied)
tgz <- pkgbuild::build(stage, dest_path = work, vignettes = FALSE, manual = FALSE,
                       quiet = TRUE)
listed <- utils::untar(tgz, list = TRUE)
top <- unique(sub("^[^/]+/([^/]+).*$", "\\1", listed))
ok["tarball_was_built"] <- file.exists(tgz)
ok["tarball_leaves_out_the_user_data"] <-
  !any(c("data", "outputs", "examples", "tests", "docs") %in% top)
ok["tarball_holds_the_package"] <- all(c("DESCRIPTION", "NAMESPACE", "R", "man") %in% top)
ok["tarball_holds_the_citation"] <- any(grepl("^[^/]+/inst/CITATION$", listed))

# ── 4. install into the temporary library ────────────────────────────────────
# R CMD INSTALL's own words when it fails: install.packages() would only warn
# "non-zero exit status" and the next step would fail for a reason it hides.
inst <- callr::rcmd("INSTALL", c("-l", lib, tgz), fail_on_status = FALSE)
if (inst$status != 0L) {
  stop("R CMD INSTALL failed (status ", inst$status, "):\n",
       paste(utils::tail(strsplit(paste(inst$stdout, inst$stderr), "\n")[[1]], 25),
             collapse = "\n"), call. = FALSE)
}
ok["installed_into_the_temporary_library"] <-
  file.exists(file.path(lib, "soilcnn", "DESCRIPTION"))

# ── 5-7. a fresh process, as a user of the installed package ─────────────────
got <- callr::r(function(lib) {
  library(soilcnn, lib.loc = lib)
  ns <- asNamespace("soilcnn")
  exports <- getNamespaceExports("soilcnn")
  loader <- get(".pkg_loader", envir = ns)()
  # A worker of THIS session, started as dsm_predict() starts one.
  worker <- callr::r(function(l) {
    ns <- l$open(l)
    list(hash = get(".pkg_state", envir = ns)$code_hash,
         path = getNamespaceInfo(ns, "path"))
  }, args = list(loader))
  list(
    api = c("dsm_prepare", "dsm_load", "spatial_cv", "knndm_cv", "dsm_train",
            "one_se", "dsm_final", "dsm_predict") %in% exports,
    # The dot helpers, and the plumbing the export list keeps out: a runner
    # (dsm_train() drives it), the store's own reader, a script's writer.
    internal_hidden = !any(c(".pkg_loader", ".predict_worker", ".check_cnn_grid",
                             "run_cnn_resample", "load_patch_store",
                             "safe_write_csv2") %in% exports),
    models = sort(list_models()$name),
    # From the global environment, where print.dsm_fit is not visible: only a
    # REGISTERED method is found from here, and only a registered one
    # dispatches when a user types the object's name.
    s3 = !is.null(utils::getS3method("print", "dsm_fit", optional = TRUE,
                                     envir = globalenv())),
    loader = loader[c("package", "dev", "path", "code_hash")],
    worker = worker,
    citation_authors = format(utils::citation("soilcnn", lib.loc = lib)$author,
                              include = c("given", "family")),
    lib_path = normalizePath(file.path(lib, "soilcnn"), winslash = "/"))
}, args = list(lib), libpath = c(lib, .libPaths()))

ok["installed_exports_the_api"] <- all(got$api)
ok["installed_hides_the_internals"] <- isTRUE(got$internal_hidden)
ok["installed_registers_the_models"] <- all(c("cnn", "mlp", "rf") %in% got$models)
ok["installed_registers_print_methods"] <- isTRUE(got$s3)
ok["installed_loader_says_installed"] <- identical(got$loader$dev, FALSE)
ok["installed_loader_points_at_the_copy"] <- identical(got$loader$path, got$lib_path)
ok["installed_worker_opens_the_same_copy"] <-
  identical(got$worker$hash, got$loader$code_hash) &&
  identical(normalizePath(got$worker$path, winslash = "/"), got$lib_path)
ok["installed_citation_names_both_authors"] <-
  identical(got$citation_authors,
            c("C\u00e1ssio Marques Moquedace", "Clara Gl\u00f3ria Oliveira Baldi"))

cat("  tarball                  : ", basename(tgz), " (", length(listed), " files)\n", sep = "")
cat("  installed models         : ", paste(got$models, collapse = ", "), "\n", sep = "")
cat("  citation authors         : ", paste(got$citation_authors, collapse = "; "), "\n", sep = "")

unlink(work, recursive = TRUE)
.report(ok, "test_package_install")
