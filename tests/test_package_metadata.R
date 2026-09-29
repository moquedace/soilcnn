# Unit test: the package's metadata says what the code does
#
# WHY THIS FILE EXISTS.
#
# NAMESPACE and DESCRIPTION are written by hand here -- roxygen2 is not what
# anyone runs after an edit -- and each can drift from the code in a way that
# shows up far from the edit:
#
#   an export whose function was renamed stops the package from LOADING;
#   a function tagged @export but missing from NAMESPACE is invisible to
#     library() users and to nobody else (the source tree exports everything);
#   a print method never registered prints the raw list;
#   a pkg:: call on a package DESCRIPTION does not declare installs fine and
#     fails on the next machine.
#
# And the guards that keep a session on the package's own code: the two for
# workers (.pkg_loader(), .pkg_check_portable() in R/utils.R), checked here
# without starting a process -- tests/test_package_install.R starts them --
# and .onAttach()'s refusal of copies source()d into the global environment.
#
# Verified:
#   1. DESCRIPTION names the package, and declares every package R/ calls with ::
#   2. every declared import is used
#   3. NAMESPACE is what the @export tags say, and names only what exists
#   4. every print.<class> in R/ is a registered S3 method
#   5. R/ holds no loader and sources nothing
#   6. the session's loader describes the source tree, with a fingerprint
#   7. the loader refuses a framework that is not a namespace
#   8. a job carrying a function of the namespace is refused; plain data passes
#   9. a copy source()d from R/ is refused at attach, with the way out; a
#      function of the user's own with the same name is not
#  10. every exported function that scores a store's predictions defaults to
#      the store's own inverse transform
#  11. .Rbuildignore keeps the user's directories out and the package in
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_package_metadata.R")

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
ns <- .load_framework(root)

ok <- logical(0)

desc <- read.dcf(file.path(root, "DESCRIPTION"))
pkg  <- unname(desc[1, "Package"])
deps <- function(field) {
  if (!field %in% colnames(desc)) return(character(0))
  x <- trimws(strsplit(desc[1, field], ",")[[1]])
  x <- sub("\\s*\\(.*$", "", x)
  x[nzchar(x) & x != "R"]
}
imports  <- deps("Imports")
suggests <- deps("Suggests")

r_files <- sort(list.files(file.path(root, "R"), pattern = "\\.R$", full.names = TRUE))

# ── 1-2. every pkg:: is declared, and every import is used ───────────────────
# From the parser's own tokens, not a regex: a "pkg::" inside a string or a
# comment is not a call.
used <- unique(unlist(lapply(r_files, function(f) {
  pd <- utils::getParseData(parse(f, keep.source = TRUE))
  pd$text[pd$token == "SYMBOL_PACKAGE"]
})))
undeclared <- setdiff(used, c(imports, suggests, "base"))
ok["description_names_the_package"] <- identical(pkg, "soilcnn")
ok["every_pkg_call_is_declared"] <- length(undeclared) == 0L
ok["every_import_is_used"] <- all(imports %in% used)

# ── 3. NAMESPACE is what the tags say ────────────────────────────────────────
# roxygen2 turns `#' @export` above `name <- function` into export(name), or
# S3method(generic,class) for a method of a generic -- here, print only. The
# directives are compared as sets; the file's order is roxygen's.
tagged <- unlist(lapply(r_files, function(f) {
  ln <- readLines(f, warn = FALSE)
  at <- which(trimws(ln) == "#' @export")
  vapply(at, function(i) {
    j <- i + 1L
    while (j <= length(ln) && grepl("^\\s*#", ln[j])) j <- j + 1L
    sub("^([A-Za-z_.][A-Za-z0-9_.]*)\\s*<-\\s*function.*$", "\\1", ln[j])
  }, character(1))
}))
from_tags <- ifelse(grepl("^print\\.", tagged),
                    sprintf("S3method(print,%s)", sub("^print\\.", "", tagged)),
                    sprintf("export(%s)", tagged))
ns_lines <- readLines(file.path(root, "NAMESPACE"), warn = FALSE)
ns_lines <- ns_lines[nzchar(ns_lines) & !grepl("^#", ns_lines)]
ok["namespace_matches_the_export_tags"] <- setequal(from_tags, ns_lines)
ok["namespace_is_sorted_as_roxygen_writes_it"] <-
  identical(ns_lines, sort(ns_lines, method = "radix"))

exported <- sub("^export\\((.*)\\)$", "\\1", grep("^export\\(", ns_lines, value = TRUE))
ok["every_export_is_a_function_of_the_namespace"] <-
  all(vapply(exported, function(n) exists(n, envir = ns, inherits = FALSE) &&
               is.function(get(n, envir = ns)), logical(1)))

# ── 4. every print method is registered ──────────────────────────────────────
methods_in_r <- grep("^print\\.", ls(ns, all.names = TRUE), value = TRUE)
registered <- sub("^S3method\\(print,(.*)\\)$", "print.\\1",
                  grep("^S3method\\(print,", ns_lines, value = TRUE))
ok["every_print_method_is_registered"] <- setequal(methods_in_r, registered)

# ── 5. R/ is a package's R/: no loader, nothing sourced ──────────────────────
sources_something <- vapply(r_files, function(f) {
  pd <- utils::getParseData(parse(f, keep.source = TRUE))
  any(pd$token == "SYMBOL_FUNCTION_CALL" & pd$text %in% c("source", "sys.source"))
}, logical(1))
ok["no_loader_in_R"] <- !file.exists(file.path(root, "R", "load_all.R"))
ok["no_file_in_R_sources_another"] <- !any(sources_something)

# ── 6. the loader, in this session: the source tree, fingerprinted ───────────
ld <- .pkg_loader()
ok["loader_names_the_package"] <- identical(ld$package, pkg)
ok["loader_says_source_tree"]  <- isTRUE(ld$dev)
ok["loader_points_at_the_root"] <-
  identical(normalizePath(ld$path, winslash = "/"), normalizePath(root, winslash = "/"))
ok["loader_carries_the_fingerprint"] <-
  is.character(ld$code_hash) && nzchar(ld$code_hash) &&
  identical(ld$code_hash, .pkg_code_hash(root))
# What travels must not drag the namespace along (see .pkg_loader()).
ok["loader_open_has_base_as_its_environment"] <-
  identical(environment(ld$open), baseenv())

# ── 7. a framework that is not a namespace is refused ────────────────────────
# What a session that source()d the files would have: the same function, its
# environment the global one.
loose <- .pkg_loader
environment(loose) <- globalenv()
msg <- tryCatch({ loose(); "" }, error = function(e) conditionMessage(e))
ok["loader_refuses_source_d_files"] <- grepl("as a package", msg)

# ── 8. a job tied to the namespace is refused ────────────────────────────────
ok["plain_job_passes"] <- isTRUE(.pkg_check_portable(
  list(a = 1, b = list(c = "x", f = identity), d = data.frame(x = 1:3)), "test"))
msg <- tryCatch({ .pkg_check_portable(list(a = 1, nested = list(fit = dsm_train)), "test"); "" },
                error = function(e) conditionMessage(e))
ok["namespace_function_in_a_job_is_refused"] <- grepl("job\\$nested\\$fit", msg)
closure <- local({ x <- 1; function() x })       # enclosed by this test, not the package
ok["foreign_closure_passes"] <- isTRUE(.pkg_check_portable(list(f = closure), "test"))

# ── 9. copies of the package source()d into the global environment ───────────
# A function whose source reference says R/metrics.R is what a session from
# before the package left behind; a user's own function of the same name
# (ccc() is a common one to write) comes from somewhere else and must pass.
# .onAttach() is called by hand: attaching again would reload the package.
if (exists("ccc", envir = globalenv(), inherits = FALSE)) {
  stop("test_package_metadata puts a ccc() in the global environment and ",
       "needs it free of one first.", call. = FALSE)
}
fn_from <- function(file, text) {
  eval(parse(text = text, keep.source = TRUE, srcfile = srcfilecopy(file, text)))
}
copy <- fn_from(file.path(root, "R", "metrics.R"), "function(x, y) 0")
own  <- fn_from(file.path(tempdir(), "my_functions.R"), "function(x, y) 1")
on_attach <- get(".onAttach", envir = ns)
probe <- function(f) {
  assign("ccc", f, envir = globalenv())
  on.exit(rm("ccc", envir = globalenv()))
  list(found  = .pkg_stale_copies(),
       attach = tryCatch({ on_attach(dirname(root), pkg); "" },
                         error = function(e) conditionMessage(e)))
}
p_copy <- probe(copy)
p_own  <- probe(own)
ok["copy_sourced_from_R_is_found"]     <- identical(p_copy$found, "ccc")
ok["attach_refuses_it_and_says_how"]   <- grepl(".pkg_stale_copies()", p_copy$attach, fixed = TRUE)
ok["own_function_of_that_name_passes"] <- length(p_own$found) == 0L && identical(p_own$attach, "")
ok["the_probe_left_nothing_behind"]    <- !exists("ccc", envir = globalenv(), inherits = FALSE)

# ── 10. what scores predictions from a store defaults to the store's inverse ──
# dsm_train() has since the audit; score_test_grid() and occlusion_report()
# defaulted to identity, and on a log1p store a call that left `transform` out
# scored in log space, in plausible numbers. NULL is the store's own inverse.
scorers <- c("dsm_train", "dsm_final", "score_test_grid", "occlusion_report")
ok["scorers_default_to_the_stores_inverse"] <- all(vapply(scorers, function(f) {
  fm <- formals(get(f, envir = ns))
  "transform" %in% names(fm) && is.null(fm$transform)
}, logical(1)))

# ── 11. .Rbuildignore keeps the user's directories out of a tarball ──────────
# By the rule R CMD build applies: each line a Perl regular expression, case
# ignored, against the path from the package root. test_package_install.R
# builds from a staged copy of the package's own files, so this is where the
# rule for a build of the whole directory is held to.
ign <- readLines(file.path(root, ".Rbuildignore"), warn = FALSE)
ign <- ign[nzchar(trimws(ign))]
kept_out <- function(path) {
  any(vapply(ign, function(p) grepl(p, path, perl = TRUE, ignore.case = TRUE), logical(1)))
}
user_parts <- c("data", "outputs", "examples", "tests", "docs", "tools", "utils",
                "README.md", "LICENSE.md")
package_parts <- c("DESCRIPTION", "NAMESPACE", "LICENSE", "R", "man", "vignettes", "inst")
ok["buildignore_keeps_the_user_parts_out"] <- all(vapply(user_parts, kept_out, logical(1)))
ok["buildignore_keeps_the_package_in"] <- !any(vapply(package_parts, kept_out, logical(1)))

cat("  imports declared/used    : ", length(imports), " / ", sum(imports %in% used), "\n", sep = "")
cat("  exports, S3 methods      : ", length(exported), ", ", length(registered), "\n", sep = "")

.report(ok, "test_package_metadata")
