# ── When the package loads ────────────────────────────────────────────────────
#
# Two things, once per session, however the package was loaded -- library(),
# or pkgload::load_all() on the source tree:
#
#   the fingerprint of the code it loaded, which every worker compares with
#     its own before it maps a unit (see .pkg_loader(), R/utils.R);
#   the built-in models, into the registry.
#
# The models used to register at the top of R/baselines.R, as it was
# source()d. A package reads its files when it is INSTALLED, in alphabetical
# order, and baselines.R comes before model_registry.R: register_model() does
# not exist yet when it is read. A Collate field in DESCRIPTION could force the
# order -- and would be the list R/load_all.R kept, a second place to keep in
# step with the files, which is what making this a package was meant to end.
.onLoad <- function(libname, pkgname) {
  .pkg_state$code_hash <- .pkg_code_hash(getNamespaceInfo(pkgname, "path"))
  .register_builtin_models()
}

# ── When it is attached: refuse copies of itself in the global environment ───
#
# Before this was a package, every session loaded it by source()ing R/ into
# the global environment (R/load_all.R), and those copies outlive the change:
# RStudio's Restart R keeps the global environment. There they answer every
# call made from the console or from a script in place of the package, with
# the code as it was when it was sourced -- and 21 of the example scripts do
# not clear the workspace. The first run after the change showed it: P4's
# dsm_predict() was the old copy, whose workers sourced an R/load_all.R that
# no longer exists.
#
# Stopped here, where every script and library() pass, rather than in each
# script: a check in 21 scripts would miss the console and the next script.
# Not removed: a package does not delete objects in the user's workspace; it
# says which and how. And a copy is told from a function of the user's own
# with the same name -- a ccc() of one's own -- by where its source says it
# came from, so only this package's R/ is refused.
.onAttach <- function(libname, pkgname) {
  stale <- .pkg_stale_copies()
  if (length(stale)) {
    stop("The global environment holds ", length(stale), " function(s) source()d ",
         "from this package's R/ directory (", paste(utils::head(stale, 4), collapse = ", "),
         if (length(stale) > 4L) ", ..." else "", "), left from before it was a ",
         "package. They would answer the console's and the scripts' calls in place ",
         "of the package, with the code as it was when it was sourced.\n  Remove them, ",
         "then load again:\n    rm(list = ", pkgname, ":::.pkg_stale_copies(), ",
         "envir = globalenv())\n  RStudio's Restart R keeps the global environment, ",
         "so a restart alone does not clear them.", call. = FALSE)
  }
}

# The functions in the global environment that share a name with this
# package's and were source()d from its R/ directory, by the filename their
# source reference carries.
.pkg_stale_copies <- function() {
  ns <- environment(sys.function())
  canon <- function(p) {
    p <- normalizePath(p, winslash = "/", mustWork = FALSE)
    if (.Platform$OS.type == "windows") tolower(p) else p
  }
  r_dir <- canon(file.path(getNamespaceInfo(ns, "path"), "R"))
  g <- globalenv()
  both <- intersect(ls(g, all.names = TRUE), ls(ns, all.names = TRUE))
  from_r <- vapply(both, function(n) {
    f <- get(n, envir = g)
    if (!is.function(f)) return(FALSE)
    sf <- attr(attr(f, "srcref"), "srcfile")
    if (!is.environment(sf) || !is.character(sf$filename) || !nzchar(sf$filename)) {
      return(FALSE)
    }
    fn <- sf$filename
    if (!grepl("^([A-Za-z]:)?[/\\\\]", fn) && is.character(sf$wd)) fn <- file.path(sf$wd, fn)
    identical(canon(dirname(fn)), r_dir)
  }, logical(1))
  both[from_r]
}
