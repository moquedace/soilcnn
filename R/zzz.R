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
