# Before submitting to CRAN

Not in cran-comments.md: submit_cran() sends that file to the reviewers.

1. Vignette: no `[DRAFT` left, figures drawn from the final run
   (`tools/vignette_results.R`, `results_from <- "sample10"`).
2. `tools/check_package.R` with `--as-cran`, the PDF manual and libtorch
   hidden: only "New submission".
3. `spelling::spell_check_package()`: empty (inst/WORDLIST).
4. `devtools::check_win_devel()`: the same single NOTE; update the dates above.
5. NEWS.md and the version agree; `git status` clean; pushed.
6. `devtools::submit_cran()`, then confirm the e-mail link.
