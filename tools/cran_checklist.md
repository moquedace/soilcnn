# Before submitting to CRAN

Not in cran-comments.md: submit_cran() sends that file to the reviewers.

When the sample10 run is done (03 -> 06 of the trial, then 07_importance.R):

0. Before 07: install the latest soilcnn from GitHub -- 06 is done, no
   worker of the run is left. The run's 8a8270f draws 07's SHAP map in one
   pass, but a map read back from 07's cache (a second run) needs b0c20e0.
1. Figures from the run, each looked at before it is kept:
   `tools/vignette_results.R` with `results_from <- "sample10"` (all five;
   the importance figure reads 07's outputs, so 07 runs first),
   and `tools/vignette_designs.R` with `plans_from <- "sample10"`. Commit the
   PNGs: their records now name sample10, so test_package_metadata lets them
   in, and `result_figure()` draws them in place of the notes.
2. Numbers from the run: `tools/vignette_numbers.R` with
   `results_from <- "sample10"`. Fill every `[DRAFT` of the vignette from its
   printout, never by hand, and settle every check it flags.
3. Vignette text: no `[DRAFT` left; the introduction's "Results marked DRAFT
   remain provisional" and section 9's "provisional" gone; "About the
   illustrations" says where each figure came from (the validation panels:
   the run on a tenth of the profiles).
4. `tests/run_all.R`: every script passes -- the full local suite, the gate
   (CI and CRAN run only the testthat subset in the tarball).
5. `tools/check_package.R` with `--as-cran`, the PDF manual and libtorch
   hidden: only "New submission".
6. `spelling::spell_check_package()`: empty (inst/WORDLIST).
7. `devtools::check_win_devel()`: the same single NOTE. The words it flags
   in DESCRIPTION are listed in cran-comments.md, and its date there is
   updated.
8. NEWS.md and the version agree; `git status` clean; pushed.
9. `devtools::submit_cran()`, then confirm the e-mail link.
