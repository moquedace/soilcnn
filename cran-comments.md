## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.
* The words flagged as possibly misspelled in DESCRIPTION are authors' names
  in the references (Apley, Duan, Erion, Linnenbrink, Pebesma, Zhu), 'et al.',
  and technical terms (SHAP, convolutional, multilayer perceptron).

## Test environments

* Local: Windows 11, R 4.6.1, `R CMD check --as-cran` with the PDF manual,
  and with torch's backend hidden (`TORCH_HOME` set to an empty folder), as
  on CRAN's machines.
* win-builder: R-devel (2026-09-30 r90605 ucrt), 1 NOTE: new submission and
  the words above.

## Notes for the reviewers

* The package depends on 'torch'. Its backend (libtorch) is not on CRAN's
  machines: every example that needs it is guarded by
  `@examplesIf torch::torch_is_installed()`, those that train are inside
  `\donttest{}`, and the tests that CRAN runs need no backend.
* Examples, tests and the vignette write only to `tempdir()` and use at most
  two threads.
