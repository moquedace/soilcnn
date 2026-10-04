## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Test environments

* Local: Windows 11, R 4.6.1, `R CMD check --as-cran` with the PDF manual,
  and with torch's backend hidden (`TORCH_HOME` set to an empty folder), as
  on CRAN's machines.
* win-builder: R-devel. [to be filled in]

## Notes for the reviewers

* The package depends on 'torch'. Its backend (libtorch) is not on CRAN's
  machines: every example that needs it is guarded by
  `@examplesIf torch::torch_is_installed()`, those that train are inside
  `\donttest{}`, and the tests that CRAN runs need no backend.
* Examples, tests and the vignette write only to `tempdir()` and use at most
  two threads.
