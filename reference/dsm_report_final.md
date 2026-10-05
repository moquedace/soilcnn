# The declaration of a final model that already exists.

dsm_final() writes final_report.md as it fits. A final model fitted by
stage 04, before dsm_final() existed, has the seeds and the files but no
declaration – and retraining ten seeds to obtain one would also change
them (T1: the thread count changes the numbers). This writes it from
what is on disk, re-assembling the seeds the way P3 proved dsm_final()
does, and changes nothing that is there: it ADDS final_report.md and
selected_hyperparameters.csv to the run directory.

## Usage

``` r
dsm_report_final(
  run_dir,
  tuning_dir,
  conformal_alpha = c(0.1, 0.05),
  verbose = TRUE
)
```

## Arguments

- run_dir:

  The final-model run (it holds comparison/final_run_summary.rds).

- tuning_dir:

  The tuning run it was selected from.

- conformal_alpha:

  As the run was calibrated; stage 04 used c(0.1, 0.05).

- verbose:

  Print the result.

## Value

A `dsm_final` describing the run, printed with the declaration.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
# the declaration of a final run, written from what is on disk
dsm_report_final(run$final$run_dir, run$fit$run_dir)
# }
}
```
