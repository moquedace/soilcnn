# The smearing factor from a tuning run's out-of-fold predictions.

ONE RESIDUAL PER POINT, not one per (point, seed).

## Usage

``` r
smearing_from_run(run_dir, config_id, role = "validation", transform = "log1p")
```

## Arguments

- run_dir:

  Tuning run directory.

- config_id:

  Which config's residuals.

- role:

  Which role. "validation" is the point.

- transform:

  Name of the forward transform. Only "log1p" and "log" are supported,
  and an unknown one is refused rather than assumed: the algebra above
  is specific to an exponential back-transform, and applying it to a
  square root or an identity would scale a number that needs no scaling.

## Value

A smearing_cal, or NULL when the run wrote no usable predictions.

## Details

The deployed prediction is the ensemble MEDIAN over seeds, so the
residual being calibrated has to be the ensemble's and not one member's.
This is the same reasoning cv_residuals() applies for the conformal
interval, and for a while this function claimed to use "the same
out-of-fold residuals" while quietly reading the per-seed rows instead:
9,276 rows for 3,092 points.

Two consequences, and the second is the one that matters:

- exp() is convex, so mean(exp(e)) grows with var(e). A single member's
  residual is noisier than the ensemble's (sd 0.6698 against 0.6507
  here), which inflated S by 0.99% – almost exactly the 1.00% that the
  variance difference alone predicts.

- `n` read 9,276 when there were 3,092 exchangeable units, overstating
  the evidence threefold in the one line a reader would use to judge it.

The 1% is NOT why this was changed. The target is bracketed between
1.259 and 1.389 by two held-out measurements of the deployed ensemble
itself, so no choice inside that range is defensible as more accurate.
It was changed because the claim in the docs was false and the printed n
was wrong.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
smearing_from_run(run$fit$run_dir, run$final$selected_config_ids)
# }
}
```
