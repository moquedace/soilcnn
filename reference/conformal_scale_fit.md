# Fit the scale of an interval: \|residual\| on covariates that say how wrong.

The scale of
[`conformal_scaled_calibrate()`](https://moquedace.github.io/soilcnn/reference/conformal_scaled_calibrate.md),
fitted on its own: a + b1 x1 + b2 x2 + ..., by least squares on \|obs -
pred\|, with a floor. Fitted on points the quantile is NOT taken on – a
calibration set's quantile, scaled by a fit to the cross-validated
residuals, keeps the whole calibration set for q; fitted and calibrated
on the same points, the fit would absorb the residuals the quantile
measures.

## Usage

``` r
conformal_scale_fit(obs, pred, covariates, floor_frac = 0.05)
```

## Arguments

- obs, pred:

  Observations and predictions, NATIVE units.

- covariates:

  A data frame of scale covariates, one row per point, e.g.
  data.frame(level = pred, di = di).

- floor_frac:

  The scale is never below this share of the median \|residual\|. A
  linear fit can go to zero or below at the edge of the covariates'
  range, and a zero-width interval there would claim a certainty the
  data never gave.

## Value

A `conformal_scale`: the coefficients, the floor, and how well the fit
explained \|residual\| (r2_fit).

## Examples

``` r
set.seed(1)
level <- runif(300, 1, 4)
di    <- runif(300)                      # a dissimilarity index
obs   <- exp(level + rnorm(300, 0, 0.1 + 0.3 * di))
pred  <- exp(level)
conformal_scale_fit(obs, pred, data.frame(level = pred, di = di))
#> 
#> <conformal_scale> |residual| ~ -2.2331 +0.2163 x level +4.3332 x di
#>   fitted on 300 point(s), R2 0.352 | floor 0.0939 (0.05 x the median |residual|)
```
