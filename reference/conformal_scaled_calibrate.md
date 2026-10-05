# Calibrate an interval whose width is a fitted scale.

Calibrate an interval whose width is a fitted scale.

## Usage

``` r
conformal_scaled_calibrate(
  obs,
  pred,
  covariates,
  alpha = 0.1,
  fit_frac = 0.5,
  floor_frac = 0.05,
  seed = 42L,
  scale = NULL,
  group = NULL
)
```

## Arguments

- obs, pred:

  Calibration observations and predictions, NATIVE units.

- covariates:

  A data frame of scale covariates, one row per point, e.g.
  data.frame(level = pred, di = di). The scale is a + b1 x1 + b2 x2 +
  ..., fitted by least squares on \|obs - pred\|.

- alpha:

  Miscoverage rate: 0.1 asks for 90%.

- fit_frac:

  Share of the points the scale is fitted on; the rest calibrate q. One
  half each is the textbook split. Not used when `scale` is given.

- floor_frac:

  The scale is never below this share of the median \|residual\| of the
  fitting half. A linear fit can go to zero or below at the edge of the
  covariates' range, and a zero-width interval there would claim a
  certainty the data never gave.

- seed:

  Seed of the split.

- scale:

  NULL: the scale is fitted on `fit_frac` of these points and q taken on
  the rest. A `conformal_scale` from
  [`conformal_scale_fit()`](https://moquedace.github.io/soilcnn/reference/conformal_scale_fit.md),
  fitted on OTHER points: every point here then calibrates q.

- group:

  Optional group of each point, as in
  [`conformal_calibrate()`](https://moquedace.github.io/soilcnn/reference/conformal_calibrate.md):
  q weighs every group alike, and the split, when there is one, keeps a
  group on one side.

## Value

A `conformal_scaled` (also a `conformal_cal`).

## Examples

``` r
set.seed(1)
level <- runif(300, 1, 4)
di    <- runif(300)                      # a dissimilarity index
obs   <- exp(level + rnorm(300, 0, 0.1 + 0.3 * di))
pred  <- exp(level)
covariates <- data.frame(level = pred, di = di)
cal <- conformal_scaled_calibrate(obs[1:200], pred[1:200], covariates[1:200, ],
                                  alpha = 0.1)
cal
#> 
#> <conformal_scaled> 90% intervals, width = q x fitted scale
#>   scale              : -2.7486 +0.2495 x level +4.5165 x di  (floor 0.1124; R2 on the fit half 0.386)
#>   points             : 100 fitted the scale, 100 calibrated q
#>   rank used          : 91 of 100  (the (n+1) correction)
#>   q                  : 3.7964 x scale
# the scale fitted on other points, and all 100 of these calibrating q
sc <- conformal_scale_fit(obs[1:200], pred[1:200], covariates[1:200, ])
conformal_scaled_calibrate(obs[201:300], pred[201:300], covariates[201:300, ],
                           scale = sc)
#> 
#> <conformal_scaled> 90% intervals, width = q x fitted scale
#>   scale              : -2.5489 +0.2269 x level +4.8155 x di  (floor 0.0974; R2 on the fit half 0.338)
#>   points             : 200 fitted the scale (other points), 100 calibrated q
#>   rank used          : 91 of 100  (the (n+1) correction)
#>   q                  : 4.0640 x scale
```
