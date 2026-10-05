# One weight per channel, for the area of applicability, from an importance.

The dissimilarity index measures how far a pixel is from the training
data, one axis per channel. Unweighted, a channel the model ignores
counts as much as the one it leans on, and a pixel unlike the training
data in an ignored channel falls outside the area of applicability for
nothing. Meyer & Pebesma (2021) weight each axis by the predictor's
importance; this gives each channel its variable's importance – every
channel of a one-hot set, or of a group of yours, its variable's – and a
negative importance, noise around zero, weight zero.

## Usage

``` r
importance_weights(x)
```

## Arguments

- x:

  A `dsm_importance` with one value per variable: by
  [`permutation_importance()`](https://moquedace.github.io/soilcnn/reference/permutation_importance.md),
  [`shap_importance()`](https://moquedace.github.io/soilcnn/reference/shap_importance.md),
  [`sage_importance()`](https://moquedace.github.io/soilcnn/reference/sage_importance.md),
  [`refit_importance()`](https://moquedace.github.io/soilcnn/reference/refit_importance.md)
  or
  [`ale_effect()`](https://moquedace.github.io/soilcnn/reference/ale_effect.md).

## Value

A named numeric vector, one weight per channel in the store's order, for
`dsm_predict(aoa_weights = )` or `aoa_reference(weights = )`.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
imp <- dsm_importance(run$final, run$data, shap_importance(samples = 20, background = 20),
                      verbose = FALSE)
importance_weights(imp)   # for dsm_predict(aoa_weights = ) or aoa_reference(weights = )
# }
}
```
