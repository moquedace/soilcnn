# Summarise a training set for dissimilarity-index computation.

Computed once and reused for every prediction chunk: the training data
does not change, and the pairwise mean is the expensive part.

## Usage

``` r
di_reference(x_train, weights = NULL, max_pairs = 2e+06, seed = 42L)
```

## Arguments

- x_train:

  Numeric matrix of SCALED training predictors (n x p), in the model's
  channel order.

- weights:

  Optional per-predictor weights, e.g. variable importance. NULL gives
  every predictor the same weight, which is what Meyer & Pebesma fall
  back to when no importance is available.

- max_pairs:

  Cap on the number of point pairs used to estimate avg_dist. The full
  n^2 is unnecessary for a mean and quadratic in memory; a large random
  sample estimates it to more precision than the quantity deserves.

- seed:

  Draw seed for that sample.

## Value

An object of class "di_reference".

## Examples

``` r
set.seed(1)
x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
ref <- di_reference(x_train)
ref
#> <di_reference> 100 training row(s) x 2 predictor(s) | unweighted
#>   mean pairwise distance (the DI's unit): 1.656
```
