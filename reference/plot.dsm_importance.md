# Plot a variable importance.

The figure that answers the importance's own question: bars for a
permutation, the cost per pixel by ring (or per variable, or per window)
for a context, the summary plot for SHAP – a point per profile at its
SHAP value, coloured by the variable's value there – the curves for ALE,
for SAGE the loss each variable takes away, and for a refit the skill a
network trained without it loses.

## Usage

``` r
# S3 method for class 'dsm_importance'
plot(x, n = 20L, ...)
```

## Arguments

- x:

  A `dsm_importance`, from
  [`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md).

- n:

  How many variables to show.

- ...:

  Ignored.

## Value

`x`, invisibly.
