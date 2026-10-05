# Plot importances side by side.

The first two importances' shares of their largest, one against the
other, each variable a point on a 1:1 line; and, for two SHAP
importances of the same points, how well they agree variable by
variable.

## Usage

``` r
# S3 method for class 'importance_comparison'
plot(x, ...)
```

## Arguments

- x:

  An `importance_comparison`, from
  [`compare_importance()`](https://moquedace.github.io/soilcnn/reference/compare_importance.md).

- ...:

  Ignored.

## Value

`x`, invisibly.
