# Plot maps of SHAP values.

One panel per variable on a diverging scale centred on zero – where the
variable raises the prediction and where it lowers it – then the
dominant variable and the prediction.

## Usage

``` r
# S3 method for class 'importance_map'
plot(x, n = 12L, ...)
```

## Arguments

- x:

  An `importance_map`, from
  [`importance_map()`](https://moquedace.github.io/soilcnn/reference/importance_map.md).

- n:

  How many variables, the most important first.

- ...:

  Ignored.

## Value

`x`, invisibly.
