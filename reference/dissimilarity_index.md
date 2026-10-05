# The dissimilarity index of new data.

The dissimilarity index of new data.

## Usage

``` r
dissimilarity_index(ref, x, chunk = 2000L)
```

## Arguments

- ref:

  From di_reference().

- x:

  Numeric matrix of new rows, scaled with the MODEL's scaling.

- chunk:

  Rows of `x` compared at a time. It bounds the memory a call takes and
  does not change the result.

## Value

Numeric vector, one DI per row. 0 means identical to a training point; 1
means as far from the training data as two training points are from each
other on average.

## Examples

``` r
set.seed(1)
x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
ref <- di_reference(x_train)
dissimilarity_index(ref, rbind(c(0, 0), c(4, 4)))   # near the data, and far from it
#> [1] 0.1040553 2.3432834
```
