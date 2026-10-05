# Is each row inside the area of applicability?

Is each row inside the area of applicability?

## Usage

``` r
inside_aoa(di, threshold)
```

## Arguments

- di:

  From dissimilarity_index().

- threshold:

  From aoa_threshold().

## Value

Logical vector. TRUE means the cross-validated error applies here.

## Examples

``` r
set.seed(1)
x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
ref <- di_reference(x_train)
threshold <- aoa_threshold(ref, folds = rep(1:5, 20))
inside_aoa(dissimilarity_index(ref, rbind(c(0, 0), c(4, 4))), threshold)
#> [1]  TRUE FALSE
```
