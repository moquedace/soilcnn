# Report what an AOA covers.

Prints the threshold, the share of cells inside it and the quantiles of
the DI, and says so plainly when less than half the map is inside.

## Usage

``` r
print_aoa(di, threshold, label = "prediction area")
```

## Arguments

- di:

  Dissimilarity index of the cells, from dissimilarity_index().

- threshold:

  From aoa_threshold().

- label:

  What the cells are, for the report.

## Value

The logical mask of the cells inside (inside_aoa()), invisibly.

## Examples

``` r
set.seed(1)
x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
ref <- di_reference(x_train)
threshold <- aoa_threshold(ref, folds = rep(1:5, 20))
print_aoa(dissimilarity_index(ref, matrix(rnorm(400, sd = 1.5), 200, 2)), threshold)
#> 
#> -- Area of applicability --
#>   DI threshold          : 0.3342  (Q75 + 1.5 x IQR of the cross-validated training DI)
#>   prediction area       : 145 of 200 cells inside (72.5%)
#>   DI quantiles          : min 0.01 | median 0.15 | q95 1.15 | max 2.45
```
