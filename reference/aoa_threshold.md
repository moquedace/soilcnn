# The DI threshold that separates the area of applicability.

Derived from the CROSS-VALIDATED training data: for each training row,
the distance to the nearest training row in a DIFFERENT fold. That set
is the dissimilarity the model actually coped with during
cross-validation, so the outlier-removed maximum of it is the largest
dissimilarity for which the reported error has been demonstrated.

## Usage

``` r
aoa_threshold(ref, folds, k_iqr = 1.5)
```

## Arguments

- ref:

  From di_reference().

- folds:

  Integer fold label per training row, in the same order as the matrix
  the reference was built from; `NA` for a row never held out.

- k_iqr:

  Outlier rule: threshold = Q75 + k_iqr \* IQR. 1.5 is Tukey's fence and
  the value Meyer & Pebesma use.

## Value

The threshold, with the cross-validated DI attached as "cv_di" (`NA` for
the rows never held out).

## Details

A row never held out – the training side of a single split – has fold
`NA`: it is a neighbour to every fold, and has no cross-validated DI of
its own, since no model predicted it.

## Examples

``` r
set.seed(1)
x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
ref <- di_reference(x_train)
th <- aoa_threshold(ref, folds = rep(1:5, 20))
as.numeric(th)                 # the threshold
#> [1] 0.3342118
summary(attr(th, "cv_di"))     # the cross-validated DI it was taken from
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#> 0.02244 0.06969 0.11721 0.13145 0.17550 0.53434 
```
