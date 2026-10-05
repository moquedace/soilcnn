# Lin's Concordance Correlation Coefficient.

Lin's Concordance Correlation Coefficient.

## Usage

``` r
ccc(obs, pred)
```

## Arguments

- obs, pred:

  Numeric vectors of the same length.

## Value

A single numeric, or NA when fewer than two finite pairs remain.

## Examples

``` r
obs <- c(10, 20, 30, 40)
ccc(obs, obs)        # 1: perfect agreement
#> [1] 1
ccc(obs, obs + 5)    # below 1: an offset is disagreement too, though r is 1
#> [1] 0.9090909
```
