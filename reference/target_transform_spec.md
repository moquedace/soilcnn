# The forward and inverse functions for a named target transform.

The forward and inverse functions for a named target transform.

## Usage

``` r
target_transform_spec(name)
```

## Arguments

- name:

  "none" or "log1p".

## Value

list(name, forward, inverse).

## Examples

``` r
tr <- target_transform_spec("log1p")
tr$forward(10)
#> [1] 2.397895
tr$inverse(tr$forward(10))
#> [1] 10
```
