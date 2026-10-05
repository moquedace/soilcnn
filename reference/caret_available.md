# Every caret method that could be borrowed here.

Every caret method that could be borrowed here.

## Usage

``` r
caret_available(pattern = NULL)
```

## Arguments

- pattern:

  Optional regular expression on the method name.

## Value

A tibble of method, label and the parameters it tunes.

## Examples

``` r
caret_available("^rf$|^ranger$")
#> # A tibble: 2 × 3
#>   method label         parameters                    
#>   <chr>  <chr>         <chr>                         
#> 1 ranger Random Forest mtry, splitrule, min.node.size
#> 2 rf     Random Forest mtry                          
```
