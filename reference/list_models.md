# Every registered model, as a table.

Every registered model, as a table.

## Usage

``` r
list_models()
```

## Value

A tibble: name, input ("table" or "patches"), tunable (whether it draws
its own grid) and description.

## Examples

``` r
list_models()
#> # A tibble: 3 × 4
#>   name  input   tunable description                                             
#>   <chr> <chr>   <lgl>   <chr>                                                   
#> 1 cnn   patches TRUE    Dual-branch CNN over the patch tensors (the model under…
#> 2 mlp   table   TRUE    Fully connected network on the tabular view (torch).    
#> 3 rf    table   TRUE    Random Forest on the tabular view (ranger, or randomFor…
```
