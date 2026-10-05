# Fetch a registered model.

Fetch a registered model.

## Usage

``` r
get_model(name)
```

## Arguments

- name:

  Registered name.

## Value

The `model_spec` registered under `name`.

## Examples

``` r
get_model("rf")
#> <model_spec> rf
#>   input       : table
#>   tunable     : yes
#>   reports size: yes
#>   Random Forest on the tabular view (ranger, or randomForest).
```
