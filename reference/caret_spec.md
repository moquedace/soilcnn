# Turn a caret method into a model_spec.

Turn a caret method into a model_spec.

## Usage

``` r
caret_spec(method, name = method, search = c("grid", "random"), ...)
```

## Arguments

- method:

  caret method name, e.g. "ranger", "xgbTree", "cubist".

- name:

  Name to register it under. Defaults to the method name.

- search:

  Passed to caret_grid() for the default grid.

- ...:

  Extra arguments forwarded to every caret::train() call – this is where
  a model's own arguments go (num.threads, nthread, ...).

## Value

A model_spec with input == "table".

## Examples

``` r
caret_spec("rf")
#> <model_spec> rf
#>   input       : table
#>   tunable     : yes
#>   reports size: no
#>   caret::train(method = "rf") -- Random Forest
```
