# Add a model to the registry.

Add a model to the registry.

## Usage

``` r
register_model(spec, overwrite = FALSE)
```

## Arguments

- spec:

  From model_spec().

- overwrite:

  FALSE (default) refuses to replace an existing name. A silent
  replacement is how two different models come to answer to the same
  name in the same session, and the results carry no mark of which ran.

## Value

`spec`, invisibly.

## Examples

``` r
mean_only <- model_spec(
  "mean_only", input = "table",
  fit = function(x, y, cfg, ...) list(mu = mean(y)),
  predict = function(object, x, ...) rep(object$mu, nrow(x)),
  description = "the training mean: a floor every model must beat")
register_model(mean_only, overwrite = TRUE)
list_models()      # dsm_train(data, model = "mean_only", ...) now fits it
#> # A tibble: 4 × 4
#>   name      input   tunable description                                         
#>   <chr>     <chr>   <lgl>   <chr>                                               
#> 1 cnn       patches TRUE    Dual-branch CNN over the patch tensors (the model u…
#> 2 mean_only table   FALSE   the training mean: a floor every model must beat    
#> 3 mlp       table   TRUE    Fully connected network on the tabular view (torch).
#> 4 rf        table   TRUE    Random Forest on the tabular view (ranger, or rando…
```
