# Build a manual grid from explicit lists of values per parameter.

Build a manual grid from explicit lists of values per parameter.

## Usage

``` r
make_manual_tune_grid(...)
```

## Arguments

- ...:

  Named arguments, each a vector or list of values to cross. Only the
  provided parameters are varied; all others use their first (default)
  value from the parameter space.

## Value

A tibble, one configuration per row, with a config_id column.

## Examples

``` r
make_manual_tune_grid(
  embedding_dim = c(256L, 384L),
  gate_type     = c("vector_featurewise", "no_gate_concat"),
  base_lr       = c(0.001, 0.0005)
)
#> # A tibble: 8 × 21
#>   config_id window_sizes conv_channels use_residual use_se_block se_reduction
#>   <chr>     <list>       <list>        <lgl>        <lgl>               <int>
#> 1 cfg_001   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> 2 cfg_002   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> 3 cfg_003   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> 4 cfg_004   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> 5 cfg_005   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> 6 cfg_006   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> 7 cfg_007   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> 8 cfg_008   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> # ℹ 15 more variables: embedding_dim <int>, embed_pool <chr>,
#> #   conv_padding <chr>, gate_type <chr>, dropout <dbl>, spatial_dropout <dbl>,
#> #   embed_dropout <dbl>, gate_dropout <dbl>, head_dropout_1 <dbl>,
#> #   head_dropout_2 <dbl>, base_lr <dbl>, weight_decay <dbl>, batch_size <int>,
#> #   loss_fn <chr>, warmup_epochs <int>
```
