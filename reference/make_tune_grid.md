# Generate a random hyperparameter grid (like caret's tuneLength).

Generate a random hyperparameter grid (like caret's tuneLength).

## Usage

``` r
make_tune_grid(
  tune_length = 20L,
  seed = NULL,
  fixed = list(),
  windows = NULL,
  n_train = NULL
)
```

## Arguments

- tune_length:

  Number of configurations to sample.

- seed:

  Random seed for reproducibility.

- fixed:

  Named list used to FIX or RESTRICT parameters:

  - a length-1 value fixes the parameter (e.g. loss_fn = "smooth_l1")

  - a length\>1 vector restricts the sampling pool (e.g. base_lr =
    c(1e-4, 3e-4))

  - for the list-valued params window_sizes / conv_channels, pass a list
    of options (e.g. window_sizes = list(c(7L), c(5L, 7L))) or a single
    vector. Example focused search: fixed = list( loss_fn = "smooth_l1",
    batch_size = 512L, base_lr = c(1e-4, 3e-4), window_sizes =
    list(c(7L), c(5L, 7L)) )

- windows:

  The patch sizes the store holds (`data$store$window_sizes`). The
  window options become every single window and every pair of them – a
  dual-branch model takes two. NULL keeps the SOC example's 3/9/15 set;
  dsm_train() always passes the store's.

- n_train:

  Training points in the smallest fold. Batch sizes that would give an
  epoch fewer than four gradient steps are left out; if none is left,
  the largest power of two that gives four is used. NULL keeps the full
  set. dsm_train() passes it from the plan.

## Value

A tibble with one row per configuration. window_sizes and conv_channels
are stored as list-columns.

## Examples

``` r
make_tune_grid(tune_length = 3, seed = 1, windows = c(3, 7))
#> # A tibble: 3 × 21
#>   config_id window_sizes conv_channels use_residual use_se_block se_reduction
#>   <chr>     <list>       <list>        <lgl>        <lgl>               <int>
#> 1 cfg_001   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> 2 cfg_002   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> 3 cfg_003   <int [1]>    <int [2]>     TRUE         TRUE                   16
#> # ℹ 15 more variables: embedding_dim <int>, embed_pool <chr>,
#> #   conv_padding <chr>, gate_type <chr>, dropout <dbl>, spatial_dropout <dbl>,
#> #   embed_dropout <dbl>, gate_dropout <dbl>, head_dropout_1 <dbl>,
#> #   head_dropout_2 <dbl>, base_lr <dbl>, weight_decay <dbl>, batch_size <int>,
#> #   loss_fn <chr>, warmup_epochs <int>
```
