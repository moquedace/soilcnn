# The interval as the blocks grow.

[`block_bootstrap()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap.md)
for blocks of growing side. Blocks smaller than the distance over which
the values resemble each other are themselves dependent, and their
interval is still too narrow; where the width stops growing, they no
longer are.

## Usage

``` r
block_bootstrap_by_size(
  x,
  y,
  obs,
  pred,
  against = NULL,
  sizes_km = c(25, 50, 100, 200, 400),
  metric = "mae",
  weights = "profile",
  n_boot = 2000L,
  conf = 0.95,
  seed = 42L,
  coords = c("lonlat", "metres"),
  centre = NULL
)
```

## Arguments

- x, y:

  Coordinates, as in
  [`equal_area_blocks()`](https://moquedace.github.io/soilcnn/reference/equal_area_blocks.md).

- obs, pred, against:

  As in
  [`block_bootstrap()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap.md).

- sizes_km:

  The block sides to try, in km.

- metric:

  One metric, as in
  [`block_bootstrap()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap.md).

- weights:

  One weighting, as in
  [`block_bootstrap()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap.md).

- n_boot, conf, seed:

  As in
  [`block_bootstrap()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap.md).

- coords, centre:

  As in
  [`equal_area_blocks()`](https://moquedace.github.io/soilcnn/reference/equal_area_blocks.md).

## Value

A `block_bootstrap_by_size`: a tibble, one row per size, with `size_km`,
`n_blocks`, the `estimate` (or the `difference`), `ci_low`, `ci_high`,
`width` and `n_effective`.

## Examples

``` r
ex <- example_landscape()
obs <- ex$profiles$soc_stock
set.seed(1)
pred_a <- obs * exp(rnorm(length(obs), 0, 0.20))   # two models' predictions
pred_b <- obs * exp(rnorm(length(obs), 0, 0.25))
s <- block_bootstrap_by_size(ex$profiles$x, ex$profiles$y, obs, pred_a, against = pred_b,
                             sizes_km = c(1, 2, 5, 10), n_boot = 500)
s
#> # A tibble: 4 × 7
#>   size_km n_blocks difference ci_low ci_high width n_effective
#> *   <dbl>    <int>      <dbl>  <dbl>   <dbl> <dbl>       <dbl>
#> 1       1       84      -2.96  -4.36  -1.37   2.99        160 
#> 2       2       60      -2.96  -4.41  -1.70   2.71        160 
#> 3       5       18      -2.96  -4.40  -1.62   2.78        160 
#> 4      10        6      -2.96  -4.03  -0.224  3.80        141.
plot(s)
```
