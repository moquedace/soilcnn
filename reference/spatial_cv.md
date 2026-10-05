# Spatially blocked k-fold: whole blocks of ground go to one fold.

Spatially blocked k-fold: whole blocks of ground go to one fold.

## Usage

``` r
spatial_cv(
  k = 5L,
  block_size = "auto",
  buffer = "auto",
  test_frac = 0.15,
  max_share = 0.1,
  buffer_metric = c("chebyshev", "euclidean"),
  seed = 42L
)
```

## Arguments

- k:

  Folds.

- block_size:

  "auto" measures it from the points (see suggest_block_size); a number
  is used as given, in x/y units.

- buffer:

  "auto" is `max(window) * cell_size`, which is the exact distance at
  which two patches stop sharing a pixel under the Chebyshev metric; a
  number is used as given; NULL applies none.

- test_frac:

  Share held out entirely, carved by the same criterion.

- max_share:

  Balance constraint for "auto": the largest share of the points one
  block may hold.

- buffer_metric:

  "chebyshev" (the default) or "euclidean". Chebyshev is exact for
  square patches: two patches share a pixel when their centres are
  within the window in both axes.

- seed:

  Seed for this split only; the training seeds do not move it.

## Value

A `resample_spec`. resolve_resampling() turns it into folds against the
points, and dsm_train() does that itself.

## Examples

``` r
spatial_cv(k = 5, block_size = "auto", buffer = "auto")
#> <resample_spec> spatial
#>   k               5
#>   block_size      auto
#>   buffer          auto
#>   test_frac       0.15
#>   max_share       0.1
#>   buffer_metric   chebyshev
#>   seed            42
```
