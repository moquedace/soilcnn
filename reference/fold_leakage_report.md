# How much does each fold still leak?

Reports, per fold, the same quantities the pipeline check reports for
the fixed split: the share of validation points sharing a raster cell
with a training point, and sharing patch pixels at each window.
Deliberately the SAME criterion, so a number here can be compared with a
number there rather than being a second, incompatible notion of leakage.

## Usage

``` r
fold_leakage_report(plan, meta, cell_size, windows = c(3L, 9L, 15L))
```

## Arguments

- plan:

  A fold_plan.

- meta:

  Patch store meta with x/y.

- cell_size:

  Raster resolution, in the units of x/y.

- windows:

  Window sizes to report overlap for.

## Value

A tibble with a `fold` column: the overlap shares per role and window,
fold by fold.

## Examples

``` r
ex <- example_landscape()
meta <- data.frame(sample_id = seq_len(nrow(ex$profiles)), x = ex$profiles$x,
                   y = ex$profiles$y)
blocks   <- spatial_folds(meta, k = 3, block_size = 0.05)
buffered <- spatial_folds(meta, k = 3, block_size = 0.05, buffer = 7 * 0.0025)
fold_leakage_report(blocks, meta, cell_size = 0.0025, windows = c(3, 7))
#> # A tibble: 9 × 8
#>    fold split      criterion                  window matters     n   pct n_split
#>   <int> <chr>      <chr>                       <dbl> <lgl>   <int> <dbl>   <int>
#> 1     1 validation identical patch (same ras…     NA TRUE        0  0         55
#> 2     1 validation shares pixels (3x3)             3 FALSE       1  1.82      55
#> 3     1 validation shares pixels (7x7)             7 FALSE      22 40         55
#> 4     2 validation identical patch (same ras…     NA TRUE        0  0         53
#> 5     2 validation shares pixels (3x3)             3 FALSE      15 28.3       53
#> 6     2 validation shares pixels (7x7)             7 FALSE      32 60.4       53
#> 7     3 validation identical patch (same ras…     NA TRUE        0  0         52
#> 8     3 validation shares pixels (3x3)             3 FALSE      12 23.1       52
#> 9     3 validation shares pixels (7x7)             7 FALSE      20 38.5       52
fold_leakage_report(buffered, meta, cell_size = 0.0025, windows = c(3, 7))
#> # A tibble: 9 × 8
#>    fold split      criterion                  window matters     n   pct n_split
#>   <int> <chr>      <chr>                       <dbl> <lgl>   <int> <dbl>   <int>
#> 1     1 validation identical patch (same ras…     NA TRUE        0     0      55
#> 2     1 validation shares pixels (3x3)             3 FALSE       0     0      55
#> 3     1 validation shares pixels (7x7)             7 FALSE       0     0      55
#> 4     2 validation identical patch (same ras…     NA TRUE        0     0      53
#> 5     2 validation shares pixels (3x3)             3 FALSE       0     0      53
#> 6     2 validation shares pixels (7x7)             7 FALSE       0     0      53
#> 7     3 validation identical patch (same ras…     NA TRUE        0     0      52
#> 8     3 validation shares pixels (3x3)             3 FALSE       0     0      52
#> 9     3 validation shares pixels (7x7)             7 FALSE       0     0      52
```
