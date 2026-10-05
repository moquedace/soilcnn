# Fold by a grouping column: no group is split across folds.

For when the unit that must not leak is named rather than geometric – a
survey campaign, a country, a laboratory, a soil map unit. With k =
number of groups this is leave-one-group-out.

## Usage

``` r
region_folds(meta, group, k = NULL, test_frac = 0, test_ids = NULL, seed = 42L)
```

## Arguments

- meta:

  Patch store meta.

- group:

  Group labels, one per row of `meta`.

- k:

  Number of folds; defaults to one per group.

- test_frac:

  Share held out as the test set, drawn first.

- test_ids:

  Sample ids that ARE the test set, in place of `test_frac`: a test set
  drawn again on every run is not a test set.

- seed:

  Seed for this split only; the training seeds do not move it.

## Value

A `fold_plan`.

## Examples

``` r
ex <- example_landscape()
meta <- data.frame(sample_id = seq_len(nrow(ex$profiles)), x = ex$profiles$x,
                   y = ex$profiles$y)
region_folds(meta, group = ex$profiles$survey, k = 3)
#> <fold_plan> region_folds | 3 fold(s) | 160 rows in the store
#>   params: n_groups=7 | k=3 | test_frac=0 | n_test=0 | seed=42
#> # A tibble: 3 × 4
#>    fold n_train n_validation n_test
#>   <int>   <int>        <int>  <int>
#> 1     1      90           70      0
#> 2     2     115           45      0
#> 3     3     115           45      0
```
