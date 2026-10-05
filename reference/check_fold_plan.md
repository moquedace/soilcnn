# Check a plan's folds are well formed, and describe them.

Run this before any training: a broken plan caught here costs a second,
caught later it costs the whole run. Checks that, within each fold,
train and validation are disjoint; that test (when present) appears in
neither; and that across the folds every pooled row is validated exactly
once – the property that makes the k metrics a partition of the pool
rather than an arbitrary set of overlapping subsets.

## Usage

``` r
check_fold_plan(plan, meta = NULL, group = "auto")
```

## Arguments

- plan:

  A `fold_plan`.

- meta:

  The point table the plan was made for. Given, the grouping is proven
  against it: no group may be split across train and validation.

- group:

  How the rows group, as in holdout().

## Value

tibble, one row per fold.

## Examples

``` r
ex <- example_landscape()
meta <- data.frame(sample_id = seq_len(nrow(ex$profiles)), x = ex$profiles$x,
                   y = ex$profiles$y)
plan <- spatial_folds(meta, k = 3, test_frac = 0.2, block_size = 0.05)
check_fold_plan(plan, meta)
#> # A tibble: 3 × 4
#>    fold n_train n_validation n_test
#>   <int>   <int>        <int>  <int>
#> 1     1      78           37     45
#> 2     2      76           39     45
#> 3     3      76           39     45
```
