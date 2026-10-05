# Random k-fold over the train+validation pool.

Ignores geography by construction. On spatially clustered data that
makes the validation metric optimistic – the pipeline check measures it:
a random split put ~27% of validation in the SAME 250 m pixel as a
training point. Offered anyway, for two reasons: it is the right answer
when the rows really are independent, and running it against
spatial_folds() on the same data is the cleanest way to MEASURE how much
geography is worth here.

## Usage

``` r
random_folds(
  meta,
  k = 5L,
  test_frac = 0,
  test_ids = NULL,
  seed = 42L,
  group = "auto"
)
```

## Arguments

- meta:

  Patch store meta.

- k:

  Number of folds.

- test_frac:

  Share held out as the test set, drawn first.

- test_ids:

  Sample ids that ARE the test set, in place of `test_frac`: a test set
  drawn again on every run is not a test set.

- seed:

  Seed for the partition only (see with_local_seed): a fixed plan
  reproduces even when the training seeds change.

- group:

  "auto" keeps the rows of one profile_id together when the table
  repeats profiles; NULL or "row" makes each row its own unit; a column
  name, or a vector with one label per row, names the groups.

## Value

A `fold_plan` with `k` folds.

## Examples

``` r
ex <- example_landscape()
meta <- data.frame(sample_id = seq_len(nrow(ex$profiles)), x = ex$profiles$x,
                   y = ex$profiles$y)
random_folds(meta, k = 3, test_frac = 0.2)
#> <fold_plan> random_folds | 3 fold(s) | 160 rows in the store
#>   params: grouping=every row is its own unit (no profile_id column) | k=3 | test_frac=0.2 | n_test=32 | seed=42
#> # A tibble: 3 × 4
#>    fold n_train n_validation n_test
#>   <int>   <int>        <int>  <int>
#> 1     1      85           43     32
#> 2     2      85           43     32
#> 3     3      86           42     32
```
