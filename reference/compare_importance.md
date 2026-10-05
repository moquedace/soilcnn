# Several importances, side by side.

Aligns importances computed by different methods, variants, rows or runs
– permutation over all rows against within blocks, the test set against
the folds, the spatial design against the random one – by what each
perturbed (the variable, for a permutation), and says how far their
rankings agree. Where they disagree is the reading: a variable that
keeps its importance within blocks discriminates locally; one that loses
it carries a regional gradient.

## Usage

``` r
compare_importance(..., n = 20L)
```

## Arguments

- ...:

  Two or more `dsm_importance`, named or not. Names become the labels;
  unnamed ones are labelled by run, rows and method.

- n:

  How many variables the print shows.

## Value

An `importance_comparison`: `table` (one row per target: each
importance, its rank, and its share of that importance's largest),
`agreement` (Spearman's correlation between the rankings, over the
variables they share), `labels`, and for two SHAP importances of the
same points `points_agreement` (per variable, the correlation of their
values point by point and their mean difference against the second's
size).

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()    # a small fitted run, made once a session
perm <- dsm_importance(run$final, run$data, permutation_importance(draws = 2),
                       verbose = FALSE)
shap <- dsm_importance(run$final, run$data, shap_importance(samples = 20, background = 20),
                       verbose = FALSE)
compare_importance(permutation = perm, shap = shap)
# }
}
```
