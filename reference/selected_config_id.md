# The configuration a final run deployed.

Read from the list of configurations in the order they were chosen – not
from the grid's order, in which a runner-up can come first.

## Usage

``` r
selected_config_id(summary, label = "this final run")
```

## Arguments

- summary:

  the list read from comparison/final_run_summary.rds.

- label:

  what to call the run in messages.

## Value

the config id, or stop() – "auto" must never travel on unresolved.

## Examples

``` r
summary <- list(selected_config_ids = c("cfg_004", "cfg_011"))
selected_config_id(summary)   # the first chosen; the grid's order can differ
#> [1] "cfg_004"
```
