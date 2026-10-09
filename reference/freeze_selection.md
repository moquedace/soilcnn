# Record which config was chosen, and when.

Writes the selection into the run directory so a later test-set report
can prove the choice preceded it. Call it at the moment the choice is
made.

## Usage

``` r
freeze_selection(
  run_dir,
  config_id,
  rule = "one_se",
  metric = "val_ccc",
  note = NA_character_
)
```

## Arguments

- run_dir:

  Tuning run directory (the one holding comparison/).

- config_id:

  The chosen config (or configs, one per family).

- rule:

  How it was chosen, e.g. "one_se" or "rank1".

- metric:

  The selection metric.

- note:

  Anything a reader would need to reconstruct the decision.

## Value

The selection record, invisibly.

## Examples

``` r
run_dir <- file.path(tempdir(), "tuning_example")
freeze_selection(run_dir, "cfg_002", note = "one_se on the cross-validation")
#> Selection frozen: cfg_002 (one_se on val_ccc) -> /tmp/RtmpcjA5yR/tuning_example/comparison/selection.rds
#>   NOTE: no git commit recorded -- /tmp/RtmpcjA5yR/tuning_example/comparison is not inside a git work tree.
#>   The record still fixes WHAT was chosen and WHEN; it cannot fix against which state of the code.
# the same choice again is accepted; another is refused
try(freeze_selection(run_dir, "cfg_005"))
#> Error : This run already has a frozen selection (cfg_002, 2026-10-09 13:43:43).
#>   Re-freezing a different config would destroy the ordering the file exists to prove.
#>   Delete /tmp/RtmpcjA5yR/tuning_example/comparison/selection.rds deliberately if the first selection was genuinely wrong, and record why in the log.
```
