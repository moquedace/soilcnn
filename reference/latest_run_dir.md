# The most recent run under `base`, by time, and only if it finished.

WHY THIS IS NOT `sort(dirs, decreasing = TRUE)[1]`.

## Usage

``` r
latest_run_dir(
  base,
  prefix,
  require_file = NULL,
  require_pattern = NULL,
  label = "run",
  on_none = c("stop", "null")
)
```

## Arguments

- base:

  directory holding the run directories.

- prefix:

  run ids must start with this ("soc\_", "final\_"). "" accepts every
  directory.

- require_file:

  path INSIDE a run that only exists when it finished, e.g.
  "comparison/comparison_ranked.csv". A run without it is not a
  candidate, and its mtime is what "newest" is measured on. NULL accepts
  any directory, which is almost never what you want.

- require_pattern:

  alternative to require_file for runs whose completion is a FAMILY of
  files rather than one (05c's shard logs): a regex that at least one
  file directly inside the run must match. "Newest" is then the newest
  matching file, so a run still being written wins over an old one.

- label:

  what to call these runs in messages.

- on_none:

  what to do when no run qualifies. "stop" (default) for a stage that
  cannot proceed without one; "null" for a CHECK that must report
  "incomplete" and carry on – 99_check_pipeline.R does that, and a
  stop() there would turn a diagnosis into a crash. Returning NULL is
  loud only if the caller tests for it; every caller that passes "null"
  here prints a line saying so.

## Value

The run id (basename), or stop() / NULL when nothing qualifies.

## Details

That expression means "last alphabetically", which equals "most recent"
only while every run id is a timestamp sharing one prefix. Runs given
names broke it silently: `soc_0_5cm_design_spatial` sorts ahead of
`soc_0_5cm_20260916_232318` because 'd' \> '2', so "latest" started
resolving to a run that had died on its first unit, and stage 04 would
have refit the final model against it without a word.

It also never asked whether the run FINISHED. Stage 04 creates its
output directory before its own validations run, so a failure leaves a
`final_<timestamp>` behind that "latest" would then deploy – which is
what B2's review found from the other direction.

## Examples

``` r
base <- file.path(tempdir(), "runs")
for (id in c("run_a", "run_b")) {
  dir.create(file.path(base, id, "comparison"), recursive = TRUE, showWarnings = FALSE)
  writeLines("done", file.path(base, id, "comparison", "comparison_ranked.csv"))
}
Sys.setFileTime(file.path(base, "run_a", "comparison", "comparison_ranked.csv"),
                Sys.time() - 3600)
dir.create(file.path(base, "run_c"), showWarnings = FALSE)   # started, never finished
latest_run_dir(base, "run_", require_file = "comparison/comparison_ranked.csv")
#> run resolved to: run_b  (newest of 2 finished, 2026-10-05 04:46)
#>   skipped as unfinished: run_c
#> [1] "run_b"
```
