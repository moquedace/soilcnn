# B2 — two configs in stage 04: the decision that blocks it

**Status 2026-09-28:** decided and implemented. Options 3 and 6 were taken on
2026-09-18 (`project_log.md`, "Options 3 and 6 of docs/b2_two_configs_decision.md,
implemented"), and `_b2_two_config_check.R` passed 13/13. Stage 04 has since
become `dsm_final()`. Kept as the record of the decision; the paragraph below
is its status when it was written.

**Status (2026-09-18):** blocked on one decision, which belongs to you. Everything below is
verified against the code and the files on disk as of 2026-09-18. Nothing here
was run; no R was executed to write it.

**Revised 2026-09-18 after review.** Four things in the first draft were wrong
and are corrected here, because a reader who saw the first version should know
which parts moved: (i) Option 5 was built on `one_se()` returning two configs,
which it cannot do at any `tune_length` — see §1c, and Option 5 is rewritten;
(ii) Option 3's snippet was given the wrong insertion point and a seeds parse
that fails at `set.seed()`; (iii) the driver in Option 6 was a bare `source()`,
which `rm(list = ls())` destroys mid-run; (iv) the headline cost was "about 12
minutes" against the note's own table, which gives ~18.

B2 in [`test_plan.md`](test_plan.md) is: *"Two configs in stage 04 — exercises
`paired_by_seed.csv`, a branch that has never run. Cost: 2 × 10 seeds."*

The branch is real and it has never run. But the cost line is wrong, the
obstacle is not the one the plan implies, and the decision is not "which second
config" — it is **whether a two-config stage-04 run on the 2026-09-16 tuning run
is a model selection or a capability test**. Those two things want opposite
plumbing, and only you can say which one it is.

---

## 1. What the code actually does

### 1a. `freeze_selection()` already accepts a set. It refuses to *change* one.

`R/test_optimism.R:47-84`. The guard is one comparison:

```r
if (!identical(sort(old$config_id), sort(config_id))) stop(...)
```

Three consequences, and the first is the one that matters:

- **A set of configs is native and always has been.** `config_id` is a character
  vector, `stopifnot(length(config_id) >= 1L)`, and the comparison sorts both
  sides. `tests/test_selection_order.R:99-109` asserts exactly this:
  `freeze_selection(run_dir2, c("cfg_002", "rf_001"))` succeeds, and freezing the
  same pair in the other order is idempotent. The header comment on the `note`
  parameter says "the chosen config (**or configs, one per family**)". So the
  option "extend `freeze_selection()` to record a SET rather than one" is already
  implemented, and there is nothing to build.
- **Re-freezing the identical set is silent and does not move the clock**
  (`return(invisible(old))`, asserted at `tests/test_selection_order.R:87-90`).
  Re-running stage 04 unchanged is free.
- **Widening a frozen record is refused, not merged.** `sort("cfg_003")` against
  `sort(c("cfg_003", "cfg_001"))` is not `identical`, so the call `stop()`s.
  There is no "add a config" path and no `force` argument.

This tuning run is frozen. From
`outputs/tuning/soc_stock_modeling/soc_stock_0_5cm/soc_0_5cm_20260916_232318/comparison/selection.rds`:

| field | value |
|---|---|
| `config_id` | `cfg_003` |
| `rule` | `one_se` |
| `metric` | `val_ccc` |
| `note` | `stage 04 on 2026-09-17 10:31:26` |
| `git_commit` | `0eef391` |

So: running
`D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/04_final_model.R`
with two configs against this tuning run **dies at line 229**, before `dsm_load()`,
before a single tensor is built, with the message that names `cfg_003` and tells
you to delete the file deliberately. It costs seconds, not a morning — but it
costs the run.

### 1b. The override routes that exist, and the one that does not

`04_final_model.R:20` does `rm(list = ls())`, so a variable set before the
`source()` is gone. The repo solved this problem already, twice, and the pattern
is documented in `05_predict_spatial.R:62-64`:

> *The `rm(list=ls())` above erases any variable set before the `source()`, so an
> environment variable is the only way to pass a parameter into a `source()`d
> run: `Sys.setenv()` survives `rm(list=ls())`, a workspace object does not.*

Verified inventory of override routes in the example scripts:

| script | `commandArgs()` | `Sys.getenv()` |
|---|---|---|
| `05_predict_spatial.R` | yes, 5 positional args (lines 71-78) | yes — `soc_row_shard_id`, `soc_col_shard_id`, `soc_n_row_shards`, `soc_n_col_shards`, `soc_max_concurrent`, `soc_predict_raster_dir` |
| `05a_run_parallel.R` | — | yes, and it forwards `soc_predict_raster_dir` to its workers explicitly (lines 217-218) |
| `07_area_of_applicability.R` | — | yes — `soc_predict_raster_dir` |
| `_b4_shard_merge_check.R` | — | it is the *caller*: `Sys.setenv()` at line 154, `Sys.unsetenv()` at line 176 |
| **`04_final_model.R`** | **none** | **none** |

**Stage 04 has no override route at all.** Today the only way to change
`selected_config_ids` is to edit line 55 of the script. The repo has already
written down why that is bad, in `_b4_shard_merge_check.R:150-153`:

> *Before that fix this step needed a patched copy of 05a — and a test that
> requires editing the script it tests is a test that runs once.*

Note what this means for B2: **the `rm(list = ls())` is not the blocker.** Adding
env vars to stage 04 is a couple of dozen lines (Option 3 — most of the count is
the seed coercion and the comments) and solves obstacle (b) completely — and
changes nothing about obstacle (a), because `freeze_selection()` refuses the same
way whether the list came from line 55 or from the environment. Anyone who
reaches for the env var first has solved the easy half and will hit the wall at
line 229 anyway.

### 1c. Two facts that reshape the question — one about this run, one about the code

**The grid has already been scored on the test set.** `A8` of the capability
sweep ran on 2026-09-17 21:30 and wrote
`.../soc_0_5cm_20260916_232318/comparison/test_optimism_by_config.csv`. The order
held — frozen 10:31, scored 21:30 — and the result is published in
`docs/reference_performance.md:439`. But the numbers are now on disk and in a
document:

| config | window | n_params | val_ccc (mean ± sd) | val_ccc_se | test_ccc (mean) | test rank |
|---|---|---|---|---|---|---|
| `cfg_003` | 15 | 340,225 | 0.4726 ± 0.0313 | 0.01042 | **0.4651** | 1 |
| `cfg_001` | 9×15 | 1,891,217 | 0.4490 ± 0.0275 | 0.00916 | 0.4309 | 2 |
| `cfg_002` | 3×9 | 2,425,217 | 0.4388 ± 0.0455 | 0.01515 | 0.4300 | 3 |

Selection optimism: **+0.0000 CCC.** Validation and test agree on the full
ordering.

The consequence is blunt and it is not hypothetical: **any second config named
today, for this tuning run, is named by someone who has already seen its test
score.** No option below can undo that. What the options differ on is whether the
written record says so.

**No automatic path in stage 04 can name two configs — on this run or on any
other.** This is stronger than "the rule happens not to tie here", and the whole
decision turns on it, so it is worth writing out in full.

- On this run the one-SE band is `mu[best] - se[best]` = 0.472620 − 0.010422 =
  **0.462198**; `cfg_001` (0.448951) and `cfg_002` (0.438788) are both outside
  it, so `within_one_se = 1` and `simpler_than_best = FALSE`.
- **But a tie would not have widened the answer either.**
  `R/resample.R:1321-1325` is `cand <- which(within & is.finite(cx))`, then
  `pick <- cand[which.min(cx[cand])]`, then
  `out <- by_config[pick, , drop = FALSE]`. `which.min()` returns a single
  index, so `pick` has length 1 and `out` is one row — which is exactly what the
  function's own header promises at `R/resample.R:1278`: *"@return One row of
  `by_config`"*. `within_one_se` is an **attribute** recording how many configs
  sat inside the band; it is not what comes back. `04_final_model.R:181` then
  takes `pick$config_id`, which is therefore length 1. This holds at
  `tune_length = 3`, at 24, and at any other length: a larger grid changes
  *which* config wins, not *how many* are returned.
- **Both fallbacks are single too.** `by_config`'s rank is
  `agg$rank <- seq_len(nrow(agg))` (`R/resample.R:1180`), so
  `04_final_model.R:193` matches exactly one row. `comparison_ranked.csv`'s rank
  is `cumsum(status == "success")` (`R/train_cnn.R:844`), so
  `04_final_model.R:195` matches exactly one row as well.

**Consequence, and it is the load-bearing fact of this note: a
`selected_config_ids` of length 2 can only ever be typed by a human** — at
`04_final_model.R:55`, or through an override. The `length(selected_config_ids)
== 2` gate at `04_final_model.R:669` fires for no other reason. That is not a
weakness in the argument for B2; it is what makes §4's question — *where that
hand-written record gets written* — the entire decision rather than a detail of
it. A pair has to be picked by hand, which is precisely the act
`freeze_selection()` exists to date-stamp.

### 1d. Two things any two-config run must handle, whichever option you pick

Neither is a reason to do or not do B2. Both are defects B2 would walk into.

- **`04_final_model.R:669` gates on `length(selected_config_ids) == 2`, exactly.**
  Three configs writes no `paired_by_seed.csv` and prints nothing about it. By
  this project's own standing rule that is a quiet skip, and it should `stop()`
  or at minimum say what it did not do. B2 with two configs never triggers it,
  but the next person with three will.
- **Stage 05 would map the wrong model.** `05_predict_spatial.R` resolves
  `config_id = "auto"` as `readRDS(final_run_summary.rds)$selected_cfgs$config_id[1]`
  — the first row of `dplyr::filter(tune_grid_full, config_id %in% selected_config_ids)`,
  which is **tune_grid order, not the order you typed and not the winner**. A
  final run holding `cfg_003` and `cfg_001` resolves "auto" to `cfg_001`. And
  `run_id <- paste0("final_", ...)` at `04_final_model.R:131` is not overridable,
  so a throwaway two-config run becomes `final_<timestamp>` and therefore becomes
  `"latest"` for both `05_predict_spatial.R:latest` and
  `07_area_of_applicability.R:84-92`. A B2 run left on disk silently redirects the
  next map and the next AOA to a config nobody selected.

### 1e. What B2 actually costs

The plan says "2 × 10 seeds". Measured, from
`.../final_model/.../final_20260917_234453/comparison/all_seed_results_test.csv`
and the tuning run's `comparison_ranked.csv`:

| | tuning, per unit (500 ep / patience 60) | stage 04, per seed (700 ep / patience 100) |
|---|---|---|
| `cfg_003` | 1.76 min (best_epoch 31.0) | **3.27 min measured**, 32.7 min for 10 seeds |
| `cfg_002` | 1.42 min (best_epoch 16.7) | ~2.6 min projected → **~26 min** for 10 seeds |
| `cfg_001` | 5.83 min (best_epoch 74.9) | ~11 min projected → **~1.8 h**, plausibly 2.5-3 h |

The stage-04 multiplier is 3.27 / 1.76 = **1.86×** (more epochs, more patience,
and the refit trains on ~85% of the non-test rows rather than a CV fold's share).
`cfg_001`'s projection is the least trustworthy of the three: at patience 60 it
stopped at epoch 75 and had not settled, so patience 100 and a 700-epoch ceiling
may let it run considerably longer.

Three things the plan's "2 × 10" hides:

1. **Stage 04 has no resume for the seed loop.** `cfg_003`'s 33 minutes are paid
   again; they are not free just because that run already exists.
2. **The cost depends entirely on which second config.** `cfg_003` + `cfg_002` is
   about **1 hour**. `cfg_003` + `cfg_001` is **2.3 to 3.5 hours**. A factor of
   three, from a choice the plan does not mention.
3. **The seed count is not load-bearing for B2.** `paired_by_seed.csv` is a
   `pivot_wider` over `seed` with four `mutate`d differences. The branch is
   byte-identical at 3 seeds and at 10. What 10 seeds buy is statistical power in
   the paired comparison — which is a *scientific* quantity, not the capability
   under test. At 3 seeds, `cfg_003` + `cfg_002` is 3 × 3.27 + 3 × 2.64 =
   **about 18 minutes of fitting, plus a few minutes of patch loading and cache
   building** that every run pays regardless of seed count.

---

## 2. The options

Costed against the guarantee that matters. That guarantee, stated plainly:

> **Nobody chose the final model after looking at the test set, and the record on
> disk proves the order to a third party who was not in the room.**

### Option 1 — let `freeze_selection()` grow a frozen set

Relax the guard from `identical(sort(old), sort(new))` to something like
`all(old$config_id %in% config_id)`, so a frozen `cfg_003` may be widened to
`c("cfg_003", "cfg_001")`.

- **CPU:** none.
- **Code:** about five lines in `R/test_optimism.R`, plus a new assertion in
  `tests/test_selection_order.R`.
- **Risk:** it is the cheapest option and the only one that is *wrong*.
- **Guarantee: destroyed, and quietly.** The refused sequence is score → dislike
  the answer → re-freeze → re-score. Widening is that sequence with an extra
  member: score the grid, see that `cfg_001` came second on test, add it, report
  "we compared the top two". The record would show a set frozen at 10:31 that did
  not exist at 10:31, and the timestamp — the entire point of the file — would be
  a lie by omission. `identical(sort(...), sort(...))` is not an accident of
  implementation; the sort is there specifically so that *membership* is the
  identity of a frozen selection, and this change makes membership mutable.

**Discard.** Note that the brief's phrasing — "extend `freeze_selection()` to
record a SET of configs rather than one" — describes something already true
(§1a). The only version of this option that is *not* already implemented is the
one that erodes the guarantee, and that is not a coincidence.

### Option 2 — a separate entry point for multi-config final fits

A `_b2_two_config_check.R` in `examples/soc_stock_0_5cm/`, following the
checking-script convention (`_b4_shard_merge_check.R`, `_capability_sweep.R`),
that does the two-config fit without going through stage 04's selection logic.

- **CPU:** whatever the fit costs (1 h to 3.5 h at 10 seeds; ~18 min at 3).
- **Code:** expensive if it duplicates `train_config_all_seeds()` — that is
  roughly 290 lines of `04_final_model.R`, including the conformal block and the
  smearing block.
- **Risk: a fork, of exactly the shape this project has been burned by twice.**
  The scaling table drifting from the weights (`04_final_model.R:335-344`) and the
  global-vs-per-fold constants are both "a second copy of the same fact, free to
  drift". A second final-fit script is a third copy. Worse, it would not exercise
  the branch B2 is about: `paired_by_seed.csv` is written by
  `04_final_model.R:669-692`, so a script that reimplements the fit tests its own
  copy of the branch and leaves the real one untested. **That is B2 failing while
  reporting PASS.**
- **Guarantee: intact.** It never touches `selection.rds`.

**Discard in the reimplementing form.** It survives only as a *thin* script that
sets parameters and *drives* the real stage 04 in a separate R process — which is
Option 3 plus a caller, and is exactly how `_b4_shard_merge_check.R` is built
(set env, run the real 05a via `processx::run()`, `stopifnot()` on the status,
unset). See Option 6 for why the driver must be `processx::run()` and not a bare
`source()`.

### Option 3 — environment-variable override in stage 04

**Where it goes, precisely: after `04_final_model.R:97`** — the closing `)` of
`training_args` — **and before the `# ── Paths` header at `:99`.**

Not after the `source(load_all.R)` at `:29`. All three variables being
overridden are assigned *after* that line: `tuning_run_id` at `:38`,
`selected_config_ids` at `:55`, `seeds` at `:76`. A block inserted at `:30` is
silently overwritten by the settings section ten lines later, and
`Sys.setenv(soc_final_config_ids = "cfg_003,cfg_002")` would have no effect at
all: the run proceeds on the `one_se` pick, `freeze_selection()` succeeds
because `cfg_003` is already frozen, and the B2 run finishes producing no
`paired_by_seed.csv` and no error — this project's own quiet-skip failure mode,
in the script whose output nobody would re-check. Line 97 is also *before* the
only early consumer, the `"latest"` resolution at `:112-118`. Defaults first,
overrides after, which is the order `05_predict_spatial.R:65-86` uses.

```r
# -- Overrides for a driven run (_b2_two_config_check.R) ----------------------
# Defaults above stand when these are unset, which is every interactive run.
# Sys.setenv() survives the rm(list = ls()) at line 20; a workspace object does
# not. Placed here, AFTER the settings block, because an override assigned
# before the defaults is overwritten by them and fails silently.

if (nzchar(Sys.getenv("soc_tuning_run_id"))) {
  tuning_run_id <- Sys.getenv("soc_tuning_run_id")   # already a string
}

if (nzchar(Sys.getenv("soc_final_config_ids"))) {
  selected_config_ids <- trimws(strsplit(Sys.getenv("soc_final_config_ids"),
                                         ",", fixed = TRUE)[[1]])
  # The frozen record must say how the choice was MADE, not only what it was.
  # See the risk list below: this is what freeze_selection() stores as `rule`.
  selection_rule <- "manual"
}

if (nzchar(Sys.getenv("soc_final_seeds"))) {
  # as.integer() is NOT optional. strsplit() yields character, and a character
  # seed survives all the way to the first training call before it dies:
  # set.seed(seed_val) at :354 with "supplied seed is not a valid integer", or
  # sprintf("seed%04d", seed_val) at :374 with "invalid format '%04d'; use
  # format %s for character objects" -- several minutes in, after dsm_load()
  # and the fold cache have already been paid.
  seeds <- as.integer(trimws(strsplit(Sys.getenv("soc_final_seeds"),
                                      ",", fixed = TRUE)[[1]]))
  # And fail HERE if it did not parse. as.integer("7a") is NA with a warning,
  # and an NA seed reaches set.seed() as a different error 400 lines away.
  if (length(seeds) == 0L || anyNA(seeds)) {
    stop("soc_final_seeds did not parse to integers: '",
         Sys.getenv("soc_final_seeds"), "'", call. = FALSE)
  }
}
```

- **CPU:** none.
- **Code:** about 30 lines for three variables, plus the `Sys.unsetenv()`
  discipline in whatever calls it.
- **Risk:** the one `_b4_shard_merge_check.R:169-176` documents — an env var
  survives into the *next* `source()` in the same session, so a later real run
  silently takes the check's parameters. `_b4` handles it by unsetting
  immediately after the call returns, and the caller here must do the same. Also:
  the choice then lives in shell history rather than in git, so
  `freeze_selection()`'s `note` should record where the list came from
  (`"stage 04 on <time>, config ids from soc_final_config_ids"`), otherwise the
  audit trail loses the only thing it was collecting.
- **Risk: without the `selection_rule <- "manual"` line, the frozen record
  misstates *how* the pair was chosen.** `04_final_model.R:230` passes
  `rule = if (is.null(selection_rule)) "manual" else selection_rule`, and
  `selection_rule` is hardcoded to `"one_se"` at `:66` — it is never `NULL`, so
  **the `"manual"` branch is unreachable**. A pair supplied by the env var, or
  by editing line 55, is frozen into `selection.rds` as
  `rule = "one_se", config_id = c("cfg_003", "cfg_002")`. Per §1c, `one_se()`
  provably cannot return a pair, so that record is self-contradictory on its
  face — to precisely the third party the guarantee is aimed at. It is the same
  defect the script already names in its own words at `:186-191`: *a report
  describing a selection that never happened.* Hence the `selection_rule <-
  "manual"` assignment in the snippet above, which must travel with the list
  wherever the list is overridden. **The unreachable branch at `:230` is a
  pre-existing defect** — it fires today for anyone who edits line 55, with no
  B2 involved — and should be fixed at the source as well, so that a
  hand-written `selected_config_ids` and `rule = "manual"` cannot come apart.
- **Guarantee: neutral on its own once `selection_rule` travels with the list,
  and it does not unblock B2.** This is the point
  most likely to be got wrong. The env var changes *how the list arrives*, not
  *what is allowed to be frozen*. Against the 2026-09-16 run it still dies at line
  229. Its real value is that stage 04 stops being the one script in the example
  directory that can only be driven by editing it — which is worth doing whether
  or not B2 ever runs.

**Do this regardless, but do not mistake it for the answer to B2.**

### Option 4 — do nothing

- **CPU:** zero.
- **Code:** zero.
- **What is lost:** `04_final_model.R:669-692` — 24 lines — never runs. Its
  plausible silent failures are narrow: `pivot_wider(names_from = config_id,
  values_from = c(ccc, mae, rmse, mqi))` produces `ccc_cfg_003` and the code reads
  `paste0("ccc_", c1)`, which agrees by construction; a seed that errored out for
  one config yields `NA` in its cell and `mean(..., na.rm = TRUE)` absorbs it
  without saying so, which is a genuine but minor hole. The sign convention is
  documented in the printed message. **This is not where a silent wrong answer is
  waiting.** Compare `rf_grid()` drawing the same config four times, or the buffer
  protecting validation and leaving the test exposed: those were defects in
  *which rows go where*. This branch reshapes a table that is already correct.
- **Risk:** it violates the project's standing rule — *"every capability that has
  not run on real data is a claim, not a fact"* — and the rule has been right three
  times.
- **Guarantee: intact, trivially.**

**There is one more argument for doing nothing that deserves to be said out loud,
because it cuts the other way from everything else in this note.**
`paired_by_seed.csv` is built from `all_seed_results`, which is
`filter(result$perf_all, dataset_role == "test")` (`04_final_model.R:394`). The
file is a **paired comparison of two configs on the test set**. That is a
legitimate thing to *report* once the selection is locked — it is the same logic
that makes `score_test_grid()` safe — and an illegitimate thing to *act on*. So
the capability is real, but the artefact it produces is one whose only tempting
use is forbidden. Anyone who runs B2 should know that before they read the output,
and if the file is ever produced it should carry that warning in the message
beside it.

**Discard as a final answer, but it is the correct answer if the cost is 2 × 10
seeds of `cfg_001`.** Three hours to exercise 24 lines of `pivot_wider` is not a
trade this project should make.

### Option 5 — B2 by hand on the `tune_length = 24` science run

**This option is not "wait for `one_se()` to return a set". It never will.**
§1c: every automatic path in stage 04 returns exactly one config *by
construction* (`R/resample.R:1278, 1321-1325`; `R/resample.R:1180`;
`R/train_cnn.R:844`). A 24-config grid changes which config wins, not how many
come back, so `length(selected_config_ids) == 2` at `04_final_model.R:669` is
`FALSE` on a 24-config run exactly as it is on a 3-config one. An earlier draft
of this note claimed the opposite, and it was the one place the note failed to
apply its own standard to itself.

What survives is the *other* half of the argument, and it is a good one: **on a
fresh 24-config run, no test score exists yet.** A pair listed by hand there is
chosen on validation alone, so the record needs no apology. That is the same
honesty Option 6 buys with a throwaway copy, bought instead by ordering.

So Option 5, stated as what it actually is: run `03_run_tuning.R` at
`tune_length = 24`, read `comparison/comparison_by_config.csv`, list two
`config_id`s at `04_final_model.R:55` (or via Option 3's env var, with
`selection_rule <- "manual"`) **before anything is scored on the test set**, and
let that stage-04 run produce `paired_by_seed.csv`.

- **CPU: a full extra 10-seed fit, not zero.** Stage 04 loops
  `for (i in seq_len(nrow(selected_cfgs)))` at `:625`, so the second config is
  trained with every seed exactly like the first — it is not a by-product of
  seeds that were being spent anyway. On the current grid that would be ~26 min
  (`cfg_002`) to ~1.8 h (`cfg_001`) on top of the run; on a 24-config grid it is
  whatever the second-named config costs, which is unknown until the grid runs.
- **Code:** zero if the pair is typed at line 55; Option 3's lines if it is not.
- **Risk:** it is a hand-pick, so it carries the `rule = "one_se"` record defect
  like every other hand-pick (see Option 3's risk list) and needs the same
  `selection_rule <- "manual"` fix. It inverts the plan's stated order ("the
  `tune_length = 24` science run comes after these are clean"). And it spends
  real seeds on a capability test inside a science run — the conflation §3 warns
  against, in the milder direction.
- **Risk, the one that matters: it only happens if somebody does it.** Nothing
  fires on its own. If the 24-config run is launched with
  `selected_config_ids <- NULL`, the `== 2` gate is `FALSE`, nothing is written,
  and — because there is no `else` branch — nothing says so. Deferring B2 to
  Option 5 without committing to the hand-pick is deferring it indefinitely,
  through exactly the silent-skip shape this project has been burned by three
  times.
- **Mitigating fact, verified:** the blast radius of a failure in that branch is
  one CSV. `04_final_model.R` writes `all_seed_results_test.csv` (652),
  `config_summary_test.csv` (653) and `final_run_summary.rds` (655-661) **before**
  the paired block at 669. A crash there loses the paired table and the closing
  report, and nothing that was expensive to compute. So deferring B2 into the
  science run risks a cosmetic failure at the end of it, not a lost night.
- **Guarantee: intact, for a different reason than Option 6.** The pair is
  hand-picked either way — no rule produces one — but on a fresh 24-config run
  nobody has seen a test score when the list is typed, so the record is clean
  provided it says `rule = "manual"`. Option 6 reaches the same honesty by
  letting its record die with the copy.

**Keep as the fallback if Option 6 is refused — but keep it knowing it is a
deliberate act that costs a second full seed loop, not something that falls out
of the science run for free.**

### Option 6 — a 3-seed pair against a non-frozen copy of the tuning run

Not in the brief; it is what falls out of §1e.3 and §1a. `04_final_model.R` reads
exactly six things from `tuning_dir` — `comparison/comparison_ranked.csv` (121),
`comparison/comparison_by_config.csv` (128), `tune_grid.rds` (129),
`fold_plan.rds` (304), and `predictions/` twice, via `cv_residuals()` (456) and
`smearing_from_run()` (581). **It never reads `models/`.** That is 161 MB of the
run's 182 MB. A copy that omits `models/` and omits `comparison/selection.rds` is
about **20 MB**, has no frozen selection, and drives a genuine stage-04 run.

So: copy the run, set `soc_tuning_run_id` / `soc_final_config_ids` /
`soc_final_seeds` (Option 3), run the real `04_final_model.R` **in a separate R
process**, assert on `paired_by_seed.csv`, unset, delete.

**Separate process, not `source()`.** This matters enough to be part of the
contract rather than a note. Plain `source()` defaults to `local = FALSE`, so it
evaluates in the global environment — and `04_final_model.R:20` is
`rm(list = ls())`. The check script's own variables (the copy directory, the
expected output paths, everything it needs to assert on and to delete) are
destroyed mid-`source()`, and every line after it fails with *object not found*
— **after** the 18-minute fit, with the copy and the `final_<timestamp>`
directory both still on disk and the cleanup unreachable. That leftover is
exactly the one §1d says redirects the next map and the next AOA. The repo has
already settled this: `_b4_shard_merge_check.R:160-167` drives 05a with
`processx::run(rscript_bin, args = ..., error_on_status = FALSE)` followed by
`stopifnot(a_res$status == 0L)`, and the one `source()` in `_b4` (line 317, on
`05c_estimate_eta.R`) is of a script that has **no** `rm(list = ls())` and is
*still* isolated with `local = new.env()`.

```r
rscript_bin <- file.path(R.home("bin"), "Rscript.exe")   # _b4:35
stopifnot(file.exists(rscript_bin))
res <- processx::run(rscript_bin,
                     args = file.path(script_dir, "04_final_model.R"),
                     stdout = "|", stderr = "|", echo = TRUE,
                     error_on_status = FALSE)
stopifnot(res$status == 0L)
```

If an in-process run is ever wanted instead, it must be
`source(..., local = new.env())` — never bare `source()`. Either way the
assertions and the cleanup must live somewhere that survives the run, and
failing to delete the copy or the `final_<timestamp>` directory is a `stop()`,
not a `message()`.

- **CPU:** `cfg_003` + `cfg_002` at 3 seeds ≈ **18 minutes of fitting**
  (3 × 3.27 + 3 × 2.64), plus a few minutes of `dsm_load()` and fold-cache
  building that any run pays. Both are the cheap configs; `cfg_001` at
  5.83 min/unit is the one to avoid.
- **Code:** Option 3's 15 lines, plus a `_b2_two_config_check.R` of maybe 80
  lines that is a *caller*, not a fork — the `_b4_shard_merge_check.R` shape.
- **Risks, all of them real and all of them nameable:**
  - **The copy must not be called `soc_*`.** `04_final_model.R:112-116` resolves
    `"latest"` by `grepl("^soc_", run_dirs)` then `sort(decreasing = TRUE)[1]`, and
    `soc_0_5cm_20260916_232318_b2check` sorts *above* the real run. Name it
    `b2_copy_soc_0_5cm_20260916_232318` and the `^soc_` filter excludes it from
    `"latest"` while an explicit `tuning_run_id` still finds it (the filter only
    applies inside the `"latest"` branch).
  - **The run it produces is still `final_<timestamp>`** (`04_final_model.R:131`,
    not overridable) and therefore becomes `"latest"` for stages 05 and 07, which
    would then map `cfg_001`-or-whichever-is-first via `config_id = "auto"` (§1d).
    The check script must delete its own output directory, and the check should
    *assert* that it did.
  - A copied run directory is a second copy of a run, which is the drift pattern.
    It is defensible only because it is created, used and deleted inside one
    script, and the note it writes says so.
- **Guarantee: intact, and it should be made loud rather than merely true.** The
  check freezes its own selection *in the copy*, so the copy's `selection.rds`
  records a two-config choice made on 2026-09-18 by someone who had seen the test
  scores — which is honest, because that record dies with the copy and the real
  run's record is untouched. The `note` should say `"b2 capability test, not a
  model selection"` in as many words, and — per Option 3's risk list — the
  `rule` field must read `"manual"`, not `"one_se"`. A record saying
  `rule = one_se, config_id = c(cfg_003, cfg_002)` is one `one_se()` cannot have
  produced (§1c); leaving it that way would put a false account of the
  *mechanism* inside the very file whose job is to prove the mechanism, even
  though the file is about to be deleted. The check should assert on both
  fields, because that assertion is the cheapest place the `:230` defect ever
  gets caught.

---

## 3. Recommendation

**Do Option 3 now, and Option 6 as the B2 run, at 3 seeds on `cfg_003` +
`cfg_002` — about 18 minutes of fitting, plus a few minutes of patch loading and
cache building. Do not spend 2 × 10 seeds, and above all do not spend them on
`cfg_001`.**

The reasoning, in order of weight:

1. **The seed count is not the capability.** B2's stated purpose is "exercises
   `paired_by_seed.csv`, a branch that has never run". That branch is a
   `pivot_wider` over `seed` and four subtractions; it executes identically at
   n = 3 and n = 10. Ten seeds buy statistical power in a paired comparison that
   nobody is allowed to act on (§ Option 4). Paying 1 to 3.5 hours for it is
   paying for the wrong thing.
2. **The second config should be the cheap one, and it does not matter which.**
   `cfg_002` costs 1.42 min/unit against `cfg_001`'s 5.83. The branch does not care
   which two `config_id` strings it pivots on.
3. **Stage 04 should be drivable without editing it, independently of B2.** It is
   the only example script that cannot be, and the project has already written down
   why that is a defect.
4. **The frozen record stays frozen.** `selection.rds` from 2026-09-17 10:31 on
   `cfg_003`, commit `0eef391`, is the thing that makes the published selection
   optimism of +0.0000 mean anything. Nothing about a capability test is worth
   touching it, and Option 1 — the only option that would — should be recorded as
   rejected rather than left as an open possibility.
5. **If Option 6 is refused, Option 5 is the answer, not Option 4 — but Option 5
   is a deliberate act, not a wait.** On the `tune_length = 24` run a two-config
   freeze is *legitimate* (no test score exists yet) but it is not *free* and it
   is not automatic: somebody must list two `config_id`s before reading anything
   from the test set, and the second config costs a full extra seed loop
   (`04_final_model.R:625`). Nothing fires on its own — §1c shows no rule ever
   produces a pair, at any `tune_length`, so the `== 2` gate at `:669` stays
   `FALSE` and, having no `else`, says nothing about it. Choosing Option 5 means
   committing to that hand-pick; choosing it as a way of waiting means choosing
   Option 4 without writing it down. Once it does run, the downside is bounded: a
   failure in the paired block costs one CSV, because everything expensive is
   already on disk two lines earlier.

What I would *not* do: run B2 at 10 seeds in order to also get a scientific
comparison out of it. That conflates a capability test with a science run, which
`test_plan.md:124-128` already warns against in the other direction — and here it
would conflate them across the one line the project has drawn hardest, because the
comparison in question is on the test set.

---

## 4. The question

> **May the B2 run point stage 04 at a throwaway copy of
> `soc_0_5cm_20260916_232318` — same fold plan, same predictions, no `models/`, no
> `selection.rds` — so that it freezes its own two-config record inside the copy
> and the 2026-09-17 record is never touched?**

- **Yes** → I add the `Sys.getenv()` overrides to `04_final_model.R` **after
  line 97** (with `selection_rule <- "manual"` travelling with the config list,
  and `as.integer()` + a `stop()` on the seeds), and write
  `examples/soc_stock_0_5cm/_b2_two_config_check.R` around Option 6: copy the
  run, set the environment, drive the real `04_final_model.R` with
  `processx::run(rscript_bin, ...)` + `stopifnot(res$status == 0L)` — **not**
  bare `source()`, which `rm(list = ls())` would gut mid-run — assert on
  `paired_by_seed.csv` and on the copy's `selection.rds` (`rule = "manual"`,
  both config ids), unset the environment, delete both the copy and the
  `final_<timestamp>` directory it produced, and `stop()` if any of that did not
  happen.
- **No** → B2 is deferred to a hand-picked pair on the `tune_length = 24` run
  (Option 5), which is a decision someone has to carry out on that run — it will
  not happen by itself, because no selection rule can produce a pair (§1c). I
  add the `Sys.getenv()` overrides anyway, fix the `== 2` gate at
  `04_final_model.R:669` so three configs cannot skip silently, fix the
  unreachable `"manual"` branch at `:230` so a hand-written selection is not
  frozen as `one_se`, and record here that B2 is deferred, on what condition,
  and at what cost (a second full seed loop).

Everything else in this note follows from that answer. Three pre-existing defects
should be fixed either way; none of them is B2's, they are just where B2 found
them:

- the exact-`== 2` gate at `04_final_model.R:669`, which skips three configs in
  silence (§1d);
- stage 05's `config_id = "auto"` taking tune_grid order rather than the
  selection order (§1d);
- the unreachable `"manual"` branch at `04_final_model.R:230`, which stamps
  `rule = "one_se"` on every hand-written selection — including one typed at
  line 55 today, with no B2 anywhere near it (Option 3's risk list).
