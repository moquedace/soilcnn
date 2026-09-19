# Capability test plan

## What "the pipeline is closed" actually proved

On 2026-09-16/17 the chain 01 → 02 → 03 → 03b → 04 → 05 → 07 ran end to end.
That proved **one path**: `spatial_cv`, one CNN config out of three, the `rf` and
`mlp` baselines, a constant-width conformal interval, and a single-tile
prediction at 20 km.

It did not prove the framework. A measured inventory of what the code offers
against what has ever executed:

```
126 public functions | 37 with no unit test | 60 never called by any script
```

Most of those 60 are internals reached through something else. The ones that
matter are the **user-facing capabilities a reader of the README would expect to
work** — a different resampling scheme, another model family, a diagnostic — and
several of those have never touched real data.

The risk is specific and it is not "the code errors out". Code that errors out is
cheap. The risk is a capability that runs, returns plausible numbers, and is
wrong — which is exactly what `rf_grid()` did for two runs, what the buffer did
for every run before 2026-09-16, and what the conformal calibration did in the
first stage-04 run.

---

## The idea that makes this affordable

**Test the framework with the cheap model.** Almost every capability below is a
property of the *plumbing* — the fold plan, the table view, the registry, the
comparison table — and not of the CNN. A Random Forest unit costs seconds where a
CNN unit costs two minutes, and it exercises the same plumbing.

So: use `model = "rf"` for everything whose question is "does this wiring work on
real data", and spend CNN time only where the CNN itself is the subject.

A single script, `examples/soc_stock_0_5cm/_capability_sweep.R`, can carry tiers
A and B below. It should write one row per capability with PASS/FAIL and the
number that was checked, so the answer is a table rather than a scroll.

---

## Tier A — cheap, and a silent wrong answer is plausible

Everything here runs in minutes with `model = "rf"` on the existing store.

| # | capability | what has never been checked on real data | how it could be silently wrong |
|---|---|---|---|
| A1 | `random_cv()` | fold construction with `group = "auto"` on 3,728 real points | profiles split across roles, so the score is inflated and nothing says so |
| A2 | `holdout_cv()` | same, single split | same |
| A3 | `region_cv()` | grouping by a derived class (argmax of the `soil_class_fao_*` dummies) | a region silently split, or an empty fold |
| A4 | `features = "window_mean"` alone | the table view path that drops the centre columns | the wrong columns selected, giving a model that quietly reads something else |
| A5 | `caret_spec()` | borrowing one method (`glmnet`, `xgbTree`) through `dsm_train()` | caret resampling leaking in, or the one-row grid not being honoured |
| A6 | `conformal_cv()` | calibrating across folds on the real 03 predictions | it is the honest coverage estimate and has only ever run on simulated data |
| A7 | `occlusion_report()` | the real trained `cfg_003` | see below — this is also a scientific result, not only a test |
| A8 | `score_test_grid()` | the real 03 run, after `freeze_selection()` | ditto: it measures selection optimism |

**A7 and A8 are not only tests.** Occlusion answers, from inside the trained
model, whether the neighbourhood is used at all — the question 03b answers from
outside. `score_test_grid()` measures how much a test-selected number would have
been overstated. Both are inference only, minutes, and both produce a number for
the paper.

Tier A is implemented as `examples/soc_stock_0_5cm/_capability_sweep.R`.
Expected cost: **1.5-2 hours**, most of it A4 (36 forests on 543 columns) and
A5 (54 xgboost units on 724).

### What the contract pass changed before a line of it ran

Every call above was derived from the source by one agent and then adversarially
refuted by another. **Thirteen of fourteen contracts came back BROKEN**, and the
dominant failure was not a wrong function name -- it was a *weak assertion*: a
check that passes on a silently wrong result. Five examples that reached the
script:

- **A1** checked the test set's SIZE (591) rather than its identity. A fresh
  draw would be `ceiling(0.15 * 3728) = 560`, so 591 catches a total failure by
  arithmetic coincidence and nothing else. It now compares the ids.
- **A3** derived the soil-class group from the point table only. A group vector
  in CSV order against a store in store order produces a plan that looks perfect
  and groups the wrong points. The label is now derived twice -- from the aligned
  table and from the patch tensors' centre pixel -- and the two must agree.
- **A4** checked column names, which are pasted from the window key and never
  read from the data, so they are right whatever the tensor held. The means are
  now recomputed in base R on 20 rows of every window.
- **A5** could not have detected caret's own resampling leaking in, which is the
  entire risk of borrowing caret. It now probes `trainControl(method = "none")`
  on the fitted object directly and compares fold membership by row index.
- **A6** asserted `all(diff(by_group$picp) >= 0)` -- a tautology, because
  `picp_report()` sorts `by_group` by `picp`.

Two corrections went the other way, against the refuters, because the code moved
under them: A2 asserted that `metrics/*_perf.csv` carries a test row (true when
it was written, false since `.drop_test_rows()`), and A6's golden PICP values
were computed outside R and are now checked as properties with the exact numbers
printed rather than asserted.

---

> **Status (2026-09-19):** tier A done; B1, B2, B4, B5 done, B3 and B6 have
> their scripts ready and have not run; C1 has both runs finished and its
> comparison script not yet run. The table with results is in
> `status_and_roadmap.md` §1.

## Tier B — real risk, real cost

| # | capability | why it is not in tier A | cost |
|---|---|---|---|
| B1 | `knndm_cv()` on the real points | needs `predpoints` from the 20 km raster and CAST on 3,728 points; the projection and the cost were measured on paper, never on this data | minutes to build the plan; a full `rf` run on it after |
| B2 | Two configs in stage 04 | exercises `paired_by_seed.csv`, a branch that has never run | 2 × 10 seeds |
| B3 | D4 augmentation on/off | a training-time axis never varied on real data; it changes what the model sees, so only a CNN can answer | 2 × 9 units |
| B4 | `05a` / `05b` / `05c` | the parallel path and the mosaic have not run since the refactor, and `05` now writes **10 bands instead of 7** (smearing added one more after this was written) — the merge walks that list | a 2 × 2 shard run at 20 km |
| B5 | `06` and `99b` | the graphical evaluation and the visual pipeline check, never run against this API | minutes |
| B6 | resume after an interruption | kill 03 midway, restart, confirm it picks up and that `check_plan_unchanged()` stays quiet | one interrupted run |

---

## Tier C — folded into work already planned

| # | capability | where it gets exercised |
|---|---|---|
| C1 | `gate_type`, `use_se_block`, `embed_pool`, `conv_padding` | the `tune_length = 24` grid covers all of them by construction |
| C2 | Normalised conformal intervals | blocked: needs a difficulty score comparable between calibration and prediction. The dissimilarity index from `R/aoa.R` is the right one — see `reference_performance.md` |
| C3 | `region_folds()` with a real region layer | no region column exists in the point table today |

**The `tune_length = 24` run is not a capability test.** It is the science run
that answers whether the CNN family was ever properly represented. Keeping the
two apart matters: a capability sweep that fails tells you the code is wrong, and
a science run that disappoints tells you the model is. Conflating them makes both
unreadable.

---

## Order

1. **Tier A as one sweep script.** It is the cheapest hour available and it is
   where a silent wrong answer is most likely, because these are the paths a
   second user would take first.
2. **B4** next, because `05` now writes two bands more than `05b` has ever seen,
   and that breaks at merge time rather than at prediction time.
3. **B1, B2, B3, B5, B6** as they become convenient.
4. **The `tune_length = 24` science run**, once the sweep is clean — no point
   spending seven hours on a grid if the framework under it has an unmeasured
   fault.

---

## The standing rule this comes from

Every capability that has not run on real data is a claim, not a fact. The three
defects that cost the most in this project were all of that shape:

- `rf_grid()` drew the same config four times and reported four results
- the buffer protected the validation set and left the test set exposed
- the conformal interval was calibrated on one spatial block and applied to
  another

None of them raised an error. All three were found by running something that had
not been run before, or by measuring a claim the code made about itself.
