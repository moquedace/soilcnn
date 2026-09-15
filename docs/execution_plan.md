# Execution plan — the one that is meant to be the last restart

Status: **active**. Started 2026-09-14.

This plan exists because the pipeline has been restarted from stage 01 five
times. Its first job is not to add features — it is to make the next discovery
cheap, so that finding something wrong stops meaning "run five hours again".

---

## Root cause

The five restarts, as far as they can be reconstructed:

| # | what triggered it |
|---|---|
| 1 | spatial leakage discovered — results invalidated |
| 2 | patches stored **scaled**, therefore tied to one split — re-extract |
| 3 | `torch_save` silently corrupting above 2 GB — re-extract |
| 4 | verification aborted 5 h of extraction on a false positive |
| 5 | buffer geometry wrong; test set still a random split |

One pattern, every time: **the decision or the defect lived UPSTREAM of where
it was found.** Not carelessness — coupling. While the expensive artefact (16 GB
of patches) carries a decision that is still going to change, changing that
decision costs five hours.

This was already solved once, without noticing it was the general solution:
storing patches **raw** decoupled scaling from storage, which is why the whole
resampling step fitted without re-extracting anything.

**The same has to be done for the split.**

---

## The change that breaks the cycle

`patch_meta.csv` — regenerable only by five hours of extraction — currently
carries `dataset_role`. Who is train, validation and test is written **inside
the expensive artefact**.

After this plan:

| artefact | holds | cost to change |
|---|---|---|
| patch store | `sample_id`, `profile_id`, `x`, `y`, target, patches | hours |
| `data_split.csv` | which points are modelling vs test | seconds |
| fold plan (stage 03) | who trains and who scores, per fold | seconds |

What still forces re-extraction, and nothing else: **predictors, windows,
target transform.** Three things. A manifest lock makes a mismatch fail
immediately instead of five hours in.

---

## Phase 0 — armour (changes no artefact, forces no rerun)

- [x] **0.0** Development profile. `run_profile <- "dev"` in stage 01 keeps 10%
      of the points by WHOLE 2-degree blocks, so the whole pipeline runs end to
      end in minutes. Patches stay at 250 m -- a 20 km window changes what the
      window means, which was tried before and caused problems. Only the
      prediction stage uses the 20 km rasters, and only to exercise the code:
      those values are a machinery test, not a map. The profile travels in
      `target_config.csv` and the 99 opens with a banner, so a 10% number can
      never be read as a result.
      *Measured:* neighbours within 1 km per point, full 29.0, block subsample
      28.8, random subsample of the same size 9.4.

- [x] **0.1** Commit everything. Two weeks of refactor existed only on disk.
- [x] **0.2** Delete `_test_gpu/R/` and `scripts/` — three core modules and a
      whole pre-refactor pipeline, diverged from the real ones. Traps.
      *Verified by:* `run_all` still 9/9.
- [ ] **0.3** English everywhere in code, tests, examples **and user-facing
      messages**. `print_noise_floor()` currently prints Portuguese from the
      framework layer.
      *Verified by:* grep for Portuguese → 0 hits.
- [ ] **0.4** Tests for the three functions whose failure mode is "everything
      is fine": `spatial_overlap_report()`, `calc_metrics()`,
      `write/compare_run_snapshot()`. The pipeline already **asserts** 0%
      leakage per fold on the word of an unverified function.
      *Verified by:* assertions 229 → ~270.
- [ ] **0.5** Own CCC implementation, removing the `DescTools` dependency —
      and it currently computes a confidence interval nobody reads, on 10k
      points, every epoch.
      *Verified by:* equality with `DescTools` to 1e-12 in the test.
- [ ] **0.6** `metric = c("chebyshev", "euclidean")` in `apply_buffer()`,
      chebyshev by default. Patch overlap is a square condition; the current
      circular buffer lets the diagonal escape.
      *Verified by:* a test with the (14, 14) case that escapes today.
- [ ] **0.7** `one_se()` as an **option**, not a default.
      *Verified by:* a fixture where it changes the choice.
- [ ] **0.8** README: it still describes patches as "scaled", which is the
      exact defect the refactor removed.
- [ ] **0.9** Translate `docs/` to English. Mechanical, no logic risk, its own
      commit so the diff is purely translation. Nothing depends on it, so it
      can happen any time in this phase.

Excluded by decision: **2.3** (test evaluated during tuning) stays as it is.
**MQI** stays.

---

## Phase 1 — the decoupling

- [ ] **1.1** Remove `dataset_role` from the patch store; roles move to
      `data_split.csv`, regenerable in seconds.
- [ ] **1.2** Vocabulary: `modelling` / `test`. The `train`/`validation`
      distinction written by stage 01 only ever meant anything for
      `holdout()` — every other plan pools them and re-splits, discarding that
      boundary. This is the caret contract: hand over one set, the control
      splits it.
- [ ] **1.3** The plan decides the **test set too**:
      `spatial_folds(..., test_frac = )`.
- [ ] **1.4** Rule: the test set is spatial **if and only if** validation is
      spatial. A spatial validation next to a random test puts two
      incomparable numbers in the same table, and this run measured the gap:
      **+0.042 CCC** in favour of the random test.
- [ ] **1.5** **Contract test**: the patch store must contain no role column.
      This is what stops the cycle from coming back — if anyone (including me,
      next month) re-bakes the split into the store, `run_all` fails.

**Applies to the existing store without re-extracting.** Running 01 and 02
again becomes a choice, not an obligation — which is the whole point.

---

## Phase 2 — manifest lock

Not a hand-written file of our values: stage 02 records what it extracted, and
the 99 compares what the **current configuration asks for** against what the
store **has**.

- [ ] **2.1** 02 writes the spec it extracted under: predictor list (+ hash),
      windows, target and transform.
- [ ] **2.2** 99 **FAILS** on mismatch, naming the fix: "the store has windows
      3, 9, 15 and this grid asks for 21 — re-extract, or change the grid."

Dynamic by construction: a package user changing windows gets a clear failure
in seconds instead of a wrong result or a wasted extraction.

---

## Phase 3 — model registry, and the baselines that make it necessary

Random Forest is not a torch model, so it forces the registry to handle
genuinely different `fit`/`predict` contracts. That is the right pressure on
the design.

- [ ] **3.1** Minimal registry: a model declares `build`, `param_space`, and
      what input it consumes (`patches` or `table`).
- [ ] **3.2** The fold cache can produce a **tabular** view (centre pixel, and
      per-channel window means) alongside the tensor view.
- [ ] **3.3** Three baselines under identical folds, seeds and noise floor:

| model | input | answers |
|---|---|---|
| RF | centre pixel | the classic DSM baseline |
| RF | centre pixel + per-channel window means | context **without** spatial structure |
| MLP | centre pixel | is it the architecture or just the covariates? |
| CNN | the whole patch | context **with** spatial structure |

The gap between the second and the last **is what the convolution is worth**.
If they match, the CNN is doing averaging, and that is falsifiable, cheap, and
measured under the same folds.

---

## Phase 4 — the cheap questions, before anything expensive

- [ ] **4.1** Is the §2.4 bias real? It was to be measured from the per-epoch
      histories of the 27 full-data units, but those were deleted with the rest
      of the outputs — only the summary in `reference_performance.md` survives.
      It now needs the histories of a **dev** run, which cost minutes. Same
      question, same zero-training method, different source. If the bias is
      small, A3 dies and 15% of the training data is saved.
- [ ] **4.2** Does the large branch accept `valid` padding? One parameter in
      the grid. Note the constraint found while auditing: with window 3 and 2
      conv blocks the centre's receptive field is already 5×5, larger than the
      patch, so **every** output position depends on the padding — a 3×3 branch
      is fundamentally incompatible with exact fully-convolutional prediction.
      The redundancy is w², so the 15 branch is 225× redundant and the 3 branch
      only 9×: applying this to the large branch alone captures nearly all of
      the gain without touching the winning architecture.
- [ ] **4.3** kNNDM feasibility. Nothing is promised before measuring: the
      points are global lon/lat, and without a projected CRS `knndm` needs full
      spherical distance matrices over 31k points. Measure, then decide.

---

## Phase 5 — run

`01` → `99` → `02` → `99` → `03` → `99`, with `run_all` before every expensive
script.

---

## Open questions

- `_scratch_1km_test/` (7 scripts, 124 KB): delete like the other stale
  scaffolding, or keep? It is tracked, so deleting is reversible.
- Commit cadence: one commit per verified step, or in batches?

## Rules that hold for the rest of the project

1. **The expensive artefact never carries a cheap decision.** If changing X
   forces re-extraction, X belongs in the manifest, not in the data.
2. **CSV is a presentation format.** When code writes for code to read, the
   format must carry the type. Paid for three times already.
3. **Every claim the pipeline makes about its own correctness earns a second,
   independent implementation.** The patch-centre cross-check is one. The
   leakage report became one the night it exposed the buffer error.
4. **Verification reports; the user aborts.** Never `stop()` on a check that
   would discard hours of work.
