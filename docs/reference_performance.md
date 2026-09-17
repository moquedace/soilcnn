# Reference performance — full data, before the rebuild

Recorded 2026-09-14, immediately before deleting every output to restart on a
fast development configuration. These are the numbers the rebuilt pipeline is
answerable to.

**Read the relations, not the absolutes.** A development run on 10% of the
points and 20 km predictors will score far lower, and that is expected, not a
signal. What must survive is the *shape* of the results — the four relations at
the bottom of this file. If one of those inverts, the cause is the code, not
the data.

---

## The run

`soc_0_5cm_20260914_030744` — the first complete spatially-validated run this
project produced.

| | |
|---|---|
| points | 36,974 prepared / 36,697 with a fully valid window |
| predictors | 181 channels at 250 m (0.0022458°) |
| windows | 3, 9, 15 |
| target | `soc_stock_0_5cm`, ton/ha, trained on `log1p` |
| resampling | `spatial_folds(k = 3, block_size = 2°, buffer = 15 px)` |
| repetitions | 3 seeds (42, 43, 44), shared across configs |
| units | 27 = 3 configs × 3 folds × 3 seeds, 0 failures |

## Cost

| | |
|---|---|
| stage 02 extraction | 5.02 h, 16.7 GB |
| stage 03 tuning | 9.5 h for 27 units |
| per unit | 13.5 min min / 15.3 median / 37.9 max |
| best epoch | 6 min / 11 median / 21 max — early stopping at 66–81 |

## Per config

| config | architecture | val_ccc (spatial) | val_mae | test_ccc (random) |
|---|---|---|---|---|
| cfg_003 | 3×9, 64_128, no gate, flatten, lr 1e-3 | **0.5257 ± 0.0251** | 16.79 | 0.5613 |
| cfg_001 | 9×15, 64_128_128, gate, gap, lr 1e-4 | 0.5170 ± 0.0219 | 17.11 | 0.5661 |
| cfg_002 | 3×9, 64_128, no gate, flatten, lr 3e-4 | 0.5071 ± 0.0279 | 16.93 | 0.5478 |

## Leakage, as measured

Two different things, never to be conflated again:

**Identical patch (same raster cell)** — two points in one 250 m pixel feed the
network the same input, bit for bit. A defect under ANY plan, because it is
about duplicate inputs, not about geography.

**Shares pixels** — two nearby points have some surrounding cells in common.
**Not a defect.** Under a random plan it is the condition being measured; a
random split is answering "how well does this predict at new points drawn from
the same spatial distribution", and neighbours sharing context is what that
question is made of. It is worth looking at only under a plan that claims to be
spatial, where it describes how well the separation held.

| split | identical patch | shares pixels 15×15 |
|---|---|---|
| validation (fixed split from 01) | 27.06% | 54.2% |
| test (fixed split from 01) | 28.08% | 55.0% |
| validation (spatial folds, per fold) | **0%** | 0.51–0.98% ¹ |

¹ measured by independent reimplementation, not by the pipeline: the circular
buffer lets the diagonal escape a square overlap condition. Phase 0.6 of the
execution plan fixes it.

---

## The four relations that must survive

These are the checks that matter in development mode. They are about structure,
not magnitude, so they should hold at any data volume or resolution.

1. **The random test set scores HIGHER than the spatial validation: +0.042 CCC.**
   Consistent across all three configs.

   This is not an error in the test set. The two numbers answer two different
   questions -- interpolation near known samples, and prediction on unvisited
   ground -- and the gap is the distance between them, which is worth knowing.
   It is listed here as a control: if a rebuilt pipeline reverses the sign, the
   split logic changed behaviour.

2. **The seed noise floor exceeds the gap between configs.** Seed sd 0.0275
   against a 1st-to-2nd gap of 0.0086 — 3.2× larger. The ranking does not
   separate, and the pipeline says so.

3. **Seed variation dominates geographic variation.** sd between fold means
   0.0086, sd between seeds within a fold 0.0275. The three spatial folds agree
   with each other more than repeated seeds of the same fold do.

4. **Early stopping fires between epochs 6 and 21.** The model exhausts what
   the labels can teach almost immediately. If a rebuild shows best epochs in
   the hundreds, the training loop changed behaviour.

---

## The trap in a 10% subsample, and how to avoid it

Dropping 90% of the points **at random** thins the spatial clusters. Leakage
falls, the buffer discards fewer points, and the folds look cleaner than
reality — so a bug in the spatial logic can hide behind a healthy-looking
report.

Subsample **whole blocks** instead: keep every point inside the blocks that are
kept. Local density, and therefore the leakage phenomenon the project exists to
control, is preserved at one tenth of the cost.

---

# The first dev run, 2026-09-15

`soc_0_5cm_20260915_122952` — the first run on the rebuilt pipeline, on the
front end, with the spec lock, the measured block size and the carved test set.

**A dev number. Comparable only with another dev number.** 10% of the points,
by whole blocks.

| | |
|---|---|
| points | 3,766 prepared / 3,728 with a fully valid window |
| resampling | `spatial_cv(k = 3, block_size = "auto" -> 0.25 deg, buffer = "auto" -> 0.0337 deg)` |
| test | 591 points, carved by the SAME criterion, frozen in `data_split.csv` |
| buffer cost | 319 training points dropped, 5.1% per fold |
| units | 27 = 3 configs x 3 folds x 3 seeds, **0 failures** |

## Results

| rank | config | architecture | val CCC | sd | SE | test CCC | val − test |
|---|---|---|---|---|---|---|---|
| 1 | cfg_002 | 3x9 · 64_128 · flatten · **valid_large** · lr 1e-3 | **0.4885** | 0.0370 | 0.0123 | 0.4768 | +0.012 |
| 2 | cfg_001 | 9x15 · 64_128_128 · gap · same · lr 1e-4 | 0.4442 | 0.0340 | 0.0113 | 0.4098 | +0.034 |
| 3 | cfg_003 | 15 · 64_128 · gap · **valid_large** · lr 3e-4 | 0.4212 | 0.0370 | 0.0123 | 0.4070 | +0.014 |

## Three things this run establishes

### 1. The grid separated, for the first time

| | full run (14 Sep) | dev run (15 Sep) |
|---|---|---|
| 1st − 2nd | 0.0086 | **0.0443** |
| noise floor (sd between seeds) | 0.0275 | 0.0383 |
| verdict | gap is 3.2x SMALLER than the noise | gap **clears** it |

Against the standard error of the mean — the right yardstick for comparing
means of 9 units — the gap is **3.6 SE**. The three configs of the full run
were indistinguishable; these are not.

### 2. The test optimism is gone, and that is the refactor's whole point

The full run's test was a stratified RANDOM draw made in stage 01, while
validation was spatial. Measured consequence: the test came out **+0.042 CCC
EASIER** than the validation, across all three configs, while carrying the name
that suggests it is the stricter number.

The test is now carved by the SAME criterion as the folds. It comes out
**0.012 to 0.034 HARDER** than validation, on all three configs. The sign
flipped, which is what it should do when both numbers answer the same question.

### 3. Zero leakage, by construction and by measurement

0 identical patches AND **0 shared pixels at window 15**, on all three folds.

That is not luck: `block_size = 0.25 deg` (~27 km) with `buffer = 0.0337 deg`
puts every validation point at least 3.75 km from any training point — exactly
the ground a 15x15 patch spans at 250 m. The buffer was derived from the window
and the resolution, and the report confirms it did what the arithmetic said.

## What this run does NOT establish

**Nothing about `conv_padding`.** cfg_002 won and uses `valid_large`, but it
also differs in window (3x9 vs 9x15), learning rate (1e-3 vs 1e-4), gate and
pooling. Three configs cannot separate five factors. The question stays open;
answering it needs a grid that varies padding while holding the rest.

**Nothing about the full data.** 0.489 here against ~0.52 on the full set is
encouraging — a tenth of the data costs ~0.03 CCC, so volume is not the binding
constraint at this scale — but it is a comparison ACROSS profiles, which this
file exists to warn against.

---

## Stage 03b -- the baselines (dev run, 10% of the points)

Same fold plan, same seeds, same scaling as stage 03. Best config of each
family, 9 units each (3 folds x 3 seeds):

| family     | config  | val_ccc | val_ccc_sd | val_mae |
|------------|---------|---------|------------|---------|
| cnn        | cfg_002 | 0.489   | 0.0370     | 17.6    |
| mlp_centre | mlp_001 | 0.484   | 0.0348     | 17.2    |
| rf_context | rf_001  | 0.479   | 0.0307     | 16.3    |
| rf_centre  | rf_001  | 0.464   | 0.0348     | 16.5    |

CNN - rf_context = +0.0091 CCC. Against the standard error of the difference
(~0.016 unpaired), that is 0.57 SE. The whole range across the four families
is 0.025, which is smaller than the CNN's own seed noise floor of 0.0383.

**Nothing separates the four families on CCC**, and MAE ranks them in the
opposite order (rf_context best at 16.3, cnn worst at 17.6, against a median
SOC stock of 29.3 ton/ha). A model that stretches its predictions to match the
observed spread buys CCC at the cost of MAE; that is what the reversal is.

Three caveats, in order of how much they should change the reading:

1. **The RF baseline was handicapped.** `rf_grid()` drew `mtry_frac` with
   replacement and never de-duplicated: all four configs drew 0.1, and three
   were literally identical. So each RF family tested TWO distinct forests, both
   at `mtry = 0.1p` -- far below the `p/3` regression default (18 features
   instead of 60 on centre, 72 instead of 241 on centre+window). The error runs
   in the direction that flatters the CNN. Fixed; 03b must be re-run before
   these RF numbers are quoted.
2. **This is 10% of the data.** Deep models typically gain more from added data
   than forests do. "The convolution buys nothing at 3,728 points" does not
   imply the same at ~37,000.
3. **The CNN grid was 9 configs.** A family represented by its best of 9 draws
   is not the same as a family at its ceiling.

The result that is robust to all three: `mlp_centre` reaches 0.484 from the 181
centre values alone, indistinguishable from the CNN on full patches. Whatever
the neighbourhood is worth here, the evidence for it is not yet visible.

---

## Caveat on every test metric recorded above

All of the runs above predate the fix to `apply_buffer()`, which protected the
validation set only. Training points adjacent to test blocks were kept, and
their patches overlapped test patches. **The test numbers above are optimistic
by an unknown amount** and should not be quoted; the validation numbers, which
are what every comparison on this page actually uses, are unaffected.

---

## Dev run on the corrected pipeline (2026-09-16/17)

Fold plan: `spatial_folds`, k = 3, block 0.25°, buffer 0.0337° Chebyshev,
`protect = validation + test`. 409 training points dropped per fold (6.5%):
319 near validation, 90 near test, plus 45 validation points near test — the
last 135 are the leak that existed until the buffer was fixed. Identical-cell
overlap and 15×15 patch overlap between train and validation: **0**.

### Stage 03 — three configs

| rank | config | val_ccc | sd | se | val_mae | n_params | architecture |
|---|---|---|---|---|---|---|---|
| 1 | cfg_003 | 0.473 | 0.031 | 0.0104 | 17.4 | 340,225 | **single** 15×15 branch |
| 2 | cfg_001 | 0.449 | 0.028 | 0.0092 | 18.4 | 1,891,217 | 9+15, vector gate |
| 3 | cfg_002 | 0.439 | 0.046 | 0.0152 | 17.6 | 2,425,217 | 3+9, no gate |

The single-branch model wins on both metrics with 5–7× fewer parameters. The
gap to second (0.024) is smaller than the seed noise floor (0.0307), so the
ranking does not separate — but the dual-branch configs are not ahead either,
which is the premise under test.

### Stage 03b — the baselines, on identical folds

| family | config | ccc | nse | mae | rmse | mqi | seed noise |
|---|---|---|---|---|---|---|---|
| rf_centre | rf_004 | 0.487 | 0.314 | 16.32 | 24.66 | 0.358 | 0.0015 |
| rf_context | rf_001 | 0.487 | 0.313 | 16.32 | 24.67 | 0.356 | 0.0019 |
| mlp_centre | mlp_005 | 0.480 | 0.215 | 17.68 | 26.30 | 0.220 | 0.0181 |
| cnn | cfg_003 | 0.473 | 0.230 | 17.42 | 26.03 | 0.236 | 0.0374 |

Paired by (fold, seed), 9 pairs:

| comparison | CCC | MAE | NSE | MQI |
|---|---|---|---|---|
| cnn − rf_context | −0.015 (t −0.9) | **+1.09 (t +6.1)** | **−0.083 (t −4.1)** | **−0.120 (t −3.4)** |
| cnn − mlp_centre | −0.008 (t −0.5) | −0.26 (t −1.5) | +0.015 (t +1.3) | +0.016 (t +0.9) |
| rf_context − rf_centre | −0.000 (t −0.1) | +0.008 (t +0.3) | −0.001 (t −0.3) | −0.002 (t −0.3) |

Three findings, mutually consistent:

1. **The neighbourhood buys nothing.** `rf_context = rf_centre` on every metric.
   Together with the single-branch CNN winning stage 03, neither window means
   for a forest nor a second convolutional branch pays for itself.
2. **The CNN loses on error.** Not separated on CCC; behind on MAE, NSE and MQI,
   all beyond t = 3.
3. **The architecture buys nothing.** An MLP on the 181 centre values ties the
   CNN on every metric.

And the cost nobody counts: the CNN's seed-to-seed noise is **~20× the
forest's** (0.037 against 0.0018). Delivering a map needs an ensemble; the
forest returns the same answer every time.

Caveats: 3 CNN configs, and 10% of the points.

### Stage 04 — the final model, and the coverage finding

cfg_003, 10 seeds, refit split (train 2559 / validation 449 / test 591).
Test: CCC 0.467 ± 0.025, MAE 18.6 ± 0.4, RMSE 28.5, NSE 0.234, MQI 0.234.

Conformal coverage, by calibration set, measured on the same 591 test points:

| calibration set | n | q (90%) | PICP |
|---|---|---|---|
| this run's validation (one fold of k = 7) | 449 | 31.98 | 83.6% |
| cross-validated residuals (all folds) | 3,092 | 39.62 | **87.8%** |

**The first was a defect, the second is a result.** The refit validation is one
spatial region and the easiest of the three sets (MAE 14.6 against 17.0 across
the CV and 18.2 on test); calibrating there and measuring elsewhere is
exchangeability failure by construction.

What survives after the fix is the finding: **even calibrated across every
block, the interval under-covers a held-out spatial block by ~2 points.** The CV
calibration is conservative by construction — those models trained on ~53% of
the points against the final model's 69%, so their residuals are larger — and it
still falls short. That gap is the part conformal cannot fix, and it is the
measurable statement that a spatially held-out region is not exchangeable with
the regions used to calibrate.
