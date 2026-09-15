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
