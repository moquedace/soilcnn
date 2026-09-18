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

At 95% the two calibrations give 92.2% and 92.7%. Interval widths: 65.1 t/ha
mean at 90% (against a median stock of 29.3), 83.8 at 95%. Those widths are the
honest cost of this model's error, and they are the number a user of the map
would have to live with.

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

### The normalised interval is not produced, and why

Locally adaptive intervals need a per-point difficulty score. The obvious one is
the ensemble spread — available on the prediction side (10 seeds of the final
model) and **not** on the calibration side, because the calibration set is now
the cross-validated residuals, from 3 seeds of models trained on ~53% of the
points.

Calibrating the ratio residual/spread on one kind of spread and applying it to
another is not a weaker guarantee, it is no guarantee: the CV models disagree
with each other more than the final ensemble does, so the ratio is
systematically wrong and the interval comes out confidently the wrong width.

The principled difficulty score is the **dissimilarity index** from `R/aoa.R`.
It is computed the same way for a calibration point and for every prediction
pixel, from the same scaling, and it measures the thing that actually makes a
point hard: distance from what the model was trained on. That is the next step,
and it is also what would address the residual under-coverage — the points the
interval misses should be the dissimilar ones.

### Stage 05 and 07 — the pipeline closed (20 km, plumbing only)

Predicted at 20 km against a model trained at 250 m: the 15×15 patch covers 300
km instead of 3.75, so the network sees a neighbourhood it never met. Stage 05
says so in a banner and the map is not a map. 1.6 M cells, 358,537 valid,
30.5 min single-tile.

| band | min | mean | max |
|---|---|---|---|
| median prediction | 0.43 | 26.22 | 113.15 |
| ensemble sd | 0.25 | **8.56** | 37.94 |
| conformal 90% lower | 0.00 | 2.16 | 73.53 |
| conformal 90% upper | 40.06 | 65.84 | 152.77 |

**The number this whole line of work was for:** the ensemble spread says ±8.6 and
the calibrated interval says ±39.6 — a factor of **4.6**. Publishing the `sd`
band as "uncertainty" would have promised four and a half times more certainty
than the model delivers, and it would have looked exactly like a correct map.

The consequence is blunt: with a median of 26.2 and a half-width of 39.6, the
lower bound is **zero over most of the map** (mean 2.16). The model cannot rule
out an empty stock almost anywhere. That is the model being honest about what it
knows, not a defect of the interval.

Stage 07: AOA threshold DI = 0.6999, **78.7%** of valid cells inside (394,406 of
501,246), 1.4 min.

---

## Spatial occlusion on cfg_003 — and the contradiction it appears to raise

Stage 03 run `soc_0_5cm_20260916_232318`, config `cfg_003` (single 15×15 branch),
fold 1, seed 42, validation rows (n = 1037), permutation occlusion.

| scope | px hidden | CCC | ΔCCC | Δ per px |
|---|---|---|---|---|
| baseline | 0 | 0.4974 | — | — |
| context_all | 224 | −0.0154 | **−0.5128** | −0.00229 |
| centre_only | 1 | 0.4958 | −0.0016 | −0.00159 |
| ring_01 | 8 | 0.4779 | −0.0195 | −0.00244 |
| ring_02 | 16 | 0.4492 | −0.0482 | −0.00301 |
| ring_03 | 24 | 0.4106 | −0.0868 | −0.00362 |
| ring_04 | 32 | 0.3734 | **−0.1240** | −0.00388 |
| ring_05 | 40 | 0.3835 | −0.1140 | −0.00285 |
| ring_06 | 48 | 0.4687 | −0.0287 | −0.00060 |
| ring_07 | 56 | 0.4969 | −0.0005 | −0.00001 |

**Read the per-pixel column, not the totals.** The first version of the verdict
compared `context_all` against `centre_only` — 224 pixels against 1 — and
announced that the neighbourhood was doing the work. Almost any convolution over
almost any patch produces that result, because 224 permuted pixels destroy every
feature map while one perturbs them. A comparison a working network cannot fail
is not a measurement, and the print now reports cost per pixel and says so.

What survives the correction is the **ring profile**, and it is not flat:

- cost per pixel rises from ring 1 to **ring 4** (≈1 km at 250 m) and then falls;
- **ring 7, the outer border, costs essentially nothing** (−0.0005 over 56 px).
  `cfg_003` uses `conv_padding = "valid_large"`, so the outermost ring is
  plausibly cropped before it reaches the output — an architectural fact, not a
  statement about soil.

### The apparent contradiction with 03b, and its resolution

03b measured `rf_context = rf_centre` (t = −0.10): the neighbourhood adds
nothing to a forest. Occlusion says the CNN collapses without the neighbourhood
and barely notices the centre. Both are correct, and the resolution is
**redundancy, not contradiction**:

> At 250 m most covariates are smooth, so a pixel three cells out is close to a
> copy of the centre. A network can lean entirely on the rim and still learn
> nothing a centre-only model would have missed.

The two questions are separable, and the framework now answers each in one place:

| question | answered by |
|---|---|
| does this network *use* the neighbourhood? | `spatial_occlusion()` — yes, almost entirely |
| does the neighbourhood *add* anything? | 03b, `rf_centre` vs `rf_context` — no |

Which is also why `rf_centre` (0.487) beats the CNN (0.473) while reading one
pixel per channel: the centre alone already carries the signal, and the
convolution spends 340,225 parameters rediscovering it in the rim.

**The falsifiable follow-up**, and the one worth running next: measure the
correlation between each channel's centre value and its ring-4 mean across the
3,728 points. If it is high, redundancy is confirmed directly rather than
inferred from two experiments agreeing about a third thing.

---

## Capability sweep, tier A — 8 of 8 (2026-09-17)

| id | capability | measured |
|---|---|---|
| A7 | `occlusion_report()` | baseline 0.497 \| context −0.513 (224 px) \| centre −0.002 (1 px) |
| A8 | `score_test_grid()` | cfg_003 test CCC 0.465, rank 1 of 3, **selection optimism +0.0000** |
| A6 | `conformal_cv()` | PICP 0.8997 over 3,092 \| per fold 0.876 / 0.926 / 0.898 |
| A1 | `random_cv()` | 36 units, val_ccc **0.622**, test set identical to the frozen 591 |
| A2 | `holdout_cv()` | 2693 / 475 / 560 — drew its own test set, not the frozen one |
| A3 | `region_cv()` | 31 FAO classes, none split, val_ccc **0.384** (sd 0.133 across folds) |
| A4 | `features = "window_mean"` | 543 columns, means verified against base R |
| A5 | `caret_spec("rf")` | `method = "none"`, folds match the CNN's, best config mean 0.483 |

### The number this sweep produced by accident: validation design is worth 0.24 CCC

The same forest, the same 181 centre values, the same points, under three
resampling designs:

| design | val_ccc | what it answers |
|---|---|---|
| `random_cv` | **0.622** | interpolation beside a known sample |
| `spatial_cv` (03b `rf_centre`) | **0.487** | prediction into unvisited ground |
| `region_cv` (leave-soil-class-out) | **0.384** | prediction into an unseen soil class |

**0.24 CCC separates the easiest from the hardest**, which is larger than every
difference this project has measured between model families (0.025), between CNN
configs (0.034), or between window sizes. The choice of validation design
dominates the choice of model, and a paper that reports one number without
naming the design is reporting the design.

`region_cv`'s spread is also the largest — sd 0.133 across folds against 0.031
for spatial — which is what leaving out a whole soil class does: some classes
are predictable from the others and some are not.

### A5 confirms the caret adapter is a shim and nothing more

caret's `rf` reaches **0.483** where the native `rf` spec reaches **0.487**, on
the same folds and the same 724 columns, with `trainControl(method = "none")`,
no resample table and no retained training data. Two independent wrappers of
`randomForest` landing 0.004 apart is a stronger statement than any structural
check: an adapter that quietly resampled, or that ignored the tuning grid, would
not arrive there.

`xgbTree` could not be used: it dies inside xgboost's own R binding
(`ALTLIST classes must provide a Set_elt method`) before reaching any code here.
That is an xgboost/R build incompatibility on this machine, not a framework
defect, and testing an adapter through a broken backend measures the backend.

---

## B4 — the sharded path and the mosaic (2026-09-17)

2×2 shards at 20 km, four workers, 16.1 min, then 05a's own merge of 36 tiles
(9 bands × 4 shards).

| band | max abs diff, 2×2 vs 1×1 | tolerance |
|---|---|---|
| ensemble_median | 4.6e-05 | 1.2e-02 |
| ensemble_mean | 2.3e-05 | 1.2e-02 |
| ensemble_sd | 2.5e-05 | 4.8e-03 |
| ensemble_mad | 1.0e-04 | 5.8e-03 |
| ensemble_min | 6.9e-05 | 9.9e-03 |
| ensemble_max | 8.4e-05 | 1.5e-02 |
| valid_mask | **0** | 1.1e-03 |
| conformal_90_lower | 4.8e-05 | 8.4e-03 |
| conformal_90_upper | 4.6e-05 | 1.6e-02 |

**The mosaic reproduces the single-tile map.** Worst case 1.0e-04 t/ha, two
orders of magnitude inside the tolerance, and the validity mask matches exactly.
No seam, no dropped tile, no tile from another grid, and the workers predicted
the 20 km grid rather than falling back to 250 m. Geometry, valid-pixel count
and the three summary statistics agree on all nine bands.

The residual 1e-04 is expected and is why the tolerance is relative: the 2×2 run
gets a different thread count per worker, its block boundaries fall elsewhere so
the inference batches hold different pixels, and `expm1()` amplifies a float32
discrepancy by about 26 at the median and 113 at the maximum.

### Two things the run taught about the scripts themselves

**05a already runs 05b.** Line 283, documented in its own header at line 29. The
first version of the check ran the merge a second time — harmless, since it is
idempotent over the same tiles, but it verified a mosaic the *check* had produced
rather than the one the *pipeline* produces. Checking your own side effect is not
checking the pipeline. The check now reads 05a's merge log instead.

**05b's geometry check is inert whenever the prediction rasters are overridden.**
It builds its template from `raster_table_used.csv`'s first entry — always the
250 m raster — and compares with `stopOnError = FALSE`, so at 20 km all nine
layers warn and a genuinely broken mosaic would warn identically. That is why the
geometry is re-checked here against a real 20 km raster.

---

## The finding nothing in the framework could see: a −24% bias

Stage 06 ran for the first time in months as part of B5. It computes one number
the framework does not, and that number is the most consequential result of the
run.

Final model `cfg_003`, 10-seed ensemble, on the 591 frozen test points:

| | mean | median | max |
|---|---|---|---|
| observed | 39.45 | 31.18 | 173.4 |
| predicted (ensemble median) | 29.84 | 25.60 | 112.9 |

```
bias = -9.61 t/ha  =  -24.4% of the observed mean  =  53% of the MAE
```

**More than half of the average error is a systematic shortfall**, not scatter.
Using the ensemble *mean* instead of the median recovers 0.4 t/ha, so this is
not the seed ensemble — it is the model.

### Why

Not a coding error: a modelling choice nobody had written down. The target is
trained on `log1p` with a SmoothL1 loss, so the network estimates a conditional
**median** in log space, and `expm1()` of that is the conditional median of the
stock — not its mean. The target is right-skewed (mean/median = 1.27 on this
test set), so the median sits systematically below the mean, and the extremes
are compressed besides (max predicted 112.9 against 173.4 observed).

### Why nothing caught it

`calc_metrics()` returned `n, ccc, r2, mae, nse, rmse, rpd, mqi` and **not one
of them carries a sign**. MAE and RMSE are unsigned by construction; R², NSE and
RPD are insensitive to an offset in the relevant range; CCC penalises bias but
mixes it with scatter, so a low CCC never says which one it is. A map that is a
quarter light passed every check the framework makes.

`calc_metrics()` now returns `bias` (signed, native units) and `bias_pct`
(relative to the observed mean), and both propagate to the comparison tables.

### What it means for the map

**Do not sum this map for a total stock.** At −24.4% the total would be a
quarter light. The 20 km map's own numbers already showed it without anyone
reading them that way: the global sampled median came out at 25.22 t/ha against
a training median of 29.3.

Three ways forward, in increasing order of work:

1. **Say what the map is.** It is a conditional median surface. That is a
   legitimate and useful product — it is the right thing for "what is the
   typical stock here" — and it is the wrong thing for a total. Labelling it
   costs nothing and is honest.
2. **Correct the back-transform.** A smearing estimator (Duan 1983) rescales
   `expm1()` by the mean of the exponentiated residuals, which recovers the mean
   without retraining. Cheap, and it can be calibrated on the same out-of-fold
   residuals the conformal interval uses.
3. **Target the mean directly** — train on the native scale with a loss whose
   minimiser is the mean, and pay for it in sensitivity to the right tail.

Option 1 is not optional whichever else is chosen: the current map is already
published in this repository's outputs and it is a median surface.

### The correction, measured on the same data

Duan's smearing estimator (1983, JASA 78:605–610), calibrated on the 9,276
cross-validated residuals of `cfg_003` in log space:

```
S = mean(exp(residual)) = 1.3594
```

The residuals are near-lognormal — `exp(mean + sd²/2) = 1.3587` against the
empirical 1.3594 — so the estimator behaves exactly as the theory says.

Applied to the 591 frozen test points:

| surface | mean | bias | MAE | RMSE | CCC | total stock |
|---|---|---|---|---|---|---|
| median (`expm1`) | 29.82 | **−24.4%** | 18.15 | 28.07 | 0.4748 | −24.4% |
| mean (smeared) | 40.90 | **+3.7%** | 19.14 | 27.07 | **0.5692** | +3.7% |

**MAE going up is the trade-off, not a regression.** The median minimises
absolute error; the mean minimises squared error. Correcting toward the mean
must improve RMSE and worsen MAE, and a change that improved both would mean
something other than a median-to-mean move had happened. The test asserts
exactly that pair of directions.

CCC improves by 0.094 — nearly three times the seed noise floor — because CCC
penalises bias, and removing a −24% bias is the largest single improvement
anything has produced in this project.

**Nothing is replaced.** Stage 04 writes `smearing.rds` beside the conformal
calibration, stage 05 reads it and writes `soc_mean_smeared_ton_ha` beside the
median band. Each answers its own question:

| question | surface |
|---|---|
| what is the typical stock here? | the median map |
| what is the total stock over this area? | the smeared mean map, and only it |

In stage 05 the correction is exact and free: `expm1` is monotone, so the median
commutes with it and `log1p(ensemble median)` *is* the ensemble median in log
space. The band is `(1 + median) × S − 1`, with no second matrix of predictions
and no second pass over the network.
