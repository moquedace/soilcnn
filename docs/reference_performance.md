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

| split | same raster cell | patches overlap 15×15 |
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

1. **The random test set is EASIER than the spatial validation: +0.042 CCC.**
   Consistent across all three configs. If a rebuilt pipeline shows the test
   harder than spatial validation, something in the split logic is wrong.

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
