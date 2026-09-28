# Status and roadmap — 2026-09-22

Written for whoever picks this project up next, including its author after a
week away. It says what exists, what has been measured, what is open, and in
what order the open items should be taken. Numbers are the measured ones; where
a number is an estimate it says so.

---

## 1. Where we are

### The framework (`R/`, 23 modules, English, tested)

A caret-like layer for dual-branch CNNs in digital soil mapping. The front end
is three calls — `dsm_load()`, a resampling spec (`spatial_cv()`, `knndm_cv()`,
`random_cv()`, `holdout_cv()`, `region_cv()`), `dsm_train()` — and everything
below it is shared by every model family the registry knows (`cnn`, `rf`,
`mlp`, and ~230 more borrowed from caret through `caret_spec()`).

What the framework guarantees, each with the test that proves it:

| guarantee | where | test |
|---|---|---|
| the test set is carved once, frozen, and never scored during tuning | `resample.R`, `test_optimism.R` | `test_selection_order` |
| validation patches share no pixel with training (Chebyshev buffer) | `resample.R` | `test_resample` |
| resume matches units by **hyperparameters**, not by label; a changed split is refused | `utils.R` | `test_run_dirs`, `test_resample_run` |
| a config is ranked against its own seed noise floor; `one_se()` is the default | `resample.R` | `test_resample` |
| conformal intervals with the `(n+1)` correction; coverage is measured, per group | `conformal.R` | `test_conformal` |
| the median surface and the mean surface are both written and labelled | `smearing.R` | `test_smearing` |
| "latest" means newest **finished** run, by time | `utils.R` | `test_run_dirs` |
| a check ledger cannot pass by doing nothing | `checks.R` | `test_checks` |

The suite is `tests/run_all.R`: 31 files, 25 fast and 6 slow, with the
package loaded once for all of them. Every file's accumulator is named and
every `.report()` refuses an empty, unnamed, NA-bearing or non-logical one.

### The worked example (`examples/soc_stock_0_5cm/`)

SOC stock 0–5 cm, ton/ha, trained on `log1p`. 181 global raster predictors at
250 m. Points: **4,154 extracted → 3,766 after QC → 3,728 in the patch store**,
at **2,711 distinct coordinates** (51% of rows share one with another row; the
largest cluster holds 24 profiles). Effective *n* is closer to 2,700 than to
3,728, and the folds group co-located points so none leaks across roles.

The pipeline is closed end to end at `tune_length = 3`:

```
01 prepare -> 02 patches -> 03 tuning -> 03b baselines -> 04 final model
   -> 05 spatial prediction (10 bands, 20 km grid) -> 07 AOA -> 99/99b checks
```

The 20 km map exercises the wiring only: the window is counted in pixels, so a
15×15 patch covers 80× the ground it was trained on, and the script says so in
a banner. The 250 m map has not been produced.

### What has been measured (the numbers a reader should carry)

| finding | value | where |
|---|---|---|
| validation design matters more than model family | random 0.622 / spatial 0.487 / region 0.384 CCC — a 0.24 spread, against 0.025 between families | `reference_performance.md` |
| the back-transform of a log-trained median under-predicts the stock | **−24.4%** bias on the frozen test set, 53% of the MAE; Duan smearing brings it to **+2.7%** | `smearing.R`, 04 output |
| Duan's independence assumption is violated here | S runs **1.79 → 1.15** across prediction quintiles; the two obvious repairs were measured and both make the held-out bias *worse* (−5.8%, −3.9%) | `smearing.R` header |
| the smearing target is bracketed, not pinned | deployed-ensemble S is 1.259 on the refit fold and 1.389 on the test set | `reference_performance.md` |
| the block folds validate a far easier job than the map does | validation-to-train median **15.7 km** vs prediction-to-train **824 km** (52×); kNNDM folds land at 837 km; W1 128 km vs 1,048 km | `_b1_knndm_folds.R` output |
| training is deterministic across processes | seed 7 of cfg_003: CCC 0.480181867591615 in three separate runs — exactly, between interactive sessions; a unit trained in an `Rscript` subprocess differs at 1.2e-4 CCC (0.15% of the seed spread), cause not measured | B2, 04, B6 output |
| a run whose process is KILLED mid-unit resumes into the same numbers | finished units untouched (mtime drift 0 s, comparison rows byte-identical), no orphan checkpoint, worst outcome column at 0.2% of the control's own seed spread | `_b6_resume_check.R` |
| selection optimism | +0.0000 (cfg_003 ranks first on validation and on test) | `score_test_grid()` |
| the two-config branch works and the frozen record survives it | B2: 13/13, original `selection.rds` byte-identical | `_b2_two_config_check.R` |
| on the test set, cfg_002 beats the deployed cfg_003 | paired ΔCCC −0.029, t = −2.80 (df 2), all three seeds agree — **and this number may not be acted on**; it was computed after the freeze | B2 output |
| the validation design costs 0.19 CCC of the reported number | the same 8 configs score 0.480 (best, block folds) and 0.322 (best, kNNDM); all 8 drop, by −0.117 to −0.228 | C1 output |
| under kNNDM the per-config uncertainty triples | mean SE 0.042 against 0.013; **0 of 28 config pairs separated at 2 SE** (blocks: 3 of 28) | C1 output |
| neither design can order this grid | ranking reliability 0.678 (block) and 0.545 (kNNDM) over 3 seeds — a ranking that does not reproduce against itself. Spearman-Brown: **6 seeds** (block), **11** (kNNDM) for rho 0.80 | C1 output |
| D4 augmentation helps, by about as much as the whole architecture search | paired +0.0300 CCC (95% CI [+0.0014, +0.0587], 9 pairs) — against a 0.042 spread between the best and worst of 8 configs, and a 0.0533 seed spread. The document that claimed this had cited a between-rounds comparison that supported nothing | `_b3_augmentation.R` |

### Capability tests (docs/test_plan.md)

| tier | status |
|---|---|
| A (8 cheap paths a second user would hit first) | done, `_capability_sweep.R` |
| B1 kNNDM on the real points | done, 13/13 — produced the 52× finding |
| B2 two configs in stage 04 | done, 13/13 |
| B3 D4 augmentation on/off | **done** — paired +0.0300 CCC, 95% CI [+0.0014, +0.0587], smaller than the 0.0533 seed spread |
| B4 2×2 shards + merge | done, mosaic equals the 1×1 map to 1e-4 t/ha |
| B5 06 and 99b | done — 06 is what exposed the −24% bias |
| B6 resume after a real interruption | **done**, 15/15 — process killed mid-unit; the resumed run matches the control to 0.2% of the seed spread |
| C1 the same grid under two validation designs | **done**, 9/9 — the level moves 0.19 CCC, and neither design separates the configs |

### Tooling that replaces what nobody here can run

Nobody working on this repository through an assistant can run R; the author
runs every script himself. Three Python tools stand in:

| tool | proves |
|---|---|
| `tools/r_skeleton.py` | an edit touched only comments and string contents (code skeleton byte-identical) — used to clear two translation sweeps over 14 files |
| `tools/r_calls.py` | every project function a script calls exists, including `do.call` targets; named arguments match formals |
| `tools/r_lint.py` | the three R mistakes that have actually cost a round trip here: a top-level `else`, the native pipe, and a string literal spanning a line break (33 of those in two days, all from heredocs). Has a `--selftest`, because it once reported "0 findings" while broken |

`tests/test_sources_parse.R` remains the authority on syntax; it runs first.

### Open questions that are scientific, not defects

1. **Which job is the map doing?** Interpolation near profiles (block folds, 16
   km) or extrapolation across a globe with 3,728 points (kNNDM, 837 km)?
   C1 has measured the price: **0.19 CCC**, on every config — and it is a
   price on the number you report, not on the model you deploy, because
   neither design separates the configs (0 of 28 pairs under kNNDM, 3 of 28
   under blocks). So the choice is about what the map claims. That remains
   the author's call; what C1 settled is the shape of the next run.
2. **The heteroscedastic smearing factor.** A global scalar over-corrects low
   predictions and under-corrects high ones, and high predictions carry 60% of
   the stock. Fixing it needs a calibration set produced by the *deployed*
   model — the CV models see ~53% of the data, the deployed one ~69%, and the
   residual structure does not transfer. The current design does not generate
   that set.
3. **Effective sample size.** With 51% of rows co-located, every standard error
   computed as if *n* = 3,728 is optimistic. Nothing in the framework accounts
   for it yet.

---

## 2. Where we go

*(Prioritised after the overnight audit; see section 3 for what the audit
found. Ordered by what a new user or a wrong result would hit first.)*

### Next, in this order

1. ~~Run C1~~ — **done**, see section 1. What it leaves open is question 1:
   which job the map claims to do. That is a decision, not a measurement.
2. ~~B3~~ — **done**. **Tier B is closed**, and so is tier C.
3. **The science run**, whose shape C1 settled: **more seeds, not more
   configs.** 8 configs at 6 seeds (block) or 11 (kNNDM) — not 24 configs at
   3, which would buy nothing a grid of 8 already fails to resolve. Cost goes
   as configs × folds × seeds: 8 × 3 × 11 = 264 units under kNNDM, against
   the 72 just run. Decide the design first.
4. **The 250 m map — not blocked by (3), and this is C1's doing.** The price
   of the design falls on the number you *report*, not on the model you
   *deploy*: no pair of configs is separable under either design, so the
   deployed `cfg_003` is as good as anything the grid would pick. The MEDIAN
   map can be produced now, and the design decision changes its caption, not
   its pixels. **The interval bands are different** (see 5): their width comes
   from the residuals they are calibrated on, so the design decision reaches
   those pixels. Start with `05a_test.R`, which measures RAM and throughput on a
   few real shards — including a dense tropical one — and reports the safe
   `max_concurrent` and the ETA before any of it is committed.

5. **Uncertainty: decided 2026-09-26.** Seeds stay, for a stable ensemble
   median and as an optimiser diagnostic; the map's uncertainty comes from
   conformal prediction, the method with a finite-sample coverage guarantee.
   The interval `05` writes today is constant-width, in native units, and
   calibrated on block-CV residuals (~16 km) for a map that predicts at
   ~824 km. Three upgrades, cheapest first, all in `project_log.md`:
   (a) conformal on the log1p residual — width grows with the predicted level;
   (b) (a) normalised by the dissimilarity index — width grows where the model
   extrapolates; (c) CQR — a quantile head plus conformal, which needs
   retraining. In every one of them the source of the calibration residuals
   is an argument, because it decides which job the interval is honest for.

### The functions that make this a package

What a user needs to go from a table of points and a folder of aligned rasters
to maps, and what exists today:

| step | package function | today |
|---|---|---|
| points + raster folder → patch store, at the windows the user declares | `dsm_prepare()` | **done 2026-09-26** — proven on the SOC data to build the identical store (P1, 19/19); 01 calls it, 02 is folded in |
| folds, buffer, tuning, selection | `dsm_load()`, `*_cv()`, `dsm_train()`, `one_se()` | **done** |
| a tuning grid drawn from the windows in the store | inside `dsm_train()` | **done 2026-09-27** — every window the store holds and every pair; batch sizes that give the smallest fold >= 4 steps an epoch; the inverse read from the store; `n_cores`. The thread count the examples pass (30) awaits `_t1_threads_benchmark.R` |
| refit the chosen config under N seeds | `dsm_final()` | **done 2026-09-27** — seeds side by side with fixed threads per seed (T1, T2), stage 04's selection and post-processing, the declaration of every hyperparameter; `tests/test_final.R` and P3 (the assembly against stage 04's own files, 7/7) passed; 04 still its own script |
| median, mean and interval maps, with the calibration source as an argument | `dsm_predict()` | **done 2026-09-27** — row bands over the whole width, each row decompressed once; the network fully convolutional where exact; every band for every calibration source; a probe at the profiles before the map. `tests/test_predict.R` 29/29; P4 10/10: stage 05's 20 km map to 2e-5, 59x faster, and the probe on the 250 m rasters to 3.8e-7. T3 passed on its fifth run: 2 workers x 7 threads at 21,200 valid px/s on full-width 250 m rows, ~33 h for the globe, 14.7 GB a worker against 14.6 estimated, flat from unit to unit -- once the step's window was made once per worker (mimalloc, under libtorch on Windows, kept every large block freed; T4-T6), with a restart for a worker over its memory as the net. `tests/test_predict.R` 32/32. `05_dsm_predict_global.R` runs the global map |

### Robustness and ease of use (the audit's list, applied)

Section 3 says what was done overnight. What remains, in the order a new user
would hit it:

1. ~~The example headers~~ — **done 2026-09-21**, 26 files. See below.
2. ~~`clamp` as a documented formal of `dsm_train()`~~ — **done 2026-09-28**
   (08d6184): checked at the door for its shape and against the data (an
   observed target outside it is refused), kept by the run (`clamp.rds`; a
   resume with another is refused), and inherited by `dsm_final()`, whose
   summary is what `dsm_predict()` reads.
3. ~~A `.check_k()` helper~~ — **done 2026-09-28** (c2616ea): every
   constructor wants a whole `k` of at least 2 (`region_cv()` may leave it
   NULL) and fractions in [0, 1), and names itself when refusing.
4. ~~`dsm_train()` validation of a user-supplied grid~~ — **done 2026-09-28**
   (0ec26aa): `.check_cnn_grid()`, the columns derived from the parameter
   space; missing and misspelt columns refused with the near name offered,
   `dropout` alone expanded into its five sites, values and ranges checked.
5. ~~Table-model `...`~~ — **done 2026-09-28** (e16fdca): `model_spec()`
   records `fit_args`, derived from `fit()`'s formals; `dsm_train()` refuses
   any other option for a table model at the door.
6. ~~Two Portuguese file names~~ -- **done 2026-09-28**, once the package
   boundary was drawn: `06_graphical_evaluation.R` and its README, its
   function (`make_evaluation_figures()`), its figures and files, all in
   English; new runs write to `outputs/graphical_evaluation/`, and the old
   `outputs/avaliacao_grafica/` is left as it was.

### Toward a package

`R/load_all.R` said it: "when this becomes a package this file disappears and
`library()` takes its place." **It has, 2026-09-28** (0616460 to da449d9):
the repository is the package `soilcnn` (SoilCNN), and `R/load_all.R` is
gone. Verified the same day:
- `tests/run_all.R` passed 31/31, including a tarball installed into a
  temporary library and loaded with `library(soilcnn)` in a fresh process;
- P4 passed 10/10 through the package's own workers: stage 05's 20 km map
  to 3.4e-05, and the probe on the 250 m grid to 1.78e-07.

`.onAttach()` refuses a session whose global environment still holds copies
of the package that were `source()`d from `R/`.

- ~~`DESCRIPTION`, `NAMESPACE` (roxygen)~~ — **done.** Imports are what a core
  call uses unconditionally; what sits behind a `requireNamespace()` guard is
  a Suggest. NAMESPACE is what roxygen2 writes from the `@export` tags: 100
  exports — every non-dot function some script calls — and the 20 print
  methods. `tests/test_package_metadata.R` checks it against the tags, and
  DESCRIPTION against the `pkg::` calls.
- ~~One way to load~~ — **done.** Scripts and tests load the source tree with
  `pkgload::load_all()`; `library(soilcnn)` loads an installed
  copy, built through a tarball (`R CMD INSTALL` of the directory copies
  `data/`). The models register in `.onLoad()`. Workers load what their
  session loaded and stop if the code changed since; `dsm_prepare()`'s band
  workers never load the package. `tests/test_package_install.R` builds,
  installs into a temporary library and loads it in a fresh process.
- **The help pages.** `roxygen2::roxygenise()` writes `man/` from the `#'`
  blocks the modules already carry; roxygen2 is not installed yet.
- **`examples/`.** The worked example stays outside the package — it is the
  SOC project, with its paths and its data, and the tarball leaves it out;
  `quickstart.R` becomes a vignette. 03, 03b and 04 still type
  `n_threads = 30` where `setup_torch_device()` would read the machine (T1
  measured 30 and 15 a wash).
- **The export list is the API as the scripts use it**, not yet as designed:
  some helpers only the check scripts call (`patch_gather()`,
  `patch_window_key()`) may not belong in it. Curated when the examples move.
- ~~**The example headers.**~~ **Done, 2026-09-21.** 26 files carried
  `project_root <- "D:/usuario_armazenamento/..."` and 15 of them fetched
  `install_load_pkg()` from a GitHub URL on every run — so the project ran on
  one machine, and needed the network to start. All 26 now use the
  self-locating snippet that `tests/*.R` and `R/load_all.R` already used, and
  the installer comes from `utils/install_load_pkg.R` on disk. The ordering
  problem this involved: the URL used to be sourced *before* any
  `project_root` existed, and the `rm(list = ls())` that followed would have
  erased one — so the snippet takes the URL's place and the wipe became
  `rm(list = setdiff(ls(), "project_root"))`.

  What is still absolute, deliberately: `predictor_raster_dir` in 01, 02 and
  `_b4`. That is where the **user's data** lives, not where the code lives,
  and no amount of self-location can find it. It is a setting and it belongs
  in sight at the top of the script that needs it.

- The seven `docs/*.md` files that predate the last month (see the audit's
  drift list).

---

## 3. The overnight audit (2026-09-19)

Six lenses over the whole repository — contracts and error paths, resume and
provenance, the checkers, tests, documentation drift, ease of use for a
second user — 105 findings. The automated refutation pass did not complete
(session limits), so every finding acted on was re-read at the line first,
and the ones below are those that survived that reading. Everything is
committed in five batches (`20d634f`, `cf8b020`, `ec27785`, `b42b7d2`,
`53ad956`) and written up in `project_log.md` under this date.

### Fixed tonight — where a wrong result could have looked right

| finding | what it did | now |
|---|---|---|
| a locked output file was silently renamed | `safe_*` wrote `<stem>_<timestamp>` beside the target and returned a path nobody read; the authoritative file kept its old contents | stops, names the file |
| a store whose own manifest said INCOMPLETE loaded | stage 02 wrote `store_complete = FALSE` and stopped; the loader never looked | refused |
| a missing seed checkpoint shrank the map's ensemble | 05 warned "using 2 available" while the summary listed 3 | 05 stops; 04 stops when a seed fails and writes `seeds_fitted`, per config |
| "which config was deployed" had three copies, two reading grid order | with two configs the runner-up could come first | `selected_config_id()`, eight sites |
| a plan that could not be read was treated as no plan | resume proceeded past the one check that exists for it | stops |
| 99 said "All clear" over four skipped stages | a skip left no row | each skip is a WARN row; FAIL is an error, not a `warning()` |
| a `_b4` with no reference snapshot printed PASS | "not compared" folded into `ok_pix = TRUE` | INCOMPLETE, wiring only |
| 05b mosaicked three tiles of a 2×2 | the hole is NA that looks like ocean | count must match the grid the filenames declare; one grid only |
| a fold in which every unit failed died inside `arrange()` | the real cause sat unread in `error_message` | stops with the first error |
| `dsm_train(data, resampling = "cv")`, a `...` typo, a `type_table` without `is_dummy` | failed minutes in, in an internal call | refused at the door, with the choices |

### Fixed tonight — ease of use

`%>%` bound in the framework (a session without `library(dplyr)` died in the
Quickstart); `setup_torch_device()` reads the machine; the installer stops on
a package that will not load and names `install_torch()`; raw `Sys.getenv()`
reads go through `env_*()`; four print methods; eleven messages; 21 strings
with literal line breaks joined; the README's dependencies, run order and
override table; four stale documents corrected.

### Tests added

`test_fold_cache.R` (22) and eight assertions in `test_run_dirs.R`: scaling
from the fold's own rows, `store_complete`, patch centres, the two torch
helpers, the deployed-config rule in both orders, a locked target.

### Verified, not fixed — waiting for the author

Listed in section 2 under *Robustness and ease of use*: the headers, `clamp`
as a formal, `.check_k()`, grid-column validation, table-model `...`.

### Findings that did not survive re-reading

- "`.timestamped_path()`'s shape is grepped for by other code" — nothing
  greps for it; it is gone.
- "`check_patch_centres` else-branch silently passes" — the `try()` around it
  already records FAIL on error.
- "`05a`'s resume marker can belong to another grid" — the marker's name
  carries the grid; what it could not tell was a marker whose tiles were
  deleted, and that is what was fixed.
