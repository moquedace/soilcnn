# Status and roadmap — 2026-09-19

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

The suite is `tests/run_all.R`: 24 files, ~790 assertions, ~3.5 min. Every
file's accumulator is named and every `.report()` refuses an empty, unnamed,
NA-bearing or non-logical one.

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

### Capability tests (docs/test_plan.md)

| tier | status |
|---|---|
| A (8 cheap paths a second user would hit first) | done, `_capability_sweep.R` |
| B1 kNNDM on the real points | done, 13/13 — produced the 52× finding |
| B2 two configs in stage 04 | done, 13/13 |
| B3 D4 augmentation on/off | script ready (`_b3_augmentation.R`), not run |
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
| `tools/r_lint.py` | the two R mistakes that have actually cost a round trip here: a top-level `else`, and the native pipe. Has a `--selftest`, because it once reported "0 findings" while broken |

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
2. **B3** (~1 h) — the last open item of tier B. B6 is done.
3. **The science run**, whose shape C1 settled: **more seeds, not more
   configs.** 8 configs at 6 seeds (block) or 11 (kNNDM) — not 24 configs at
   3, which would buy nothing a grid of 8 already fails to resolve. Cost goes
   as configs × folds × seeds: 8 × 3 × 11 = 264 units under kNNDM, against
   the 72 just run. Decide the design first.
4. **The 250 m map**, once a final model exists that was selected for the job
   the map does.

### Robustness and ease of use (the audit's list, applied)

Section 3 says what was done overnight. What remains, in the order a new user
would hit it:

1. **The example headers** (below): the self-locating `project_root` and the
   local installer. One pass, 24 files, suite immediately after.
2. **`clamp` as a documented formal of `dsm_train()`.** It is the one
   argument that can silently destroy predictions and it lives in `...`.
   Additive, safe, not done overnight because it touches the signature.
3. **A `.check_k()` helper** for the five resampling constructors: `k = 2.5`
   or `k = "5"` currently fails somewhere below the constructor.
4. **`dsm_train()` validation of a user-supplied grid** beyond windows: the
   column set of `make_tune_grid()`, derived from the parameter space rather
   than hardcoded.
5. **Table-model `...`:** `model_spec()` could record `fit_args` so a typo
   for `rf` is refused at the door as the CNN's now is.
6. **Two Portuguese file names** kept by decision (`06_avaliacao_grafica.R`
   and its output slugs); rename when the package boundary is drawn.

### Toward a package

`R/load_all.R` says it: "when this becomes a package this file disappears and
`library()` takes its place." What stands between here and that:

- **The example headers.** 24 scripts hardcode
  `project_root <- "D:/usuario_armazenamento/..."`, in at least four header
  variants (some define it twice, two not at all, some without `rm(list =
  ls())`). Fifteen of them fetch `install_load_pkg()` from a GitHub URL — a
  network dependency on every run — while `utils/install_load_pkg.R` in the
  repository is the same file. The fix is one self-locating snippet at the top
  of each script, the one `tests/*.R` and `R/load_all.R` already use:

  ```r
  project_root <- (function() {
    cand <- character(0)
    a <- commandArgs(trailingOnly = FALSE)
    f <- sub("^--file=", "", a[grep("^--file=", a)])
    if (length(f)) cand <- c(cand, dirname(normalizePath(f[1], mustWork = FALSE)))
    for (i in seq_len(sys.nframe())) {
      of <- sys.frame(i)$ofile
      if (!is.null(of) && is.character(of)) cand <- c(cand, dirname(normalizePath(of, mustWork = FALSE)))
    }
    cand <- c(cand, getwd())
    for (d in cand) for (up in c(".", "..", "../..")) {
      r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
      if (file.exists(file.path(r, "R", "load_all.R"))) return(r)
    }
    stop("Project root not found; source() this script by its full path.", call. = FALSE)
  })()
  source(file.path(project_root, "utils", "install_load_pkg.R"))
  ```

  followed by `rm(list = setdiff(ls(), "project_root"))` where a script clears
  its workspace. This was **not** done overnight: it is a 24-file change to the
  scripts the author runs every morning, and nothing here can parse-check it.
  It should be done in one pass, with `tests/run_all.R` run immediately after.
- `DESCRIPTION`, `NAMESPACE` (roxygen), moving `examples/` to `inst/` or a
  vignette, and `setup_torch_device(n_threads = 30)` becoming a default that
  reads the machine.
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
