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

The suite is `tests/run_all.R`: 23 files, ~760 assertions, ~3.5 min. Every
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
| training is deterministic across processes | seed 7 of cfg_003: CCC 0.480181867591615 in three separate runs | B2 output vs 04 output |
| selection optimism | +0.0000 (cfg_003 ranks first on validation and on test) | `score_test_grid()` |
| the two-config branch works and the frozen record survives it | B2: 13/13, original `selection.rds` byte-identical | `_b2_two_config_check.R` |
| on the test set, cfg_002 beats the deployed cfg_003 | paired ΔCCC −0.029, t = −2.80 (df 2), all three seeds agree — **and this number may not be acted on**; it was computed after the freeze | B2 output |

### Capability tests (docs/test_plan.md)

| tier | status |
|---|---|
| A (8 cheap paths a second user would hit first) | done, `_capability_sweep.R` |
| B1 kNNDM on the real points | done, 13/13 — produced the 52× finding |
| B2 two configs in stage 04 | done, 13/13 |
| B3 D4 augmentation on/off | script ready (`_b3_augmentation.R`), not run |
| B4 2×2 shards + merge | done, mosaic equals the 1×1 map to 1e-4 t/ha |
| B5 06 and 99b | done — 06 is what exposed the −24% bias |
| B6 resume after a real interruption | script ready (`_b6_resume_check.R`), not run |
| C1 the same grid under two validation designs | both runs **finished** (8 configs × 3 folds × 3 seeds each); `_c1_design_comparison.R` not yet run |

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
   C1 measures how much the answer changes; it cannot say which question is
   the right one. That is the author's call and it decides the science run.
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

1. **Run C1** (seconds) on the two finished design runs. Then decide question
   1 above. Everything about the science run depends on it.
2. **B3 and B6** (~1 h and one deliberate interruption). They close tier B.
3. **The science run** at the design C1 argues for — or both, if the ranking
   ceiling C1 reports says the extra configs would buy nothing.
4. **The 250 m map**, once a final model exists that was selected for the job
   the map does.

### Robustness and ease of use (the audit's list, applied)

See section 3. Items marked *done tonight* are already committed.

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

## 3. The overnight audit

*(Filled in when the six-lens audit completes: what it found, what survived
refutation, what was fixed tonight, what waits for the author.)*
