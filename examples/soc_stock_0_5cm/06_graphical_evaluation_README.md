# Graphical evaluation of the final model

`06_graphical_evaluation.R` reads the results that already exist and draws
seven figures, each in PNG and SVG, with their supporting tables and an HTML
gallery. Nothing is trained again. The figures are base R; the framework is
loaded only to find the run.

## Running it in RStudio

```r
source("examples/soc_stock_0_5cm/06_graphical_evaluation.R", encoding = "UTF-8")
make_evaluation_figures()
```

The results go to `outputs/graphical_evaluation/`; open `gallery.html` there.
(Runs before 2026-09-28 wrote to `outputs/avaliacao_grafica/`, which is left
as it was.)

Another destination, or another final run:

```r
make_evaluation_figures(
  output_dir   = "outputs/graphical_evaluation",
  final_run_id = "final_20260918_150311",
  config_id    = "cfg_003"
)
```

`final_run_id = "latest"` (the default) takes the newest finished final run,
and `config_id = "auto"` the configuration it deployed. The tuning run is read
from the final run's summary.

The script expects the result layout of `soc_stock_0_5cm`, and it needs the
project's own files: the figures alone do not carry the predictions they were
drawn from. Scoring every combination of seeds can take a few minutes.

## The figures

1. The configurations ranked on validation.
2. Performance against training time and the size of the architecture.
3. The seeds' learning curves.
4. Stability on validation, against the original tuning.
5. Observed against predicted, and the ensemble's residuals, on test.
6. What combining seeds gains, over every subset.
7. The size and the direction of the error, by band of the observed stock.

## What they do not say

- The ensemble is the median of the seeds' predictions, in native units.
- The band across combinations is not a confidence interval.
- The training budget differs between tuning and retraining, so figure 4 does
  not isolate the effect of the seed.
- The test figures are descriptive. They are not a way to choose seeds.

## Maintaining it

The parameter count is analytic, from the current architecture. A change to the
architecture needs the counting function updated. The figures are drawn again,
over the old ones, on every run.

## Output files

- `01_ranking` to `07_errors_by_band`, each as `.png` and `.svg`
- `metrics_combinations.csv`, `metrics_bands.csv`,
  `metrics_seeds_validation.csv`, `metrics_ensemble.csv`
- `ranking_with_parameters.csv`
- `summary.rds`
- `gallery.html`
