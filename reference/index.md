# Package index

## All functions

- [`ale_effect()`](https://moquedace.github.io/soilcnn/reference/ale_effect.md)
  : ALE: how the prediction changes along each variable's range.

- [`aoa_di()`](https://moquedace.github.io/soilcnn/reference/aoa_di.md)
  : The DI of new rows of RAW predictor values, QC'd and scaled as the
  reference was.

- [`aoa_reference()`](https://moquedace.github.io/soilcnn/reference/aoa_reference.md)
  : The dissimilarity reference of a fitted model.

- [`aoa_threshold()`](https://moquedace.github.io/soilcnn/reference/aoa_threshold.md)
  : The DI threshold that separates the area of applicability.

- [`block_bootstrap()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap.md)
  : A metric, or two models' difference in it, with an interval from
  whole blocks.

- [`block_bootstrap_by_size()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap_by_size.md)
  : The interval as the blocks grow.

- [`calc_metrics()`](https://moquedace.github.io/soilcnn/reference/calc_metrics.md)
  : Compute all regression metrics for one obs/pred pair.

- [`caret_available()`](https://moquedace.github.io/soilcnn/reference/caret_available.md)
  : Every caret method that could be borrowed here.

- [`caret_spec()`](https://moquedace.github.io/soilcnn/reference/caret_spec.md)
  : Turn a caret method into a model_spec.

- [`ccc()`](https://moquedace.github.io/soilcnn/reference/ccc.md) :
  Lin's Concordance Correlation Coefficient.

- [`check_fold_plan()`](https://moquedace.github.io/soilcnn/reference/check_fold_plan.md)
  : Check a plan's folds are well formed, and describe them.

- [`compare_importance()`](https://moquedace.github.io/soilcnn/reference/compare_importance.md)
  : Several importances, side by side.

- [`conformal_calibrate()`](https://moquedace.github.io/soilcnn/reference/conformal_calibrate.md)
  : Calibrate a conformal interval from held-out residuals.

- [`conformal_cv()`](https://moquedace.github.io/soilcnn/reference/conformal_cv.md)
  : Calibrate on the folds of a resampling run, then report coverage.

- [`conformal_interval()`](https://moquedace.github.io/soilcnn/reference/conformal_interval.md)
  : Turn predictions into intervals.

- [`conformal_scale_fit()`](https://moquedace.github.io/soilcnn/reference/conformal_scale_fit.md)
  : Fit the scale of an interval: \|residual\| on covariates that say
  how wrong.

- [`conformal_scaled_calibrate()`](https://moquedace.github.io/soilcnn/reference/conformal_scaled_calibrate.md)
  : Calibrate an interval whose width is a fitted scale.

- [`conformal_scaled_interval()`](https://moquedace.github.io/soilcnn/reference/conformal_scaled_interval.md)
  : Turn predictions into intervals with a fitted scale.

- [`context_importance()`](https://moquedace.github.io/soilcnn/reference/context_importance.md)
  : Context importance: how far from the point, and at which scale, the
  model reads.

- [`cv_plus_calibrate()`](https://moquedace.github.io/soilcnn/reference/cv_plus_calibrate.md)
  : Calibrate a CV+ interval from the folds of a cross-validation.

- [`cv_plus_interval()`](https://moquedace.github.io/soilcnn/reference/cv_plus_interval.md)
  : Turn the fold models' predictions into CV+ intervals.

- [`cv_residuals()`](https://moquedace.github.io/soilcnn/reference/cv_residuals.md)
  : Cross-validated residuals from a tuning run, for calibration.

- [`cv_residuals_for_config()`](https://moquedace.github.io/soilcnn/reference/cv_residuals_for_config.md)
  : Cross-validated residuals of a configuration, from any tuning run.

- [`di_reference()`](https://moquedace.github.io/soilcnn/reference/di_reference.md)
  : Summarise a training set for dissimilarity-index computation.

- [`dissimilarity_index()`](https://moquedace.github.io/soilcnn/reference/dissimilarity_index.md)
  : The dissimilarity index of new data.

- [`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)
  : Refit the configuration a tuning run selects, under N seeds.

- [`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md)
  : Variable importance of a fitted model, by the method you choose.

- [`dsm_load()`](https://moquedace.github.io/soilcnn/reference/dsm_load.md)
  : Load everything a run needs, and check that it fits together.

- [`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md)
  : Predict a map from a final model.

- [`dsm_prepare()`](https://moquedace.github.io/soilcnn/reference/dsm_prepare.md)
  : Build a patch store from a point table and a folder of aligned
  rasters.

- [`dsm_report_final()`](https://moquedace.github.io/soilcnn/reference/dsm_report_final.md)
  : The declaration of a final model that already exists.

- [`dsm_train()`](https://moquedace.github.io/soilcnn/reference/dsm_train.md)
  : Fit and tune a model over a resampling plan.

- [`equal_area_blocks()`](https://moquedace.github.io/soilcnn/reference/equal_area_blocks.md)
  : Square blocks of equal area, in kilometres.

- [`example_landscape()`](https://moquedace.github.io/soilcnn/reference/example_landscape.md)
  : A small synthetic landscape, for the examples and the tests.

- [`example_run()`](https://moquedace.github.io/soilcnn/reference/example_run.md)
  : A small fitted run on the example landscape, made once a session.

- [`fold_leakage_report()`](https://moquedace.github.io/soilcnn/reference/fold_leakage_report.md)
  : How much does each fold still leak?

- [`freeze_selection()`](https://moquedace.github.io/soilcnn/reference/freeze_selection.md)
  : Record which config was chosen, and when.

- [`get_model()`](https://moquedace.github.io/soilcnn/reference/get_model.md)
  : Fetch a registered model.

- [`holdout()`](https://moquedace.github.io/soilcnn/reference/holdout.md)
  : A single train/validation/test split of a point table.

- [`holdout_cv()`](https://moquedace.github.io/soilcnn/reference/holdout_cv.md)
  : A single train/validation/test split.

- [`importance_groups()`](https://moquedace.github.io/soilcnn/reference/importance_groups.md)
  : Which channels the importance treats as one variable.

- [`importance_map()`](https://moquedace.github.io/soilcnn/reference/importance_map.md)
  : Maps of SHAP values, from an importance computed at points of the
  map.

- [`importance_points()`](https://moquedace.github.io/soilcnn/reference/importance_points.md)
  : Points of the map to explain: a regular grid of the model's raster
  cells.

- [`importance_weights()`](https://moquedace.github.io/soilcnn/reference/importance_weights.md)
  : One weight per channel, for the area of applicability, from an
  importance.

- [`inside_aoa()`](https://moquedace.github.io/soilcnn/reference/inside_aoa.md)
  : Is each row inside the area of applicability?

- [`knndm_cv()`](https://moquedace.github.io/soilcnn/reference/knndm_cv.md)
  : Folds matched to where the map will be predicted (kNNDM).

- [`knndm_folds()`](https://moquedace.github.io/soilcnn/reference/knndm_folds.md)
  : Fold plan from k-fold Nearest Neighbour Distance Matching.

- [`latest_run_dir()`](https://moquedace.github.io/soilcnn/reference/latest_run_dir.md)
  :

  The most recent run under `base`, by time, and only if it finished.

- [`list_models()`](https://moquedace.github.io/soilcnn/reference/list_models.md)
  : Every registered model, as a table.

- [`make_manual_tune_grid()`](https://moquedace.github.io/soilcnn/reference/make_manual_tune_grid.md)
  : Build a manual grid from explicit lists of values per parameter.

- [`make_tune_grid()`](https://moquedace.github.io/soilcnn/reference/make_tune_grid.md)
  : Generate a random hyperparameter grid (like caret's tuneLength).

- [`model_spec()`](https://moquedace.github.io/soilcnn/reference/model_spec.md)
  : Describe a model the framework can fit.

- [`occlusion_report()`](https://moquedace.github.io/soilcnn/reference/occlusion_report.md)
  : What each part of the patch is worth to one trained unit of a tuning
  run.

- [`one_se()`](https://moquedace.github.io/soilcnn/reference/one_se.md)
  : Pick the simplest config within one standard error of the best.

- [`paired_family_test()`](https://moquedace.github.io/soilcnn/reference/paired_family_test.md)
  : Paired comparison between two model families.

- [`permutation_importance()`](https://moquedace.github.io/soilcnn/reference/permutation_importance.md)
  : Permutation importance: how much a fitted model relies on each
  variable.

- [`picp_report()`](https://moquedace.github.io/soilcnn/reference/picp_report.md)
  : Coverage, and whether it holds where it is needed.

- [`plot(`*`<block_bootstrap_by_size>`*`)`](https://moquedace.github.io/soilcnn/reference/plot.block_bootstrap_by_size.md)
  : Plot the interval as the blocks grow.

- [`plot(`*`<dsm_importance>`*`)`](https://moquedace.github.io/soilcnn/reference/plot.dsm_importance.md)
  : Plot a variable importance.

- [`plot(`*`<importance_comparison>`*`)`](https://moquedace.github.io/soilcnn/reference/plot.importance_comparison.md)
  : Plot importances side by side.

- [`plot(`*`<importance_map>`*`)`](https://moquedace.github.io/soilcnn/reference/plot.importance_map.md)
  : Plot maps of SHAP values.

- [`plot(`*`<spatial_correlogram>`*`)`](https://moquedace.github.io/soilcnn/reference/plot.spatial_correlogram.md)
  : Plot a spatial correlogram.

- [`prediction_sample()`](https://moquedace.github.io/soilcnn/reference/prediction_sample.md)
  :

  A regular sample of the prediction area, as kNNDM's `predpoints`.

- [`print_aoa()`](https://moquedace.github.io/soilcnn/reference/print_aoa.md)
  : Report what an AOA covers.

- [`print_noise_floor()`](https://moquedace.github.io/soilcnn/reference/print_noise_floor.md)
  : Print a noise-floor report in the terms it should be read in.

- [`print_one_se()`](https://moquedace.github.io/soilcnn/reference/print_one_se.md)
  : Say what one_se() did, including when it did nothing.

- [`random_cv()`](https://moquedace.github.io/soilcnn/reference/random_cv.md)
  : Random k-fold. Ignores geography by construction.

- [`random_folds()`](https://moquedace.github.io/soilcnn/reference/random_folds.md)
  : Random k-fold over the train+validation pool.

- [`refit_importance()`](https://moquedace.github.io/soilcnn/reference/refit_importance.md)
  : Refit importance: what the model loses when trained without a
  variable.

- [`region_cv()`](https://moquedace.github.io/soilcnn/reference/region_cv.md)
  : Leave-region-out, on a grouping that already exists (biome,
  catchment, ...).

- [`region_folds()`](https://moquedace.github.io/soilcnn/reference/region_folds.md)
  : Fold by a grouping column: no group is split across folds.

- [`register_model()`](https://moquedace.github.io/soilcnn/reference/register_model.md)
  : Add a model to the registry.

- [`resolve_resampling()`](https://moquedace.github.io/soilcnn/reference/resolve_resampling.md)
  : Turn a spec into a fold plan against real points.

- [`sage_importance()`](https://moquedace.github.io/soilcnn/reference/sage_importance.md)
  : SAGE: how much of the model's skill each variable carries, shared
  fairly.

- [`score_test_grid()`](https://moquedace.github.io/soilcnn/reference/score_test_grid.md)
  : Score every trained unit of a tuning run on the held-out test set.

- [`seed_noise_floor()`](https://moquedace.github.io/soilcnn/reference/seed_noise_floor.md)
  : How much does a metric move when ONLY the seed changes?

- [`selected_config_id()`](https://moquedace.github.io/soilcnn/reference/selected_config_id.md)
  : The configuration a final run deployed.

- [`set_torch_threads()`](https://moquedace.github.io/soilcnn/reference/set_torch_threads.md)
  : Set torch's thread pools.

- [`setup_torch_device()`](https://moquedace.github.io/soilcnn/reference/setup_torch_device.md)
  : Configure torch threads and select compute device.

- [`shap_importance()`](https://moquedace.github.io/soilcnn/reference/shap_importance.md)
  : SHAP importance: how each variable pushes each prediction, up or
  down.

- [`smear()`](https://moquedace.github.io/soilcnn/reference/smear.md) :
  The conditional MEAN surface, from transform-space predictions.

- [`smear_map()`](https://moquedace.github.io/soilcnn/reference/smear_map.md)
  : The mean surface of a median map already made.

- [`smearing_check()`](https://moquedace.github.io/soilcnn/reference/smearing_check.md)
  : The smearing factors on held-out points: which one unbiases the mean
  surface.

- [`smearing_factor()`](https://moquedace.github.io/soilcnn/reference/smearing_factor.md)
  : The smearing factor, from held-out residuals in transform space.

- [`smearing_from_run()`](https://moquedace.github.io/soilcnn/reference/smearing_from_run.md)
  : The smearing factor from a tuning run's out-of-fold predictions.

- [`spatial_correlogram()`](https://moquedace.github.io/soilcnn/reference/spatial_correlogram.md)
  : Moran's I by distance class: how far apart two points stop
  resembling each other.

- [`spatial_cv()`](https://moquedace.github.io/soilcnn/reference/spatial_cv.md)
  : Spatially blocked k-fold: whole blocks of ground go to one fold.

- [`spatial_folds()`](https://moquedace.github.io/soilcnn/reference/spatial_folds.md)
  : Spatially blocked folds of a point table.

- [`summarise_resamples()`](https://moquedace.github.io/soilcnn/reference/summarise_resamples.md)
  : Aggregate a unit-level comparison table into one row per config.

- [`target_transform_spec()`](https://moquedace.github.io/soilcnn/reference/target_transform_spec.md)
  : The forward and inverse functions for a named target transform.
