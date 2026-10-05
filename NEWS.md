# soilcnn 0.1.0

First release.

* Patch store from points and aligned rasters (`dsm_prepare()`, `dsm_load()`),
  with a subsample by whole spatial blocks, stratified by region on request.
* Resampling specs for spatial blocks, kNNDM, random folds, a holdout and
  groups (`spatial_cv()`, `knndm_cv()`, `random_cv()`, `holdout_cv()`,
  `region_cv()`), with buffers that keep training patches from touching
  validation and test patches.
* Tuning of a dual-branch convolutional network over the plan
  (`dsm_train()`), a seed noise floor and the one-standard-error rule
  (`seed_noise_floor()`, `one_se()`), and random forest and multilayer
  perceptron baselines on the same folds; caret models through
  `caret_spec()`.
* A frozen test set scored only after the choice (`freeze_selection()`,
  `score_test_grid()`).
* The selected configuration refitted under several seeds (`dsm_final()`),
  and maps of any extent (`dsm_predict()`): ensemble median and spread, the
  mean by smearing (`smear()`, `smear_map()`, `smearing_check()`), conformal
  intervals calibrated on cross-validated residuals, with coverage checked on
  the test set (`conformal_calibrate()`, `conformal_scaled_calibrate()`)
  and the area of applicability (`aoa_reference()`, `dissimilarity_index()`).
* Variable importance by permutation, SHAP (expected gradients, integrated
  gradients, kernel), SAGE, refits without each variable and accumulated local
  effects (`dsm_importance()` and its methods), with maps of SHAP values.
* Intervals for clustered test points from whole blocks of equal area
  (`block_bootstrap()`, `equal_area_blocks()`, `spatial_correlogram()`).
* A small synthetic landscape for examples and tests (`example_landscape()`,
  `example_run()`).
