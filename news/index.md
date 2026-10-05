# Changelog

## soilcnn 0.1.0

First release.

- Patch store from points and aligned rasters
  ([`dsm_prepare()`](https://moquedace.github.io/soilcnn/reference/dsm_prepare.md),
  [`dsm_load()`](https://moquedace.github.io/soilcnn/reference/dsm_load.md)),
  with a subsample by whole spatial blocks, stratified by region on
  request.
- Resampling specs for spatial blocks, kNNDM, random folds, a holdout
  and groups
  ([`spatial_cv()`](https://moquedace.github.io/soilcnn/reference/spatial_cv.md),
  [`knndm_cv()`](https://moquedace.github.io/soilcnn/reference/knndm_cv.md),
  [`random_cv()`](https://moquedace.github.io/soilcnn/reference/random_cv.md),
  [`holdout_cv()`](https://moquedace.github.io/soilcnn/reference/holdout_cv.md),
  [`region_cv()`](https://moquedace.github.io/soilcnn/reference/region_cv.md)),
  with buffers that keep training patches from touching validation and
  test patches.
- Tuning of a dual-branch convolutional network over the plan
  ([`dsm_train()`](https://moquedace.github.io/soilcnn/reference/dsm_train.md)),
  a seed noise floor and the one-standard-error rule
  ([`seed_noise_floor()`](https://moquedace.github.io/soilcnn/reference/seed_noise_floor.md),
  [`one_se()`](https://moquedace.github.io/soilcnn/reference/one_se.md)),
  and random forest and multilayer perceptron baselines on the same
  folds; caret models through
  [`caret_spec()`](https://moquedace.github.io/soilcnn/reference/caret_spec.md).
- A frozen test set scored only after the choice
  ([`freeze_selection()`](https://moquedace.github.io/soilcnn/reference/freeze_selection.md),
  [`score_test_grid()`](https://moquedace.github.io/soilcnn/reference/score_test_grid.md)).
- The selected configuration refitted under several seeds
  ([`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md)),
  and maps of any extent
  ([`dsm_predict()`](https://moquedace.github.io/soilcnn/reference/dsm_predict.md)):
  ensemble median and spread, the mean by smearing
  ([`smear()`](https://moquedace.github.io/soilcnn/reference/smear.md),
  [`smear_map()`](https://moquedace.github.io/soilcnn/reference/smear_map.md),
  [`smearing_check()`](https://moquedace.github.io/soilcnn/reference/smearing_check.md)),
  conformal intervals calibrated on cross-validated residuals, with
  coverage checked on the test set
  ([`conformal_calibrate()`](https://moquedace.github.io/soilcnn/reference/conformal_calibrate.md),
  [`conformal_scaled_calibrate()`](https://moquedace.github.io/soilcnn/reference/conformal_scaled_calibrate.md))
  and the area of applicability
  ([`aoa_reference()`](https://moquedace.github.io/soilcnn/reference/aoa_reference.md),
  [`dissimilarity_index()`](https://moquedace.github.io/soilcnn/reference/dissimilarity_index.md)).
- Variable importance by permutation, SHAP (expected gradients,
  integrated gradients, kernel), SAGE, refits without each variable and
  accumulated local effects
  ([`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md)
  and its methods), with maps of SHAP values.
- Intervals for clustered test points from whole blocks of equal area
  ([`block_bootstrap()`](https://moquedace.github.io/soilcnn/reference/block_bootstrap.md),
  [`equal_area_blocks()`](https://moquedace.github.io/soilcnn/reference/equal_area_blocks.md),
  [`spatial_correlogram()`](https://moquedace.github.io/soilcnn/reference/spatial_correlogram.md)).
- A small synthetic landscape for examples and tests
  ([`example_landscape()`](https://moquedace.github.io/soilcnn/reference/example_landscape.md),
  [`example_run()`](https://moquedace.github.io/soilcnn/reference/example_run.md)).
