# The dissimilarity reference of a fitted model.

The centre pixel of every point that trained or validated in the tuning
plan, QC'd and scaled with the MODEL's scaling (stage 07's
construction). Each point's fold is the one it was held out in, so its
cross-validated DI is the distance to the nearest point OUTSIDE its
fold: the dissimilarity the model that predicted it during
cross-validation actually faced. That is the DI a calibration residual
comes with. A point that only ever trained – the training side of a
holdout – has fold `NA` and no cross-validated DI: it is in the
reference, a neighbour to every fold, and out of the threshold.

## Usage

``` r
aoa_reference(points, predictors, qc_table, scaling, plan, weights = NULL)
```

## Arguments

- points:

  Point table aligned to the store (align_points_to_meta()).

- predictors:

  Channel names, in the model's order.

- qc_table:

  QC rules, in the same order.

- scaling:

  The fitted model's predictor_scaling, in the same order.

- plan:

  The tuning run's fold plan.

- weights:

  NULL (every channel alike), or one non-negative weight per channel, in
  the same order – an importance, from importance_weights(), so a point
  is unlike the training data in what the model uses (Meyer & Pebesma
  2021).

## Value

An `aoa_reference`: the DI reference, the AOA threshold, and each used
point's fold and cross-validated DI (`NA` for a point never held out).

## Examples

``` r
ex <- example_landscape()
store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
                     windows = c(3, 7), out_dir = file.path(tempdir(), "landscape"),
                     percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
                     overwrite = TRUE, verbose = FALSE)
data <- dsm_load(store, windows = integer(0), verbose = FALSE)
predictors <- as.character(data$store$predictors)
qc <- utils::read.csv2(file.path(data$patch_dir, "qc_table.csv"))
qc <- qc[match(predictors, qc$predictor), ]
# a fitted model's scaling is its predictor_scaling.csv; here, the profiles' own
x <- as.matrix(data$points[, predictors])
scaling <- data.frame(predictor = predictors, center = colMeans(x),
                      scale = apply(x, 2, sd))
plan <- spatial_folds(data$store$meta, k = 3, block_size = 0.05)
aref <- aoa_reference(data$points, predictors, qc, scaling, plan)
as.numeric(aref$threshold)     # beyond it, the cross-validated error does not apply
#> [1] 0.7424028
head(aref$cv)
#> # A tibble: 6 × 3
#>   sample_id  fold cv_di
#>       <dbl> <int> <dbl>
#> 1         1     2 0.264
#> 2         2     2 0.120
#> 3         3     2 0.167
#> 4         4     2 0.236
#> 5         5     2 0.318
#> 6         6     2 0.236
```
