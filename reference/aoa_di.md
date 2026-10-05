# The DI of new rows of RAW predictor values, QC'd and scaled as the reference was.

The DI of new rows of RAW predictor values, QC'd and scaled as the
reference was.

## Usage

``` r
aoa_di(aref, values)
```

## Arguments

- aref:

  From aoa_reference().

- values:

  Matrix or data frame of raw values, columns in the model's order.

## Value

The DI of each row; NA where a channel is missing after QC.

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
aoa_di(aref, data$points[1:5, predictors])
#> [1] 0 0 0 0 0
```
