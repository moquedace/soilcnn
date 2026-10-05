# Which channels the importance treats as one variable.

A categorical raster arrives as several 0/1 channels, one per class.
Permuting one of them alone makes points that belong to two classes or
to none, so the channels of one categorical are one variable. The
package is not told which raster a dummy came from, so it infers the
sets: dummies that share a name prefix (cut at "\_") and are never 1
together at the points. Every other channel is a variable of its own.
The sets found are listed; check them, and give your own grouping when
they are not what the rasters were.

## Usage

``` r
importance_groups(data, groups = "auto")
```

## Arguments

- data:

  The `dsm_data` the model was fitted on, from
  [`dsm_load()`](https://moquedace.github.io/soilcnn/reference/dsm_load.md).

- groups:

  "auto" (the default), "channel" (every channel alone, dummies too –
  see above for what that measures), or your grouping: a data frame with
  columns `channel` and `variable`, or a named list (variable = channel
  names). Channels it does not name keep the automatic rule.

## Value

A tibble in channel order: `channel`, `variable`, and `rule` ("alone",
"one-hot set" or "yours").

## Details

A grouping of your own can be anything that makes sense to perturb as
one: the classes of a categorical named outright, or themes (climate,
relief, vegetation...), which answers how much the model relies on each
theme as a block – the way to read many correlated variables.

## Examples

``` r
ex <- example_landscape()
store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
                     windows = c(3, 7), out_dir = file.path(tempdir(), "landscape"),
                     percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
                     overwrite = TRUE, verbose = FALSE)
data <- dsm_load(store, windows = integer(0), verbose = FALSE)
importance_groups(data)          # the geology dummies, found as one variable
#> # A tibble: 8 × 3
#>   channel           variable      rule       
#>   <chr>             <chr>         <chr>      
#> 1 clay_pct          clay_pct      alone      
#> 2 elevation         elevation     alone      
#> 3 geology_basalt    geology       one-hot set
#> 4 geology_granite   geology       one-hot set
#> 5 geology_sandstone geology       one-hot set
#> 6 ndvi              ndvi          alone      
#> 7 precipitation     precipitation alone      
#> 8 temperature       temperature   alone      
themes <- data.frame(channel = c("temperature", "precipitation"), variable = "climate")
importance_groups(data, themes)  # and two channels made one by you
#> # A tibble: 8 × 3
#>   channel           variable  rule       
#>   <chr>             <chr>     <chr>      
#> 1 clay_pct          clay_pct  alone      
#> 2 elevation         elevation alone      
#> 3 geology_basalt    geology   one-hot set
#> 4 geology_granite   geology   one-hot set
#> 5 geology_sandstone geology   one-hot set
#> 6 ndvi              ndvi      alone      
#> 7 precipitation     climate   yours      
#> 8 temperature       climate   yours      
```
