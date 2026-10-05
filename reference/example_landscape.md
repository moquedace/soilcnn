# A small synthetic landscape, for the examples and the tests.

80 x 80 cells of 0.0025 degrees (about 250 m) in coordinates of
south-eastern Brazil – the place is borrowed, the landscape is made up –
with eight predictors and 160 soil profiles, 90 of them in six survey
clusters. The soil organic carbon stock (t/ha, 0-30 cm) was drawn from
vegetation, clay, temperature, geology and the topographic position of
the cell in its 7 x 7 neighbourhood: a network that reads the
neighbourhood has something to find that the centre pixel does not show.

## Usage

``` r
example_landscape()
```

## Value

A list: `raster_dir`, the folder of the eight GeoTIFFs (elevation,
temperature, precipitation, ndvi, clay_pct, and geology as three 0/1
dummies); `profiles`, a data frame of `profile_id`, `x`, `y`, `survey`
and `soc_stock`; and `truth`, the formula the stock was drawn from.

## Examples

``` r
ex <- example_landscape()
head(ex$profiles)
#>   profile_id         x         y   survey soc_stock
#> 1       p001 -49.57785 -20.16780 survey_1     43.21
#> 2       p002 -49.58180 -20.15074 survey_1     43.06
#> 3       p003 -49.58201 -20.15571 survey_1     40.98
#> 4       p004 -49.57326 -20.15419 survey_1     77.12
#> 5       p005 -49.57106 -20.15406 survey_1     51.35
#> 6       p006 -49.57359 -20.15379 survey_1     63.81
list.files(ex$raster_dir)
#> [1] "clay_pct.tif"          "elevation.tif"         "geology_basalt.tif"   
#> [4] "geology_granite.tif"   "geology_sandstone.tif" "ndvi.tif"             
#> [7] "precipitation.tif"     "temperature.tif"      
```
