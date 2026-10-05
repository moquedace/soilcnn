# Square blocks of equal area, in kilometres.

Each point falls in a square cell of `size_km` on a Lambert azimuthal
equal-area projection centred on the points (on the sphere of WGS84's
authalic radius), so every block covers the same ground wherever it lies
– which a grid in degrees does not: a degree of longitude is 111 km at
the equator and 71 km at 50 degrees.

## Usage

``` r
equal_area_blocks(x, y, size_km, coords = c("lonlat", "metres"), centre = NULL)
```

## Arguments

- x, y:

  Longitude and latitude in degrees (`coords = "lonlat"`), or easting
  and northing in metres of an equal-area projection of yours
  (`coords = "metres"`).

- size_km:

  The side of a block, in km.

- coords:

  "lonlat" (the default) or "metres".

- centre:

  For "lonlat": the projection's centre, `c(longitude, latitude)`. NULL:
  the points' own means.

## Value

A character vector, one block per point (the cell's column and row,
"i_j"), with the size and the projection as attributes.

## Examples

``` r
ex <- example_landscape()
blk <- equal_area_blocks(ex$profiles$x, ex$profiles$y, size_km = 2)
length(unique(blk))    # blocks holding a profile
#> [1] 60
table(table(blk))      # how many blocks hold 1, 2, ... profiles
#> 
#>  1  2  3  4  6  8 10 16 21 
#> 34 12  5  1  3  1  2  1  1 
```
