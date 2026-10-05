# A small fitted run on the example landscape, made once a session.

The store, a tuning run and a final model on
[`example_landscape()`](https://moquedace.github.io/soilcnn/reference/example_landscape.md),
in `dir`: what the examples of the functions that need a fitted model
start from. Made at the first call – a minute or two on one core – and
kept for the session; later calls return it at once. It is an example,
not a model: 160 profiles, one small configuration, 30 epochs.

## Usage

``` r
example_run(dir = file.path(tempdir(), "soilcnn_example"), verbose = FALSE)
```

## Arguments

- dir:

  Where the run is made. The default is in
  [`tempdir()`](https://rdrr.io/r/base/tempfile.html), so it goes with
  the session.

- verbose:

  Report each step.

## Value

A list: `data` (from
[`dsm_load()`](https://moquedace.github.io/soilcnn/reference/dsm_load.md)),
`fit` (from
[`dsm_train()`](https://moquedace.github.io/soilcnn/reference/dsm_train.md)),
`final` (from
[`dsm_final()`](https://moquedace.github.io/soilcnn/reference/dsm_final.md))
and `dir`.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
# \donttest{
run <- example_run()
run$final
# }
}
```
