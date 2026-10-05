# Configure torch threads and select compute device.

Configure torch threads and select compute device.

## Usage

``` r
setup_torch_device(n_threads = NULL, use_cuda = TRUE)
```

## Arguments

- n_threads:

  Number of intra-op threads; NULL for the physical cores minus one (see
  resolve_cores()).

- use_cuda:

  Use GPU if available.

## Value

A torch_device object.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
before <- Sys.getenv(c("OMP_NUM_THREADS", "MKL_NUM_THREADS"), unset = NA)
n <- torch::torch_get_num_threads()
device <- setup_torch_device(n_threads = 2, use_cuda = FALSE)
device
# the session as it was: torch's threads, and the two variables set here
torch::torch_set_num_threads(n)
Sys.unsetenv(names(before)[is.na(before)])
if (any(!is.na(before))) do.call(Sys.setenv, as.list(before[!is.na(before)]))
}
```
