# Set torch's thread pools.

Split out of setup_torch_device() so that a caller who already holds a
device can change the threads without building a second one –
dsm_train(device = d, n_cores = 8) is that caller. The threads belong to
the R session, not to a device: whatever set them last is what every
later torch call gets.

## Usage

``` r
set_torch_threads(n_threads = NULL)
```

## Arguments

- n_threads:

  NULL for the physical cores minus one (see resolve_cores()), or a
  whole number \>= 1.

## Value

The number of threads, invisibly.

## Details

The interop pool can be sized once per session, before torch first runs
work in parallel; later calls leave it as it is and say so once.

## Examples

``` r
if (FALSE) { # torch::torch_is_installed()
before <- Sys.getenv(c("OMP_NUM_THREADS", "MKL_NUM_THREADS"), unset = NA)
n <- torch::torch_get_num_threads()
set_torch_threads(2)
torch::torch_get_num_threads()
# the session as it was: torch's threads, and the two variables set here
torch::torch_set_num_threads(n)
Sys.unsetenv(names(before)[is.na(before)])
if (any(!is.na(before))) do.call(Sys.setenv, as.list(before[!is.na(before)]))
}
```
