# Describe a model the framework can fit.

Describe a model the framework can fit.

## Usage

``` r
model_spec(
  name,
  input,
  fit,
  predict,
  default_grid = NULL,
  count_params = NULL,
  description = "",
  fit_args = NULL
)
```

## Arguments

- name:

  Unique identifier, e.g. "rf", "mlp", "cnn".

- input:

  "table" or "patches" – which view of the fold cache the model
  consumes.

- fit:

  function(x, y, cfg, ...) returning a fitted object.

- predict:

  function(object, x, ...) returning a numeric vector.

- default_grid:

  function(tune_length, seed, x, y) returning a tibble with a config_id
  column, or NULL if the model has nothing to tune.

  x AND y ARE THE REAL TRAINING DATA OF THE FIRST FOLD, and they are in
  the signature because some grid generators need them. mtry is a
  fraction of ncol(x); glmnet's lambda path is computed from the VALUES.
  A grid built against a synthetic matrix of the right width is correct
  for the first kind and quietly wrong for the second, so the framework
  hands over the real thing and a generator takes what it needs.

  A generator that needs neither still has to accept them – an argument
  it ignores costs nothing, and a signature that varies per model is a
  signature the runner cannot call.

  A PATCH model's generator may also take `windows`, the patch sizes the
  loaded store holds, and `n_train`, the training points of the smallest
  fold. dsm_train() passes each one the generator declares, so a default
  grid never asks for a window the store does not have – the CNN's grid
  used to draw 3/9/15, the SOC example's, from any store – or for a
  batch no fold can fill.

- count_params:

  function(object) returning the number of free parameters, used as the
  complexity axis in one_se(). NULL means the model cannot report it and
  one_se() falls back to its other rule.

- description:

  One line, printed by list_models().

- fit_args:

  The arguments a caller may hand fit() through dsm_train()'s `...`.
  NULL derives them from fit()'s formals, less the ones the framework
  supplies itself (x, y, cfg, x_val, y_val, device, n_cores,
  points_valid, transform, model_name) and `...`. dsm_train() refuses
  any other: a misspelt option for a table model used to vanish into
  fit()'s `...` and train as if it had not been given.

## Value

A model_spec.

## Examples

``` r
mean_only <- model_spec(
  "mean_only", input = "table",
  fit = function(x, y, cfg, ...) list(mu = mean(y)),
  predict = function(object, x, ...) rep(object$mu, nrow(x)),
  description = "the training mean: a floor every model must beat")
mean_only
#> <model_spec> mean_only
#>   input       : table
#>   tunable     : no
#>   reports size: no
#>   the training mean: a floor every model must beat
```
