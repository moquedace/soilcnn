# Context importance: how far from the point, and at which scale, the model reads.

The patch is a neighbourhood. This permutes part of it – the point's own
pixel, a ring of pixels at one distance from it, every ring but the
centre, or one window's whole input – and measures the loss of skill, as
[`permutation_importance()`](https://moquedace.github.io/soilcnn/reference/permutation_importance.md)
does for a variable. The windows are cut around the same point, so a
ring is the same ground in every window, and it takes its values from
one donor in all of them.

## Usage

``` r
context_importance(
  by = c("ring", "window"),
  bands = NULL,
  per_variable = FALSE,
  draws = 5L,
  metric = c("ccc", "rmse", "rmse_transform"),
  seed = 42L
)
```

## Arguments

- by:

  "ring" (the default): one row per distance band. "window": one row per
  window the model reads, that window's whole input permuted – how much
  the branch carries – and, when the model has a gate between its two
  branches, the gate read point by point beside it (`$gate`).

- bands:

  For `by = "ring"`: the bands, a named list of ring numbers – 0 is the
  point's own pixel, ring d the 8d pixels d steps from it. NULL: the
  centre, each ring alone, and the context (every ring but the centre);
  with `per_variable`, the centre and the context.

- per_variable:

  For `by = "ring"`: one row per variable and band – which variables the
  model reads through the neighbourhood, and which at the point alone.
  The variables are those of `groups` in
  [`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md).

- draws:

  Permutations per band (or variable and band, or window) and per model.

- metric:

  What the table is ranked by, as in
  [`permutation_importance()`](https://moquedace.github.io/soilcnn/reference/permutation_importance.md).

- seed:

  Seed of the permutations, shared by every model.

## Value

An `importance_spec`, for
[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md).

## Details

A large importance of the context says the network reads the
neighbourhood. It does not say the neighbourhood adds what the centre
lacks: at 250 m most covariates are smooth, a pixel three cells away is
close to a copy of the centre, and a network can lean on the rim and
learn nothing the centre would not have told it. Ring d holds 8d pixels,
so the table gives the cost per pixel beside the total: a wider ring
costs more for its area alone.

## Examples

``` r
context_importance(by = "ring")
#> <importance_spec> context by ring, 5 draw(s) | ranked by ccc
```
