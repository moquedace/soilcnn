# ALE: how the prediction changes along each variable's range.

Accumulated local effects (Apley & Zhu 2020). A variable's range is cut
into bins by its quantiles at the points; within each bin, every point
is moved to the bin's lower edge and to its upper edge, and the
difference of the two predictions is the local effect. Accumulated
across the bins and centred, it is the curve. A point is moved only
across its own bin, so the model is asked about values close to the ones
it saw – which is why ALE is read with correlated predictors, where a
partial dependence plot would ask about combinations that do not occur.

## Usage

``` r
ale_effect(variables = NULL, bins = 20L)
```

## Arguments

- variables:

  Which variables, named as
  [`importance_groups()`](https://moquedace.github.io/soilcnn/reference/importance_groups.md)
  names them (a channel, or a one-hot set). NULL: all of them. Groups of
  your own are not used: an effect needs one variable's values.

- bins:

  Bins of a continuous variable's range, cut at its quantiles at the
  points – fewer where it has fewer distinct values.

## Value

An `importance_spec`, for
[`dsm_importance()`](https://moquedace.github.io/soilcnn/reference/dsm_importance.md).
Its importance is the spread of the curve over the points – the standard
deviation of the effect at their values; flat is no effect (Greenwell et
al. 2018 for the idea, on partial dependence).

## Details

A point's whole patch is moved, every pixel and every window by the same
amount: the neighbourhood keeps its texture and changes its level.
Moving the centre pixel alone would build a spike no landscape has. A
categorical – a one-hot set, or a binary map alone – has no range: its
effect is the prediction with the whole patch made each class, less the
prediction as it is. That is a partial dependence, which asks about a
class also where it does not occur; the number of points in each class
says where it does.

## Examples

``` r
ale_effect(bins = 10)
#> <importance_spec> ALE, 10 bin(s)
```
