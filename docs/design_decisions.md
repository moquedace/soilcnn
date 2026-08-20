# Design Decisions

This document explains every non-trivial architectural and training choice in the framework.
For each component the structure is: **what it is → why we use it → what the alternative is → connection to digital soil mapping (DSM)**.

Intended audience: practitioners who want to understand, reproduce, or adapt the framework.
For parameter ranges and tuning guidance see [`tuning_guide.md`](tuning_guide.md).
For the architecture diagram and module structure see [`architecture.md`](architecture.md).

---

## 1. Target transform — `log1p` / `expm1`

### What it is
The raw target (SOC stock, ton/ha) is transformed to `log1p(y) = log(1 + y)` before training.
All model outputs are in log1p space. Predictions are back-transformed via `expm1(ŷ) = exp(ŷ) − 1`.

### Why
SOC stock distributions are strongly right-skewed: the bulk of profiles cluster between 0 and
50 ton/ha, but a long tail of organic-rich soils (peatlands, histosols) reaches several hundred.
Training directly on the raw scale means:

- The loss is dominated by the extreme tail — a single high-value outlier has disproportionate
  gradient weight.
- The network spends most of its capacity fitting the rare high end, at the expense of the
  common range.
- Residuals in the common range (0–50 ton/ha) look small in absolute terms but are large
  proportionally.

`log1p` compresses the tail and expands the common range, making the loss landscape more
symmetric and the gradients more informative across all strata.

### Alternative considered
Training on raw scale with a robust loss (SmoothL1). This mitigates the outlier problem but
does not address the disproportionate gradient attention to high values. The two approaches are
complementary: we use both (log1p transform AND SmoothL1 loss).

### DSM connection
Peatlands and high-SOC tropical soils are scientifically important but statistically rare.
Log-transform lets the model learn the common-range pattern well without completely ignoring the
tail — the tail still matters in absolute carbon accounting, but it should not distort the
whole optimisation.

### Ensemble back-transform note
`median(expm1(ẑ)) = expm1(median(ẑ))` — the median commutes with `expm1` because `expm1` is
monotone. So the ensemble median computed in log1p space is identical to the median computed in
native space. **Mean does not commute**: `mean(expm1(ẑ)) > expm1(mean(ẑ))` by Jensen's
inequality — averaging in log1p space and then back-transforming systematically underestimates
the ensemble mean. This is one of the reasons we use the median as the headline map.

---

## 2. Loss function — SmoothL1 in log1p space

### What it is
SmoothL1 (also called Huber loss) is a piecewise function:

```
L(r) = 0.5 × r²          if |r| < 1
L(r) = |r| − 0.5         otherwise
```

where `r = ŷ − y` in log1p space. The threshold of 1 corresponds to roughly a factor of
`e ≈ 2.7` in native space — a residual of 1 in log1p space means the prediction is off by
a multiplicative factor of ~2.7, which is already a large error in soil carbon.

### Why
- **Robustness to outliers**: for residuals beyond the threshold, the gradient is constant
  (like MAE/L1), not growing (like MSE/L2). A single extreme profile cannot dominate the
  gradient.
- **Smooth gradients near convergence**: unlike pure L1, the gradient shrinks linearly to zero
  as `r → 0`. Pure L1 has a constant gradient of ±1 near zero, which causes oscillation
  ("bouncing") at convergence.
- **Combined with log1p**: the log1p transform already compresses the tail, so SmoothL1
  operates on a more symmetric residual distribution. The two choices reinforce each other.

### Alternative considered
- **MSE (L2)**: quadratic penalty amplifies large residuals. With right-skewed soil data this
  makes the model chase peatland outliers. Rejected.
- **MAE (L1)**: robust to outliers but non-smooth at zero, causing gradient oscillation.
  Acceptable but SmoothL1 is strictly better in practice.
- **Quantile loss (pinball)**: would directly target a specific quantile (e.g. the median).
  More principled but harder to tune and less commonly used in spatial regression. Not explored.

### DSM connection
SOC stock datasets often contain field measurement errors, spline extrapolation artefacts, and
bulk density estimation uncertainty. These produce "soft outliers" — not wrong enough to remove,
but noisy enough to distort MSE. SmoothL1 downweights them automatically.

---

## 3. Activation function — SiLU

### What it is
SiLU (Sigmoid Linear Unit, also called Swish):

```
SiLU(x) = x × σ(x) = x / (1 + exp(−x))
```

It is smooth everywhere (unlike ReLU), bounded below but unbounded above, and has a small
negative region for x < 0 (unlike ReLU, which is exactly zero for all x < 0).

### Why
- **Smooth gradient**: no zero-gradient region. ReLU produces exactly zero gradient for all
  negative activations — a unit that fires negative is "dead" and receives no learning signal.
  SiLU avoids this entirely.
- **Self-gating**: the output is modulated by its own sigmoid, giving each activation a
  data-dependent nonlinearity rather than a hard threshold. This adds representational capacity
  without extra parameters.
- **Empirical superiority**: on most modern architectures SiLU marginally but consistently
  outperforms ReLU and GELU in regression tasks.

### Alternative considered
- **ReLU**: simplest, fastest. Risk of dying neurons especially in deep networks with high
  weight_decay. Rejected in favour of SiLU for deeper configs.
- **GELU**: similar properties to SiLU, slightly different shape. Performance is equivalent;
  SiLU was chosen as it is already available in PyTorch/torch as `nn_silu`.
- **Leaky ReLU**: solves the dying ReLU problem by allowing a small negative slope. Acceptable
  but less expressive than SiLU.

### DSM connection
With 200+ input channels, the first convolutional block must learn highly mixed representations
from very different predictor types (climate, topography, vegetation, geology). Smooth
activations help gradients flow uniformly across a heterogeneous input space.

---

## 4. Batch normalisation — `BatchNorm2d` and `BatchNorm1d`

### What it is
After each convolution and after the embedding linear layer, the activations are normalised to
zero mean and unit variance within the current mini-batch, then rescaled by learnable parameters
`γ` and `β`:

```
BN(x) = γ × (x − μ_batch) / √(σ²_batch + ε) + β
```

`BatchNorm2d` normalises 2D feature maps (one per channel across spatial positions).
`BatchNorm1d` normalises 1D embedding vectors (one per feature across samples).

### Why
- **Training stability**: normalising activations prevents them from growing or collapsing as
  the network deepens. Without BN, deep networks with many conv blocks require very careful
  learning-rate tuning to avoid divergence.
- **Higher learning rates**: BN smooths the loss landscape, allowing larger learning rates
  and therefore faster convergence.
- **Implicit regularisation**: the batch-level statistics add a small amount of noise during
  training, acting as a mild regulariser.
- **Reduces sensitivity to weight initialisation**: the network is less dependent on the
  initial scale of weights.

### Critical detail: BN and input scaling
Because BN normalises activations inside the network, the model can train from raw (unscaled)
predictor inputs — BN in the first conv block will normalise the activations regardless of the
input scale. However, this is not free: if input channels differ by orders of magnitude
(e.g., elevation in metres vs. NDVI in [−1, 1]), the first convolutional layer must learn very
different weight magnitudes per channel, which slows convergence. Pre-scaling the inputs (z-score
for continuous, /100 for proportions) puts all channels in the same dynamic range *before* the
first conv, making BN's job easier and improving gradient flow from the very first epoch.

### Alternative considered
Layer normalisation (LayerNorm): normalises across features within one sample rather than across
the batch. More stable with very small batch sizes, but loses the implicit noise regularisation.
Not used here because batch sizes ≥ 128 keep BN statistics stable.

---

## 5. Squeeze-and-Excitation (SE) block

### What it is
After the last convolutional block, the SE block applies channel-wise attention:

1. **Squeeze**: global average pooling over spatial dimensions → one scalar per channel:
   `s_c = (1/H×W) Σ x_{c,h,w}`
2. **Excitation**: a small 2-layer MLP (linear → SiLU → linear → sigmoid) maps the
   C-dimensional squeeze vector to a weight vector `w ∈ (0,1)^C`.
3. **Recalibration**: each channel of the feature map is multiplied by its weight:
   `x_c ← w_c × x_c`.

The bottleneck ratio `se_reduction` controls the MLP size: hidden layer has `C / reduction`
neurons.

### Why
Convolutional layers share weights across spatial positions — they learn *what* to detect but
not *how important* that detection is relative to other channels for the current spatial context.
SE adds this second level of selectivity: the network learns to up-weight channels that carry
relevant information for the target, and down-weight channels that are noisy or irrelevant for
that sample.

### DSM connection
With 200+ input raster layers (climate indices, topographic derivatives, spectral bands,
soil proxies), the convolutional feature maps encode very different types of environmental
information. For a sample in a humid tropical location, vegetation and precipitation channels
should dominate; for an arid site, temperature and aridity channels matter more. SE allows the
model to learn these context-dependent weights rather than treating all channels equally after
the last conv block.

### Tuning note
In round-2 tuning results, the top two configs had `use_se_block = FALSE`. This suggests that
with the current dataset size (~23 000 training patches) and network depth (2 conv blocks),
SE adds complexity without consistent benefit — the convolutional layers already compress 200+
channels to 64/128 feature maps, and the head may be sufficient to learn the relevant weighting.
SE is kept as a tuneable parameter rather than removed, because it may matter for larger networks
or more diverse datasets.

---

## 6. Dual-branch design and multi-scale patches

### What it is
Two independent CNN branches process patches of different spatial sizes centred on the same
profile location:

- **Branch 1 (small)**: the smaller window (e.g. 3×3) — captures local-scale spatial structure.
- **Branch 2 (large)**: the larger window (e.g. 9×9 or 15×15) — captures landscape-scale structure.

Each branch has its own conv blocks, SE block (optional), and embedding linear layer. Both
produce an embedding vector of the same size (`embedding_dim`).

### Why
Soil properties are driven by processes operating at multiple scales simultaneously. A single
window size forces a choice between local and landscape context. The dual-branch design
captures both within the same forward pass and lets the gate (see below) decide, per sample,
how much to draw from each scale.

### DSM connection
A window's physical extent is **`window_size × raster resolution`** — it scales with the
resolution of your predictors, it is not a fixed distance. The applied example runs at **250 m**:

| Window | Extent at 250 m | Typical processes |
|--------|-----------------|-------------------|
| 3 × 3   | ~0.75 km | Immediate neighbourhood, land cover, micro-relief |
| 9 × 9   | ~2.25 km | Hillslope, drainage, soil–landscape position |
| 15 × 15 | ~3.75 km | Local landscape, catchment context |

(The same windows at 20 km resolution would span ~60 / ~180 / ~300 km — see `tuning_guide.md`.)

For SOC stock (0–5 cm), local processes (litter input, drainage, land use) interact with
landscape position and local climate. No single scale captures this interaction; the dual-branch
design captures both and lets the gate decide, per sample, how much to draw from each.

### Empirical finding (resolution-dependent — read with care)
In an early prototype **at 20 km resolution**, round-1 tuning with 10 random configurations
showed a large gap: configs with a 7×7 window (≈140 km of context) reached CCC ~0.57–0.61,
while 3×3 / 5×5 configs reached ~0.20–0.25. At that resolution the large window captured the
dominant climate and parent-material gradients driving global SOC.

**This finding is tied to 20 km pixels and does not transfer to 250 m.** At 250 m a 7×7 window
covers only ~1.75 km, so the climate-scale signal that made it win is no longer inside the patch.
When the resolution changes the window search must be redone from scratch — which is why the
example now extracts **3 / 9 / 15 at 250 m** and re-tunes, rather than reusing the 20 km winner.

---

## 7. Gate mechanism — vector, scalar, and no-gate

### What it is
After both branches produce embeddings `f1` (small) and `f2` (large), a gate combines them:

**Vector gate (`vector_featurewise`):**
```
gate_input = [f1, f2, |f1 − f2|, f1 ⊙ f2]   (4 × embedding_dim)
gate = sigmoid(MLP(gate_input))               (embedding_dim values ∈ (0,1))
fused = gate ⊙ f1 + (1 − gate) ⊙ f2
```
One weight per embedding dimension: different "features" in the embedding can have different
branch preferences.

**Scalar gate (`scalar_per_sample`):**
Same structure but the MLP outputs one scalar per sample, not one per dimension.
```
gate ∈ (0,1)   (scalar)
fused = gate × f1 + (1 − gate) × f2
```

**No gate (`no_gate_concat`):**
No gating network. Both embeddings are concatenated directly:
```
head_input = [f1, f2]   (2 × embedding_dim)
```
The head learns to fuse them implicitly.

### Absolute difference `|f1 − f2|` as gate input

The absolute difference between the two branch embeddings is included as an explicit input to
the gate network. This encodes *how much the two scales disagree* for a given sample.

**Why this matters:** A sample in a uniform agricultural plain will have very similar f1 and f2
(small disagreement → |f1 − f2| ≈ 0). A wetland surrounded by dryland will have very different
small-scale and large-scale environments (large disagreement). Including |f1 − f2| tells the
gate not just what each scale sees, but how different their views are — which may itself be
informative and influence how confidently the gate should commit to one scale.

### Product interaction `f1 ⊙ f2` as gate input

The element-wise product captures *concordance* between the two embeddings — dimensions where
both branches agree and are both strongly activated produce large products; dimensions where
one branch is inactive produce near-zero products. This is complementary to the absolute
difference (which captures disagreement) and provides the gate with a richer view of the
inter-scale relationship.

### Head input — why `[fused, |f1 − f2|]` and not just `fused`

For vector and scalar gates, the head receives both the fused embedding and the absolute
difference:
```
head_input = [fused, |f1 − f2|]   (2 × embedding_dim)
```
The disagreement signal `|f1 − f2|` is passed to the head even after fusion because it may
carry predictive information independently: locations where small and large scales strongly
disagree (e.g., a small wetland embedded in a dry landscape) may systematically differ in
SOC regardless of the fused representation.

### Tuning results
Round-2 top configs used `no_gate_concat` (cfg_004 and cfg_012). This suggests that, at the
current dataset size, the simpler fusion is competitive with or superior to the learned gate.
The gate adds a trainable MLP and regularisation burden; with ~23 000 samples the signal may
be insufficient to train it reliably. The three gate types remain in the search space because
this conclusion is dataset-specific and the situation may change with more data or targeted
spatial splits.

> **Historical note:** cfg_004/cfg_012 were selected under the initial 20 km-resolution
> prototype (window sizes 3/5/7). After moving to the 250 m production run (window sizes
> 3/9/15, re-tuned end to end), the selected final model is **cfg_022** (dual-branch,
> windows 9×9 + 15×15, `vector_featurewise` gate, `embedding_dim=384`). The discussion above
> is kept for the qualitative conclusion (gate vs. concat), not as a pointer to the current
> production config.

---

## 8. D4 augmentation (patch orientation)

### What it is
Raster patches have no preferred orientation — North, South, East, West are arbitrary
conventions with no physical meaning for the CNN. The D4 symmetry group has 8 elements:
4 rotations (0°, 90°, 180°, 270°) × 2 reflections (identity, horizontal flip).

During training, **each sample draws its own random D4 operation** (per-sample, not one
per mini-batch), so a single gradient step already mixes all 8 orientations. This
multiplies the effective training set size by 8 without collecting new data, and gives
smoother gradients and more representative BatchNorm statistics than rotating the whole
batch the same way. In a dual-branch model the two windows of a given sample receive the
*same* symmetry, keeping their spatial scales geometrically consistent. Every D4 element
fixes the centre cell of an odd-sized patch, so the centre-point label is preserved
(verified by `tests/test_augmentation.R`).

### Why
- **Invariance**: the spatial pattern within a 3×3 or 7×7 patch should predict the same SOC
  regardless of which cardinal direction is "up" in the patch. Training without augmentation
  forces the network to learn this invariance implicitly (which costs capacity). Augmentation
  encodes it directly.
- **Regularisation**: the network never sees the same patch in the same orientation twice,
  reducing memorisation of training samples.

### Alternative considered
No augmentation (the version before round 2). Round-1 to round-2 improvement in CCC
(0.569 → 0.585–0.605) and large RMSE reduction confirm that D4 augmentation meaningfully
helped generalisation.

### DSM connection
Unlike natural images (where "cat upside down" is unusual), environmental raster patches are
truly invariant to orientation: the climate, topographic, and soil gradient at a location looks
the same regardless of map rotation convention.

---

## 9. Learning rate schedule — warmup + ReduceLROnPlateau

### What it is
Two mechanisms modify the learning rate during training:

**1. Linear warmup**  
For the first `warmup_epochs` (fixed at 5), LR increases linearly from `warmup_start_lr`
(1e-5) to `base_lr`. Only after warmup does the full learning rate apply.

**2. ReduceLROnPlateau**  
If validation loss does not improve by at least `lr_plateau_min_delta` for
`lr_plateau_patience` epochs, LR is multiplied by `lr_plateau_factor` (0.5), down to a
minimum of `min_lr`. This is applied at epoch end.

### Why warmup
At the start of training, weights are randomly initialised and gradients are noisy. A large LR
at this stage can push weights far from a good initialisation, destabilising training. Warmup
prevents this by starting with a very small step and gradually increasing it once early
representations are formed.

### Why ReduceLROnPlateau
As the model approaches convergence, a fixed LR may cause the optimiser to oscillate around a
local minimum without descending further. Reducing LR allows finer updates when progress slows,
often recovering additional performance without restarting training.

### Why not cosine annealing or cyclic LR
Both are valid alternatives. ReduceLROnPlateau was chosen because it is *data-driven*: LR only
drops when the model actually stagnates, making it robust across configurations with very
different convergence speeds (a slow-converging config gets more full-LR epochs before the
first reduction). Cosine and cyclic schedules have a fixed period that must be tuned.

---

## 10. Early stopping

### What it is
Training is halted if validation loss does not improve by at least `es_min_delta` for
`patience` consecutive epochs. The model weights from the best validation-loss epoch are
restored at the end.

### Why
Neural network training always eventually overfits on the training set if run long enough.
Validation loss is a proxy for generalisation: it typically decreases for the first phase
of training, then reaches a minimum, then starts increasing as the model memorises training
samples. Early stopping detects this minimum automatically.

### Why validation loss (not CCC or MAE)
The early stopping monitor is the training loss function (SmoothL1 on log1p scale), evaluated
on the validation set. This is intentional:

- **Consistency**: the stopping criterion uses the same scale as the optimisation objective.
  Monitoring a different metric (e.g., CCC in native space) could stop training at a point
  that is locally optimal for CCC but not for the overall loss surface, potentially leaving the
  model in a suboptimal state.
- **Smoothness**: SmoothL1 loss is smoother than CCC, which can fluctuate epoch-to-epoch.

CCC and MAE are computed for diagnosis but do not drive early stopping.

### Tuning vs. final model patience
During tuning: `patience = 60` epochs (shorter — fast exploration of many configs).  
During final model training: `patience = 100` epochs (longer — full convergence for the chosen
architecture, not a budget-constrained screen).

---

## 11. Multi-seed ensemble

### What it is
The winning architecture is trained N times from scratch (different random seed each time).
Each seed independently randomises:

| Source | Effect |
|--------|--------|
| Weight initialisation | Different starting point → different local minimum after convergence |
| Batch shuffle order | Different gradient path through the loss landscape |
| Dropout masks | Different sub-networks regularised per forward pass |

Spatial prediction aggregates all N seed models per pixel.

### Median vs. mean — why median

The **median** is the headline ensemble map. Reasons:

1. **Back-transform invariance**: `median(expm1(ẑ_1, ..., ẑ_N)) = expm1(median(ẑ_1, ..., ẑ_N))`
   because `expm1` is monotone. Median can be computed in log1p space and back-transformed
   without bias.
2. **Robust to divergent seeds**: if one seed converges to a poor local minimum and produces
   systematically high or low predictions, the median is unaffected (it requires the majority
   to agree). Mean would be pulled toward the outlier seed.
3. **Consistency with SmoothL1**: SmoothL1 is an approximation of L1, whose population
   minimiser is the conditional median. The model is optimised toward median prediction;
   averaging predictions from median-targeting models is less coherent than taking the median.

Mean, SD, MAD, min, and max are also written as separate layers for users who need them.

### What the ensemble uncertainty represents
SD and MAD between seeds measure **epistemic uncertainty due to training stochasticity**:
how stable is the prediction across different weight initialisation and training trajectories.
This is useful for identifying regions where the model is inconsistent.

It does **not** capture: measurement error in SOC stock, spline interpolation uncertainty,
bulk density estimation error, coarse fragment uncertainty, or spatial extrapolation uncertainty
(the latter is the dominant uncertainty source in many DSM applications).

---

## 12. Model selection — validation only

### What it is
After training all configurations in the tuning grid, configs are ranked by **validation CCC**
(descending) and then validation MAE (ascending). The test set is opened only once, after the
winning architecture is fixed.

### Why not rank by test metrics
Ranking by test performance is a subtle form of data leakage: the test set is used to inform
the model selection decision, not just to evaluate the final chosen model. Over multiple
configurations, the config that happens to be best on the test set will be selected — but this
"best" is partly due to chance fit on that specific test sample, not true generalisation.

This is analogous to multiple comparisons: selecting the maximum CCC from 30 test-set
evaluations without correction inflates the apparent performance.

### Practical implication
Test CCC is written to the comparison CSV for diagnostic reference but explicitly labelled as
not used for selection. The final reported test CCC (from script 04, multi-seed) is a genuine
out-of-sample estimate because it was not used at any decision point.

---

## 13. MQI — Model Quality Index

### What it is
```
MQI = (CCC × NSE) / (MAE / mean(obs))
```

A composite metric that combines three complementary performance aspects:

| Component | What it captures |
|-----------|-----------------|
| **CCC** | Accuracy + precision: both correlation and proximity to the 1:1 line |
| **NSE** | Skill relative to using the observed mean as prediction (0 = mean-only model, 1 = perfect) |
| **MAE / mean(obs)** | Normalised absolute error: relative magnitude of mistakes |

Higher MQI = better. MQI penalises bias (via CCC), skill deficit (via NSE), and large errors
relative to the mean (via normalised MAE). No single standard metric captures all three
simultaneously.

### Why not just CCC
CCC is the primary metric but has a blind spot: a model with high correlation but systematic
bias can achieve moderate CCC while having poor absolute accuracy. NSE penalises models that
perform only marginally better than the mean. Normalised MAE adds a scale-relative error
dimension. MQI integrates all three.

### Note
MQI is an original composite defined for this project. It does not appear in the standard
pedometrics literature under this name. It is retained because it is statistically interpretable
and penalises failure modes that matter for SOC stock mapping (bias, scale, and skill).

---

## 14. Channel-wise input scaling

### What it is
Before the patches reach the CNN, each predictor channel is scaled using statistics computed
**from the training split only**:

| Predictor type | Transform |
|----------------|-----------|
| Continuous (z-score) | `(x − μ_train) / σ_train` |
| Proportions (0–100) | `x / 100` |
| Binary/dummy (0–1) | identity (no change) |

The same parameters are stored in `predictor_scaling.csv` and applied identically during
spatial prediction.

### Why
Although BatchNorm normalises activations inside the network, all channels arrive at the first
conv layer in their original physical units. A channel in metres (elevation: 0–8000) has
weights ~8000× larger than a channel in [0, 1] (NDVI). The first convolutional layer must
learn radically different weight scales per channel, which:
- Slows convergence (gradients are dominated by large-magnitude channels).
- May cause numerical instability in BatchNorm statistics if one channel dwarfs all others.

Pre-scaling puts all channels in the same dynamic range before the first operation, making
gradient flow uniform from the start.

### Why training-split statistics only
Using the full dataset (or the validation/test splits) to compute scaling parameters would
constitute a subtle data leakage: the model would implicitly "know" the global distribution
of the validation and test sets. Computing from training only is the methodologically correct
approach.

### Critical consistency requirement
The scaling transform applied during patch extraction (script 02) **must be identical** to the
transform applied during spatial prediction (script 05). Any mismatch — different statistics,
different column order, omitted QC steps — will cause the spatial prediction to receive inputs
from a different distribution than the training data, producing systematic map artefacts.

### Out-of-range proportions: clamp, don't discard
Proportion channels (e.g. potential-natural-vegetation class fractions, clay mineralogy) are
continuous, interpolated surfaces. Near sharp spatial transitions — a biome boundary, a mosaic
seam — they can legitimately overshoot slightly past their `[0, 100]` bounds (a small negative
value where the true signal is ~0%, or slightly over 100% at the other extreme). This is
Gibbs-like ringing from whatever smoothing produced the surface, not sensor error and not
genuinely missing data.

An earlier version of the scaling step set any out-of-range value to `NA` instead of clamping
it. Because the CNN requires every cell of the receptive window to be finite, a single discarded
pixel invalidated up to a 15×15 neighbourhood around it. Since near-zero noise is common
wherever a class's true probability is low — i.e. almost everywhere, for most classes — this
cascaded into large, spurious gaps in both the training set (patches dropped that had no real
missing data) and the spatial prediction map (holes where coverage should have been complete).
Validated on a 250 m test tile: coverage went from 0.72% to 100% after switching from discard to
clamp, with no change to the predicted value distribution in the pixels that were already valid.

**Fix:** clamp to `[0, 100]` (`pmin(pmax(x, 0), 100)`) instead of discarding. Genuine `NA`/`Inf`
— actual missing data in the source raster — still propagate to `NA` and still correctly
invalidate the window; only the harmless boundary noise is preserved instead of manufactured
into missingness. Applied identically in scripts 01, 02, and 05, per the consistency requirement
above.
The `predictor_scaling.csv` file is the single source of truth for both scripts.
