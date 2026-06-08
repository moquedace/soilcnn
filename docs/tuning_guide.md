# Hyperparameter Tuning Guide

This document explains every tuneable parameter in the framework: what it controls architecturally, why the search range was defined the way it is, and how each choice relates to digital soil mapping (DSM).

---

## 1. Spatial architecture

### `window_sizes`

**What it is:** The spatial extent of the patch extracted around each profile location.  
A `window_size = 3` means a 3 × 3 grid of cells is extracted → 9 values per predictor layer.  
A `window_size = 5` means a 5 × 5 grid → 25 values per predictor layer.

**DSM connection:**  
Soil properties are driven by processes operating at multiple spatial scales simultaneously.

The physical extent a window covers is **`window_size × raster resolution`** — it is *not* a fixed distance, it scales with the resolution of your predictors. The **same** window therefore means very different things at different resolutions:

| Window | at 20 km px | at 1 km px | at 250 m px |
|--------|-------------|------------|-------------|
| 3 × 3   | ~60 km  | ~3 km  | ~0.75 km |
| 9 × 9   | ~180 km | ~9 km  | ~2.25 km |
| 15 × 15 | ~300 km | ~15 km | ~3.75 km |

So pick windows by the **physical process** you want to capture, then convert to pixels for *your* resolution. The applied example runs at **250 m**, where the three windows span the local pixel neighbourhood up to local-landscape / small-catchment scale:

| Window | Extent at 250 m | Typical processes captured |
|--------|-----------------|---------------------------|
| 3 × 3   | ~0.75 × 0.75 km | Immediate neighbourhood, land cover, micro-relief |
| 9 × 9   | ~2.25 × 2.25 km | Hillslope, drainage, soil–landscape position |
| 15 × 15 | ~3.75 × 3.75 km | Local landscape, catchment context, local climate |

**Options:** `c(3)`, `c(9)`, `c(15)` (single branch), or `c(3,9)`, `c(3,15)`, `c(9,15)` (dual branch).

**Why this range:**  
Smaller than 3 gives no spatial context (equivalent to point extraction). Each window must be **odd** so the profile sits at the exact centre. The upper bound is a compute/coverage trade-off: cost grows with window², and at a finer resolution you need a *larger* pixel window to reach the same physical extent. **Re-pick the windows whenever you change resolution** — a set tuned at one resolution does not transfer to another.

---

### `n_branches` (derived from `window_sizes`)

**What it is:** 1 if `window_sizes` has one element; 2 if it has two.

**DSM connection:**  
Using two branches allows the model to explicitly learn from local *and* landscape context.  
The gate (see below) learns how to weight each scale per sample — some locations may be strongly driven by local processes (e.g., a wetland in a dry region), while others are governed by regional climate.

---

## 2. Convolutional blocks

### `conv_channels`

**What it is:** A vector of integers specifying the number of feature maps (filters) in each convolutional block.  
The **length** of the vector determines the number of blocks (network depth).

| Example | Depth | Parameters (approx., 204 input ch.) |
|---------|-------|--------------------------------------|
| `c(32, 64)` | 2 blocks | ~400 k |
| `c(64, 128)` | 2 blocks | ~1.7 M |
| `c(64, 128, 128)` | 3 blocks | ~2.4 M |
| `c(128, 256, 256)` | 3 blocks | ~8.5 M |

**DSM connection:**  
With 200+ predictor channels, the first conv block is doing a major dimension reduction (204 → 64 channels). More subsequent blocks allow the network to learn higher-order interactions between predictor-derived features. However, more depth = more risk of overfitting, especially with < 30 000 training samples.

**Recommendation:**  
- Start with `c(64, 128)` or `c(64, 128, 128)`.  
- Enable `use_residual = TRUE` if using 3 blocks.

---

### `use_residual`

**What it is:** If `TRUE`, each conv block has a skip connection: the block input is added to its output before activation (ResNet style). When input and output channels differ, a 1 × 1 conv aligns the dimensions before adding.

**Why it helps:**  
Deep networks suffer from the vanishing gradient problem: gradients shrink as they backpropagate through many layers, making early layers learn very slowly. Skip connections provide a gradient "highway" that bypasses each block, keeping early layers trainable.

**DSM recommendation:**  
Enable (`TRUE`) whenever `n_conv_blocks >= 2`. The cost is minor (a few extra 1 × 1 conv parameters). The benefit is more stable training and the ability to go deeper without accuracy degradation.

**Range:** `TRUE`, `FALSE`

---

## 3. Channel attention

### `use_se_block`

**What it is:** A Squeeze-and-Excitation (SE) block applied after the last conv block. It computes a global average over spatial dimensions (squeeze), then a small 2-layer MLP predicts a per-channel weight (excitation) in [0, 1]. Each channel of the feature map is multiplied by its weight.

**DSM connection:**  
With 200+ input predictors compressed into feature maps, some channels will represent climate variables, others vegetation indices, others soil texture proxies, etc. The SE block lets the network learn *which types of features matter most* for a given spatial context — not just spatially, but across the channel dimension.

For example, in humid tropical areas, vegetation indices may dominate SOC prediction, while in dry regions temperature and aridity predictors matter more. SE allows the model to adaptively reweight without manual feature selection.

**Range:** `TRUE`, `FALSE`

---

### `se_reduction`

**What it is:** Bottleneck ratio of the SE MLP. The hidden layer has `channels / reduction` neurons.

| Reduction | Hidden units (128 ch.) | Parameters |
|-----------|------------------------|------------|
| 8  | 16 | ~4 k |
| 16 | 8  | ~2 k |
| 32 | 4  | ~1 k |

**Fixed at:** `16`

This is a 2nd-order knob whose effect on accuracy is negligible relative to window size, depth, and learning rate. Varying it wastes tuning budget. Override via `fixed = list(se_reduction = 8L)` if you specifically want to study it.

---

### `embedding_dim`

**What it is:** Size of the vector produced by each branch after flattening and the linear projection.  
All gate and head operations happen in this space.

**Range:** `128`, `256`, `384`, `512`

**Trade-off:**  
Larger embedding → more representational capacity, but quadratically more parameters in the linear projection (e.g., for a 5×5 patch with 128 final channels: flatten = 128 × 25 = 3200 → linear 3200 × embedding_dim).  
For datasets with < 30 000 samples, 256–384 is usually a good balance.

---

## 4. Gate (dual-branch fusion)

### `gate_type`

Controls how the two branch embeddings are combined into a single representation.

**`"vector_featurewise"` (most expressive)**  
The gate is a vector of `embedding_dim` values (one gate weight per embedding dimension).  
It is computed from the concatenation of:
- `f1` (small-branch embedding)
- `f2` (large-branch embedding)  
- `|f1 − f2|` (absolute difference: how much the scales disagree)
- `f1 * f2` (element-wise product: interaction between scales)

This 4 × `embedding_dim` input goes through a small MLP → sigmoid → gate weights ∈ (0, 1).

The fused representation is: `gate * f1 + (1 − gate) * f2`  
(gate = 1 → full weight to small branch; gate = 0 → full weight to large branch).

The head then receives: `[fused, |f1 − f2|]` — including the disagreement signal, which itself can be informative (e.g., a wetland surrounded by drier landscape will show high disagreement between scales).

**`"scalar_per_sample"` (simpler)**  
Same as vector gate but the MLP outputs a single scalar per sample rather than one per embedding dimension. Less expressive but fewer parameters.

**`"no_gate_concat"` (baseline)**  
No gating. Both embeddings are concatenated: `[f1, f2]`. The head learns to fuse them implicitly through its own weights.

**DSM recommendation:**  
`"vector_featurewise"` is the best starting point. It consistently outperforms simpler fusion in spatial regression tasks because different embedding dimensions (corresponding to different feature groups) should be weighted differently depending on spatial context.

---

## 5. Regularisation

### `dropout` (single knob)

**What it is:** A single scalar that controls overall dropout strength across the model.  
The model has five internal dropout sites. Tuning them independently makes the search space huge and non-identifiable — many combinations are practically equivalent. Instead, one `dropout` value is expanded to all five sites with fixed sensible ratios:

| Site | Formula | Interpretation |
|------|---------|----------------|
| `spatial_dropout` | 0.25 × dropout | 2D channel dropout before flatten (mild) |
| `embed_dropout` | 0.0 | Disabled — marginal effect right after BatchNorm |
| `gate_dropout` | 0.50 × dropout | Inside the gate MLP, before sigmoid |
| `head_dropout_1` | 1.00 × dropout | First head layer (`2 × embed_dim` → `embed_dim`) |
| `head_dropout_2` | 0.50 × dropout | Second head layer (128 units) |

**Range:** `0.0`, `0.1`, `0.2`, `0.3`

`0.0` = no dropout anywhere; `0.3` = strong regularisation.  
For datasets with ~25 000 samples and a moderately-sized network, `0.1` or `0.2` is a good starting point.

**Power users:** `make_manual_tune_grid()` accepts the five individual parameters directly if you need fine-grained control.

---

### `weight_decay`

**What it is:** L2 regularisation on all model weights (added to the loss as λ × ||W||²). Shrinks weights toward zero, reducing overfitting.

**Range:** `0.0`, `1e-5`, `1e-4`, `1e-3`

For CNNs trained with Adam, small values (1e-5 to 1e-4) are typical. Values ≥ 1e-3 may under-fit.

---

## 6. Optimiser and training dynamics

### `base_lr` (learning rate)

**What it is:** The peak learning rate used by the Adam optimiser after the warm-up phase.

**Range:** `1e-4`, `3e-4`, `5e-4`, `1e-3`, `2e-3`, `3e-3`

The framework uses two LR management strategies on top of `base_lr`:

1. **Warm-up:** LR starts at `warmup_start_lr` (default: 1e-5) and increases linearly to `base_lr` over `warmup_epochs`. This prevents gradient explosions when weights are random.

2. **Reduce on plateau:** If validation loss does not improve for `lr_plateau_patience` epochs, LR is multiplied by `lr_plateau_factor` (default: 0.5), down to `min_lr`. This allows fine-tuning as the model nears convergence.

**DSM recommendation:**  
`1e-3` is a good default. Higher values (2e-3, 3e-3) risk instability; lower values (1e-4, 3e-4) are safe but may require more epochs.

---

### `warmup_epochs`

**What it is:** Number of epochs for the linear LR warm-up.

**Fixed at:** `5`

An optimisation detail, not a model hyperparameter — 5 epochs is sufficient across datasets and varying it wastes tuning budget. Override via `fixed = list(warmup_epochs = 10L)` if you train with small batches (128) on a noisy loss surface.

---

### `batch_size`

**What it is:** Number of samples per gradient update step.

**Range:** `128`, `256`, `512`

| Batch size | Gradient stability | Updates / epoch (n=25k) | Memory (GPU) |
|------------|-------------------|-------------------------|--------------|
| 128 | Lower (noisier) | ~195 | Low |
| 256 | Medium | ~98 | Medium |
| 512 | High | ~49 | High |

Smaller batch sizes can act as an implicit regulariser but require more compute per epoch (more updates).

---

### `loss_fn`

**What it is:** The training objective function. Note: this is **separate from the model selection metrics** (CCC, MAE, etc.), which are always computed for all splits regardless of which loss is used for training.

**`"smooth_l1"` (Huber loss) — recommended for soil data**  
Behaves like MAE (L1) for large residuals and like MSE (L2) near zero:  
- L(x) = 0.5 × x² if |x| < 1, else |x| − 0.5

This makes it robust to extreme values (high-SOC peatlands, outliers) while still giving smooth gradients near convergence. **Strongly recommended for right-skewed soil attributes**.

**`"mse"`**  
Penalises large errors quadratically. More sensitive to outliers.

**`"mae"`**  
Pure L1 loss. Equally robust to outliers, but the gradient is constant (not smooth), which can cause oscillation near convergence.

**Early stopping** is based on `validation_loss` (using whichever `loss_fn` is configured), so the same loss that drives training also drives the early stopping decision. This is intentional: it keeps the stopping criterion consistent with the training objective.

---

## 7. Early stopping and model selection

**Early stopping monitor:** validation SmoothL1 loss (or the configured `loss_fn`).  
**Patience:** number of epochs without improvement before stopping.  
**Model selection across configs:** ranked by **validation CCC** (descending), then validation MAE (ascending). Test metrics are written to the comparison CSV for diagnostic reference only — they are never used to choose between configurations. The test set is opened once, after the winning architecture is locked in.

The separation between early stopping metric (loss) and selection metric (CCC/MAE) is deliberate:  
- Loss guides training stability (smooth, differentiable, robust to outliers).  
- CCC captures agreement between predictions and observations and is more interpretable for DSM.

---

## How to build your own tuning grid

```r
# Fix some parameters, vary others:
grid <- make_tune_grid(
  tune_length = 40,
  seed        = 42,
  fixed = list(
    loss_fn    = "smooth_l1",    # always use Huber loss
    batch_size = 512L,           # fixed by GPU memory
    use_se_block = TRUE          # always use SE
  )
)

# Or define a manual grid:
grid <- make_manual_tune_grid(
  embedding_dim = c(256L, 384L),
  gate_type     = c("vector_featurewise", "no_gate_concat"),
  base_lr       = c(0.001, 0.0005),
  use_residual  = c(TRUE, FALSE)
)
```
