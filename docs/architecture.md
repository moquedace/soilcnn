# CNN Architecture

## Overview

```
Input (N profiles)
│
├── Patch 1: N × C × w1 × w1        ├── Patch 2: N × C × w2 × w2
│   (small window, e.g. 3×3)        │   (large window, e.g. 15×15)
│                                   │
├── conv_block × n_blocks            ├── conv_block × n_blocks
│   Conv2d → BN → Act               │   Conv2d → BN → Act
│   (+ skip connection if residual) │   (+ skip connection if residual)
│                                   │
├── SE block (optional)              ├── SE block (optional)
│   channel reweighting             │   channel reweighting
│                                   │
├── Spatial Dropout 2D               ├── Spatial Dropout 2D
│                                   │
├── Flatten: C_final × w1²           ├── Flatten: C_final × w2²
│                                   │
├── Linear → embedding_dim           ├── Linear → embedding_dim
│   BN → Act → Dropout              │   BN → Act → Dropout
│                                   │
│           f1 (N × embed_dim)      │           f2 (N × embed_dim)
│                └──────────┬───────┘
│                           │
│               ┌───────────▼──────────────┐
│               │  Gate network            │
│               │  input: [f1, f2,         │
│               │          |f1-f2|, f1*f2] │
│               │  → Linear → BN → Act     │
│               │  → Dropout → Linear      │
│               │  → Sigmoid               │
│               │  gate ∈ (0,1)^embed_dim  │
│               └───────────┬──────────────┘
│                           │
│           fused = gate * f1 + (1-gate) * f2
│           abs_dif = |f1 - f2|
│
│           head_input = [fused, abs_dif]   (2 × embed_dim)
│
└── Regression head
    Linear(2e → e) → BN → Act → Dropout
    Linear(e → 128) → BN → Act → Dropout
    Linear(128 → 64) → Act
    Linear(64 → 1)
    │
    └── Prediction (N × 1)
```

*C = number of predictor channels (raster layers). w1, w2 = window sizes. e = embedding_dim.*

---

## Single-branch mode

When `window_sizes` has length 1, only Branch 1 is built. There is no gate. The head receives only `f1` (embed_dim inputs instead of 2 × embed_dim).

---

## Conv block detail

### Without residual connection (`use_residual = FALSE`)
```
x → Conv2d(in_ch, out_ch, k=3, p=1) → BatchNorm2d → Activation → output
```

### With residual connection (`use_residual = TRUE`)
```
x ──────────────────────────────────────────────────► (+) → output
│                                                      ▲
└→ Conv2d(in_ch, out_ch, k=3, p=1) → BatchNorm2d ────┘
                                                  ↑
                         shortcut: if in_ch ≠ out_ch:
                           Conv2d(in_ch, out_ch, k=1) → BN
                         else: identity
```

The activation is applied **after** the addition (standard ResNet order).

---

## SE block detail

```
x (N × C × H × W)
│
├── AdaptiveAvgPool2d(1,1) → squeeze: (N × C)
│
├── Linear(C → C/reduction) → Activation → Linear(C/reduction → C)
│                                           ↓
│                                        Sigmoid → weights (N × C)
│                                           ↓ reshape to (N × C × 1 × 1)
│
└── x * weights  → output (N × C × H × W)
```

---

## Gate detail (vector_featurewise)

```
f1: N × embed_dim
f2: N × embed_dim

gate_input = concat([f1, f2, |f1-f2|, f1*f2], dim=1)
           → N × (4 × embed_dim)

gate_net:
  Linear(4e → e) → BN → Act → Dropout → Linear(e → e) → Sigmoid

gate: N × embed_dim   (each value ∈ (0, 1))

fused = gate * f1 + (1 - gate) * f2
```

When `gate_type = "scalar_per_sample"`, the final `Linear(e → e)` becomes `Linear(e → 1)`, producing one scalar per sample instead of one per dimension.

When `gate_type = "no_gate_concat"`, there is no gate network and the head receives `[f1, f2]` directly.

---

## Activation function

SiLU (Sigmoid Linear Unit, also called Swish) is used if available in the installed `torch` version:  
`SiLU(x) = x * sigmoid(x)`

SiLU is smooth, non-monotonic, and empirically outperforms ReLU in most deep learning tasks. If SiLU is not available, ReLU is used as a fallback.

---

## Parameter count example

Configuration: `conv_channels = c(64, 128, 128)`, `embedding_dim = 384`, dual branch `c(3, 15)`, SE reduction = 16.

| Component | Parameters (approx.) |
|---|---|
| Branch 1 (3×3): 3 conv blocks | ~200 k |
| Branch 2 (15×15): 3 conv blocks | ~200 k |
| SE blocks (×2) | ~4 k |
| Linear 3×3 → embed: (128×9) → 384 | ~0.44 M |
| Linear 15×15 → embed: (128×225) → 384 | ~11.1 M |
| Gate network: (4×384) → 384 → 384 | ~0.74 M |
| Head | ~0.35 M |
| **Total** | **~13 M** |

> **Note — the flatten→embedding linear scales with window².** The conv blocks are
> window-size independent (kernel × in × out), but with `embed_pool = "flatten"`
> each branch flattens `C_final × w × w` and projects it to `embedding_dim`, so that
> one linear layer grows quadratically with the window: ~0.44 M for 3×3, ~1.2 M for
> 5×5, **~11 M for 15×15**. Large windows therefore concentrate most parameters in a
> single layer. If a large-window config overfits, the levers are a smaller
> `embedding_dim`, fewer final conv channels, or — most directly — `embed_pool`.

### `embed_pool`: flatten vs. global average pool

Each branch reduces its `C_final × w × w` feature map before the linear projection
in one of two ways, selectable per model (and searchable in the tuning grid):

| `embed_pool` | Linear input size | 15×15 branch linear | Keeps within-patch detail? |
|---|---|---|---|
| `"flatten"` (default) | `C_final × w × w` | ~11 M params | Yes (every cell) |
| `"gap"` | `C_final` | ~0.05 M params | No (spatial mean only) |

With `"gap"` (global average pool), the linear size no longer depends on the window,
so a single-branch 15×15 model drops from ~11.5 M to ~0.46 M parameters (~25×
lighter), and dual `c(3, 15)` from ~12.3 M to ~0.86 M (~14× lighter). This makes
large windows tractable at any resolution and far less prone to overfitting, at the
cost of discarding the fine spatial arrangement inside the patch. `"flatten"`
preserves that detail and is the right default for small windows; let tuning compare
the two when large windows are in play. Both branches of a dual model share the
choice. Verified by `tests/test_architecture.R`.

With ~25 000 training samples the `"flatten"` variant is a sizeable network, so the
regularisers (dropout, weight decay, SE, D4 augmentation, BatchNorm) and early
stopping matter — especially for the larger windows. The single-branch 3×3 variant,
or any `"gap"` variant, is by contrast tiny (~0.5–1 M total).
