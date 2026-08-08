# LGC: Legendre Graph Convolution

**Long-range molecular graph modeling via a Cayley-stabilized Legendre spectral filter**

This repository contains the reference implementation accompanying the paper
*"LGC: Long-range Modeling on Molecular Graphs with Legendre-basis"*
(submitted to the LoG 2026 Proceedings Track). It provides two companion
codebases — one for MoleculeNet-style graph classification, one for the
Long Range Graph Benchmark (LRGB) Peptides tasks — that implement the same
underlying layer described in the paper.

```
.
├── lgc-molnet/              # MoleculeNet (BACE, BBBP, HIV)
└── cayley-legendre-gnn/     # LRGB (Peptides-func, Peptides-struct)
```

---

## The problem

Message-passing GNNs propagate information one hop per layer, so modeling an
interaction between two nodes at graph distance *d* requires depth ≥ *d*.
Pushing depth to reach distant nodes runs straight into **over-squashing**
(exponentially growing receptive fields compressed into fixed-size vectors)
and **over-smoothing** (node representations collapsing toward
indistinguishability). Spectral GNNs sidestep this by defining convolution as
a polynomial filter of the graph Laplacian, decoupling receptive field size
from network depth — but the field has mostly used Chebyshev, Bernstein, or
monomial bases, leaving the **Legendre basis** comparatively unexplored.

## What this work shows

**1. Legendre polynomials are a naturally more stable long-range basis than Chebyshev.**
Through a Marchenko–Pastur analysis of the layer-wise Jacobian, the paper
proves that under random weight initialization the expected squared singular
values of a degree-*K* spectral filter grow as `Θ(K)` for Chebyshev but only
`Θ(ln K)` for Legendre. Concretely, the ratio of Legendre to Chebyshev
spectral energy vanishes as `O(ln K / K)` — Legendre filters accumulate
signal energy far more slowly as the polynomial order grows, which is
exactly the property you want when stacking many hops of propagation.

**2. But unbounded polynomial order still eventually destabilizes signal propagation.**
Because Legendre's spectral energy is unbounded (just slower-growing), layer
Jacobian eigenvalues still drift outside the unit circle at sufficiently
high *K*, degrading gradient flow.

**3. The fix — Legendre Graph Convolution (LGC).**
The paper reformulates Legendre propagation through a **Cayley transform**:

```
X^(l+1) = C(A) X^(l),      C(A) = (I − ε/2·A)^(−1) (I + ε/2·A)
```

where `A = Σ_k Wₖᵀ ⊗ Pₖ(L)` is a **skew-symmetric generator** built from
Legendre-polynomial propagation of the normalized graph Laplacian `L`, and
`ε` is a (learnable) step size. Because `A` is skew-symmetric, `C(A)` is
provably **orthogonal at every layer**, for *any* `ε`, polynomial order `K`,
or graph — which gives an exact, closed-form guarantee (not an empirical
regularization) that the layer-wise Jacobian has unit spectral norm. Chained
across depth, this eliminates vanishing/exploding gradients through the
propagation operator entirely, with no approximation.

The matrix inverse in `C(A)` is never formed explicitly — it's approximated
per forward pass with an iterative solver operating only through
matrix-vector products:

- **Krylov (Arnoldi/GMRES-style)** solver — the default; builds an orthonormal
  Krylov basis and solves the resulting Hessenberg least-squares system.
  Because `(I − ε/2·A)` is well-conditioned for *any* step size when `A` is
  skew-symmetric, this converges in relatively few iterations even for large,
  learnable `ε`.
- **Truncated Neumann / Richardson series** — cheaper per iteration, but only
  convergent when `ε/2·‖A‖ₒₚ < 1`, making it less robust at larger step sizes.

The Legendre propagation itself uses the standard three-term recurrence
`(k+1)Pₖ₊₁ = (2k+1)x·Pₖ − k·Pₖ₋₁`, computed with sparse matrix-vector products
against the graph Laplacian — bringing the cost of building the filter down
from `O(K²|E|)` in naive implementations to `O(K|E|)`.

## Empirical results (from the paper)

- **Synthetic long-range benchmarks** (graph property prediction — Diameter,
  SSSP, Eccentricity): LGC achieves the best log₁₀MAE on Diameter (a ~37%
  error reduction over the next-best model, SWAN) and the second-best on
  Eccentricity, behind only ChebNet.
- **LRGB Peptides**: 71.28 AP on Peptides-func (second only to S²GNN, which
  relies on a full Laplacian eigendecomposition that LGC avoids) and 0.2528
  MAE on Peptides-struct — competitive with or ahead of GRIT, TIGT,
  Exphormer, DRew-GCN, and several state-space and rewiring-based baselines.
- **MoleculeNet**: best-in-class ROC-AUC on BACE (82.79), third on BBBP, and
  second on HIV, outperforming both spectral (ChebNet, BernNet, JacobiConv)
  and message-passing (GCN, GAT, GraphSAGE) baselines on average.
- **Ring/feature-transfer tests**: LGC shows strong signal retention over
  long source-target distances on Line and Crossed-Ring graphs without
  relying on expensive eigendecomposition.

## Repository contents

| Folder | Task | Datasets | Entry point |
|---|---|---|---|
| [`lgc-molnet/`](./lgc-molnet) | Graph classification | MoleculeNet: BACE, BBBP, HIV | `main.py` |
| [`cayley-legendre-gnn/`](./cayley-legendre-gnn) | Graph classification + regression | LRGB: Peptides-func, Peptides-struct | `scripts/train_peptides.py` |

Both packages implement the same `SkewLinear` / `GeneratorA` /
`CayleyLegendreLayer` building blocks and the same Krylov/Neumann/Richardson
solvers described above. `cayley-legendre-gnn` is the more complete
refactor — a proper `src/` package layout, a unified training engine
supporting both classification (BCE loss, Average Precision) and regression
(L1/MAE loss), validation-based checkpoint selection, and multi-GPU support
— while `lgc-molnet` is the MoleculeNet-focused variant with its own
scaffold-split data pipeline and ROC-AUC evaluation.

```
lgc-molnet/
├── main.py                  # CLI entry point (training / sweeps)
├── lgc_molnet/
│   ├── layers.py             # Laplacian utils, Legendre propagation, Cayley layer
│   ├── solvers.py            # Neumann / Richardson / Krylov iterative solvers
│   ├── data.py                # MoleculeNet pre-transform, scaffold/random split
│   ├── model.py                # LGC_MoleNet (full model)
│   ├── train.py                 # train/eval loops and experiment runner
│   └── utils.py                  # seeding helper
└── requirements.txt / setup.py

cayley-legendre-gnn/
├── scripts/train_peptides.py    # CLI entry point
├── src/cayley_legendre/
│   ├── laplacian.py              # normalized Laplacian construction/caching
│   ├── solvers.py                # Legendre propagation + Neumann/Richardson/Krylov solvers
│   ├── layers.py                 # SkewLinear, GeneratorA, CayleyLegendreLayer
│   ├── model.py                   # CayleyLegendreGNN
│   ├── metrics.py                  # average precision helper
│   ├── data.py                      # LRGB dataset loading + dataloaders
│   ├── engine.py                     # train/eval loops (classification + regression)
│   └── utils.py                       # seeding, device selection
└── requirements.txt / pyproject.toml
```

See the README inside each folder for full installation instructions, CLI
flags, and the Python API.

## Quickstart

```bash
# MoleculeNet
cd lgc-molnet && pip install -r requirements.txt
python main.py --dataset bace --eps_mode variable

# LRGB Peptides
cd cayley-legendre-gnn && pip install -r requirements.txt
python scripts/train_peptides.py --tasks func --hidden 150 --K 10
```

> `torch_geometric` and `rdkit` can be finicky to install depending on your
> `torch`/CUDA version — see the
> [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> if `pip install torch_geometric` doesn't pull in the right wheels.

## Citation

```bibtex
@inproceedings{lgc2026,
  title     = {LGC: Long-range Modeling on Molecular Graphs with Legendre-basis},
  author    = {Anonymous},
  booktitle = {Proceedings of the Fifth Learning on Graphs Conference (LoG)},
  year      = {2026},
  note      = {Proceedings Track submission}
}
```

## License

MIT (or your license of choice — update this section as needed).
