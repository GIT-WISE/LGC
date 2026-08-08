# LGC: Legendre Graph Convolution

**Long-range molecular graph modeling via a Cayley-stabilized Legendre spectral filter**

This repository contains the official implementation of the paper
*"LGC: Long-range Modeling on Molecular Graphs with Legendre-basis"*
(submitted to the LoG 2026 Proceedings Track). It provides two companion
codebases: one for MoleculeNet based graph classification, one for the
Long Range Graph Benchmark (LRGB) Peptides tasks.

```
.
├── lgc-molnet/              # MoleculeNet (BACE, BBBP, HIV)
└── cayley-legendre-gnn/     # LRGB (Peptides-func, Peptides-struct)
```

---

## Summary

Message-passing GNNs propagate information one hop per layer, so modeling an
interaction between two nodes at graph distance *d* requires depth ≥ *d*.
Pushing depth to reach distant nodes causes **over-squashing**
and **over-smoothing**. Spectral GNNs sidestep this by defining convolution as
a polynomial filter of the graph Laplacian, decoupling receptive field size
from network depth, but the field has mostly used Chebyshev, Bernstein, or
monomial bases, leaving the **Legendre basis** comparatively unexplored.

The polynomial bases used so far (Chebyshev, Bernstein, monomial) become
numerically unstable at higher polynomial orders needed for long-range
signal propagation. The paper addresses this instability by exploring the
Legendre polynomial basis for GNNs, which is theoretically shown to be more
stable than Chebyshev at high order. Since Legendre alone doesn't fully solve
the problem, the paper also introduces propagation as a Cayley transform of a
skew-symmetric Legendre generator (LGC), which is provably orthogonal at
every layer, guaranteeing stable signal propagation regardless of depth or
polynomial order.

## Repository contents

| Folder | Task | Datasets |
|---|---|---|
| [`lgc-molnet/`](./lgc-molnet) | Graph classification | MoleculeNet: BACE, BBBP, HIV |
| [`cayley-legendre-gnn/`](./cayley-legendre-gnn) | Graph classification + regression | LRGB: Peptides-func, Peptides-struct |

## Installation and requirements

Both sub-projects require Python 3.9+, PyTorch, and PyTorch Geometric. Each
has its own `requirements.txt`, so install them separately depending on
which one you're using.

**`lgc-molnet`** (MoleculeNet):

```bash
cd lgc-molnet
pip install -r requirements.txt
# requirements: torch>=2.0, torch_geometric>=2.4, rdkit>=2023.3.1,
#               tqdm>=4.65, numpy>=1.23, scikit-learn>=1.2
```

**`cayley-legendre-gnn`** (LRGB Peptides):

```bash
cd cayley-legendre-gnn
pip install -r requirements.txt
# requirements: torch, torch_geometric, numpy, scikit-learn, tqdm
```

Both can optionally be installed as editable packages (`pip install -e .`)
using the `setup.py` / `pyproject.toml` in each folder.
