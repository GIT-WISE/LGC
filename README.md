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
