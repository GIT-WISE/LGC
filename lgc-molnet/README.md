# LGC-MoleNet

A Cayley-Legendre graph convolutional network (**LGC**) for graph-level
molecular property prediction on [MoleculeNet](https://moleculenet.org/)
datasets (BACE, BBBP, HIV).

Each layer applies a Cayley transform `Y = (I - eps/2 * A)^{-1} (I + eps/2 * A) x`,
where `A` is a skew-symmetric operator built from Legendre-polynomial graph
propagation of the (rescaled, symmetric-normalised) graph Laplacian. The
matrix inverse is never formed explicitly; it's approximated with an
iterative Krylov (GMRES-style) or truncated Neumann series solver.

## Project structure

```
lgc-molnet/
├── main.py                  # CLI entry point (training / sweeps)
├── requirements.txt
├── setup.py
└── lgc_molnet/
    ├── __init__.py          # public API
    ├── layers.py            # Laplacian utils, Legendre propagation, Cayley layer
    ├── solvers.py            # Neumann / Richardson / Krylov iterative solvers
    ├── data.py               # MoleculeNet pre-transform, scaffold/random split
    ├── model.py               # LGC_MoleNet (full model)
    ├── train.py               # train/eval loops and experiment runner
    └── utils.py                # seeding helper
```

## Installation

```bash
git clone <this-repo-url>
cd lgc-molnet
pip install -r requirements.txt
# or, to install as an importable package:
pip install -e .
```

> Note: `torch_geometric` and `rdkit` can be finicky to install depending on
> your CUDA/torch version. See the
> [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
> if `pip install torch_geometric` doesn't work out of the box.

## Usage

Train on a single dataset:

```bash
python main.py --dataset bace --eps_mode variable
```

Sweep over all three datasets and both eps modes (learnable vs. fixed):

```bash
python main.py --dataset all --eps_mode both
```

Key options (see `python main.py --help` for the full list):

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `all` | `bace`, `bbbp`, `hiv`, or `all` |
| `--eps_mode` | `both` | `variable`, `fixed`, or `both` |
| `--hidden_dim` | `128` | Hidden feature dimension |
| `--num_layers` | `4` | Number of Cayley-Legendre layers |
| `--K` | `8` | Legendre propagation order |
| `--solver` | `krylov` | `krylov` or `neumann` |
| `--solver_iters` | `12` | Number of solver iterations |
| `--epochs` | `100` | Max training epochs |
| `--patience` | `30` | Early-stopping patience |

## Using the library directly

```python
from lgc_molnet import run

test_auc = run(dataset="bbbp", eps_mode="variable", epochs=50)
print(test_auc)
```

Or build the model manually:

```python
from lgc_molnet import LGC_MoleNet

model = LGC_MoleNet(in_dim=9, hidden_dim=128, num_tasks=1, num_layers=4, K=8)
```

## License

MIT (or your license of choice — update this section as needed).
