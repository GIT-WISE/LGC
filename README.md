# Cayley-Legendre GNN

An implicit graph neural network layer built from a Legendre-polynomial expansion
of the (skew-symmetrized) graph Laplacian, combined via a Cayley transform and
solved with a Krylov (or Neumann) linear solver. Benchmarked on the
[Long Range Graph Benchmark](https://github.com/vijaydwivedi75/lrgb) Peptides
datasets (`peptides-func`, `peptides-struct`).

## Repository layout

```
cayley-legendre-gnn/
├── src/
│   └── cayley_legendre/
│       ├── __init__.py     # public API
│       ├── laplacian.py    # normalized Laplacian construction/caching
│       ├── solvers.py      # Legendre propagation + Neumann/Richardson/Krylov solvers
│       ├── layers.py       # SkewLinear, GeneratorA, CayleyLegendreLayer
│       ├── model.py        # CayleyLegendreGNN
│       ├── metrics.py      # average precision helper
│       ├── data.py         # LRGB dataset loading + dataloaders
│       ├── engine.py       # train/eval loops (classification + regression)
│       └── utils.py        # seeding, device selection
├── scripts/
│   └── train_peptides.py   # CLI entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Install

```bash
pip install -r requirements.txt
# or, for an editable install of the package:
pip install -e .
```

`torch_geometric` typically requires matching wheels for your `torch` + CUDA
version — see the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
if `pip install torch_geometric` doesn't pull in the right extensions.

## Usage

Train on both Peptides-func and Peptides-struct with default settings:

```bash
python scripts/train_peptides.py
```

Train on a single task, and tweak the model / solver:

```bash
python scripts/train_peptides.py \
  --tasks func \
  --hidden 150 --K 10 --num_layers 3 \
  --solver krylov --solver_iters 12 \
  --epochs 550 --batch_size 200
```

Key flags:

| Flag | Description |
|---|---|
| `--tasks` | `func`, `struct`, or `all` (default) |
| `--K` | Legendre expansion order for the Laplacian generator |
| `--solver` | `krylov`, `neumann`, or `richardson` for the implicit Cayley half-step |
| `--solver_iters` | number of solver iterations per layer forward pass |
| `--eps_init` | initial (learnable) Cayley step size |
| `--gpus` / `--single_gpu` | multi-GPU (`torch_geometric.nn.DataParallel`) or forced single-GPU |

Model selection is done on the validation split (`ReduceLROnPlateau` +
best-validation checkpointing); the reported test metric always uses the
val-selected checkpoint, not a metric chosen by peeking at the test set
during training.

## Package API

```python
from cayley_legendre import CayleyLegendreGNN, load_dataset, make_loader

model = CayleyLegendreGNN(in_channels=9, hidden=150, out_channels=10, K=10)
```
