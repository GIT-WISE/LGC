"""
lgc_molnet
==========

A Cayley-Legendre graph convolution model for molecular property prediction
on MoleculeNet datasets (BACE, BBBP, HIV).
"""

from .layers import (
    normalise_laplacian,
    legendre_propagate_sparse,
    SkewLinear,
    GeneratorA,
    CayleyLegendreLayer,
    sparse_laplacian_from_cache,
)
from .solvers import neumann_solve, richardson_solve, krylov_solve
from .data import (
    make_pre_transform,
    random_split,
    scaffold_split,
    DATASET_NAME_MAP,
)
from .model import LGC_MoleNet
from .utils import set_seed
from .train import train_epoch, evaluate, run

__all__ = [
    "normalise_laplacian",
    "legendre_propagate_sparse",
    "SkewLinear",
    "GeneratorA",
    "CayleyLegendreLayer",
    "sparse_laplacian_from_cache",
    "neumann_solve",
    "richardson_solve",
    "krylov_solve",
    "make_pre_transform",
    "random_split",
    "scaffold_split",
    "DATASET_NAME_MAP",
    "LGC_MoleNet",
    "set_seed",
    "train_epoch",
    "evaluate",
    "run",
]

__version__ = "0.1.0"
