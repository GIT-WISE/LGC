"""Data utilities: MoleculeNet pre-transforms and scaffold/random splitting."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .layers import normalise_laplacian

__all__ = [
    "make_pre_transform",
    "random_split",
    "scaffold_split",
    "DATASET_NAME_MAP",
]

#: Maps the CLI-friendly dataset key to the name expected by
#: ``torch_geometric.datasets.MoleculeNet``.
DATASET_NAME_MAP = {"bace": "BACE", "bbbp": "BBBP", "hiv": "HIV"}


def make_pre_transform(lambda_max: float = 2.0):
    """Return a PyG ``pre_transform`` that caches the rescaled normalised
    Laplacian on each graph as ``lap_edge_index`` / ``lap_edge_weight``.
    """

    def _transform(data):
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        edge_weight = getattr(data, "edge_weight", None)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
        lap_idx, lap_w = normalise_laplacian(edge_index, edge_weight, num_nodes, lambda_max)
        data.lap_edge_index = lap_idx
        data.lap_edge_weight = lap_w
        return data

    return _transform


def _generate_scaffold(smiles: str, include_chirality: bool = False):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def random_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """Simple random train/valid/test split by index."""
    n = len(dataset)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)
    train_idx = idx[:n_train].tolist()
    valid_idx = idx[n_train:n_train + n_valid].tolist()
    test_idx = idx[n_train + n_valid:].tolist()
    return train_idx, valid_idx, test_idx


def scaffold_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0):
    """Deterministic Bemis-Murcko scaffold split (falls back to a random
    split if RDKit is unavailable).
    """
    try:
        import rdkit  # noqa: F401
    except ImportError:
        print("[warn] RDKit not found -- falling back to a random split.")
        return random_split(dataset, frac_train, frac_valid, frac_test, seed)

    scaffolds = defaultdict(list)
    for i in range(len(dataset)):
        smiles = dataset[i].smiles
        scaffolds[_generate_scaffold(smiles)].append(i)

    scaffold_sets = sorted(scaffolds.values(), key=lambda idxs: (len(idxs), idxs[0]), reverse=True)

    n_total = len(dataset)
    n_train = int(frac_train * n_total)
    n_valid = int(frac_valid * n_total)

    train_idx, valid_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, valid_idx, test_idx
