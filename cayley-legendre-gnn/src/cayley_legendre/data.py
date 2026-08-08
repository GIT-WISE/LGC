from __future__ import annotations

from torch_geometric.loader import DataLoader as PyGDataLoader, DataListLoader

from .laplacian import add_normalized_laplacian


def load_dataset(name: str, root: str = "/tmp/LRGB"):

    from torch_geometric.datasets import LRGBDataset

    name_map = {
        "peptides-func": "Peptides-func",
        "peptides-struct": "Peptides-struct",
    }
    if name not in name_map:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {list(name_map)}")
    pyg_name = name_map[name]

    train_ds = LRGBDataset(root=root, name=pyg_name, split="train",
                            pre_transform=add_normalized_laplacian)
    val_ds = LRGBDataset(root=root, name=pyg_name, split="val",
                          pre_transform=add_normalized_laplacian)
    test_ds = LRGBDataset(root=root, name=pyg_name, split="test",
                           pre_transform=add_normalized_laplacian)
    return train_ds, val_ds, test_ds


def make_loader(dataset, batch_size: int, shuffle: bool, multi_gpu: bool):

    if multi_gpu:
        return DataListLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
