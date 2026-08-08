from __future__ import annotations

import torch
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data


def normalise_laplacian(edge_index, edge_weight, num_nodes, lambda_max=2.0):
    lap_idx, lap_w = get_laplacian(
        edge_index, edge_weight, normalization="sym", num_nodes=num_nodes
    )
    scale = 2.0 / float(lambda_max)
    scaled_w = lap_w * scale
    diag_mask = lap_idx[0] == lap_idx[1]
    scaled_w = scaled_w.clone()
    scaled_w[diag_mask] -= 1.0
    return lap_idx, scaled_w


def add_normalized_laplacian(data: Data) -> Data:

    lap_idx, lap_w = normalise_laplacian(data.edge_index, None, data.num_nodes)
    data.lap_edge_index = lap_idx
    data.lap_edge_weight = lap_w
    return data


def build_L_sparse(edge_index, edge_weight, num_nodes):

    L_idx, L_w = normalise_laplacian(edge_index, edge_weight, num_nodes)
    return torch.sparse_coo_tensor(L_idx, L_w, size=(num_nodes, num_nodes)).coalesce()


def sparse_laplacian_from_cache(lap_edge_index, lap_edge_weight, num_nodes):

    return torch.sparse_coo_tensor(
        lap_edge_index, lap_edge_weight, size=(num_nodes, num_nodes)
    ).coalesce()
