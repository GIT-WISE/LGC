"""LGC_MoleNet: a Cayley-Legendre graph convolutional network for
graph-level molecular property prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .layers import CayleyLegendreLayer, sparse_laplacian_from_cache

__all__ = ["LGC_MoleNet"]


class LGC_MoleNet(nn.Module):
    """Stack of ``CayleyLegendreLayer`` blocks with residual connections,
    batch norm, and a final MLP classifier head over mean-pooled graph
    embeddings.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_tasks: int,
                 num_layers: int = 4, K: int = 8, solver: str = "krylov",
                 solver_iters: int = 12, eps_init: float = 1.0,
                 eps_mode: str = "variable", dropout: float = 0.5):
        super().__init__()
        assert eps_mode in ("variable", "fixed")
        learnable_eps = eps_mode == "variable"

        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            CayleyLegendreLayer(hidden_dim, K=K, eps_init=eps_init, solver=solver,
                                 solver_iters=solver_iters, learnable_eps=learnable_eps)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, data) -> torch.Tensor:
        x = data.x.float()
        batch = data.batch
        num_nodes = x.size(0)

        L_sparse = sparse_laplacian_from_cache(data.lap_edge_index, data.lap_edge_weight, num_nodes)

        h = self.atom_encoder(x)
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, L_sparse)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new

        g = global_mean_pool(h, batch)
        return self.classifier(g)
