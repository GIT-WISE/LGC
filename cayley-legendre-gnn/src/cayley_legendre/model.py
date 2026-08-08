from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data

from .layers import CayleyLegendreLayer
from .laplacian import build_L_sparse, sparse_laplacian_from_cache


class CayleyLegendreGNN(nn.Module):

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 K: int = 5, num_layers: int = 4, dropout: float = 0.2,
                 eps_init: float = 1.0, solver: str = "krylov",
                 solver_iters: int = 12):
        super().__init__()
        self.dropout = dropout
        self.encoder = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList([
            CayleyLegendreLayer(hidden, K=K, eps_init=eps_init, solver=solver,
                                 solver_iters=solver_iters)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )

    def forward(self, *args, **kwargs) -> torch.Tensor:

        if len(args) == 1 and isinstance(args[0], Data):
            data = args[0]
            x = data.x.float()
            edge_index = data.edge_index
            batch = data.batch
            lap_edge_index = getattr(data, "lap_edge_index", None)
            lap_edge_weight = getattr(data, "lap_edge_weight", None)
            edge_weight = getattr(data, "edge_weight", None)
        else:
            x, edge_index, batch = args[0], args[1], args[2]
            lap_edge_index = kwargs.get("lap_edge_index")
            lap_edge_weight = kwargs.get("lap_edge_weight")
            edge_weight = kwargs.get("edge_weight")

        num_nodes = x.size(0)
        if lap_edge_index is not None and lap_edge_weight is not None:

            L_sparse = sparse_laplacian_from_cache(lap_edge_index, lap_edge_weight, num_nodes)
        else:

            L_sparse = build_L_sparse(edge_index, edge_weight, num_nodes)

        x = self.encoder(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, L_sparse)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.head(x)

    def __repr__(self) -> str:
        return (f"CayleyLegendreGNN(layers={len(self.convs)}, "
                f"K={self.convs[0].A.K if self.convs else None}, "
                f"dropout={self.dropout})")
