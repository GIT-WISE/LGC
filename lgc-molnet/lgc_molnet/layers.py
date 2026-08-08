"""Core building blocks: normalised Laplacian utilities, Legendre polynomial
propagation, skew-symmetric linear maps, and the Cayley-Legendre graph layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian

from .solvers import krylov_solve, neumann_solve

__all__ = [
    "normalise_laplacian",
    "legendre_propagate_sparse",
    "SkewLinear",
    "GeneratorA",
    "CayleyLegendreLayer",
    "sparse_laplacian_from_cache",
]


def normalise_laplacian(edge_index, edge_weight, num_nodes, lambda_max=2.0):
    """Compute the symmetric-normalised, rescaled graph Laplacian.

    Returns edge indices/weights of ``(2 / lambda_max) * L_sym - I``, which
    has spectrum in ``[-1, 1]``.
    """
    lap_idx, lap_w = get_laplacian(
        edge_index, edge_weight, normalization="sym", num_nodes=num_nodes
    )
    scale = 2.0 / float(lambda_max)
    scaled_w = lap_w * scale
    diag_mask = lap_idx[0] == lap_idx[1]
    scaled_w = scaled_w.clone()
    scaled_w[diag_mask] -= 1.0
    return lap_idx, scaled_w


def legendre_propagate_sparse(L_sparse: torch.Tensor, X: torch.Tensor, K: int):
    """Generate Legendre-polynomial propagated features P_0(L)X, ..., P_K(L)X
    using the three-term recurrence, with all matrix products done via
    sparse matmul.
    """
    outs = [X]
    if K == 0:
        return outs
    Pkm1 = X
    Pk = torch.sparse.mm(L_sparse, X)
    outs.append(Pk)
    for n in range(1, K):
        Pk_new = ((2 * n + 1) * torch.sparse.mm(L_sparse, Pk) - n * Pkm1) / (n + 1)
        outs.append(Pk_new)
        Pkm1, Pk = Pk, Pk_new
    return outs


class SkewLinear(nn.Module):
    """A learnable skew-symmetric linear map, parameterised as R - R^T."""

    def __init__(self, d: int):
        super().__init__()
        self.R = nn.Parameter(torch.empty(d, d))
        nn.init.orthogonal_(self.R)
        with torch.no_grad():
            self.R.mul_(1.0 / math.sqrt(d))

    def weight(self) -> torch.Tensor:
        return self.R - self.R.T

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        W = self.weight()
        return X @ W.T


class GeneratorA(nn.Module):
    """Skew-symmetric generator built from a bank of Legendre-propagated,
    per-order skew-symmetric linear maps.
    """

    def __init__(self, hidden_dim: int, K: int):
        super().__init__()
        self.K = K
        self.skews = nn.ModuleList([SkewLinear(hidden_dim) for _ in range(K + 1)])

    def matvec(self, L_sparse: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        Pks = legendre_propagate_sparse(L_sparse, X, self.K)
        out = self.skews[0](Pks[0])
        for k in range(1, self.K + 1):
            out = out + self.skews[k](Pks[k])
        return out


class CayleyLegendreLayer(nn.Module):
    """A single Cayley-transform graph propagation layer.

    Applies ``Y = (I - eps/2 * A)^{-1} (I + eps/2 * A) x`` where ``A`` is a
    skew-symmetric (hence orthogonal-generating) operator built from
    Legendre-propagated graph features. The inverse is approximated with an
    iterative solver (Krylov or truncated Neumann series) so no explicit
    matrix inversion is required.
    """

    def __init__(self, hidden_dim: int, K: int = 8, eps_init: float = 1.0,
                 solver: str = "krylov", solver_iters: int = 12,
                 learnable_eps: bool = True):
        super().__init__()
        self.A = GeneratorA(hidden_dim, K)
        raw = math.log(math.exp(eps_init) - 1.0)
        self.raw_eps = nn.Parameter(torch.tensor(float(raw)), requires_grad=learnable_eps)
        self.solver = solver
        self.solver_iters = solver_iters

    def eps(self) -> torch.Tensor:
        return F.softplus(self.raw_eps)

    def forward(self, x: torch.Tensor, L_sparse: torch.Tensor) -> torch.Tensor:
        eps = self.eps()
        scale = eps / 2

        def matvec(Z):
            return self.A.matvec(L_sparse, Z)

        B = x + scale * matvec(x)

        if self.solver == "neumann":
            Y = neumann_solve(matvec, B, scale, self.solver_iters)
        else:
            Y = krylov_solve(matvec, B, scale, self.solver_iters)
        return Y

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(hidden={self.A.skews[0].R.shape[0]}, "
                f"K={self.A.K}, solver={self.solver})")


def sparse_laplacian_from_cache(lap_edge_index, lap_edge_weight, num_nodes):
    """Reassemble a coalesced sparse COO Laplacian tensor from cached
    (precomputed) edge index/weight tensors.
    """
    return torch.sparse_coo_tensor(
        lap_edge_index, lap_edge_weight, size=(num_nodes, num_nodes)
    ).coalesce()
