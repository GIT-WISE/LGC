from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .solvers import legendre_propagate_sparse, neumann_solve, krylov_solve


class SkewLinear(nn.Module):

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
