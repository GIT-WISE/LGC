from __future__ import annotations

import torch


def legendre_propagate_sparse(L_sparse: torch.Tensor, X: torch.Tensor, K: int):

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


def neumann_solve(matvec, B: torch.Tensor, scale: torch.Tensor, n_terms: int) -> torch.Tensor:
    term = B
    total = B.clone()
    for _ in range(n_terms):
        term = scale * matvec(term)
        total = total + term
    return total


def richardson_solve(matvec, B: torch.Tensor, scale: torch.Tensor, n_iter: int) -> torch.Tensor:

    Y = B
    for _ in range(n_iter):
        Y = B + scale * matvec(Y)
    return Y


def krylov_solve(matvec, B: torch.Tensor, scale: torch.Tensor, n_iter: int = 12,
                  tol: float = 1e-8) -> torch.Tensor:

    beta = B.norm()
    if beta < 1e-12:
        return B.clone()
    V = [B / beta]
    H_cols = []
    for j in range(n_iter):
        w = V[j] - scale * matvec(V[j])
        col = []
        for i in range(j + 1):
            h_ij = (w * V[i]).sum()
            col.append(h_ij)
            w = w - h_ij * V[i]
        h_next = w.norm()
        col.append(h_next)
        H_cols.append(col)
        if h_next < tol:
            n_iter = j + 1
            break
        V.append(w / h_next)

    m = len(V) - 1
    cols = []
    for j, col in enumerate(H_cols[:m]):
        col_t = torch.stack(col)
        pad = m + 1 - col_t.shape[0]
        if pad > 0:
            col_t = torch.cat([col_t, torch.zeros(pad, device=B.device, dtype=B.dtype)])
        cols.append(col_t)
    H_full = torch.stack(cols, dim=1)

    e1 = torch.zeros(m + 1, device=B.device, dtype=B.dtype)
    e1 = e1.clone()
    e1[0] = beta
    y = torch.linalg.lstsq(H_full, e1.unsqueeze(-1)).solution.squeeze(-1)
    Y = sum(y[i] * V[i] for i in range(len(y)))
    return Y
