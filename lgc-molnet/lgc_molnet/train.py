"""Training loop, evaluation, and the top-level experiment runner."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from .data import DATASET_NAME_MAP, make_pre_transform, scaffold_split
from .model import LGC_MoleNet
from .utils import set_seed

__all__ = ["train_epoch", "evaluate", "run"]


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n_examples = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        y = batch.y.view(out.shape).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n_examples += batch.num_graphs
    return total_loss / max(n_examples, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        y_true.append(batch.y.view(out.shape).cpu())
        y_pred.append(out.cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    aucs = []
    for t in range(y_true.shape[1]):
        col_true = y_true[:, t]
        mask = ~np.isnan(col_true)
        if mask.sum() > 0 and len(np.unique(col_true[mask])) > 1:
            aucs.append(roc_auc_score(col_true[mask], y_pred[mask, t]))
    return float(np.mean(aucs)) if aucs else float("nan")


def run(dataset="bace", eps_mode="variable", root="./data", hidden_dim=128,
        num_layers=4, K=8, solver="krylov", solver_iters=12, eps_init=1.0,
        lambda_max=2.0, dropout=0.5, lr=1e-3, weight_decay=0.0, batch_size=32,
        epochs=100, patience=30, seed=0, device=None):
    """Train and evaluate LGC_MoleNet on a single MoleculeNet dataset,
    returning the final test ROC-AUC.
    """
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    mol_dataset = MoleculeNet(
        root=root, name=DATASET_NAME_MAP[dataset],
        pre_transform=make_pre_transform(lambda_max=lambda_max),
    )
    train_idx, valid_idx, test_idx = scaffold_split(mol_dataset, seed=seed)

    train_loader = PyGDataLoader(mol_dataset[train_idx], batch_size=batch_size, shuffle=True)
    valid_loader = PyGDataLoader(mol_dataset[valid_idx], batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(mol_dataset[test_idx], batch_size=batch_size, shuffle=False)

    num_tasks = mol_dataset[0].y.numel()
    in_dim = mol_dataset.num_node_features

    model = LGC_MoleNet(in_dim=in_dim, hidden_dim=hidden_dim, num_tasks=num_tasks,
                         num_layers=num_layers, K=K, solver=solver, solver_iters=solver_iters,
                         eps_init=eps_init, eps_mode=eps_mode, dropout=dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    best_valid_auc, best_epoch, bad_epochs = -1.0, -1, 0
    pbar = tqdm(range(1, epochs + 1), desc=f"{dataset}/{eps_mode}")
    for epoch in pbar:
        loss = train_epoch(model, train_loader, optimizer, device)
        valid_auc = evaluate(model, valid_loader, device)
        scheduler.step(valid_auc)

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        pbar.set_postfix(loss=f"{loss:.4f}", valid_auc=f"{valid_auc:.4f}")

        print(f"[{dataset}/{eps_mode}] epoch {epoch:03d}/{epochs}  "
              f"loss={loss:.4f}  valid_auc={valid_auc:.4f}  "
              f"best_valid_auc={best_valid_auc:.4f} (epoch {best_epoch})  "
              f"bad_epochs={bad_epochs}/{patience}", flush=True)
        if bad_epochs >= patience:
            print(f"[{dataset}/{eps_mode}] early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)", flush=True)
            break

    final_test_auc = evaluate(model, test_loader, device)

    print(f"[{dataset} | {eps_mode}] best epoch={best_epoch}  "
          f"valid AUC={best_valid_auc:.4f}  final test AUC={final_test_auc:.4f}",
          flush=True)
    return final_test_auc
