from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import average_precision_score_torch


def train_epoch_classification(model, loader, optimizer, device, multi_gpu: bool = False) -> float:

    model.train()
    total_loss = 0.0
    for data in loader:
        optimizer.zero_grad()
        if multi_gpu:
            out = model(data)
            y = torch.cat([d.y for d in data], dim=0).to(out.device).float()
            num_graphs = len(data)
        else:
            data = data.to(device)
            out = model(
                data.x.float(), data.edge_index, data.batch,
                lap_edge_index=data.lap_edge_index, lap_edge_weight=data.lap_edge_weight,
            )
            y = data.y.float()
            num_graphs = data.num_graphs
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_classification(model, loader, device, multi_gpu: bool = False) -> dict:

    model.eval()
    ys, scores = [], []
    for data in loader:
        if multi_gpu:
            out = model(data)
            y = torch.cat([d.y for d in data], dim=0)
        else:
            data = data.to(device)
            out = model(
                data.x.float(), data.edge_index, data.batch,
                lap_edge_index=data.lap_edge_index, lap_edge_weight=data.lap_edge_weight,
            )
            y = data.y
        ys.append(y.cpu().numpy())
        scores.append(torch.sigmoid(out).cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(scores, axis=0)
    ap = average_precision_score_torch(y_true, y_score)
    return {"AP": ap}


def train_epoch_regression(model, loader, optimizer, device, multi_gpu: bool = False) -> float:

    model.train()
    total_loss = 0.0
    for data in loader:
        optimizer.zero_grad()
        if multi_gpu:
            out = model(data)
            y = torch.cat([d.y for d in data], dim=0).to(out.device).float()
            num_graphs = len(data)
        else:
            data = data.to(device)
            out = model(
                data.x.float(), data.edge_index, data.batch,
                lap_edge_index=data.lap_edge_index, lap_edge_weight=data.lap_edge_weight,
            )
            y = data.y.float()
            num_graphs = data.num_graphs
        loss = F.l1_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_regression(model, loader, device, multi_gpu: bool = False) -> dict:

    model.eval()
    total_mae, total_n = 0.0, 0
    for data in loader:
        if multi_gpu:
            out = model(data)
            y = torch.cat([d.y for d in data], dim=0).to(out.device).float()
        else:
            data = data.to(device)
            out = model(
                data.x.float(), data.edge_index, data.batch,
                lap_edge_index=data.lap_edge_index, lap_edge_weight=data.lap_edge_weight,
            )
            y = data.y.float()
        mae = F.l1_loss(out, y, reduction="sum")
        total_mae += mae.item()
        total_n += y.numel()
    return {"MAE": total_mae / total_n}
