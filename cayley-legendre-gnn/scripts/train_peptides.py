from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
from torch_geometric.nn import DataParallel
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cayley_legendre import (
    CayleyLegendreGNN,
    load_dataset,
    make_loader,
    train_epoch_classification,
    evaluate_classification,
    train_epoch_regression,
    evaluate_regression,
    set_seed,
    get_device_ids,
)


def run_peptides(dataset_name: str, args, device, device_ids: list[int]):
    print("\n" + "=" * 60)
    print(f"LRGB {dataset_name}  (Cayley-LegendreNet)")
    print("=" * 60)

    set_seed(args.seed)

    multi_gpu = len(device_ids) > 1

    train_ds, val_ds, test_ds = load_dataset(dataset_name, root=args.root)

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, multi_gpu=multi_gpu)
    val_loader   = make_loader(val_ds,   args.batch_size, shuffle=False, multi_gpu=multi_gpu)
    test_loader  = make_loader(test_ds,  args.batch_size, shuffle=False, multi_gpu=multi_gpu)

    in_channels = train_ds.num_features
    is_classification = dataset_name == "peptides-func"
    out_channels = 10 if is_classification else 11

    model = CayleyLegendreGNN(
        in_channels=in_channels, hidden=args.hidden, out_channels=out_channels,
        K=args.K, num_layers=args.num_layers, dropout=args.dropout,
        eps_init=args.eps_init, solver=args.solver, solver_iters=args.solver_iters,
    ).to(device)

    if multi_gpu:
        print(f"Multi-GPU: wrapping model with torch_geometric.nn.DataParallel "
              f"over GPUs {device_ids}.")
        print(f"Note: --batch_size={args.batch_size} graphs are split across "
              f"{len(device_ids)} GPUs per step (~{args.batch_size // len(device_ids)} "
              f"graphs/GPU); raise --batch_size if you want each GPU to see as many "
              f"graphs per step as a single-GPU run would.")
        model = DataParallel(model, device_ids=device_ids)

    underlying_model = model.module if multi_gpu else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max" if is_classification else "min",
        factor=0.5, patience=20, min_lr=1e-5)

    print(f"Model: {underlying_model}")
    print(f"Dataset: {dataset_name} | "
          f"Train/Val/Test graphs: {len(train_ds)}/{len(val_ds)}/{len(test_ds)} | "
          f"Features: {in_channels} | Targets: {out_channels}\n")

    train_fn, eval_fn, metric_key, better = (
        (train_epoch_classification, evaluate_classification, "AP", max)
        if is_classification else
        (train_epoch_regression, evaluate_regression, "MAE", min)
    )

    best_val = -math.inf if better is max else math.inf
    best_state = None

    pbar = tqdm(range(1, args.epochs + 1), desc=f"{dataset_name}", leave=False)
    for epoch in pbar:
        epoch_start = time.perf_counter()
        loss = train_fn(model, train_loader, optimizer, device, multi_gpu=multi_gpu)
        epoch_time = time.perf_counter() - epoch_start

        postfix = {"loss": f"{loss:.4f}", "s/epoch": f"{epoch_time:.2f}"}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = eval_fn(model, val_loader, device, multi_gpu=multi_gpu)
            val_score = val_metrics[metric_key]
            improved = (val_score > best_val) if better is max else (val_score < best_val)
            if improved:
                best_val = val_score

                best_state = {k: v.detach().clone() for k, v in underlying_model.state_dict().items()}
            scheduler.step(val_score)
            postfix[f"val_{metric_key}"] = f"{val_score:.4f}"
            postfix[f"best_val_{metric_key}"] = f"{best_val:.4f}"

        pbar.set_postfix(postfix)

    if best_state is not None:
        underlying_model.load_state_dict(best_state)
    final_test_metrics = eval_fn(model, test_loader, device, multi_gpu=multi_gpu)
    print(f"\n{dataset_name}: Final Test {metric_key} (val-selected checkpoint): "
          f"{final_test_metrics[metric_key]:.4f}")
    return {
        **final_test_metrics,
        f"best_val_{metric_key}": best_val,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LRGB Peptides-func / Peptides-struct on Cayley-LegendreNet"
    )
    parser.add_argument("--hidden",       type=int,   default=150)
    parser.add_argument("--K",            type=int,   default=10)
    parser.add_argument("--num_layers",   type=int,   default=3)
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size",   type=int,   default=200)
    parser.add_argument("--epochs",       type=int,   default=550)
    parser.add_argument("--eval_every",   type=int,   default=1,
                        help="run validation every N epochs (1 = every epoch, like the LRGB baseline)")
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--root",         type=str,   default="/tmp/LRGB",
                        help="root directory for the LRGB Peptides dataset download/cache")

    parser.add_argument("--eps_init",     type=float, default=1.0,
                        help="initial Cayley step size (softplus-parameterized, learnable)")
    parser.add_argument("--solver",       type=str,   default="krylov",
                        choices=["krylov", "neumann", "richardson"],
                        help="linear solver for the implicit Cayley half-step")
    parser.add_argument("--solver_iters", type=int,   default=12,
                        help="number of solver iterations per layer forward pass")

    parser.add_argument("--gpus",         type=str,   default=None,
                        help="comma-separated GPU ids to use, e.g. '0,1'. "
                             "Defaults to every visible GPU.")
    parser.add_argument("--single_gpu",   action="store_true",
                        help="force single-GPU training even if multiple GPUs are visible "
                             "(e.g. for a single-vs-multi-GPU timing comparison)")

    parser.add_argument("--tasks", nargs="+",
                        choices=["func", "struct", "all"], default=["all"])

    args, _ = parser.parse_known_args()

    run_func_flag   = "all" in args.tasks or "func"   in args.tasks
    run_struct_flag = "all" in args.tasks or "struct" in args.tasks

    device_ids = get_device_ids(args)
    device = torch.device(f"cuda:{device_ids[0]}") if device_ids else torch.device("cpu")

    if device_ids:
        gpu_names = ", ".join(f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in device_ids)
        print(f"Device: {device} | Visible GPUs ({len(device_ids)}): {gpu_names}")
        if len(device_ids) > 1:
            print("Running in multi-GPU mode via torch_geometric.nn.DataParallel.")
    else:
        print(f"Device: {device} (no CUDA GPU visible)")

    results = {}
    if run_func_flag:
        results["peptides-func"] = run_peptides("peptides-func", args, device, device_ids)
    if run_struct_flag:
        results["peptides-struct"] = run_peptides("peptides-struct", args, device, device_ids)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for name, metrics in results.items():
            key = "AP" if name == "peptides-func" else "MAE"
            print(f"{name:<18} test {key} (val-selected) = {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
