#!/usr/bin/env python
"""Command-line entry point for training/evaluating LGC_MoleNet on
MoleculeNet datasets.

Usage:
    python main.py --dataset bace --eps_mode variable
    python main.py --dataset all --eps_mode both
"""

from __future__ import annotations

import argparse

from lgc_molnet import DATASET_NAME_MAP, run


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate LGC_MoleNet on MoleculeNet datasets.")

    parser.add_argument("--dataset", type=str, default="all",
                         choices=["bace", "bbbp", "hiv", "all"],
                         help="Which dataset to run. 'all' sweeps bace/bbbp/hiv.")
    parser.add_argument("--eps_mode", type=str, default="both",
                         choices=["variable", "fixed", "both"],
                         help="Whether eps is learnable ('variable'), fixed, or 'both' (sweep).")
    parser.add_argument("--root", type=str, default="./data",
                         help="Root directory for the MoleculeNet dataset cache.")

    parser.add_argument("--hidden_dim", type=int, default=128,
                         help="Hidden dimension size.")
    parser.add_argument("--num_layers", type=int, default=4,
                         help="Number of CayleyLegendreLayer blocks.")
    parser.add_argument("--K", type=int, default=8,
                         help="Legendre propagation order.")
    parser.add_argument("--solver", type=str, default="krylov",
                         choices=["krylov", "neumann"],
                         help="Linear solver used inside the Cayley layer.")
    parser.add_argument("--solver_iters", type=int, default=12,
                         help="Number of solver iterations.")
    parser.add_argument("--eps_init", type=float, default=1.0,
                         help="Initial value of eps before the softplus reparameterisation.")
    parser.add_argument("--lambda_max", type=float, default=2.0,
                         help="Lambda max used for Laplacian rescaling.")
    parser.add_argument("--dropout", type=float, default=0.5,
                         help="Dropout probability.")

    parser.add_argument("--lr", type=float, default=1e-3,
                         help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                         help="Weight decay (L2 regularisation).")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100,
                         help="Maximum number of training epochs.")
    parser.add_argument("--patience", type=int, default=30,
                         help="Early-stopping patience (epochs without validation improvement).")

    parser.add_argument("--seed", type=int, default=0,
                         help="Random seed.")
    parser.add_argument("--device", type=str, default=None,
                         help="Device to use, e.g. 'cuda' or 'cpu'. Defaults to auto-detect.")

    return parser


def main():
    args, _unknown = build_arg_parser().parse_known_args()

    datasets = ["bace", "bbbp", "hiv"] if args.dataset == "all" else [args.dataset]
    eps_modes = ["variable", "fixed"] if args.eps_mode == "both" else [args.eps_mode]

    shared_kwargs = dict(
        root=args.root,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        K=args.K,
        solver=args.solver,
        solver_iters=args.solver_iters,
        eps_init=args.eps_init,
        lambda_max=args.lambda_max,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )

    results = {}
    for dataset in datasets:
        for eps_mode in eps_modes:
            auc = run(dataset=dataset, eps_mode=eps_mode, **shared_kwargs)
            results[(dataset, eps_mode)] = auc

    print("\n=== Summary (final test ROC-AUC) ===", flush=True)
    header = f"{'Dataset':<8}" + "".join(f"{('Variable c' if m == 'variable' else 'Fixed c'):<14}" for m in eps_modes)
    print(header, flush=True)
    for dataset in datasets:
        row = f"{DATASET_NAME_MAP[dataset]:<8}"
        for eps_mode in eps_modes:
            row += f"{results[(dataset, eps_mode)] * 100:<14.2f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
