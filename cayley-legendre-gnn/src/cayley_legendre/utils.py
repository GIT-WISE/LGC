from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_ids(args) -> list[int]:

    if not torch.cuda.is_available():
        return []
    if args.single_gpu:
        return [0]
    if args.gpus is not None:
        return [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    return list(range(torch.cuda.device_count()))
