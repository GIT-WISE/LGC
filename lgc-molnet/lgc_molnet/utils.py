"""Miscellaneous shared utilities."""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["set_seed"]


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch (CPU + all CUDA devices) RNGs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
