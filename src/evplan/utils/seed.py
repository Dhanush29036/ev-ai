"""Seed utility — set global random seeds for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and environment variables."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
    except ImportError:
        pass
