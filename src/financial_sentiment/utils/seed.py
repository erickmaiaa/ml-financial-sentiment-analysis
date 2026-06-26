"""Reproducibility helpers.

Centralises seeding of every source of randomness used by the project so that
runs are deterministic. Heavy optional dependencies (NumPy, PyTorch,
Transformers) are imported lazily and skipped silently when not installed.
"""

from __future__ import annotations

import os
import random

from financial_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42, *, deterministic_torch: bool = True) -> None:
    """Seed all known random number generators.

    Args:
        seed: The seed value applied across ``random``, NumPy, PyTorch and the
            ``PYTHONHASHSEED`` environment variable.
        deterministic_torch: When ``True`` and PyTorch is available, request
            deterministic cuDNN behaviour (slower but reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug("NumPy not available; skipping its seed.")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug("PyTorch not available; skipping its seed.")

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug("Transformers not available; skipping its seed.")

    logger.info("Global random seed set to %d.", seed)


__all__ = ["set_global_seed"]
