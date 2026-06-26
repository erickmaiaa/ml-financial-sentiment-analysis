"""Dataset ingestion.

Responsible *only* for reading the raw CSV, validating its schema, removing
duplicates and encoding the sentiment labels. It performs no text cleaning —
that concern lives in :mod:`financial_sentiment.preprocessing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from financial_sentiment.config import DataConfig
from financial_sentiment.config.settings import PROJECT_ROOT
from financial_sentiment.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical column name for the integer-encoded target.
LABEL_COLUMN = "labels"


def _resolve_path(path: Path) -> Path:
    """Resolve a possibly-relative data path against the repository root.

    This makes the dataset location independent of the current working
    directory, so the notebook (run from ``notebooks/``) and the CLI scripts
    (run from the repo root) both find the same file.
    """
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass(frozen=True)
class LoadedDataset:
    """Container bundling the dataframe with its label encoder.

    Attributes:
        frame: The cleaned dataframe with an added ``labels`` column.
        label_encoder: The fitted encoder, exposing ``classes_`` for decoding
            integer predictions back to their textual sentiment.
    """

    frame: pd.DataFrame
    label_encoder: LabelEncoder

    @property
    def class_names(self) -> list[str]:
        """Human-readable class names ordered by their encoded index."""
        return list(self.label_encoder.classes_)


def load_dataset(config: DataConfig) -> LoadedDataset:
    """Read and prepare the raw sentiment dataset.

    Args:
        config: Data configuration describing the file path and columns.

    Returns:
        A :class:`LoadedDataset` with duplicates removed and labels encoded.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        KeyError: If the expected text/label columns are missing.
    """
    path = _resolve_path(config.data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    logger.info("Loading dataset from %s", path)
    frame = pd.read_csv(path)

    missing = {config.text_column, config.label_column} - set(frame.columns)
    if missing:
        raise KeyError(f"Dataset is missing required column(s): {sorted(missing)}")

    if config.drop_duplicates:
        before = len(frame)
        frame = frame.drop_duplicates(subset=[config.text_column], keep="first")
        logger.info("Dropped %d duplicate rows.", before - len(frame))

    encoder = LabelEncoder()
    frame[LABEL_COLUMN] = encoder.fit_transform(frame[config.label_column])

    frame = frame.reset_index(drop=True)
    logger.info(
        "Loaded %d rows across %d classes: %s",
        len(frame),
        len(encoder.classes_),
        list(encoder.classes_),
    )
    return LoadedDataset(frame=frame, label_encoder=encoder)


__all__ = ["LABEL_COLUMN", "LoadedDataset", "load_dataset"]
