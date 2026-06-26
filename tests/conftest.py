"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    """A tiny, representative sentiment dataframe."""
    return pd.DataFrame(
        {
            "Sentence": [
                "Profits soared to a record high this quarter!",
                "The company reported heavy losses and layoffs.",
                "Shares were unchanged during the session.",
                "The company reported heavy losses and layoffs.",  # duplicate
            ],
            "Sentiment": ["positive", "negative", "neutral", "negative"],
        }
    )


@pytest.fixture()
def sample_csv(tmp_path, sample_frame) -> str:
    """Write the sample frame to a temporary CSV and return its path."""
    path = tmp_path / "data.csv"
    sample_frame.to_csv(path, index=False)
    return str(path)
