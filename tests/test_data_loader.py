"""Tests for dataset loading."""

from __future__ import annotations

import pandas as pd
import pytest

from financial_sentiment.config import DataConfig
from financial_sentiment.data import LABEL_COLUMN, load_dataset


def test_load_dataset_encodes_and_dedupes(sample_csv) -> None:
    result = load_dataset(DataConfig(data_path=sample_csv))
    # One duplicate row removed (4 -> 3).
    assert len(result.frame) == 3
    assert LABEL_COLUMN in result.frame.columns
    assert set(result.class_names) == {"positive", "negative", "neutral"}
    assert pd.api.types.is_integer_dtype(result.frame[LABEL_COLUMN])


def test_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(DataConfig(data_path=tmp_path / "nope.csv"))


def test_missing_column_raises(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"foo": ["a"], "bar": ["b"]}).to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_dataset(DataConfig(data_path=path))
