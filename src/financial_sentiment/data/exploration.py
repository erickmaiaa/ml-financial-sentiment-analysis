"""Lightweight, dependency-free exploratory data analysis.

Profiles a text-classification dataset using only pandas — shape, dtypes,
missing values, duplicates, class balance and text-length statistics — so EDA
works on any Python version with no extra dependencies. Returns plain dataframes
that render nicely in a notebook and are easy to assert on in tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from financial_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSummary:
    """Structured EDA results for a text-classification dataset.

    Attributes:
        overview: One-row table with shape, duplicate and missing counts.
        dtypes: Column data types and per-column missing/unique counts.
        class_distribution: Label counts and percentages.
        text_length: Character/word-length statistics, overall and per class.
    """

    overview: pd.DataFrame
    dtypes: pd.DataFrame
    class_distribution: pd.DataFrame
    text_length: pd.DataFrame

    def display(self) -> None:
        """Pretty-print every section (useful outside notebooks)."""
        for title, frame in (
            ("Overview", self.overview),
            ("Columns", self.dtypes),
            ("Class distribution", self.class_distribution),
            ("Text length", self.text_length),
        ):
            print(f"\n=== {title} ===")
            print(frame.to_string())


def _overview(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rows": [len(frame)],
            "columns": [frame.shape[1]],
            "duplicate_texts": [int(frame[text_column].duplicated().sum())],
            "total_missing": [int(frame.isna().sum().sum())],
            "memory_kb": [round(frame.memory_usage(deep=True).sum() / 1024, 1)],
        }
    )


def _column_info(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing": frame.isna().sum(),
            "missing_pct": (frame.isna().mean() * 100).round(2),
            "unique": frame.nunique(),
        }
    )


def _class_distribution(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    counts = frame[label_column].value_counts()
    return pd.DataFrame(
        {
            "count": counts,
            "percentage": (counts / len(frame) * 100).round(2),
        }
    )


def _text_length(frame: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    text = frame[text_column].fillna("").astype(str)
    lengths = pd.DataFrame(
        {
            label_column: frame[label_column].values,
            "char_count": text.str.len().values,
            "word_count": text.str.split().map(len).values,
        }
    )

    metrics = ["char_count", "word_count"]
    # Overall statistics (rows) share the same two metric columns as the
    # per-class means, so the final table is a single, NaN-free block.
    overall = lengths[metrics].agg(["mean", "min", "max"]).round(1)
    overall.index = [f"overall_{stat}" for stat in overall.index]

    per_class = lengths.groupby(label_column)[metrics].mean().round(1)
    per_class.index = [f"mean[{idx}]" for idx in per_class.index]

    return pd.concat([overall, per_class])


def summarize_dataset(
    frame: pd.DataFrame,
    text_column: str = "Sentence",
    label_column: str = "Sentiment",
) -> DatasetSummary:
    """Compute a compact, pandas-only profile of the dataset.

    Args:
        frame: The dataset to profile.
        text_column: Name of the free-text column.
        label_column: Name of the categorical label column.

    Returns:
        A :class:`DatasetSummary` bundling the individual report tables.

    Raises:
        KeyError: If ``text_column`` or ``label_column`` is absent.
    """
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise KeyError(f"Columns not found in dataframe: {sorted(missing)}")

    logger.info("Profiling dataset (%d rows).", len(frame))
    return DatasetSummary(
        overview=_overview(frame, text_column),
        dtypes=_column_info(frame),
        class_distribution=_class_distribution(frame, label_column),
        text_length=_text_length(frame, text_column, label_column),
    )


__all__ = ["DatasetSummary", "summarize_dataset"]
