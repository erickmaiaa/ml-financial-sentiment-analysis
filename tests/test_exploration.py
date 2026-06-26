"""Tests for the pandas-based EDA helper."""

from __future__ import annotations

import pytest

from financial_sentiment.data import summarize_dataset


def test_summarize_dataset_sections(sample_frame) -> None:
    summary = summarize_dataset(sample_frame, "Sentence", "Sentiment")

    # Overview reflects the one duplicated sentence in the fixture.
    assert int(summary.overview.loc[0, "rows"]) == 4
    assert int(summary.overview.loc[0, "duplicate_texts"]) == 1

    # Class distribution sums to 100% and covers every label.
    assert set(summary.class_distribution.index) == {"positive", "negative", "neutral"}
    assert summary.class_distribution["percentage"].sum() == pytest.approx(100.0)

    # Text-length table exposes char/word statistics.
    assert {"char_count", "word_count"} <= set(summary.text_length.columns)


def test_summarize_dataset_missing_column(sample_frame) -> None:
    with pytest.raises(KeyError):
        summarize_dataset(sample_frame, "Nope", "Sentiment")
