"""Tests for the text cleaner / normaliser."""

from __future__ import annotations

import importlib.util

import pytest

from financial_sentiment.preprocessing import TextCleaner

_HAS_NLTK = importlib.util.find_spec("nltk") is not None
_HAS_EMOJI = importlib.util.find_spec("emoji") is not None


def test_expand_contractions() -> None:
    assert TextCleaner.expand_contractions("can't won't isn't") == (
        "can not will not is not"
    )
    assert "are" in TextCleaner.expand_contractions("they're happy")


def test_clean_dataframe_strips_urls_mentions_hashtags(sample_frame) -> None:
    frame = sample_frame.copy()
    frame.loc[0, "Sentence"] = "Great news http://x.co @ceo #bullish 100%"
    cleaned = TextCleaner.clean_dataframe(frame, "Sentence")
    text = cleaned.loc[0, "Sentence"]
    assert "http" not in text
    assert "@" not in text and "#" not in text
    assert text == text.lower()
    assert "100" not in text  # digits removed


def test_clean_dataframe_does_not_mutate_input(sample_frame) -> None:
    original = sample_frame["Sentence"].tolist()
    TextCleaner.clean_dataframe(sample_frame, "Sentence")
    assert sample_frame["Sentence"].tolist() == original


@pytest.mark.skipif(not _HAS_EMOJI, reason="emoji not installed")
def test_strip_emojis() -> None:
    assert TextCleaner.strip_emojis("up 📈 today 🚀") .strip() in {"up  today", "up today"}


@pytest.mark.skipif(not _HAS_NLTK, reason="nltk not installed")
def test_normalize_produces_tokens() -> None:
    cleaner = TextCleaner()
    out = cleaner.normalize("The companies reported soaring profits!!!")
    assert isinstance(out, str)
    assert out  # non-empty
    assert "the" not in out.split()  # stop word removed


def test_normalize_handles_empty() -> None:
    cleaner = TextCleaner()
    assert cleaner.normalize("") == ""
    assert cleaner.normalize(None) == ""  # type: ignore[arg-type]
