"""Feature extraction for the classical ML pipeline."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from financial_sentiment.config import MLConfig


def build_tfidf_vectorizer(config: MLConfig) -> TfidfVectorizer:
    """Construct a TF-IDF vectoriser from configuration.

    Uses the default (regex) tokenizer rather than ``nltk.word_tokenize`` for
    speed and to avoid an extra NLTK dependency at vectorisation time; the text
    has already been normalised upstream by :class:`TextCleaner`.

    Args:
        config: Machine-learning configuration with TF-IDF thresholds.

    Returns:
        An unfitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    """
    return TfidfVectorizer(
        use_idf=config.use_idf,
        min_df=config.tfidf_min_df,
        max_df=config.tfidf_max_df,
    )


__all__ = ["build_tfidf_vectorizer"]
