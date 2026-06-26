"""Text cleaning and normalisation for financial sentiment analysis.

This module consolidates the cleaning logic that the original notebook split
across ``clean_text`` and ``preprocess_text`` and fixes several latent bugs in
the process (documented inline). It exposes a single configurable
:class:`TextCleaner`.

Design notes
------------
* **Two-stage cleaning.** ``clean_dataframe`` runs cheap, vectorised regex
  cleaning suitable for *both* the transformer and ML pipelines. ``normalize``
  performs the heavier token-level normalisation (contraction expansion,
  stop-word removal, lemmatisation, stemming) that only benefits the classical
  ML models — transformers work best on minimally processed text.
* **Lazy heavy imports.** ``emoji`` and ``nltk`` are imported on first use so
  the cheap regex helpers (and their unit tests) need no heavy dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from financial_sentiment.utils.logging import get_logger
from financial_sentiment.utils.nltk_resources import ensure_nltk_resources

logger = get_logger(__name__)

# Extra domain/noise tokens to discard on top of the standard English stop-word
# list (carried over from the original notebook).
_EXTRA_STOPWORDS: frozenset[str] = frozenset(
    {
        "rt", "mkr", "didn", "bc", "n", "m", "im", "ll", "y", "ve", "u", "ur",
        "don", "p", "t", "s", "aren", "kp", "o", "kat", "de", "re", "amp", "will",
    }
)

# Ordered contraction expansions. Order matters: the broad ``n't`` rule must run
# after the specific ``won't``/``can't`` rules.
_CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, flags=re.IGNORECASE), repl)
    for pattern, repl in (
        (r"won't", "will not"),
        (r"can't", "can not"),
        (r"n't", " not"),
        (r"'re", " are"),
        (r"'s", " is"),
        (r"'d", " would"),
        (r"'ll", " will"),
        (r"'ve", " have"),
        (r"'m", " am"),
    )
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_NON_LETTER_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


@dataclass
class TextCleaner:
    """Configurable text cleaner / normaliser.

    Args:
        language: Stemmer language passed to NLTK's :class:`SnowballStemmer`.
        remove_stopwords: Whether ``normalize`` drops stop words.
        apply_lemmatization: Whether ``normalize`` lemmatises tokens.
        apply_stemming: Whether ``normalize`` stems tokens (applied *after*
            lemmatisation, which is the correct order — the original notebook
            stemmed first, which mangled words before the lemmatiser could act).
        min_token_length: Tokens shorter than this are dropped during
            normalisation.
    """

    language: str = "english"
    remove_stopwords: bool = True
    apply_lemmatization: bool = True
    apply_stemming: bool = True
    min_token_length: int = 2
    extra_stopwords: frozenset[str] = field(default=_EXTRA_STOPWORDS)

    # --- Lazily constructed heavy resources -----------------------------------

    @cached_property
    def _stopwords(self) -> frozenset[str]:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return frozenset(ENGLISH_STOP_WORDS) | self.extra_stopwords

    @cached_property
    def _lemmatizer(self):  # noqa: ANN202 - third-party type
        ensure_nltk_resources()
        from nltk.stem import WordNetLemmatizer

        return WordNetLemmatizer()

    @cached_property
    def _stemmer(self):  # noqa: ANN202 - third-party type
        from nltk.stem.snowball import SnowballStemmer

        return SnowballStemmer(self.language)

    # --- Stage 1: cheap, vectorised cleaning ----------------------------------

    @staticmethod
    def clean_dataframe(
        frame: pd.DataFrame, column: str, *, inplace: bool = False
    ) -> pd.DataFrame:
        """Apply fast, vectorised regex cleaning to a text column.

        Removes URLs, @-mentions and #-hashtags, strips characters outside the
        basic Latin letter set and lower-cases the result.

        Note:
            The original ``clean_text`` called ``Series.str.replace`` with regex
            patterns but without ``regex=True``; on modern pandas those patterns
            were treated as *literals* and silently did nothing. This version
            passes ``regex=True`` explicitly.

        Args:
            frame: Source dataframe.
            column: Name of the text column to clean.
            inplace: Mutate ``frame`` when ``True``; otherwise operate on a copy.

        Returns:
            The dataframe with the cleaned column.
        """
        target = frame if inplace else frame.copy()
        series = target[column].fillna("").astype(str)
        series = series.str.replace(_URL_RE, " ", regex=True)
        series = series.str.replace(_MENTION_RE, " ", regex=True)
        series = series.str.replace(_HASHTAG_RE, " ", regex=True)
        series = series.str.lower()
        series = series.str.replace(_NON_LETTER_RE, " ", regex=True)
        series = series.str.replace(_MULTISPACE_RE, " ", regex=True).str.strip()
        target[column] = series
        return target

    # --- Stage 2: token-level normalisation -----------------------------------

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expand common English contractions (e.g. ``can't`` -> ``can not``)."""
        for pattern, replacement in _CONTRACTIONS:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def strip_emojis(text: str) -> str:
        """Remove emoji characters using the modern ``emoji`` API.

        Note:
            Replaces the removed ``emoji.get_emoji_regexp()`` call (deleted in
            ``emoji>=2.0``) with the supported ``emoji.replace_emoji``.
        """
        import emoji

        return emoji.replace_emoji(text, replace="")

    def normalize(self, text: str) -> str:
        """Fully normalise a single text for the classical ML pipeline.

        Pipeline: lower-case -> expand contractions -> strip emojis/non-ASCII
        -> keep only letters -> tokenise -> drop short/stop tokens ->
        lemmatise -> stem.

        Args:
            text: Raw input text.

        Returns:
            A normalised, whitespace-joined token string.
        """
        if not isinstance(text, str) or not text:
            return ""

        text = self.expand_contractions(text.lower())
        text = self.strip_emojis(text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = _NON_LETTER_RE.sub(" ", text)

        tokens: list[str] = []
        for token in text.split():
            if len(token) < self.min_token_length:
                continue
            if self.remove_stopwords and token in self._stopwords:
                continue
            if self.apply_lemmatization:
                token = self._lemmatizer.lemmatize(token)
            if self.apply_stemming:
                token = self._stemmer.stem(token)
            if token and (not self.remove_stopwords or token not in self._stopwords):
                tokens.append(token)
        return " ".join(tokens)

    def normalize_series(self, series: pd.Series) -> pd.Series:
        """Vectorised wrapper applying :meth:`normalize` over a Series."""
        return series.fillna("").astype(str).map(self.normalize)


__all__ = ["TextCleaner"]
