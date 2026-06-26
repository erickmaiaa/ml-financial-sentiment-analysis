"""Lazy, cached download of the NLTK corpora the project depends on.

The original notebook called ``nltk.download('punkt')`` unconditionally on
every run. Here we download each resource only if it is missing, and only the
ones actually required, keeping start-up fast and offline-friendly.
"""

from __future__ import annotations

from financial_sentiment.utils.logging import get_logger

logger = get_logger(__name__)

# Mapping of NLTK ``find`` paths to their downloadable package id. ``punkt_tab``
# is required by NLTK >= 3.8.2 word tokenisation; ``wordnet`` powers the
# lemmatizer.
_REQUIRED_RESOURCES: dict[str, str] = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
}

_ensured = False


def ensure_nltk_resources(force: bool = False) -> None:
    """Download any missing NLTK resources exactly once per process.

    Args:
        force: Re-check every resource even if this function already ran.
    """
    global _ensured
    if _ensured and not force:
        return

    import nltk

    for find_path, package_id in _REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(find_path)
        except LookupError:
            logger.info("Downloading missing NLTK resource: %s", package_id)
            nltk.download(package_id, quiet=True)

    _ensured = True


__all__ = ["ensure_nltk_resources"]
