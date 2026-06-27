"""End-to-end classical ML pipeline.

Orchestrates: load data -> clean -> normalise -> TF-IDF -> train suite ->
evaluate -> persist results. Exposes both a programmatic ``run`` function and a
``main`` CLI entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from financial_sentiment.config import Settings, load_settings
from financial_sentiment.data import load_dataset
from financial_sentiment.data.loader import LABEL_COLUMN
from financial_sentiment.evaluation import plot_model_accuracies, results_to_frame
from financial_sentiment.features import build_tfidf_vectorizer
from financial_sentiment.models import (
    ModelResult,
    build_classifiers,
    save_models,
    train_and_evaluate,
)
from financial_sentiment.preprocessing import TextCleaner
from financial_sentiment.utils import get_logger, set_global_seed

logger = get_logger(__name__)


def run(settings: Settings) -> list[ModelResult]:
    """Execute the classical ML pipeline and return ranked results.

    Args:
        settings: Fully-resolved application settings.

    Returns:
        Model results sorted by descending accuracy.
    """
    from sklearn.model_selection import train_test_split

    set_global_seed(settings.random_seed)
    settings.ensure_output_dir()

    dataset = load_dataset(settings.data)
    text_col = settings.data.text_column

    cleaner = TextCleaner()
    frame = cleaner.clean_dataframe(dataset.frame, text_col)
    frame[text_col] = cleaner.normalize_series(frame[text_col])

    x_train, x_test, y_train, y_test = train_test_split(
        frame[text_col].to_numpy(),
        frame[LABEL_COLUMN].to_numpy(),
        test_size=settings.ml.test_size,
        random_state=settings.random_seed,
        stratify=frame[LABEL_COLUMN].to_numpy(),
    )

    vectorizer = build_tfidf_vectorizer(settings.ml)
    x_train_tf = vectorizer.fit_transform(x_train.astype("U"))
    x_test_tf = vectorizer.transform(x_test.astype("U"))
    logger.info(
        "TF-IDF features — train: %s, test: %s", x_train_tf.shape, x_test_tf.shape
    )

    classifiers = build_classifiers(settings.random_seed)
    results = train_and_evaluate(classifiers, x_train_tf, y_train, x_test_tf, y_test)

    _persist(results, settings.output_dir)
    _persist_models(classifiers, vectorizer, settings.output_dir)
    return results


def _persist(results: list[ModelResult], output_dir: Path) -> None:
    """Write the results table (CSV) and accuracy bar chart (PNG)."""
    frame = results_to_frame(results)
    csv_path = output_dir / "ml_results.csv"
    frame.to_csv(csv_path, index=False)
    logger.info("Wrote ML results to %s", csv_path)
    plot_model_accuracies(frame, save_path=output_dir / "ml_accuracies.png")


def _persist_models(classifiers, vectorizer, output_dir: Path) -> None:
    """Serialise the fitted TF-IDF vectoriser and classifiers with joblib.

    Artefacts land in ``<output_dir>/models/`` so they can be reloaded for
    inference (``joblib.load``) without retraining.
    """
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = models_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    logger.info("Saved TF-IDF vectoriser -> %s", vectorizer_path)

    save_models(classifiers, models_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the classical ML baseline suite.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    parser.add_argument("--data-path", type=Path, default=None, help="Override dataset path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the ML pipeline."""
    args = _parse_args(argv)
    overrides: dict = {}
    if args.data_path is not None:
        overrides["data"] = {"data_path": str(args.data_path)}
    settings = load_settings(args.config, **overrides)

    results = run(settings)
    print("\n=== Model ranking ===")
    print(pd.DataFrame([{"Model": r.name, "Accuracy": r.accuracy} for r in results]))


if __name__ == "__main__":
    main()
