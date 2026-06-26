"""Tests for the classical ML model utilities."""

from __future__ import annotations

import numpy as np

from financial_sentiment.evaluation import results_to_frame
from financial_sentiment.models import ModelResult, build_classifiers, train_and_evaluate


def test_build_classifiers_seeded() -> None:
    classifiers = build_classifiers(random_seed=42)
    assert "Random Forest" in classifiers
    assert "Naive Bayes" in classifiers
    # Core sklearn models are always present (boosting libs are optional).
    assert len(classifiers) >= 7


def test_train_and_evaluate_ranks_results() -> None:
    rng = np.random.default_rng(0)
    # Linearly separable toy data so models actually learn.
    x_train = np.vstack([rng.normal(0, 1, (20, 4)), rng.normal(5, 1, (20, 4))])
    y_train = np.array([0] * 20 + [1] * 20)
    x_test = np.vstack([rng.normal(0, 1, (5, 4)), rng.normal(5, 1, (5, 4))])
    y_test = np.array([0] * 5 + [1] * 5)

    subset = {"Naive Bayes": build_classifiers()["Naive Bayes"]}
    # MultinomialNB needs non-negative features.
    results = train_and_evaluate(
        subset, np.abs(x_train), y_train, np.abs(x_test), y_test
    )
    assert len(results) == 1
    assert isinstance(results[0], ModelResult)
    assert 0.0 <= results[0].accuracy <= 1.0


def test_results_to_frame_sorted() -> None:
    results = [
        ModelResult("A", 0.5),
        ModelResult("B", 0.9),
        ModelResult("C", 0.7),
    ]
    frame = results_to_frame(results)
    assert list(frame["Model"]) == ["B", "C", "A"]
