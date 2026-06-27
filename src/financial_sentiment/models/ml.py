"""Classical machine-learning baselines.

Builds the suite of scikit-learn (plus XGBoost/LightGBM) classifiers, trains
each on TF-IDF features and reports accuracy. Optional boosting libraries are
imported lazily so the module remains importable without them.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from financial_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelResult:
    """Accuracy of a single trained classifier."""

    name: str
    accuracy: float


def build_classifiers(random_seed: int = 42) -> dict[str, ClassifierMixin]:
    """Instantiate the classifier suite with a shared random seed.

    XGBoost and LightGBM are optional; they are included only when installed.

    Args:
        random_seed: Seed applied to every stochastic estimator.

    Returns:
        Mapping of human-readable model name to an unfitted estimator.
    """
    classifiers: dict[str, ClassifierMixin] = {
        "Random Forest": RandomForestClassifier(random_state=random_seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_seed),
        "AdaBoost": AdaBoostClassifier(random_state=random_seed),
        "Decision Tree": DecisionTreeClassifier(random_state=random_seed),
        "Support Vector Machine": SVC(random_state=random_seed),
        "Naive Bayes": MultinomialNB(),
        "Multilayer Perceptron": MLPClassifier(random_state=random_seed, max_iter=1000),
    }

    try:
        from lightgbm import LGBMClassifier

        classifiers["LightGBM"] = LGBMClassifier(random_state=random_seed, verbose=-1)
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("lightgbm not installed; skipping LightGBM.")

    try:
        from xgboost import XGBClassifier

        classifiers["XGBoost"] = XGBClassifier(
            eval_metric="mlogloss", random_state=random_seed
        )
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("xgboost not installed; skipping XGBoost.")

    return classifiers


def train_and_evaluate(
    classifiers: dict[str, ClassifierMixin],
    x_train,
    y_train,
    x_test,
    y_test,
) -> list[ModelResult]:
    """Fit each classifier and measure test accuracy.

    Args:
        classifiers: Mapping of name to unfitted estimator.
        x_train: Training feature matrix.
        y_train: Training labels.
        x_test: Test feature matrix.
        y_test: Test labels.

    Returns:
        Results sorted by descending accuracy.
    """
    results: list[ModelResult] = []
    for name, clf in classifiers.items():
        logger.info("Training %s ...", name)
        clf.fit(x_train, y_train)
        with warnings.catch_warnings():
            # LightGBM auto-names features ("Column_0", ...) when fitted on the
            # unnamed sparse TF-IDF matrix, so predicting on an equally unnamed
            # matrix triggers a benign feature-name mismatch warning. We train
            # and predict on the same format, so suppress that specific notice.
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            predictions = clf.predict(x_test)
        accuracy = float(accuracy_score(y_test, predictions))
        logger.info("%s accuracy: %.4f", name, accuracy)
        results.append(ModelResult(name=name, accuracy=accuracy))

    return sorted(results, key=lambda r: r.accuracy, reverse=True)


def _slugify(name: str) -> str:
    """Turn a human-readable model name into a safe filename stem."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_models(
    classifiers: dict[str, ClassifierMixin],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist each fitted classifier to disk with joblib.

    Each estimator is written to ``<output_dir>/<slug>.joblib`` so it can be
    reloaded later with :func:`joblib.load` for inference without retraining.

    Args:
        classifiers: Mapping of name to a *fitted* estimator.
        output_dir: Directory to write the ``.joblib`` files into; created if
            it does not yet exist.

    Returns:
        Mapping of model name to the path it was written to.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, clf in classifiers.items():
        path = directory / f"{_slugify(name)}.joblib"
        joblib.dump(clf, path)
        logger.info("Saved %s -> %s", name, path)
        paths[name] = path

    return paths


__all__ = ["ModelResult", "build_classifiers", "train_and_evaluate", "save_models"]
