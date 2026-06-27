"""Model definitions: classical ML baselines and the RoBERTa transformer."""

from financial_sentiment.models.ml import (
    ModelResult,
    build_classifiers,
    save_models,
    train_and_evaluate,
)

__all__ = ["ModelResult", "build_classifiers", "save_models", "train_and_evaluate"]
