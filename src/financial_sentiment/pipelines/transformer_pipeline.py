"""End-to-end RoBERTa fine-tuning pipeline.

Loads and lightly cleans the data (transformers prefer minimally processed
text), then delegates training to :class:`TransformerTrainer`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from financial_sentiment.config import Settings, load_settings
from financial_sentiment.data import load_dataset
from financial_sentiment.preprocessing import TextCleaner
from financial_sentiment.utils import get_logger, set_global_seed

logger = get_logger(__name__)


def run(settings: Settings) -> dict[str, float]:
    """Execute the transformer fine-tuning pipeline.

    Args:
        settings: Fully-resolved application settings.

    Returns:
        Per-epoch validation accuracy history.
    """
    # Imported lazily so the heavy DL stack is only required for this pipeline.
    from financial_sentiment.models.transformer import TransformerTrainer

    set_global_seed(settings.random_seed)
    settings.ensure_output_dir()

    dataset = load_dataset(settings.data)
    text_col = settings.data.text_column

    # Only light, vectorised cleaning — no stemming/lemmatisation for the LLM.
    frame = TextCleaner.clean_dataframe(dataset.frame, text_col)

    trainer = TransformerTrainer(settings.transformer, text_column=text_col)
    history = trainer.train(frame, random_seed=settings.random_seed)
    logger.info("Training complete. History: %s", history)
    return history


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for sentiment.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the transformer pipeline."""
    args = _parse_args(argv)
    overrides: dict = {}
    if args.epochs is not None:
        overrides["transformer"] = {"epochs": args.epochs}
    settings = load_settings(args.config, **overrides)
    run(settings)


if __name__ == "__main__":
    main()
