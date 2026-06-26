"""RoBERTa fine-tuning pipeline.

Refactors the notebook's transformer cells into small, testable functions and a
single :class:`TransformerTrainer`. Heavy deep-learning dependencies (torch,
transformers, datasets, accelerate, evaluate) are imported lazily inside the
methods so that importing this module is cheap and side-effect free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import train_test_split

from financial_sentiment.config import TransformerConfig
from financial_sentiment.data.loader import LABEL_COLUMN
from financial_sentiment.utils.logging import get_logger
from financial_sentiment.utils.seed import set_global_seed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datasets import DatasetDict

logger = get_logger(__name__)


def build_dataset_splits(
    frame: pd.DataFrame, config: TransformerConfig, random_seed: int
) -> "DatasetDict":
    """Split a dataframe into stratified train/validation/test HF datasets.

    Args:
        frame: Cleaned dataframe containing the text and ``labels`` columns.
        config: Transformer configuration with the split fractions.
        random_seed: Seed for reproducible splitting.

    Returns:
        A :class:`datasets.DatasetDict` with ``train``/``validation``/``test``.
    """
    from datasets import Dataset, DatasetDict

    train_val, test_df = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=random_seed,
        stratify=frame[LABEL_COLUMN].values,
    )
    train_df, val_df = train_test_split(
        train_val,
        test_size=config.validation_size,
        random_state=random_seed,
        stratify=train_val[LABEL_COLUMN].values,
    )
    return DatasetDict(
        train=Dataset.from_pandas(train_df, preserve_index=False),
        validation=Dataset.from_pandas(val_df, preserve_index=False),
        test=Dataset.from_pandas(test_df, preserve_index=False),
    )


class TransformerTrainer:
    """Encapsulates tokenisation, data loading and the fine-tuning loop."""

    def __init__(self, config: TransformerConfig, text_column: str = "Sentence") -> None:
        self.config = config
        self.text_column = text_column
        self._tokenizer = None

    @property
    def tokenizer(self):  # noqa: ANN201 - third-party type
        """Lazily-loaded fast tokenizer for the configured checkpoint."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_checkpoint, use_fast=True
            )
        return self._tokenizer

    def tokenize(self, dataset: "DatasetDict") -> "DatasetDict":
        """Tokenise every split, dropping the raw text columns.

        Args:
            dataset: The split dataset dictionary.

        Returns:
            Tokenised datasets formatted as PyTorch tensors.
        """
        text_column = self.text_column
        max_length = self.config.max_length
        tokenizer = self.tokenizer

        def _tokenize(batch: dict) -> dict:
            return tokenizer(
                batch[text_column],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        drop = [c for c in dataset["train"].column_names if c != LABEL_COLUMN]
        tokenized = dataset.map(_tokenize, batched=True, remove_columns=drop)
        tokenized.set_format("torch")
        return tokenized

    def train(self, frame: pd.DataFrame, random_seed: int = 42) -> dict[str, float]:
        """Fine-tune the model and return per-epoch validation accuracy.

        Uses HuggingFace Accelerate so the same code runs on CPU/GPU/TPU. The
        trained model and tokenizer are saved to ``config.model_path``.

        Args:
            frame: Cleaned dataframe with text and ``labels`` columns.
            random_seed: Seed for full reproducibility.

        Returns:
            Mapping of ``"epoch_<n>"`` to its validation accuracy.
        """
        import torch
        from accelerate import Accelerator
        from evaluate import load as load_metric
        from torch.utils.data import DataLoader
        from tqdm.auto import tqdm
        from transformers import (
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )

        set_global_seed(random_seed)
        accelerator = Accelerator()
        metric = load_metric("accuracy")

        tokenized = self.tokenize(build_dataset_splits(frame, self.config, random_seed))
        train_loader = DataLoader(
            tokenized["train"], shuffle=True, batch_size=self.config.train_batch_size
        )
        eval_loader = DataLoader(
            tokenized["validation"], shuffle=False, batch_size=self.config.eval_batch_size
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_checkpoint, num_labels=self.config.num_labels
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, eps=self.config.adam_epsilon
        )
        model, optimizer, train_loader, eval_loader = accelerator.prepare(
            model, optimizer, train_loader, eval_loader
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=len(train_loader) * self.config.epochs,
        )

        progress = tqdm(
            range(self.config.epochs * len(train_loader)),
            disable=not accelerator.is_main_process,
        )

        history: dict[str, float] = {}
        for epoch in range(self.config.epochs):
            model.train()
            for batch in train_loader:
                loss = model(**batch).loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update(1)

            accuracy = self._evaluate(model, eval_loader, accelerator, metric)
            history[f"epoch_{epoch}"] = accuracy
            accelerator.print(f"epoch {epoch}: accuracy={accuracy:.4f}")

        self._save(model, accelerator)
        return history

    @staticmethod
    def _evaluate(model, loader, accelerator, metric) -> float:
        """Run a validation pass and return accuracy."""
        import torch

        model.eval()
        predictions, references = [], []
        for batch in loader:
            with torch.no_grad():
                logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            predictions.append(accelerator.gather(preds))
            references.append(accelerator.gather(batch[LABEL_COLUMN]))

        preds = torch.cat(predictions)
        refs = torch.cat(references)
        return float(metric.compute(predictions=preds, references=refs)["accuracy"])

    def _save(self, model, accelerator) -> None:
        """Persist the unwrapped model and tokenizer to disk."""
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        path = self.config.model_path
        path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Saved fine-tuned model to %s", path)


__all__ = ["TransformerTrainer", "build_dataset_splits"]
