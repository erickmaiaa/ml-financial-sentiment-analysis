"""Centralised, validated configuration.

All tunable values live here instead of being scattered as "magic numbers"
across the codebase. Configuration is resolved with the following precedence
(highest first):

1. Explicit keyword arguments / a YAML file passed to :func:`load_settings`.
2. Environment variables (prefixed with ``FIN_``) and a local ``.env`` file.
3. The defaults declared on the models below.

The models are :class:`pydantic.BaseModel` based, which gives us free input
validation and clear error messages when a value is out of range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (…/src/financial_sentiment/config/settings.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DataConfig(BaseModel):
    """Locations and column names for the input dataset."""

    data_path: Path = Field(default=PROJECT_ROOT / "notebooks" / "data.csv")
    text_column: str = "Sentence"
    label_column: str = "Sentiment"
    drop_duplicates: bool = True


class MLConfig(BaseModel):
    """Settings for the classical TF-IDF + scikit-learn baseline."""

    test_size: float = Field(default=0.25, gt=0, lt=1)
    tfidf_min_df: float = Field(default=2e-5, ge=0)
    tfidf_max_df: float = Field(default=0.70, gt=0, le=1.0)
    use_idf: bool = True


class TransformerConfig(BaseModel):
    """Hyper-parameters for the RoBERTa fine-tuning pipeline."""

    model_checkpoint: str = "roberta-base"
    model_path: Path = Field(default=PROJECT_ROOT / "outputs" / "roberta-finetuned")
    num_labels: PositiveInt = 3
    max_length: PositiveInt = 128

    test_size: float = Field(default=0.1, gt=0, lt=1)
    validation_size: float = Field(default=0.2, gt=0, lt=1)
    train_batch_size: PositiveInt = 32
    eval_batch_size: PositiveInt = 32

    epochs: PositiveInt = 5
    lr: PositiveFloat = 3e-5
    adam_epsilon: PositiveFloat = 1e-8
    num_warmup_steps: int = Field(default=10, ge=0)


class Settings(BaseSettings):
    """Top-level application settings.

    Environment variables use the ``FIN_`` prefix with ``__`` as the nested
    delimiter (e.g. ``FIN_TRANSFORMER__EPOCHS=3``). Flat aliases for the most
    common values are also exposed for convenience in ``.env`` files.
    """

    model_config = SettingsConfigDict(
        env_prefix="FIN_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    random_seed: int = 42
    output_dir: Path = Field(default=PROJECT_ROOT / "outputs")

    data: DataConfig = Field(default_factory=DataConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)

    def ensure_output_dir(self) -> Path:
        """Create the output directory if it does not yet exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings(
    config_file: str | Path | None = None,
    **overrides: Any,
) -> Settings:
    """Build a :class:`Settings` instance.

    Args:
        config_file: Optional path to a YAML file whose contents seed the
            configuration. Environment variables still take precedence for any
            key they define.
        **overrides: Explicit values that win over both the YAML file and the
            environment (useful for tests and notebooks).

    Returns:
        A fully validated :class:`Settings` object.
    """
    data: dict[str, Any] = {}
    if config_file is not None:
        data.update(_read_yaml(config_file))
    data.update(overrides)
    return Settings(**data)


__all__ = [
    "PROJECT_ROOT",
    "DataConfig",
    "MLConfig",
    "TransformerConfig",
    "Settings",
    "load_settings",
]
