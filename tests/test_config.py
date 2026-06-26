"""Tests for the centralised configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_sentiment.config import Settings, load_settings


def test_defaults() -> None:
    settings = load_settings()
    assert settings.random_seed == 42
    assert settings.transformer.model_checkpoint == "roberta-base"
    assert settings.ml.test_size == 0.25


def test_overrides_win() -> None:
    settings = load_settings(random_seed=7, ml={"test_size": 0.3})
    assert settings.random_seed == 7
    assert settings.ml.test_size == 0.3


def test_yaml_loading(tmp_path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("random_seed: 123\nml:\n  test_size: 0.4\n", encoding="utf-8")
    settings = load_settings(cfg)
    assert settings.random_seed == 123
    assert settings.ml.test_size == 0.4


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("FIN_RANDOM_SEED", "99")
    assert Settings().random_seed == 99


@pytest.mark.parametrize("bad", [0, 1, 1.5])
def test_invalid_test_size_rejected(bad) -> None:
    with pytest.raises(ValidationError):
        load_settings(ml={"test_size": bad})
