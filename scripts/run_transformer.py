#!/usr/bin/env python
"""Thin CLI wrapper around the RoBERTa fine-tuning pipeline.

Usage:
    python scripts/run_transformer.py --config configs/default.yaml --epochs 5
"""

from financial_sentiment.pipelines.transformer_pipeline import main

if __name__ == "__main__":
    main()
