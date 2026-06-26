#!/usr/bin/env python
"""Thin CLI wrapper around the classical ML pipeline.

Usage:
    python scripts/run_ml.py --config configs/default.yaml
"""

from financial_sentiment.pipelines.ml_pipeline import main

if __name__ == "__main__":
    main()
