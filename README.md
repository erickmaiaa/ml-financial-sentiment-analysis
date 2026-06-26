# Financial Market Sentiment Prediction

Predict the sentiment (**positive / neutral / negative**) of financial texts
using two complementary approaches:

1. **Classical ML baselines** — TF-IDF features fed to nine scikit-learn /
   gradient-boosting classifiers.
2. **Fine-tuned transformer** — a `roberta-base` model fine-tuned with
   HuggingFace Accelerate, which substantially outperforms the baselines
   (~**87%** validation accuracy after 5 epochs).

This repository was refactored from a single exploratory notebook into a
modular, tested, production-oriented Python package. See
[`docs/refactoring.md`](docs/refactoring.md) for the full before/after report.

---

## Architecture

The code follows a clean, layered design — each stage of the ML lifecycle is an
isolated, independently testable module. The notebook and CLI scripts are thin
orchestrators on top of the `financial_sentiment` package.

```
ingestion → preprocessing → feature engineering → training → evaluation → visualisation
 (data/)     (preprocessing/)     (features/)        (models/)  (evaluation/) (evaluation/)
                                   ────────── orchestrated by pipelines/ ──────────
```

## Project structure

```
ml-financial-sentiment-analysis/
├── notebooks/
│   ├── financial_sentiment.ipynb     # Thin orchestrator notebook
│   └── data.csv                      # Raw dataset (Sentence, Sentiment)
├── src/financial_sentiment/
│   ├── config/        # Centralised, validated settings (pydantic + YAML + .env)
│   ├── data/          # CSV ingestion, dedup, label encoding
│   ├── preprocessing/ # Text cleaning & normalisation (TextCleaner)
│   ├── features/      # TF-IDF vectoriser
│   ├── models/        # Classical ML suite + RoBERTa trainer
│   ├── evaluation/    # Metrics tables & plots
│   ├── pipelines/     # End-to-end ML / transformer orchestration + CLIs
│   └── utils/         # Logging, seeding, NLTK resource management
├── tests/             # pytest suite
├── configs/           # default.yaml
├── scripts/           # run_ml.py / run_transformer.py
├── docs/              # Refactoring report
├── outputs/           # Generated models, tables and figures (git-ignored)
├── pyproject.toml
└── requirements.txt
```

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/erickmaiia/ml-financial-sentiment-analysis.git
cd ml-financial-sentiment-analysis

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install everything (ML baseline, preprocessing, EDA, RoBERTa stack, tests):
pip install -r requirements.txt

# Then install the package itself (editable):
pip install -e .
```

Alternatively, install via the extras declared in `pyproject.toml` to pull only
what you need:

```bash
pip install -e ".[boosting]"                   # classical ML only
pip install -e ".[boosting,transformers,dev]"  # everything
```

> **Python version note:** the transformer stack (torch / transformers) may not
> yet ship wheels for the newest CPython (e.g. 3.14). If `pip install -r
> requirements.txt` fails there, use Python 3.10–3.12.

Copy `.env.example` to `.env` to override any setting via environment variables.

## Usage

### Command line

```bash
# Classical ML baseline (writes outputs/ml_results.csv and a bar chart)
python scripts/run_ml.py --config configs/default.yaml

# Fine-tune RoBERTa (GPU recommended)
python scripts/run_transformer.py --config configs/default.yaml --epochs 5
```

Or via the installed console scripts: `fin-sentiment-ml`, `fin-sentiment-transformer`.

### Programmatic

```python
from financial_sentiment.config import load_settings
from financial_sentiment.pipelines.ml_pipeline import run

settings = load_settings("configs/default.yaml")
results = run(settings)        # list[ModelResult] ranked by accuracy
```

### Notebook

Open [`notebooks/financial_sentiment.ipynb`](notebooks/financial_sentiment.ipynb)
— it imports the package and orchestrates the full experiment end to end.

## Pipeline details

| Stage | Module | Notes |
|-------|--------|-------|
| Ingestion | `data.loader` | Reads CSV, drops duplicate sentences, label-encodes target |
| Cleaning | `preprocessing.TextCleaner.clean_dataframe` | Vectorised regex: strips URLs/mentions/hashtags/non-letters, lower-cases |
| Normalisation | `preprocessing.TextCleaner.normalize` | Contraction expansion, emoji/non-ASCII stripping, stop-word removal, lemmatisation → stemming (ML only) |
| Features | `features.build_tfidf_vectorizer` | Configurable `min_df`/`max_df`/`use_idf` |
| ML models | `models.ml` | RF, GB, AdaBoost, Decision Tree, SVM, Naive Bayes, MLP (+ XGBoost/LightGBM if installed) |
| Transformer | `models.transformer.TransformerTrainer` | RoBERTa fine-tuning with Accelerate; saves model to `outputs/` |
| Evaluation | `evaluation.plots` | Ranked accuracy table + bar chart |

## Configuration

All tunables live in [`src/financial_sentiment/config/settings.py`](src/financial_sentiment/config/settings.py).
Precedence (highest first): explicit args / YAML → environment (`FIN_` prefix) /
`.env` → defaults. Values are validated by pydantic, so an out-of-range setting
fails fast with a clear error.

## Testing

```bash
pip install -e ".[dev]"
pytest                # or: pytest --cov=financial_sentiment
```

Tests covering optional heavy dependencies (NLTK, emoji) skip gracefully when
those packages are not installed.

## Reproducibility

`set_global_seed` seeds `random`, NumPy, PyTorch and Transformers, and sets
`PYTHONHASHSEED`. Dependency versions are pinned in `requirements.txt` /
`pyproject.toml`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: financial_sentiment` | Run `pip install -e .` from the repo root |
| `LookupError` for an NLTK resource | It auto-downloads on first use; ensure internet access, or pre-run `python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet')"` |
| `XGBoost`/`LightGBM` missing from results | Install the boosting extra: `pip install -e ".[boosting]"` |
| Transformer pipeline import errors | Install the transformers extra: `pip install -e ".[transformers]"` |
| Out-of-memory while fine-tuning | Lower `transformer.train_batch_size` / `max_length` in the config |

## License

See [LICENSE](LICENSE).
