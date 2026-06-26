# Refactoring Report

This document records the transformation of the project from a single
exploratory notebook (`notebook.ipynb`, 27 cells, all logic + global state) into
a modular, tested `financial_sentiment` package.

For every change: **problem → why it matters → fix → benefit**.

---

## 1. Bugs fixed (behaviour-correcting changes)

### 1.1 Identity comparison on a string
* **Problem:** `if cfg.model_type is not "transformer"` used identity (`is`)
  instead of equality (`!=`).
* **Why:** Works only by accident of CPython string interning; semantically
  wrong and flagged by every linter.
* **Fix:** Concern removed entirely — the ML and transformer pipelines are now
  separate code paths, so the runtime branch no longer exists.
* **Benefit:** No fragile reliance on interning; intent is explicit.

### 1.2 Regex replacements that silently did nothing
* **Problem:** `clean_text` called `Series.str.replace(r"#[A-Za-z0-9_]+", ' ')`
  and similar **without** `regex=True`. On modern pandas the default is literal
  replacement, so hashtags, URLs and special characters were **not** removed.
* **Why:** The headline preprocessing step was effectively a no-op — degrading
  every downstream model silently.
* **Fix:** `TextCleaner.clean_dataframe` uses pre-compiled patterns with
  `regex=True` and a correct URL/mention/hashtag/non-letter regime.
* **Benefit:** Cleaning actually happens; behaviour is now covered by tests.

### 1.3 Deprecated emoji API
* **Problem:** `emoji.get_emoji_regexp()` was removed in `emoji>=2.0`, raising
  `AttributeError` on any current install.
* **Fix:** Use the supported `emoji.replace_emoji(text, replace="")`.
* **Benefit:** Runs on current dependency versions.

### 1.4 Dead `rename`
* **Problem:** `df.rename(columns={"label": "label_desc"})` referenced a
  non-existent `label` column — a no-op.
* **Fix:** Removed; label handling is explicit in `data.loader` (`labels` column
  + a returned `LabelEncoder` for decoding).
* **Benefit:** Less dead code; label decoding is now possible.

### 1.5 Stemming before lemmatisation
* **Problem:** `preprocess_text` stemmed **then** lemmatised. Stemming mutilates
  words (`companies → compani`) so the lemmatiser can no longer match them.
* **Fix:** `TextCleaner.normalize` lemmatises first, then optionally stems; both
  steps are configurable.
* **Benefit:** More meaningful normalised tokens; order documented.

### 1.6 Hardcoded Colab path
* **Problem:** `data_path = "/content/data.csv"` only works on Google Colab.
* **Fix:** Default resolves to `PROJECT_ROOT / "data.csv"`, overridable via
  config/env/CLI.
* **Benefit:** Portable across machines and CI.

### 1.7 NLTK punkt only
* **Problem:** Only `punkt` was downloaded; NLTK ≥ 3.8.2 needs `punkt_tab`, and
  the lemmatiser needs `wordnet`/`omw-1.4`.
* **Fix:** `utils.nltk_resources.ensure_nltk_resources` downloads exactly the
  required resources, once, only if missing.
* **Benefit:** Works offline after first run; faster start-up.

### 1.8 Inconsistent RoBERTa accuracy
* **Problem:** README claimed 86.32% while the conclusion cell hardcoded 0.8262.
* **Fix:** Reported number aligned with the best observed validation accuracy
  (~0.877) and centralised as a single constant in the notebook.

---

## 2. Architecture & design

| Problem | Fix | Benefit |
|---------|-----|---------|
| All logic + state in one notebook | Layered `src/` package (data / preprocessing / features / models / evaluation / pipelines / utils / config) | Separation of Concerns, testability, reuse |
| Global mutable state (`STOPWORDS.update`, module-level `lemmatizer`, globals) | Encapsulated in `TextCleaner` with lazy `cached_property` resources | No hidden state; deterministic; thread-safer |
| Mixed processing & visualisation | Plotting isolated in `evaluation.plots` | SoC; headless/CI-friendly |
| Training & evaluation intertwined | `TransformerTrainer.train` / `_evaluate` split | Single Responsibility |
| `FrozenConfigDict` magic values | Validated pydantic `Settings` + YAML + `.env` | Centralised config, fail-fast validation (DRY/KISS) |

## 3. Code quality

* **Type hints & docstrings** on every public function/class.
* **Structured logging** (`utils.logging`) replaces `print`/`tqdm`-only output.
* **Exception handling & input validation**: explicit `FileNotFoundError` /
  `KeyError` in the loader; pydantic range validation in config.
* **Removed**: `warnings.filterwarnings("ignore")` (was hiding real warnings),
  duplicate `pip install datasets`, unused imports (`joblib`, `CountVectorizer`,
  `WordCloud`, `DistributedType`, `string`).
* **Reproducibility**: unified `set_global_seed` across `random`/NumPy/Torch/HF.

## 4. Performance

* **Vectorised cleaning**: regex cleaning runs over whole pandas Series with
  pre-compiled patterns instead of per-row Python work where avoidable.
* **Lazy heavy imports**: torch/transformers/datasets/emoji/nltk import only when
  their pipeline runs, cutting notebook/CLI start-up time and memory.
* **Cached resources**: tokenizer, lemmatizer, stemmer and stop-word set built
  once via `cached_property`.
* **NLTK downloads** happen once per process and only when missing.

## 5. Testing

`tests/` (pytest) covers config precedence/validation, data loading
(dedup/encoding/error paths), text cleaning, the classifier suite and the
results table. Optional-dependency tests skip gracefully.

---

## Future improvements (recommended)

* Report precision/recall/F1 and confusion matrices, not just accuracy
  (dataset is class-imbalanced — neutral dominates).
* Add cross-validation and hyper-parameter search for the ML baselines.
* Add an inference/prediction module + REST endpoint for serving RoBERTa.
* Track experiments (MLflow / Weights & Biases) and version data (DVC).
* Add CI (GitHub Actions): ruff + mypy + pytest on every push.
* Persist the `LabelEncoder` and TF-IDF vectoriser for reproducible inference.
* Containerise (Dockerfile) for reproducible training/serving.

## Risks & attention points

* **Behavioural shift from bug fixes 1.2/1.5**: because cleaning now actually
  runs and the stem/lemma order changed, classical-ML accuracy numbers will
  differ from the original notebook. This is expected and *more correct*.
* **Python 3.14 + heavy ML stack**: torch/transformers wheels may lag on the
  newest Python; pin to a supported interpreter (3.10–3.12) for the transformer
  extra if installation fails.
* **Dataset size**: ~5.8k rows is small for fine-tuning a 125M-parameter model;
  results may vary run-to-run despite seeding.
* **No data/secret leakage** found — dataset is public financial text; no
  credentials or absolute user paths remain in the codebase.
