## Overview

This repository provides a compact, extensible passage retrieval and learning-to-rank toolkit built in two parts:

- Retrieval and language-modeling over a candidate passage set (classic IR models)
- Learning-based re-ranking models (logistic regression, LambdaMART, and a neural ranker)

It is organized around three main classes and a shared utilities module:

- `retrieval_system.py` → `PassageRetrievalEngine`
- `ranking_engine.py` → `RankingEngine`
- `neural_ranker.py` → `NeuralRanker` (+ helpers)
- `text_utils.py` → shared text/IR utilities


## Components

### PassageRetrievalEngine
Path: `retrieval_system.py`

Implements classic IR pipelines for candidate passage re-ranking:

- Text statistics and Zipf analysis: `analyze_zipf()`
- Index building: `build_indexes()` produces `pidpassage.tsv`, `invindex.csv`, and `index.csv`
- TF-IDF vector space ranking: `rank_tfidf()` writes `tfidf.csv`
- BM25 ranking: `rank_bm25()` writes `bm25.csv`
- Query-likelihood language models (scores are log-probabilities):
  - Laplace: `rank_laplace()` → `laplace.csv`
  - Lidstone: `rank_lidstone()` → `lidstone.csv`
  - Dirichlet: `rank_dirichlet()` → `dirichlet.csv`

Inputs expected in the same directory:

- `passage-collection.txt`
- `candidate-passages-top1000.tsv`
- `test-queries.tsv`

This engine uses shared utilities from `text_utils.py` for tokenization, TF-IDF, BM25 scoring, and metrics.


### RankingEngine
Path: `ranking_engine.py`

Implements learning-to-rank baselines and metrics for a training/validation setting:

- Metrics (proxies calling `text_utils`):
  - `average_precision_at_k`, `mean_average_precision`
  - `dcg_at_k`, `ndcg_at_k`
- Logistic Regression baseline on averaged word embeddings:
  - Train: `train_logistic_regression()`
  - Rank test candidates: `rank_with_logistic_regression()` → `LR.txt`
- LambdaMART (XGBoost ranker) on the same feature representation:
  - Train: `train_lambdamart()` (optional grid search)
  - Rank test candidates: `rank_with_lambdamart()` → `LM.txt`
- Utility to construct NN-ready inputs: `build_nn_input()`

Inputs expected in the `same directory`:

- `train_data.tsv`, `validation_data.tsv`
- `candidate_passages_top1000.tsv`
- `test-queries.tsv`

Embeddings are computed with `gensim.downloader` (e.g., `word2vec-google-news-300`) and averaged across tokens.


### NeuralRanker (PyTorch)
Path: `neural_ranker.py`

Defines a neural re-ranker and helpers:

- `NeuralRanker`: Multi-layer perceptron (MLP) over concatenated query and passage embeddings (averaged word vectors)
- Training: `train_neural_ranker(engine, train_path, val_path, ...)`
- Validation metrics: `evaluate_on_validation(engine, model, k_values)` → prints mAP@k and NDCG@k
- Candidate re-ranking: `rank_candidates_with_nn(engine, model, output_path='NN.txt')`

Architecture choice rationale:

- Averaged word embeddings yield fixed-size continuous inputs. An MLP efficiently models non-linear query–passage interactions with low overhead and minimal engineering. The scaffold can be swapped for CNN/RNN/Transformer encoders to capture richer sequence structure if desired.


## Shared Utilities
Path: `text_utils.py`

Provides:

- Text processing: `tokenize` (lowercase, punctuation strip, stemming, optional stopword removal), `basic_tokenize` (no stem/stopword removal)
- Set ops: `unique`
- Math: `cosine_similarity`, `sigmoid`
- IR helpers: `tf`, `idf`, `tfidf_vector`, `bm25_score`
- Metrics: `average_precision_at_k`, `mean_average_precision`, `dcg_at_k`, `ndcg_at_k`


## Dependencies

Python ≥ 3.9 is recommended.

Core packages:

- `numpy`
- `matplotlib` (Zipf plots)
- `nltk` (stopwords, stemming)
- `gensim` (pre-trained embeddings via `gensim.downloader`)
- `xgboost` (LambdaMART ranker)
- `scikit-learn` (GridSearchCV)
- `torch` (PyTorch; neural ranker)

Install all at once:

```bash
pip install numpy matplotlib nltk gensim xgboost scikit-learn torch
```

Notes:

- Gensim will download `word2vec-google-news-300` on first use (~1.5GB). Ensure you have disk space and a stable connection.
- NLTK stopwords are fetched automatically at import in `text_utils.py`. If needed, run an interactive `nltk.download('stopwords')` once.
- On some platforms, `xgboost` and `torch` may need platform-specific wheels. See their official install guides if defaults fail.


## Usage

### Classic IR (TF-IDF, BM25, Language Models)

Run from root:

```bash
python retrieval_system.py
```

Outputs: `pidpassage.tsv`, `invindex.csv`, `index.csv`, and the ranking CSVs (`tfidf.csv`, `bm25.csv`, `laplace.csv`, `lidstone.csv`, `dirichlet.csv`).


### Learning-to-Rank (LR, LambdaMART, Neural)

Run from root:

```bash
python ranking_engine.py
```

This will:

- Train LR and produce `LR.txt`
- Train LambdaMART and produce `LM.txt`
- Train the PyTorch `NeuralRanker`, print validation mAP@k and NDCG@k, and produce `NN.txt`

Required inputs: `train_data.tsv`, `validation_data.tsv`, `candidate_passages_top1000.tsv`, `test-queries.tsv` placed in root directory.


## Data Format Expectations

- Candidate rows: `<qid> <pid> <query> <passage>` (tab-separated)
- Training/validation rows: `<qid> <pid> <query> <passage> <label>` with a header row
- Test queries: `<qid> <query>` (tab-separated)


## Extensibility Tips

- Swap `basic_tokenize` for `tokenize` if you want stemming/stopword removal in learning models.
- Replace averaged embeddings with contextual encoders for improved performance.
- The neural ranker can be replaced by CNN/RNN/Transformer architectures without changing the data plumbing.


