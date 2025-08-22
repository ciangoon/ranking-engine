import csv
import importlib.util
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import xgboost as xgb
from gensim.downloader import load as gensim_load
from sklearn.model_selection import GridSearchCV

from text_utils import basic_tokenize, sigmoid, average_precision_at_k, mean_average_precision, dcg_at_k, ndcg_at_k
from neural_ranker import train_neural_ranker, evaluate_on_validation, rank_candidates_with_nn
 


@dataclass
class Paths:
    train_path: str = 'train_data.tsv'
    validation_path: str = 'validation_data.tsv'
    candidate_path: str = 'candidate_passages_top1000.tsv'
    test_queries_path: str = 'test-queries.tsv'
    qidquery_path: str = 'qidquery.tsv'
    pidpassage_path: str = 'pidpassagetest.tsv'


class RankingEngine:
    """Implements metrics, LR, LambdaMART, and NN-ready interfaces for ranking tasks."""

    def __init__(self, paths: Optional[Paths] = None, embedding_name: str = 'word2vec-google-news-300') -> None:
        self.paths = paths or Paths()
        # Lazy load embedding model
        self._embedding_model = None
        self._embedding_name = embedding_name

    # ---------------- Metrics ----------------
    def average_precision_at_k(self, relevancies: Sequence[int], k: int) -> float:
        """Proxy to shared utils AP@k."""
        return average_precision_at_k(relevancies, k)

    def mean_average_precision(self, ranked_relevancies_by_qid: Dict[str, Sequence[int]], k: int) -> float:
        """Proxy to shared utils mAP@k."""
        return mean_average_precision(ranked_relevancies_by_qid, k)

    def dcg_at_k(self, gains: Sequence[float], k: int) -> float:
        """Proxy to shared utils DCG@k."""
        return dcg_at_k(gains, k)

    def ndcg_at_k(self, gains: Sequence[float], ideal_gains: Sequence[float], k: int) -> float:
        """Proxy to shared utils NDCG@k."""
        return ndcg_at_k(gains, ideal_gains, k)

    # ---------------- Embeddings ----------------
    def _ensure_embeddings(self) -> None:
        if self._embedding_model is None:
            self._embedding_model = gensim_load(self._embedding_name)

    def sentence_embedding(self, text: str) -> Optional[np.ndarray]:
        """Average word embeddings for a text; returns None if no tokens are in vocab."""
        self._ensure_embeddings()
        toks = basic_tokenize(text)
        vectors: List[np.ndarray] = []
        for t in toks:
            try:
                vectors.append(self._embedding_model[t])
            except KeyError:
                continue
        if not vectors:
            return None
        return np.mean(np.vstack(vectors), axis=0)

    # ---------------- Data loading helpers ----------------
    def load_labeled_pairs(self, path: Optional[str] = None) -> List[Tuple[str, str, str, str, int]]:
        """Load labeled pairs (qid, pid, query, passage, label) from TSV with header."""
        rows: List[Tuple[str, str, str, str, int]] = []
        p = path or self.paths.train_path
        with open(p, encoding='utf8') as f:
            next(f)  # skip header
            reader = csv.reader(f, delimiter='\t')
            for r in reader:
                label = int(float(r[4]))
                rows.append((r[0], r[1], r[2], r[3], label))
        return rows

    def load_validation_pairs(self) -> List[Tuple[str, str, str, str, int]]:
        return self.load_labeled_pairs(self.paths.validation_path)

    def load_candidate_rows(self) -> List[List[str]]:
        with open(self.paths.candidate_path, encoding='utf8') as f:
            return list(csv.reader(f, delimiter='\t'))

    def load_test_queries(self) -> List[Tuple[str, str]]:
        with open(self.paths.test_queries_path, encoding='utf8') as f:
            return list(csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE))

    # ---------------- LR ----------------
    def train_logistic_regression(self, lr: float = 0.01, epochs: int = 200) -> Tuple[np.ndarray, float]:
        """Train a simple LR on averaged embeddings; returns (weights, bias)."""
        data = self.load_labeled_pairs(self.paths.train_path)
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        for _, _, q, p, label in data:
            qe = self.sentence_embedding(q)
            pe = self.sentence_embedding(p)
            if qe is None or pe is None:
                continue
            X_list.append(np.concatenate([qe, pe], axis=0))
            y_list.append(label)
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=float)
        w = np.random.randn(X.shape[1])
        b = float(np.random.randn())
        for _ in range(epochs):
            linear = np.dot(X, w) + b
            y_pred = sigmoid(linear)
            dw = np.dot(X.T, (y_pred - y)) / len(X)
            db = float(np.mean(y_pred - y))
            w -= lr * dw
            b -= lr * db
        return w, b

    def rank_with_logistic_regression(self, weights: np.ndarray, bias: float,
                                      output_path: str = 'LR.txt') -> None:
        """Rank candidate passages using trained LR; write to text file format."""
        candidates = self.load_candidate_rows()
        queries = self.load_test_queries()
        rows_by_qid: Dict[str, List[List[str]]] = defaultdict(list)
        for row in candidates:
            rows_by_qid[row[0]].append(row)

        results: List[List[Tuple[str, str, int, float, str]]] = []
        for qid, qtext in queries:
            scored: List[Tuple[str, str, float]] = []
            qe = self.sentence_embedding(qtext)
            if qe is None:
                continue
            for _, pid, _, passage in rows_by_qid.get(qid, []):
                pe = self.sentence_embedding(passage)
                if pe is None:
                    continue
                x = np.concatenate([qe, pe], axis=0)
                prob = float(sigmoid(np.dot(x, weights) + bias))
                scored.append((pid, prob,))
            scored.sort(key=lambda t: t[1], reverse=True)
            ranked = [(qid, "A2", pid, i + 1, score, "LR") for i, (pid, score) in enumerate(scored[:100])]
            if ranked:
                results.append(ranked)

        with open(output_path, 'w', encoding='utf8') as f:
            for group in results:
                for tpl in group:
                    line = ' '.join(str(v) for v in tpl)
                    f.write(line + '\n')

    # ---------------- LambdaMART ----------------
    def train_lambdamart(self, params: Optional[Dict] = None,
                          grid_search: bool = False) -> xgb.XGBRanker:
        """Train a LambdaMART ranker using averaged embeddings as features."""
        params = params or {
            'objective': 'rank:ndcg',
            'learning_rate': 0.01,
            'reg_lambda': 0.01,
            'max_depth': 8,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'min_child_weight': 3,
            'n_estimators': 200,
        }

        data = self.load_labeled_pairs(self.paths.train_path)
        # Group by qid for ranking
        by_qid: Dict[str, List[Tuple[str, str, str, str, int]]] = defaultdict(list)
        for row in data:
            by_qid[row[0]].append(row)

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        group_sizes: List[int] = []
        for qid, rows in by_qid.items():
            group_sizes.append(len(rows))
            for _, _, q, p, label in rows:
                qe = self.sentence_embedding(q)
                pe = self.sentence_embedding(p)
                if qe is None or pe is None:
                    # maintain group sizes consistency: skip entirely if any missing
                    continue
                X_list.append(np.concatenate([qe, pe], axis=0))
                y_list.append(label)

        X = np.vstack(X_list)
        y = np.array(y_list)
        model = xgb.XGBRanker(**params)

        if grid_search:
            search = GridSearchCV(model, params, scoring='neg_mean_squared_error', n_jobs=-1, cv=3, verbose=1)
            search.fit(X, y, group=[len(g) for g in by_qid.values()])
            model = search.best_estimator_

        model.fit(X, y, group=group_sizes)
        return model

    def rank_with_lambdamart(self, model: xgb.XGBRanker, output_path: str = 'LM.txt') -> None:
        """Rank candidate passages using trained LambdaMART; write to text file format."""
        candidates = self.load_candidate_rows()
        queries = self.load_test_queries()
        rows_by_qid: Dict[str, List[List[str]]] = defaultdict(list)
        for row in candidates:
            rows_by_qid[row[0]].append(row)

        results: List[List[Tuple[str, str, int, float, str]]] = []
        for qid, qtext in queries:
            qe = self.sentence_embedding(qtext)
            if qe is None:
                continue
            scored: List[Tuple[str, float]] = []
            for _, pid, _, passage in rows_by_qid.get(qid, []):
                pe = self.sentence_embedding(passage)
                if pe is None:
                    continue
                x = np.concatenate([qe, pe], axis=0).reshape(1, -1)
                score = float(model.predict(x)[0])
                scored.append((pid, score))
            scored.sort(key=lambda t: t[1], reverse=True)
            ranked = [(qid, "A2", pid, i + 1, score, "LM") for i, (pid, score) in enumerate(scored[:100])]
            if ranked:
                results.append(ranked)

        with open(output_path, 'w', encoding='utf8') as f:
            for group in results:
                for tpl in group:
                    line = ' '.join(str(v) for v in tpl)
                    f.write(line + '\n')

    # ---------------- Placeholder NN interface ----------------
    # The full NN training/inference can be plugged similarly. Provide a simple scaffold.
    def build_nn_input(self, path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create (X, y) arrays from labeled pairs using averaged embeddings."""
        labeled = self.load_labeled_pairs(path or self.paths.train_path)
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        for _, _, q, p, label in labeled:
            qe = self.sentence_embedding(q)
            pe = self.sentence_embedding(p)
            if qe is None or pe is None:
                continue
            X_list.append(np.concatenate([qe, pe], axis=0))
            y_list.append(label)
        return np.vstack(X_list), np.array(y_list)


if __name__ == "__main__":
    engine = RankingEngine()

    # Train and run Logistic Regression
    print("Training Logistic Regression...")
    weights, bias = engine.train_logistic_regression(lr=0.01, epochs=200)
    print("Ranking with Logistic Regression -> LR.txt")
    engine.rank_with_logistic_regression(weights, bias, output_path='LR.txt')

    # Train and run LambdaMART
    print("Training LambdaMART...")
    lm_model = engine.train_lambdamart()
    print("Ranking with LambdaMART -> LM.txt")
    engine.rank_with_lambdamart(lm_model, output_path='LM.txt')


    print("Training Neural Network ranker...")
    nn_model = train_neural_ranker(engine, engine.paths.train_path, engine.paths.validation_path,
                                   batch_size=256, lr=1e-3, epochs=5,
                                   hidden_dim=256, num_layers=2, dropout=0.2)
    print("Evaluating Neural Network ranker on validation...")
    evaluate_on_validation(engine, nn_model, k_values=[3, 10, 100])
    print("Ranking with Neural Network -> NN.txt")
    rank_candidates_with_nn(engine, nn_model, output_path='NN.txt')


