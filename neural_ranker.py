import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from text_utils import ndcg_at_k, mean_average_precision


class NeuralRanker(nn.Module):
    """Feed-forward neural ranker over averaged word embeddings.

    Justification:
    - We use a multi-layer perceptron (MLP) because the input is a fixed-size
      continuous representation (averaged word embeddings for query and passage).
      MLPs are well-suited to model non-linear interactions between query and
      passage embeddings with low computational overhead.
    - For larger gains, this scaffold can be extended to CNN/RNN/Transformer
      encoders to capture sequence structure beyond averaging.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_dataset(engine: Any, path: Optional[str]) -> TensorDataset:
    labeled = engine.load_labeled_pairs(path)
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for _, _, q, p, label in labeled:
        qe = engine.sentence_embedding(q)
        pe = engine.sentence_embedding(p)
        if qe is None or pe is None:
            continue
        X_list.append(np.concatenate([qe, pe], axis=0))
        y_list.append(label)
    X = torch.from_numpy(np.vstack(X_list)).float()
    y = torch.from_numpy(np.array(y_list)).float()
    return TensorDataset(X, y)


def train_neural_ranker(engine: Any,
                        train_path: Optional[str],
                        val_path: Optional[str],
                        batch_size: int = 256,
                        lr: float = 1e-3,
                        epochs: int = 5,
                        hidden_dim: int = 256,
                        num_layers: int = 2,
                        dropout: float = 0.2) -> NeuralRanker:
    train_ds = build_dataset(engine, train_path)
    val_ds = build_dataset(engine, val_path)
    input_dim = train_ds.tensors[0].shape[1]
    model = NeuralRanker(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        avg_train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item())
        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{epochs} - train loss: {avg_train_loss:.4f} - val loss: {avg_val_loss:.4f}")

    return model


def evaluate_on_validation(engine: Any, model: NeuralRanker,
                           k_values: List[int] = [3, 10, 100]) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    val_rows = engine.load_validation_pairs()
    by_qid: Dict[str, List[Tuple[float, int]]] = {}
    for qid, pid, q, p, label in val_rows:
        qe = engine.sentence_embedding(q)
        pe = engine.sentence_embedding(p)
        if qe is None or pe is None:
            continue
        with torch.no_grad():
            x = torch.from_numpy(np.concatenate([qe, pe], axis=0)).float().to(device)
            logit = float(model(x.unsqueeze(0)).squeeze(0).cpu().item())
        score = float(1.0 / (1.0 + math.exp(-logit)))
        by_qid.setdefault(qid, []).append((score, int(label)))

    for k in k_values:
        ranked_rels: Dict[str, List[int]] = {}
        ndcg_vals: List[float] = []
        for qid, pairs in by_qid.items():
            pairs_sorted = sorted(pairs, key=lambda t: t[0], reverse=True)[:k]
            rels = [lbl for _, lbl in pairs_sorted]
            ranked_rels[qid] = rels
            ideal = sorted([lbl for _, lbl in pairs], reverse=True)[:k]
            ndcg_vals.append(ndcg_at_k(rels, ideal, k))
        map_k = mean_average_precision(ranked_rels, k)
        ndcg_k = float(np.mean(ndcg_vals)) if ndcg_vals else 0.0
        print(f"Validation mAP@{k}: {map_k:.4f} | NDCG@{k}: {ndcg_k:.4f}")


def rank_candidates_with_nn(engine: Any, model: NeuralRanker, output_path: str = 'NN.txt') -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    candidates = engine.load_candidate_rows()
    queries = engine.load_test_queries()
    rows_by_qid: Dict[str, List[List[str]]] = {}
    for row in candidates:
        rows_by_qid.setdefault(row[0], []).append(row)

    results: List[List[Tuple[str, str, int, float, str]]] = []
    with torch.no_grad():
        for qid, qtext in queries:
            qe = engine.sentence_embedding(qtext)
            if qe is None:
                continue
            scored: List[Tuple[str, float]] = []
            for _, pid, _, passage in rows_by_qid.get(qid, []):
                pe = engine.sentence_embedding(passage)
                if pe is None:
                    continue
                x = torch.from_numpy(np.concatenate([qe, pe], axis=0)).float().to(device)
                logit = float(model(x.unsqueeze(0)).squeeze(0).cpu().item())
                score = float(1.0 / (1.0 + math.exp(-logit)))
                scored.append((pid, score))
            scored.sort(key=lambda t: t[1], reverse=True)
            ranked = [(qid, "A2", pid, i + 1, score, "NN") for i, (pid, score) in enumerate(scored[:100])]
            if ranked:
                results.append(ranked)

    with open(output_path, 'w', encoding='utf8') as f:
        for group in results:
            for tpl in group:
                line = ' '.join(str(v) for v in tpl)
                f.write(line + '\n')

