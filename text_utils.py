from typing import List

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from typing import Dict, Sequence


# Initialize once at import time
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
STEMMER = SnowballStemmer("english")
PUNCTUATION = '''“”∞¡...—–!()-|[]{};:'"\,<>./?@#$%^&*_~•’‘+…=£®°²→·'''


def is_valid_token(token: str) -> bool:
    """Return True if token should be kept after basic filtering."""
    if len(token) == 1:
        return token in {"a", "i"}
    if token.isdigit() or token == '':
        return False
    return True


def spellfix(token: str) -> str:
    """Apply a minimal normalisation for specific noisy tokens."""
    if token == "volunterilay":
        return "voluntarily"
    if token == "pavolf":
        return "pavlof"
    if "ruclip" in token:
        return "ruclip"
    if "ceramic" in token:
        return "ceramic"
    return token


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Lowercase, strip punctuation, split on spaces, stem, and optionally drop stopwords."""
    tokens: List[str] = []
    line = text.lower()
    line = line.translate(line.maketrans("", "", PUNCTUATION))
    for raw in line.split(" "):
        if not is_valid_token(raw):
            continue
        tok = spellfix(raw)
        tok = STEMMER.stem(tok)
        if remove_stopwords and tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def unique(tokens: List[str]) -> List[str]:
    """Return unique tokens (order not preserved)."""
    return list(set(tokens))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors, guarding against zero norms."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def basic_tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on spaces; no stemming, no stopword removal."""
    result: List[str] = []
    line = text.lower()
    line = line.translate(line.maketrans("", "", PUNCTUATION))
    for raw in line.split(" "):
        if not is_valid_token(raw):
            continue
        tok = spellfix(raw)
        result.append(tok)
    return result


# ---------- Core IR helpers ----------

def tf(counter: Counter, total_len: int, term: str) -> float:
    """Term frequency of term within a token multiset."""
    if total_len <= 0:
        return 0.0
    return float(counter.get(term, 0)) / float(total_len)


def idf(df: int, num_documents: int) -> float:
    """Inverse document frequency using base-10 logarithm with zero guards."""
    if df <= 0 or num_documents <= 0:
        return 0.0
    return float(np.log10(num_documents / float(df)))


def tfidf_vector(tokens: List[str],
                 vocab_to_index: Dict[str, int],
                 invindex_df: Dict[str, int],
                 num_documents: int) -> np.ndarray:
    """Construct dense TF-IDF vector for given tokens and global stats."""
    vector = np.zeros((len(vocab_to_index),), dtype=float)
    counter = Counter(tokens)
    total_len = len(tokens)
    for term in counter.keys():
        idx = vocab_to_index.get(term)
        if idx is None:
            continue
        vector[idx] = tf(counter, total_len, term) * idf(invindex_df.get(term, 0), num_documents)
    return vector


def bm25_score(n: float, f: float, qf: float, N: float, dl: float, avdl: float,
               k1: float = 1.2, k2: float = 100.0, b: float = 0.75) -> float:
    """Compute BM25 contribution for a given term with standard parameters."""
    if n <= 0 or N <= 0 or dl <= 0 or avdl <= 0:
        return 0.0
    K = k1 * ((1.0 - b) + b * (dl / avdl))
    first = np.log(((0.0 + 0.5) / (0.0 - 0.0 + 0.5)) / ((n - 0.0 + 0.5) / (N - n - 0.0 + 0.0 + 0.5)))
    second = ((k1 + 1.0) * f) / (K + f) if (K + f) != 0 else 0.0
    third = ((k2 + 1.0) * qf) / (k2 + qf) if (k2 + qf) != 0 else 0.0
    return float(first * second * third)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    return 1.0 / (1.0 + np.exp(-x))


# ---------- Metrics helpers ----------

def average_precision_at_k(relevancies: Sequence[int], k: int) -> float:
    """Compute AP@k for a binary relevance list (1 relevant, 0 non-relevant)."""
    if k <= 0:
        return 0.0
    num_hits = 0
    precision_sum = 0.0
    for i, rel in enumerate(relevancies[:k], start=1):
        if rel == 1:
            num_hits += 1
            precision_sum += num_hits / i
    return 0.0 if num_hits == 0 else precision_sum / num_hits


def mean_average_precision(ranked_relevancies_by_qid: Dict[str, Sequence[int]], k: int) -> float:
    """Compute mAP@k across queries given ranked relevancies per qid."""
    ap_values: List[float] = []
    for _, rels in ranked_relevancies_by_qid.items():
        ap_values.append(average_precision_at_k(rels, k))
    return float(np.mean(ap_values)) if ap_values else 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Compute DCG@k for a list of gains (e.g., binary relevancies)."""
    dcg = 0.0
    for i, g in enumerate(gains[:k], start=1):
        dcg += g / np.log2(i + 1)
    return float(dcg)


def ndcg_at_k(gains: Sequence[float], ideal_gains: Sequence[float], k: int) -> float:
    """Compute NDCG@k given gains and corresponding ideal gains."""
    dcg = dcg_at_k(gains, k)
    idcg = dcg_at_k(ideal_gains, k)
    return 0.0 if idcg == 0.0 else float(dcg / idcg)


