import csv
import math
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from text_utils import tokenize, unique, cosine_similarity, tfidf_vector, bm25_score


class PassageRetrievalEngine:
    """Simple passage retrieval engine.

    Provides utilities for building simple indices, TF-IDF ranking, BM25 ranking,
    and query-likelihood language model ranking (Laplace, Lidstone, Dirichlet).
    All I/O paths are configurable via constructor or per-method overrides.
    """

    def __init__(self,
                 passage_collection_path: str = 'passage-collection.txt',
                 candidate_passages_path: str = 'candidate-passages-top1000.tsv',
                 queries_path: str = 'test-queries.tsv',
                 pidpassage_path: str = 'pidpassage.tsv',
                 invindex_path: str = 'invindex.csv',
                 index_counts_path: str = 'index.csv',
                 tfidf_out: str = 'tfidf.csv',
                 bm25_out: str = 'bm25.csv',
                 laplace_out: str = 'laplace.csv',
                 lidstone_out: str = 'lidstone.csv',
                 dirichlet_out: str = 'dirichlet.csv') -> None:
        """Initialize engine and configure input/output paths."""

        # Cached data structures populated lazily
        self._pid_to_tokens: Dict[str, List[str]] = {}
        self._pid_to_counter: Dict[str, Counter] = {}
        self._passage_length: Dict[str, int] = {}
        self._invindex_df: Dict[str, int] = {}
        self._term_corpus_counts: Dict[str, int] = {}
        self._vocab_to_index: Dict[str, int] = {}
        self._num_passages: int = 0
        self._avg_passage_len: float = 0.0
        self._total_corpus_terms: int = 0
        self._vocab_size: int = 0

        # Filenames (configurable)
        self.file_passage_collection = passage_collection_path
        self.file_candidate_top1000 = candidate_passages_path
        self.file_test_queries = queries_path
        self.file_pid_passage = pidpassage_path
        self.file_invindex = invindex_path
        self.file_index_counts = index_counts_path
        self.file_tfidf_out = tfidf_out
        self.file_bm25_out = bm25_out
        self.file_laplace_out = laplace_out
        self.file_lidstone_out = lidstone_out
        self.file_dirichlet_out = dirichlet_out

    # Text statistics
    def analyze_zipf(self, remove_stopwords: bool = True,
                     passage_collection_path: Optional[str] = None) -> Tuple[float, float]:
        """Compute token frequencies and plot Zipf log-log curve.

        Returns a tuple (mean(rank*freq), std(rank*freq)).
        """
        counts: Dict[str, int] = {}
        total_terms = 0

        path = passage_collection_path or self.file_passage_collection
        with open(path, encoding='utf8') as fh:
            for line in fh:
                line = line.strip()
                for tok in tokenize(line, remove_stopwords=remove_stopwords):
                    total_terms += 1
                    counts[tok] = counts.get(tok, 0) + 1

        # Sort by frequency desc
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        ranks = list(range(1, len(items) + 1))
        freqs = [c / total_terms for _, c in items]

        rank_times_freq = [r * f for r, f in zip(ranks, freqs)]
        mean_val = float(np.mean(rank_times_freq)) if rank_times_freq else 0.0
        std_val = float(np.std(rank_times_freq, ddof=1)) if len(rank_times_freq) > 1 else 0.0

        # Plot empirical vs. heuristic Zipf curve (scaled as original example)
        plt.figure(figsize=(12, 8))
        plt.grid(True)
        plt.loglog(ranks, freqs, linestyle='-', linewidth=1, marker='o', markersize=1, label="data")

        N = len(items)
        rank_arr = np.arange(1, N + 1)
        zipf_freq = (1 / rank_arr) / 10  # heuristic scaling to resemble original plot
        plt.loglog(rank_arr, zipf_freq, '-', label="Zipf's curve")

        plt.legend()
        title_suffix = "(stop words removed)" if remove_stopwords else "(stop words kept)"
        plt.title(f"Frequency vs Rank {title_suffix}")
        plt.xlabel("Frequency Ranking")
        plt.ylabel("Probability of occurrence")
        plt.show()

        return mean_val, std_val

    # Inverted index
    def build_indexes(self,
                      candidate_passages_path: Optional[str] = None,
                      pidpassage_out: Optional[str] = None,
                      invindex_out: Optional[str] = None,
                      index_counts_out: Optional[str] = None) -> None:
        """Build pid->tokens, inverted index (document frequency), and corpus term counts.

        Outputs: pidpassage.tsv, invindex.csv, index.csv (configurable via args).
        """
        pid_to_tokens: Dict[str, List[str]] = {}

        cpath = candidate_passages_path or self.file_candidate_top1000
        with open(cpath, encoding='utf8') as fh:
            reader = csv.reader(fh, delimiter='\t')
            for row in reader:  # qid, pid, query, passage
                pid = row[1]
                passage_text = row[3]
                pid_to_tokens[pid] = tokenize(passage_text, remove_stopwords=False)

        # Persist pid->tokens as a space-joined token string (avoid eval on read)
        pidpassage_path = pidpassage_out or self.file_pid_passage
        with open(pidpassage_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f, delimiter='\t')
            for pid, toks in pid_to_tokens.items():
                w.writerow([pid, " ".join(toks)])

        # Build inverted index (document frequencies) and overall term counts
        invindex_df: Dict[str, int] = {}
        term_counts: Dict[str, int] = {}

        for toks in pid_to_tokens.values():
            for term in unique(toks):
                invindex_df[term] = invindex_df.get(term, 0) + 1
            for term in toks:
                term_counts[term] = term_counts.get(term, 0) + 1

        invindex_path = invindex_out or self.file_invindex
        with open(invindex_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerows(invindex_df.items())

        index_counts_path = index_counts_out or self.file_index_counts
        with open(index_counts_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerows(term_counts.items())

        # Cache for later use
        self._pid_to_tokens = pid_to_tokens
        self._pid_to_counter = {pid: Counter(toks) for pid, toks in pid_to_tokens.items()}
        self._invindex_df = invindex_df
        self._term_corpus_counts = term_counts

    def _ensure_indexes_loaded(self) -> None:
        """Load cached indices and statistics from disk if not already in memory."""
        if not self._pid_to_tokens or not self._invindex_df or not self._term_corpus_counts:
            # Load from files generated by build_indexes
            # pidpassage.tsv
            self._pid_to_tokens = {}
            with open(self.file_pid_passage, encoding='utf8') as fh:
                reader = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE)
                for row in reader:
                    pid = row[0]
                    raw = row[1]
                    # Backwards-compatible parser: eval if looks like list, else split on space
                    tokens_list: List[str]
                    if raw.startswith('[') and raw.endswith(']'):
                        try:
                            tokens_list = list(eval(raw))
                        except Exception:
                            tokens_list = raw.split(" ") if raw else []
                    else:
                        tokens_list = raw.split(" ") if raw else []
                    self._pid_to_tokens[pid] = tokens_list

            # invindex.csv (doc frequencies)
            self._invindex_df = {}
            with open(self.file_invindex, encoding='utf8') as fh:
                reader = csv.reader(fh, delimiter=',', quoting=csv.QUOTE_NONE)
                for row in reader:
                    self._invindex_df[row[0]] = int(row[1])

            # index.csv (corpus term counts)
            self._term_corpus_counts = {}
            with open(self.file_index_counts, encoding='utf8') as fh:
                reader = csv.reader(fh, delimiter=',', quoting=csv.QUOTE_NONE)
                for row in reader:
                    self._term_corpus_counts[row[0]] = int(row[1])

        # Passage lengths, stats
        if not self._passage_length:
            self._passage_length = {pid: len(tokens) for pid, tokens in self._pid_to_tokens.items()}
            self._pid_to_counter = {pid: Counter(tokens) for pid, tokens in self._pid_to_tokens.items()}
            lengths = list(self._passage_length.values())
            self._num_passages = len(lengths)
            self._avg_passage_len = (sum(lengths) / len(lengths)) if lengths else 0.0
            self._total_corpus_terms = sum(lengths)
            self._vocab_size = len(self._invindex_df)

        # Vocabulary index for TF-IDF vector construction
        if not self._vocab_to_index:
            self._vocab_to_index = {}
            for i, term in enumerate(self._invindex_df.keys()):
                self._vocab_to_index[term] = i

    def _tfidf_vector(self, tokens: List[str]) -> np.ndarray:
        """Construct a sparse TF-IDF vector in dense numpy form for given tokens."""
        return tfidf_vector(tokens, self._vocab_to_index, self._invindex_df, self._num_passages)

    # Retrieval models (TF-IDF and BM25)
    def rank_tfidf(self,
                   candidate_passages_path: Optional[str] = None,
                   queries_path: Optional[str] = None,
                   output_path: Optional[str] = None) -> None:
        """Rank candidate passages using TF-IDF cosine similarity and write CSV qid,pid,score."""
        self._ensure_indexes_loaded()

        cpath = candidate_passages_path or self.file_candidate_top1000
        qpath = queries_path or self.file_test_queries
        with open(cpath, encoding='utf8') as fh:
            doc_rows = list(csv.reader(fh, delimiter='\t'))
        with open(qpath, encoding='utf8') as fh:
            query_rows = list(csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE))

        # Group candidate rows by qid to avoid repeated scans
        rows_by_qid: Dict[str, List[List[str]]] = defaultdict(list)
        for row in doc_rows:
            rows_by_qid[row[0]].append(row)

        results: List[List[Tuple[str, str, float]]] = []
        for qid, query_text in query_rows:
            per_query: List[Tuple[str, str, float]] = []
            query_vec = self._tfidf_vector(tokenize(query_text))
            for docqid in [qid]:
                for _, pid, _, passage in rows_by_qid.get(docqid, []):
                    p_tokens = self._pid_to_tokens.get(pid) or tokenize(passage)
                    passage_vec = self._tfidf_vector(p_tokens)
                    score = cosine_similarity(query_vec, passage_vec)
                    per_query.append((qid, pid, score))
            per_query.sort(key=lambda x: x[2], reverse=True)
            results.append(per_query[:100])

        out = output_path or self.file_tfidf_out
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for group in results:
                for row in group:
                    w.writerow(row)

    def _bm25_score(self, term: str, passage_tokens: List[str], query_tokens: List[str], pid: str) -> float:
        """Compute BM25 term contribution for a single term and passage/query pair."""
        n = float(self._invindex_df.get(term, 0))
        f = float(passage_tokens.count(term))
        qf = float(query_tokens.count(term))
        N = float(self._num_passages)
        dl = float(self._passage_length[pid])
        avdl = float(self._avg_passage_len) if self._avg_passage_len > 0 else 0.0
        return bm25_score(n, f, qf, N, dl, avdl)

    def rank_bm25(self,
                  candidate_passages_path: Optional[str] = None,
                  queries_path: Optional[str] = None,
                  output_path: Optional[str] = None) -> None:
        """Rank candidate passages using BM25 and write CSV qid,pid,score."""
        self._ensure_indexes_loaded()

        cpath = candidate_passages_path or self.file_candidate_top1000
        qpath = queries_path or self.file_test_queries
        with open(cpath, encoding='utf8') as fh:
            doc_rows = list(csv.reader(fh, delimiter='\t'))
        with open(qpath, encoding='utf8') as fh:
            query_rows = list(csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE))

        rows_by_qid: Dict[str, List[List[str]]] = defaultdict(list)
        for row in doc_rows:
            rows_by_qid[row[0]].append(row)

        results: List[List[Tuple[str, str, float]]] = []
        for qid, query_text in query_rows:
            per_query: List[Tuple[str, str, float]] = []
            q_tokens = tokenize(query_text)
            for _, pid, _, passage in rows_by_qid.get(qid, []):
                p_tokens = self._pid_to_tokens.get(pid) or tokenize(passage)
                score = 0.0
                for term in set(q_tokens):
                    score += self._bm25_score(term, p_tokens, q_tokens, pid)
                per_query.append((qid, pid, score))
            per_query.sort(key=lambda x: x[2], reverse=True)
            results.append(per_query[:100])

        out = output_path or self.file_bm25_out
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for group in results:
                for row in group:
                    w.writerow(row)

    # Query likelihood language models
    def _get_occurrence(self, pid: str, term: str) -> int:
        """Get raw term frequency for a term in a passage by id."""
        return int(self._pid_to_counter.get(pid, Counter()).get(term, 0))

    def rank_laplace(self,
                     candidate_passages_path: Optional[str] = None,
                     queries_path: Optional[str] = None,
                     output_path: Optional[str] = None) -> None:
        """Rank using query-likelihood with Laplace smoothing; write log-prob scores."""
        self._ensure_indexes_loaded()

        cpath = candidate_passages_path or self.file_candidate_top1000
        qpath = queries_path or self.file_test_queries
        with open(cpath, encoding='utf8') as fh:
            doc_rows = list(csv.reader(fh, delimiter='\t'))
        with open(qpath, encoding='utf8') as fh:
            query_rows = list(csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE))

        rows_by_qid: Dict[str, List[List[str]]] = defaultdict(list)
        for row in doc_rows:
            rows_by_qid[row[0]].append(row)

        V = float(self._vocab_size)
        results: List[List[Tuple[str, str, float]]] = []
        for qid, query_text in query_rows:
            per_query: List[Tuple[str, str, float]] = []
            q_tokens = tokenize(query_text)
            for _, pid, _, _ in rows_by_qid.get(qid, []):
                dl = float(self._passage_length[pid])
                logprob = 0.0
                for term in q_tokens:
                    count = self._get_occurrence(pid, term)
                    prob = (count + 1.0) / (dl + V)
                    logprob += math.log(prob) if prob > 0 else -1e12
                per_query.append((qid, pid, logprob))
            per_query.sort(key=lambda x: x[2], reverse=True)
            results.append(per_query[:100])

        out = output_path or self.file_laplace_out
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for group in results:
                for row in group:
                    w.writerow(row)

    def rank_lidstone(self,
                      epsilon: float = 0.1,
                      candidate_passages_path: Optional[str] = None,
                      queries_path: Optional[str] = None,
                      output_path: Optional[str] = None) -> None:
        """Rank using query-likelihood with Lidstone correction; write log-prob scores."""
        self._ensure_indexes_loaded()

        cpath = candidate_passages_path or self.file_candidate_top1000
        qpath = queries_path or self.file_test_queries
        with open(cpath, encoding='utf8') as fh:
            doc_rows = list(csv.reader(fh, delimiter='\t'))
        with open(qpath, encoding='utf8') as fh:
            query_rows = list(csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE))

        rows_by_qid: Dict[str, List[List[str]]] = defaultdict(list)
        for row in doc_rows:
            rows_by_qid[row[0]].append(row)

        V = float(self._vocab_size)
        results: List[List[Tuple[str, str, float]]] = []
        for qid, query_text in query_rows:
            per_query: List[Tuple[str, str, float]] = []
            q_tokens = tokenize(query_text)
            for _, pid, _, _ in rows_by_qid.get(qid, []):
                dl = float(self._passage_length[pid])
                logprob = 0.0
                denom = dl + epsilon * V
                for term in q_tokens:
                    count = self._get_occurrence(pid, term)
                    prob = (count + epsilon) / denom if denom > 0 else 0.0
                    logprob += math.log(prob) if prob > 0 else -1e12
                per_query.append((qid, pid, logprob))
            per_query.sort(key=lambda x: x[2], reverse=True)
            results.append(per_query[:100])

        out = output_path or self.file_lidstone_out
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for group in results:
                for row in group:
                    w.writerow(row)

    def rank_dirichlet(self,
                       mu: float = 50.0,
                       candidate_passages_path: Optional[str] = None,
                       queries_path: Optional[str] = None,
                       output_path: Optional[str] = None) -> None:
        """Rank using query-likelihood with Dirichlet smoothing; write log-prob scores."""
        self._ensure_indexes_loaded()

        cpath = candidate_passages_path or self.file_candidate_top1000
        qpath = queries_path or self.file_test_queries
        with open(cpath, encoding='utf8') as fh:
            doc_rows = list(csv.reader(fh, delimiter='\t'))
        with open(qpath, encoding='utf8') as fh:
            query_rows = list(csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE))

        rows_by_qid: Dict[str, List[List[str]]] = defaultdict(list)
        for row in doc_rows:
            rows_by_qid[row[0]].append(row)

        C = float(self._total_corpus_terms)
        results: List[List[Tuple[str, str, float]]] = []
        for qid, query_text in query_rows:
            per_query: List[Tuple[str, str, float]] = []
            q_tokens = tokenize(query_text)
            for _, pid, _, _ in rows_by_qid.get(qid, []):
                N = float(self._passage_length[pid])
                logprob = 0.0
                denom = (N + mu)
                for term in q_tokens:
                    cf = float(self._term_corpus_counts.get(term, 0))
                    pwc = (cf / C) if C > 0 else 0.0
                    tf = float(self._get_occurrence(pid, term))
                    if denom <= 0 or N <= 0:
                        comp = pwc
                    else:
                        comp = (N / denom) * (tf / N) + (mu / denom) * pwc
                    logprob += math.log(comp) if comp > 0 else -1e12
                per_query.append((qid, pid, logprob))
            per_query.sort(key=lambda x: x[2], reverse=True)
            results.append(per_query[:100])

        out = output_path or self.file_dirichlet_out
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for group in results:
                for row in group:
                    w.writerow(row)


if __name__ == "__main__":
    engine = PassageRetrievalEngine()
    engine.analyze_zipf(remove_stopwords=True)
    engine.build_indexes()
    engine.rank_tfidf()
    engine.rank_bm25()
    engine.rank_laplace()
    engine.rank_lidstone(epsilon=0.1)
    engine.rank_dirichlet(mu=50.0)


