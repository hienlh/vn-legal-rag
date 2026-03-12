"""
Traditional baseline retrievers for legal article retrieval.

Provides BM25, TF-IDF, Semantic (vector), and Keyword matching baselines.
All retrievers work with doc-qualified article IDs (e.g., "59-2020-QH14:d206")
from the vn_legal_rag database (840+ articles across enterprise + traffic law).

Usage:
    from scripts import traditional_baseline_retrievers as trad
    baselines = trad.init_traditional_baselines("data/legal_docs.db", embedding_gen)
    ranked = baselines["bm25"].search("câu hỏi pháp luật", top_k=30)
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Vietnamese text processing
# ---------------------------------------------------------------------------

VIETNAMESE_STOPWORDS = {
    "và", "hoặc", "của", "là", "trong", "có", "được", "cho", "với",
    "đến", "từ", "về", "để", "khi", "như", "theo", "vì", "nếu",
    "thì", "mà", "bởi", "qua", "lại", "ra", "vào", "đi", "lên",
    "một", "các", "những", "này", "đó", "đây", "ấy", "kia",
    "ai", "gì", "nào", "đâu", "sao", "bao", "mấy", "tại",
    "ạ", "à", "ơi", "nhé", "nha", "vậy", "thế", "rồi",
}


def tokenize_vi(text: str) -> List[str]:
    """Simple Vietnamese tokenization with stopword removal."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    return [w for w in words if len(w) >= 2 and w not in VIETNAMESE_STOPWORDS]


@dataclass
class ArticleDoc:
    """Article document for baseline indexing."""
    article_id: str      # doc-qualified: "59-2020-QH14:d206"
    article_name: str    # "Điều 206. Tên điều"
    content: str
    tokens: List[str]


def load_articles_for_baselines(db_path: str) -> List[ArticleDoc]:
    """Load all articles from vn_legal_rag DB with doc-qualified IDs."""
    from vn_legal_rag.offline import LegalDocumentDB

    db = LegalDocumentDB(db_path)
    articles = db.get_all_articles()

    docs = []
    for article in articles:
        article_name = article.title or f"Điều {article.article_number}"

        # Build content from title + clauses
        content_parts = [article_name]
        if hasattr(article, 'clauses') and article.clauses:
            for clause in article.clauses:
                if hasattr(clause, 'content') and clause.content:
                    content_parts.append(clause.content)
        # Fallback to raw_text/content if no clauses
        if len(content_parts) == 1:
            if article.content:
                content_parts.append(article.content)
            elif article.raw_text:
                content_parts.append(article.raw_text)

        content = " ".join(content_parts)
        tokens = tokenize_vi(content)

        docs.append(ArticleDoc(
            article_id=article.id,
            article_name=article_name,
            content=content,
            tokens=tokens,
        ))

    return docs


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 keyword-based retrieval with doc-qualified article IDs."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[ArticleDoc] = []
        self.avg_doc_len = 0
        self.idf: Dict[str, float] = {}
        self.N = 0

    def index(self, docs: List[ArticleDoc]):
        self.docs = docs
        self.N = len(docs)

        doc_freqs = Counter()
        total_len = 0
        for doc in docs:
            total_len += len(doc.tokens)
            for token in set(doc.tokens):
                doc_freqs[token] += 1

        self.avg_doc_len = total_len / self.N if self.N > 0 else 1

        for token, df in doc_freqs.items():
            self.idf[token] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 30) -> List[str]:
        """Return ranked list of doc-qualified article IDs."""
        query_tokens = tokenize_vi(query)
        scores = []

        for doc in self.docs:
            doc_len = len(doc.tokens)
            term_freqs = Counter(doc.tokens)
            score = 0
            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = term_freqs.get(token, 0)
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * numerator / denominator
            scores.append((doc.article_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores[:top_k]]


# ---------------------------------------------------------------------------
# TF-IDF Retriever
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """TF-IDF cosine similarity retrieval."""

    def __init__(self):
        self.docs: List[ArticleDoc] = []
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[Dict[str, float]] = []

    def index(self, docs: List[ArticleDoc]):
        self.docs = docs
        N = len(docs)

        doc_freqs = Counter()
        for doc in docs:
            for token in set(doc.tokens):
                doc_freqs[token] += 1

        for token, df in doc_freqs.items():
            self.idf[token] = math.log(N / (df + 1)) + 1

        self.doc_vectors = []
        for doc in docs:
            tf = Counter(doc.tokens)
            vector = {}
            for token, count in tf.items():
                vector[token] = (1 + math.log(count)) * self.idf.get(token, 0)
            self.doc_vectors.append(vector)

    def search(self, query: str, top_k: int = 30) -> List[str]:
        query_tokens = tokenize_vi(query)
        query_tf = Counter(query_tokens)

        query_vector = {}
        for token, count in query_tf.items():
            query_vector[token] = (1 + math.log(count)) * self.idf.get(token, 0)

        scores = []
        for i, doc in enumerate(self.docs):
            sim = self._cosine_sim(query_vector, self.doc_vectors[i])
            scores.append((doc.article_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores[:top_k]]

    def _cosine_sim(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        norm1 = math.sqrt(sum(x * x for x in v1.values()))
        norm2 = math.sqrt(sum(x * x for x in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# Semantic (Vector) Retriever
# ---------------------------------------------------------------------------

class SemanticRetriever:
    """Pure semantic/vector search using vn_legal_rag embedding provider."""

    def __init__(self):
        self.docs: List[ArticleDoc] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_gen = None

    def index(self, docs: List[ArticleDoc], embedding_gen):
        self.docs = docs
        self.embedding_gen = embedding_gen

        # Generate embeddings for all articles (title + first 500 chars)
        texts = [f"{doc.article_name}: {doc.content[:500]}" for doc in docs]

        # Batch embed
        all_embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embs = embedding_gen.embed(batch)
            all_embeddings.extend(batch_embs)
        self.embeddings = np.array(all_embeddings)

    def search(self, query: str, top_k: int = 30) -> List[str]:
        if self.embedding_gen is None or self.embeddings is None:
            return []

        query_emb = np.array(self.embedding_gen.embed([query])[0])

        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)
        similarities = np.dot(self.embeddings, query_emb) / (norms * query_norm + 1e-8)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.docs[idx].article_id for idx in top_indices]


# ---------------------------------------------------------------------------
# Keyword Retriever
# ---------------------------------------------------------------------------

class KeywordRetriever:
    """Simple keyword substring matching."""

    def __init__(self):
        self.docs: List[ArticleDoc] = []

    def index(self, docs: List[ArticleDoc]):
        self.docs = docs

    def search(self, query: str, top_k: int = 30) -> List[str]:
        query_tokens = tokenize_vi(query)
        scores = []

        for doc in self.docs:
            score = 0
            content_lower = doc.content.lower()
            name_lower = doc.article_name.lower()

            for token in query_tokens:
                if token in name_lower:
                    score += 3.0
                if token in content_lower:
                    score += 1.0

            if score > 0:
                match_ratio = score / (len(query_tokens) * 4)
                score = score * (1 + match_ratio)

            scores.append((doc.article_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores[:top_k]]


# ---------------------------------------------------------------------------
# Initialization helper
# ---------------------------------------------------------------------------

def init_traditional_baselines(db_path: str, embedding_gen=None) -> Dict[str, object]:
    """Initialize all traditional baseline retrievers.

    Returns dict: { method_name: retriever_instance }
    """
    print("      Loading articles from DB...")
    docs = load_articles_for_baselines(db_path)
    print(f"      {len(docs)} articles loaded")

    retrievers = {}

    print("      Indexing BM25...")
    bm25 = BM25Retriever()
    bm25.index(docs)
    retrievers["bm25"] = bm25

    print("      Indexing TF-IDF...")
    tfidf = TFIDFRetriever()
    tfidf.index(docs)
    retrievers["tfidf"] = tfidf

    print("      Indexing Keyword...")
    keyword = KeywordRetriever()
    keyword.index(docs)
    retrievers["keyword"] = keyword

    if embedding_gen:
        print("      Indexing Semantic (generating embeddings for all articles)...")
        semantic = SemanticRetriever()
        semantic.index(docs, embedding_gen)
        retrievers["semantic"] = semantic
    else:
        print("      Skipping Semantic (no embedding provider)")

    return retrievers
