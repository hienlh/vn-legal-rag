"""
Text embedding utilities for semantic search.

Supports:
- sentence-transformers (local models)
- OpenAI embeddings API
"""

import numpy as np
from typing import List, Optional, Union


class EmbeddingProvider:
    """
    Unified embedding provider.

    Supports:
    - sentence-transformers (local, free)
    - openai (API, paid)
    """

    def __init__(
        self,
        provider: str = "sentence-transformers",
        model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        **kwargs,
    ):
        """
        Initialize embedding provider.

        Args:
            provider: Provider name (sentence-transformers, openai)
            model: Model name
            **kwargs: Additional provider-specific arguments
        """
        self.provider = provider.lower()
        self.model = model
        self.config = kwargs
        self._model = None

    def _init_model(self):
        """Lazy load model on first use."""
        if self._model is not None:
            return

        if self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

        elif self.provider == "openai":
            try:
                import os
                from openai import OpenAI
                api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
                self._model = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        self._init_model()

        # Ensure texts is a list
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # Generate embeddings
        if self.provider == "sentence-transformers":
            embeddings = self._model.encode(texts, convert_to_numpy=True)

        elif self.provider == "openai":
            response = self._model.embeddings.create(
                model=self.model,
                input=texts,
            )
            embeddings = np.array([item.embedding for item in response.data])

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Return single embedding if input was single text
        if is_single:
            return embeddings[0]

        return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._init_model()

        if self.provider == "sentence-transformers":
            return self._model.get_sentence_embedding_dimension()
        elif self.provider == "openai":
            # Model dimensions (hardcoded for known models)
            if "text-embedding-3-small" in self.model:
                return 1536
            elif "text-embedding-3-large" in self.model:
                return 3072
            elif "text-embedding-ada-002" in self.model:
                return 1536
            else:
                # Default - generate sample embedding to get dimension
                sample = self.embed("sample")
                return len(sample)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_matrix(
    queries: np.ndarray,
    documents: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between queries and documents.

    Args:
        queries: Query embeddings (shape: [n_queries, embedding_dim])
        documents: Document embeddings (shape: [n_docs, embedding_dim])

    Returns:
        Similarity matrix (shape: [n_queries, n_docs])
    """
    # Normalize vectors
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    documents_norm = documents / np.linalg.norm(documents, axis=1, keepdims=True)

    # Compute dot product
    return np.dot(queries_norm, documents_norm.T)


# Factory function
def create_embedding_provider(
    provider: str = "sentence-transformers",
    model: Optional[str] = None,
    **kwargs,
) -> EmbeddingProvider:
    """
    Create embedding provider instance.

    Args:
        provider: Provider name (sentence-transformers, openai)
        model: Model name (optional, uses default if not specified)
        **kwargs: Additional provider-specific arguments

    Returns:
        EmbeddingProvider instance
    """
    # Default models
    if model is None:
        if provider == "sentence-transformers":
            model = "paraphrase-multilingual-MiniLM-L12-v2"
        elif provider == "openai":
            model = "text-embedding-3-small"

    return EmbeddingProvider(provider=provider, model=model, **kwargs)
