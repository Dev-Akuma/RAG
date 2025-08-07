import faiss
import numpy as np
import uuid
from typing import List, Dict, Union

class FaissIndex:
    def __init__(self, embedding_dim: int = None, use_ivf: bool = False, nlist: int = 100):
        """
        :param embedding_dim: dimension of embedding vectors
        :param use_ivf: whether to use IVF index (recommended for large datasets)
        :param nlist: number of clusters for IVF (affects recall/speed tradeoff)
        """
        print("📚 Initializing FAISS index...")
        self.embedding_dim = embedding_dim
        self.use_ivf = use_ivf
        self.default_nlist = nlist  # Keep original nlist default
        self.index = None
        self.text_chunks: List[str] = []
        self.ids: List[str] = []
        self.is_trained = False

    def _init_index(self, embedding_count: int):
        if self.use_ivf:
            # Dynamically cap nlist to be ≤ number of vectors
            nlist = min(self.default_nlist, embedding_count)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            print(f"🆕 Initialized IVF FAISS index with dim: {self.embedding_dim}, nlist: {nlist}")
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            print(f"🆕 Initialized Flat FAISS index with dim: {self.embedding_dim}")

    def add(self, embeddings: Union[np.ndarray, List[List[float]]], chunks: List[str]):
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must be the same length.")

        print(f"📦 Adding {len(embeddings)} vectors to FAISS")

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype='float32')

        if self.index is None:
            self.embedding_dim = embeddings.shape[1]
            self._init_index(len(embeddings))  # 🧠 Pass vector count here

        elif embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")

        faiss.normalize_L2(embeddings)

        if self.use_ivf and not self.is_trained:
            print(f"⚙️ Training IVF index on {len(embeddings)} vectors...")
            self.index.train(embeddings)
            self.is_trained = True
            print("✅ IVF index trained successfully.")

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)
        self.ids.extend([str(uuid.uuid4()) for _ in chunks])

    def query(self, query_vector: Union[np.ndarray, List[float]], top_k: int = 5) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype='float32')

        query_vector = np.expand_dims(query_vector, axis=0)
        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.text_chunks):
                results.append({
                    "text": self.text_chunks[idx],
                    "score": float(score)
                })
        return results
