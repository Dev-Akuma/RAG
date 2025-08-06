import faiss
import numpy as np
import uuid
from typing import List, Dict, Union

class FaissIndex:
    def __init__(self, embedding_dim: int = 384):
        print("📚 Initializing FAISS index...")
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Cosine similarity via normalized vectors
        self.text_chunks: List[str] = []
        self.ids: List[str] = []

    def add(self, embeddings: Union[np.ndarray, List[List[float]]], chunks: List[str]):
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must be the same length.")

        print(f"📦 Adding {len(embeddings)} vectors to FAISS")

        # Convert to np.ndarray if not already
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype='float32')

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)
        self.ids.extend([str(uuid.uuid4()) for _ in chunks])

    def query(self, query_vector: Union[np.ndarray, List[float]], top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        # Convert to np.ndarray if needed
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype='float32')

        # Normalize query vector
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
