import faiss
import numpy as np
import uuid
from typing import List, Dict

class FaissIndex:
    def __init__(self, embedding_dim: int = 384):
        print("📚 Initializing FAISS index...")
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Cosine-like with normalized vectors
        self.text_chunks = []  # stores chunk metadata
        self.ids = []  # mapping FAISS vector order to UUIDs

    def add(self, embeddings: np.ndarray, chunks: List[str]):
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must be the same length.")

        print(f"📦 Adding {len(embeddings)} vectors to FAISS")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.text_chunks.extend(chunks)
        self.ids.extend([str(uuid.uuid4()) for _ in chunks])

    def query(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

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
