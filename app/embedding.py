import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"🔧 Loading model '{model_name}' locally...")
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        print("🔍 embed_chunks() called with", len(chunks), "chunks")
        if not chunks:
            raise ValueError("Input chunk list is empty.")

        # Encode the chunks locally using the model
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        return embeddings
