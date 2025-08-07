import os
import numpy as np
from typing import List
from google import genai

class RemoteEmbedder:
    def __init__(self, model_name: str = "models/embedding-001", google_api_key: str = None):
        self.model_name = model_name
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        print(f"🌐 Initializing Gemini embedding model: {self.model_name}")
        self.client = genai.Client(api_key=self.api_key)

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        print(f"🔍 Embedding {len(chunks)} chunks...")

        if not chunks:
            raise ValueError("Input chunk list is empty.")

        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=chunks,
            )

            embeddings = [embedding.values for embedding in result.embeddings]

            if not embeddings or not embeddings[0]:
                raise RuntimeError("No embeddings returned.")

            print(f"✅ Received embeddings of shape ({len(embeddings)}, {len(embeddings[0])})")
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            # Return zero embeddings with correct dimension (assume 3072)
            return np.zeros((len(chunks), 3072), dtype=np.float32)
