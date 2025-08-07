import os
import numpy as np
from typing import List
from google import genai


class RemoteEmbedder:
    def __init__(self, model_name: str = "gemini-embedding-001", google_api_key: str = None):
        print(f"🌐 Initializing Google Gemini embeddings with model '{model_name}'")
        self.model_name = model_name
        self.api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

        self.client = genai.Client(api_key=self.api_key)

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        print("🔍 embed_chunks() called with", len(chunks), "chunks")
        if not chunks:
            raise ValueError("Input chunk list is empty.")

        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=chunks  # ✅ Corrected from `content=`
                # ❌ Removed task_type: not currently accepted by API
            )

            embeddings = [embedding.values for embedding in result.embeddings]
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            print(f"❌ Error calling Gemini embed_content: {e}")
            return np.zeros((len(chunks), 768), dtype=np.float32)
