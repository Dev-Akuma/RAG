import os
import requests
import numpy as np
from typing import List

class RemoteEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", hf_api_key=None):
        print(f"🌐 Initializing remote Hugging Face Inference API embeddings with model '{model_name}'")
        self.model_name = model_name
        self.api_key = hf_api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.api_key:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set")

        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        print("🔍 embed_chunks() called with", len(chunks), "chunks")
        if not chunks:
            raise ValueError("Input chunk list is empty.")

        embeddings = []
        for chunk in chunks:
            payload = {"inputs": chunk}
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            # The API returns a nested list of embeddings (token embeddings), average them:
            token_embeddings = response.json()
            # Average across tokens to get a single vector for the chunk
            vector = np.mean(token_embeddings, axis=0)
            embeddings.append(vector)
        
        return np.vstack(embeddings)
