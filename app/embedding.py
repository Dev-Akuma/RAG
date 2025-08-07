import os
import numpy as np
import asyncio
import time
import random
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

    async def _embed_batch(self, batch: List[str], retries=3) -> List[List[float]]:
        for attempt in range(retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                )
                embeddings = [embedding.values for embedding in result.embeddings]
                if not embeddings or not embeddings[0]:
                    raise RuntimeError("No embeddings returned.")
                return embeddings
            except Exception as e:
                print(f"❌ Error generating embeddings (attempt {attempt+1}): {e}")
                if attempt < retries - 1:
                    wait = min(5, 2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Retrying in {wait:.1f}s...")
                    await asyncio.sleep(wait)
                else:
                    print("⚠️ Max retries reached, returning zero embeddings for this batch.")
                    # Return zero embeddings if all retries fail
                    return [[0.0] * 3072 for _ in batch]

    async def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        print(f"🔍 Embedding {len(chunks)} chunks...")

        if not chunks:
            raise ValueError("Input chunk list is empty.")

        all_embeddings = []
        batch_size = 20

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"🔍 Embedding batch {i // batch_size + 1} with {len(batch)} chunks...")
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            await asyncio.sleep(1.0)  # Pause between batches to reduce rate limit hits

        print(f"✅ Completed embedding all {len(chunks)} chunks.")
        return np.array(all_embeddings, dtype=np.float32)
