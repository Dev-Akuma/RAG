from typing import List
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
import os

class RemoteEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hf_api_key: str = None):
        print(f"🌐 Initializing remote Hugging Face embeddings with model '{model_name}'")

        # Set the HF API key as env var if provided
        if hf_api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

        self.model_name = model_name
        # Initialize without passing token explicitly
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        print("🔍 embed_chunks() called with", len(chunks), "chunks")
        if not chunks:
            raise ValueError("Input chunk list is empty.")
        
        # Use langchain's embed_documents to get list of vectors
        vectors = self.embeddings.embed_documents(chunks)
        
        # Convert list of lists to numpy array
        return np.array(vectors)
