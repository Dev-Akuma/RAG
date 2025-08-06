from typing import List
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

class RemoteEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", hf_api_key: str = None):
        print(f"🌐 Initializing remote Hugging Face embeddings with model '{model_name}'")
        # Pass your HF API key here or set it in environment variable HUGGINGFACEHUB_API_TOKEN
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, huggingfacehub_api_token=hf_api_key)

    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        print("🔍 embed_chunks() called with", len(chunks), "chunks")
        if not chunks:
            raise ValueError("Input chunk list is empty.")
        
        # Use langchain's embed method to get list of vectors
        vectors = self.embeddings.embed_documents(chunks)
        
        # Convert list of lists to numpy array
        return np.array(vectors)
