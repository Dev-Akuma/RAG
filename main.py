import os
from io import BytesIO
from typing import List
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from google import genai
from app.document_parser import extract_text_from_upload
from app.context_chunker import chunk_by_sentences
from app.faiss_index import FaissIndex

# Import your custom RemoteEmbedder (make sure it's in app/embeddings.py or adjust import path)
from app.embeddings import RemoteEmbedder

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = None
vector_index = None
genai_client = None

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@app.on_event("startup")
async def startup_event():
    global embedder, vector_index, genai_client

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in env variables")

    print("🚀 Initializing Gemini client...")
    genai_client = genai.Client(api_key=GOOGLE_API_KEY)

    print("🌐 Initializing remote Hugging Face embedder via RemoteEmbedder (Inference API)...")
    # No asyncio.to_thread needed, just instantiate
    embedder = RemoteEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("📡 Initializing FAISS Index...")
    vector_index = FaissIndex()

    print("✅ All dependencies initialized successfully.")

async def call_gemini_api(prompt: str) -> str:
    if not genai_client:
        raise HTTPException(status_code=503, detail="Service not ready. Please try again in a moment.")
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Sorry, I couldn't generate a response at this time."

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    if not embedder or not vector_index:
        raise HTTPException(status_code=503, detail="Service not ready. Please try again in a moment.")

    try:
        content = await file.read()
        file_stream = BytesIO(content)

        raw_text = extract_text_from_upload(file_stream, file.filename)
        chunks = chunk_by_sentences(raw_text, max_tokens=150)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found to chunk.")

        embeddings = embedder.embed_chunks(chunks)  # Use your embed_chunks method here
        vector_index.add(embeddings, chunks)

        return {
            "message": f"✅ {file.filename} processed and stored with {len(chunks)} chunks.",
            "chunks_count": len(chunks),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to process document: {e}")

@app.post("/query")
async def query_document(query: str):
    if not embedder or not vector_index:
        raise HTTPException(status_code=503, detail="Service not ready. Please try again in a moment.")
    
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query string is empty.")

        query_embedding = embedder.embed_chunks([query])[0]  # embed_query alternative
        top_matches = vector_index.query(query_embedding, top_k=5)

        context = "No relevant context found." if not top_matches else "\n\n".join([match["text"] for match in top_matches])

        prompt = f"""
You are an intelligent assistant. Use the following context to answer the user's query.
Context:
{context}
User query:
{query}
"""

        response_text = await call_gemini_api(prompt)

        return {
            "response": response_text,
            "query": query,
            "retrieved_chunks": [match["text"] for match in top_matches],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to generate response: {e}")
