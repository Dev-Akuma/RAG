import os
from io import BytesIO
from typing import List
import asyncio # New import

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Your existing imports
from google import genai
from app.document_parser import extract_text_from_upload
from app.context_chunker import chunk_by_sentences
from app.faiss_index import FaissIndex
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals - they are initialized once at startup
embedder = None
vector_index = None
genai_client = None # Also initialize this here

# Config from env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@app.on_event("startup")
async def startup_event():
    """
    Initialize all heavy, blocking dependencies at startup.
    This runs on a separate thread pool and won't block the event loop.
    """
    global embedder, vector_index, genai_client

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in env variables")

    # Initialize Gemini client
    print("🚀 Initializing Gemini client...")
    genai_client = genai.Client(api_key=GOOGLE_API_KEY)

    # Initialize Hugging Face embedder using asyncio.to_thread
    # This ensures the blocking model download doesn't freeze the server.
    print("🌐 Initializing remote Hugging Face embedder via LangChain...")
    embedder = await asyncio.to_thread(
        HuggingFaceEmbeddings,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    
    # Initialize FAISS Index (this should be fast)
    print("📡 Initializing FAISS Index...")
    vector_index = FaissIndex()

    print("✅ All dependencies initialized successfully.")

# Removed the get_embedder and get_vector_index functions
# We now use the global variables directly

async def call_gemini_api(prompt: str) -> str:
    # ... (no change here)
    if not genai_client:
        # In case a request comes in before startup is complete
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
    """
    Upload a document, extract text, chunk it, generate embeddings (remote),
    and store them in FAISS.
    """
    # Now just use the global variables directly
    if not embedder or not vector_index:
        raise HTTPException(status_code=503, detail="Service not ready. Please try again in a moment.")

    try:
        content = await file.read()
        file_stream = BytesIO(content)

        raw_text = extract_text_from_upload(file_stream, file.filename)
        chunks = chunk_by_sentences(raw_text, max_tokens=150)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found to chunk.")

        # The embed_documents call is still synchronous, but it's okay because
        # the model is already loaded and it runs inside an endpoint task.
        embeddings = embedder.embed_documents(chunks)
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
    """
    Query documents using FAISS-based retrieval and get Gemini LLM response.
    """
    # Now just use the global variables directly
    if not embedder or not vector_index:
        raise HTTPException(status_code=503, detail="Service not ready. Please try again in a moment.")
    
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query string is empty.")

        query_embedding = embedder.embed_query(query)
        top_matches = vector_index.query(query_embedding, top_k=5)

        if not top_matches:
            context = "No relevant context found."
        else:
            context = "\n\n".join([match["text"] for match in top_matches])

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