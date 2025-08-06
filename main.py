import os
from io import BytesIO
from typing import List

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.document_parser import extract_text_from_upload
from app.context_chunker import chunk_by_sentences
from app.faiss_index import FaissIndex

# For remote Hugging Face embedding (LangChain)
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
embedder = None
vector_index = None

# Config from env
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")  # e.g. https://api.generativeai.googleapis.com/v1beta2/models/gemini-2.5-flash-lite:generateContent
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not HUGGINGFACEHUB_API_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set in env variables")
if not GEMINI_API_URL:
    raise RuntimeError("GEMINI_API_URL not set in env variables")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in env variables")

def get_embedder():
    global embedder
    if embedder is None:
        print("🌐 Initializing remote Hugging Face embedder via LangChain...")
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        )
    return embedder

def get_vector_index():
    global vector_index
    if vector_index is None:
        print("📡 Initializing FAISS Index...")
        vector_index = FaissIndex()
    return vector_index

async def call_gemini_api(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GOOGLE_API_KEY}",
        "Content-Type": "application/json",
    }
    json_payload = {
        "prompt": prompt,
        "maxTokens": 256,
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GEMINI_API_URL, headers=headers, json=json_payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            # Adjust depending on actual response structure
            return data.get("text") or data.get("result") or ""
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "Sorry, I couldn't generate a response at this time."

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, extract text, chunk it, generate embeddings (remote),
    and store them in FAISS.
    """
    try:
        content = await file.read()
        file_stream = BytesIO(content)

        # 1. Extract text
        raw_text = extract_text_from_upload(file_stream, file.filename)

        # 2. Chunk text
        chunks = chunk_by_sentences(raw_text, max_tokens=150)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text found to chunk.")

        # 3. Generate embeddings (remote)
        embedder_instance = get_embedder()
        embeddings = embedder_instance.embed_documents(chunks)  # returns List[List[float]]

        # 4. Store in FAISS
        vector_db = get_vector_index()
        vector_db.add(embeddings, chunks)

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
    try:
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query string is empty.")

        # 1. Embed user query remotely
        embedder_instance = get_embedder()
        query_embedding = embedder_instance.embed_query(query)  # returns List[float]

        # 2. Retrieve relevant chunks from FAISS
        vector_db = get_vector_index()
        top_matches = vector_db.query(query_embedding, top_k=5)  # should return list of dicts with 'text'

        # 3. Format context from top chunks
        if not top_matches:
            context = "No relevant context found."
        else:
            context = "\n\n".join([match["text"] for match in top_matches])

        # 4. Build Gemini prompt
        prompt = f"""
You are an intelligent assistant. Use the following context to answer the user's query.

Context:
{context}

User query:
{query}
"""

        # 5. Call Gemini API asynchronously
        response_text = await call_gemini_api(prompt)

        return {
            "response": response_text,
            "query": query,
            "retrieved_chunks": [match["text"] for match in top_matches],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to generate response: {e}")
