import os
import requests
import uuid
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.document_parser import extract_text_from_upload
from app.context_chunker import chunk_by_sentences
from app.embedding import LocalEmbedder
from app.faiss_index import FaissIndex
import google.generativeai as genai
load_dotenv()

# Setup Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")



app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy globals
embedder = None
vector_index = None
EMBEDDING_DIM = 384  # for all-MiniLM-L6-v2

# Gemini API
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "http://localhost:5000/gemini_api_endpoint")

def get_embedder():
    global embedder
    if embedder is None:
        print("🔧 Initializing Local Embedder...")
        embedder = LocalEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedder

def get_vector_index():
    global vector_index
    if vector_index is None:
        print("📡 Initializing FAISS Index...")
        vector_index = FaissIndex()
    return vector_index

def call_gemini_api(prompt: str) -> str:
    try:
        response = requests.post(GEMINI_API_URL, json={"prompt": prompt, "max_tokens": 256})
        response.raise_for_status()
        return response.json().get("text", "")
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Sorry, I couldn't generate a response at this time."

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, extract text, chunk it, and store embeddings in Pinecone.
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

        # 3. Generate embeddings
        embedder_instance = get_embedder()
        embeddings = embedder_instance.embed_chunks(chunks)

        # 4. Store in Pinecone
        vector_db = get_vector_index()
        vector_db.add(embeddings, chunks)

        return {
            "message": f"✅ {file.filename} processed and stored with {len(chunks)} chunks.",
            "chunks_count": len(chunks),
            "embedding_dimension": len(embeddings[0]),
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

        # 1. Embed user query
        embedder_instance = get_embedder()
        query_embedding = embedder_instance.embed_chunks([query])[0]

        # 2. Retrieve relevant chunks from FAISS
        vector_db = get_vector_index()
        top_matches = vector_db.query(query_embedding, top_k=5)

        # 3. Format context from top chunks
        if not top_matches:
            context = "No relevant context found."
        else:
            context = "\n\n".join([match["text"] for match in top_matches])

        # 4. Build Gemini prompt
        prompt = f"""You are an intelligent assistant. Use the following context to answer the user's query.

Context:
{context}

User query:
{query}
"""

        # 5. Call Gemini directly
        response = gemini_model.generate_content(prompt)

        return {
            "response": response.text,
            "query": query,
            "retrieved_chunks": [match["text"] for match in top_matches],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to generate response: {e}")