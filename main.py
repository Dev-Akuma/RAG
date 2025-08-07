import httpx  # add this at the top near other imports if not already

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import traceback
import numpy as np
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai

from app.document_parser import extract_text_from_upload
from app.context_chunker import chunk_by_sentences
from app.faiss_index import FaissIndex
from app.embedding import RemoteEmbedder

# Load .env variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Globals
embedder = None
vector_index = None
genai_client = None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@app.on_event("startup")
async def startup_event():
    global embedder, vector_index, genai_client

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set in environment variables")

    print("🚀 Initializing Gemini API client...")
    genai_client = genai.Client(api_key=GOOGLE_API_KEY)

    print("🌐 Initializing RemoteEmbedder with Gemini model...")
    embedder = RemoteEmbedder(
        model_name="gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    print("📡 Initializing FAISS Index with IVF...")
    vector_index = FaissIndex(embedding_dim=3072, use_ivf=True, nlist=100)

    print("✅ All dependencies initialized successfully.")

async def call_gemini_api(prompt: str) -> str:
    if not genai_client:
        raise HTTPException(status_code=503, detail="Service not ready.")
    try:
        response = genai_client.models.generate_content(
            model="models/gemini-1.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"❌ Error calling Gemini API: {e}")
        return "Sorry, I couldn't generate a response at this time."

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    if not embedder or not vector_index:
        raise HTTPException(status_code=503, detail="Service not ready.")

    try:
        content = await file.read()
        print(f"📄 Read file '{file.filename}' of size {len(content)} bytes")
        file_stream = BytesIO(content)

        raw_text = extract_text_from_upload(file_stream, file.filename)
        print(f"📝 Extracted text length: {len(raw_text)} characters")

        chunks = chunk_by_sentences(raw_text, max_tokens=150)
        print(f"📦 Split text into {len(chunks)} chunks")

        if not chunks:
            raise HTTPException(status_code=400, detail="No text found to chunk.")

        # Embed chunks in batches asynchronously
        all_embeddings = []
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeds = await embedder.embed_chunks(batch)
            all_embeddings.append(batch_embeds)
            # Optional: slight delay to avoid quota issues
            # await asyncio.sleep(1)

        embeddings = np.vstack(all_embeddings)
        print(f"🔢 Generated embeddings of shape: {embeddings.shape}")

        vector_index.add(embeddings, chunks)
        print("✅ Added embeddings to FAISS index")

        return {
            "message": f"✅ {file.filename} processed and stored with {len(chunks)} chunks.",
            "chunks_count": len(chunks),
            "embedding_dimension": int(embeddings.shape[1]) if embeddings.size else 0,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Failed to process document: {str(e) or 'Unknown error'}")

# Pydantic model for query
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_document(request: QueryRequest):
    if not embedder or not vector_index:
        raise HTTPException(status_code=503, detail="Service not ready.")

    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query string is empty.")

        # Await embedding for query since embed_chunks is async now
        query_embedding = (await embedder.embed_chunks([query]))[0]
        top_matches = vector_index.query(query_embedding, top_k=5)

        context = "No relevant context found." if not top_matches else "\n\n".join(
            [match["text"] for match in top_matches]
        )

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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Failed to generate response: {str(e) or 'Unknown error'}")



class HackRxRequest(BaseModel):
    documents: str  # URL of the document
    questions: list[str]

@app.post("/hackrx/run")
async def hackrx_run(request: HackRxRequest):
    if not embedder or not vector_index:
        raise HTTPException(status_code=503, detail="Service not ready.")

    doc_url = request.documents
    questions = request.questions

    if not doc_url:
        raise HTTPException(status_code=400, detail="Document URL is required.")
    if not questions:
        raise HTTPException(status_code=400, detail="Questions list is empty.")

    try:
        # Download document from URL
        async with httpx.AsyncClient() as client:
            resp = await client.get(doc_url)
            resp.raise_for_status()
            content = resp.content

        # Extract text from downloaded file
        file_stream = BytesIO(content)
        # Attempt to infer filename extension from URL for extraction logic
        filename = doc_url.split("?")[0].split("/")[-1] or "document.pdf"

        raw_text = extract_text_from_upload(file_stream, filename)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from the document.")

        # Chunk the document text
        chunks = chunk_by_sentences(raw_text, max_tokens=150)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to chunk document text.")

        # Embed and add to FAISS index
        all_embeddings = []
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_embeds = await embedder.embed_chunks(batch)
            all_embeddings.append(batch_embeds)
        embeddings = np.vstack(all_embeddings)
        vector_index.add(embeddings, chunks)

        # Now answer each question
        answers = []
        for q in questions:
            q = q.strip()
            if not q:
                answers.append("Empty question.")
                continue

            query_embedding = (await embedder.embed_chunks([q]))[0]
            top_matches = vector_index.query(query_embedding, top_k=5)

            context = "No relevant context found." if not top_matches else "\n\n".join(
                [match["text"] for match in top_matches]
            )
            prompt = f"""
You are an intelligent assistant. Use the following context to answer the user's query.

Context:
{context}

User query:
{q}
"""
            response_text = await call_gemini_api(prompt)
            answers.append(response_text)

        return {"answers": answers}

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process request: {str(e)}")
