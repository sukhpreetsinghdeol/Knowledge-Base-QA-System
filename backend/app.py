import os
import tempfile
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
import uvicorn
from pathlib import Path
from fastapi.responses import StreamingResponse

# Create FastAPI app
app = FastAPI(title="Document QA API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage - in production, use a proper database
sessions = {}

# Define the KB folder path
KB_FOLDER = os.environ.get("KB_FOLDER", "kb")

# Ensure the KB folder exists
os.makedirs(KB_FOLDER, exist_ok=True)

class Question(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str

# === Load Text from Files ===
def load_text_from_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type. Only .txt files are supported.")

# === Split into Chunks ===
def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# === Embed Chunks ===
def embed_chunks(chunks, model):
    return model.encode(chunks)

# === Store Embeddings in FAISS ===
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# === Search Top-k Similar Chunks ===
def search_index(index, query_embedding, chunks, top_k=3):
    D, I = index.search(np.array([query_embedding]), top_k)
    return [chunks[i] for i in I[0]]

# === Query Ollama (Streaming) ===
def query_ollama_stream(model_name, prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": True
    }

    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if 'response' in json_response:
                    yield json.dumps({"text": json_response["response"]}) + "\n"
                
                # If this is the final response, break
                if json_response.get("done", False):
                    break
                    
    except Exception as e:
        yield json.dumps({"error": str(e)}) + "\n"

# === Query Ollama (Non-Streaming) ===
def query_ollama(model_name, prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        raise Exception(f"Error querying Ollama: {str(e)}")

# Initialize the sentence transformer model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("Loading the sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully!")

@app.get("/kb")
async def list_kb_files():
    """List all text files in the knowledge base folder"""
    try:
        # Get all files in the KB folder
        kb_path = Path(KB_FOLDER)
        
        # Check if the directory exists
        if not kb_path.exists() or not kb_path.is_dir():
            return {"files": [], "message": f"Knowledge base folder '{KB_FOLDER}' not found or not a directory"}
        
        # List only text files (.txt)
        supported_extensions = ['.txt']
        files = [
            {
                "name": file.name,
                "size": file.stat().st_size,
                "last_modified": file.stat().st_mtime,
                "type": file.suffix
            }
            for file in kb_path.iterdir()
            if file.is_file() and file.suffix.lower() in supported_extensions
        ]
        
        return {"files": files, "count": len(files)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing KB files: {str(e)}")

@app.post("/process")
async def process_document(file: UploadFile = File(...), save_to_kb: bool = False):
    """Process an uploaded document and prepare it for question answering"""
    global model
    
    if not model:
        raise HTTPException(
            status_code=500, 
            detail="Model not initialized. Please try again later."
        )
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext != '.txt':
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Please upload a TXT file."
        )
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # If save_to_kb is True, save the file to the KB folder
        if save_to_kb:
            kb_file_path = os.path.join(KB_FOLDER, file.filename)
            with open(kb_file_path, 'wb') as kb_file:
                # Reset file position
                await file.seek(0)
                kb_file.write(content)
        
        # Process the file
        text = load_text_from_file(temp_file_path)
        chunks = split_text(text)
        embeddings = embed_chunks(chunks, model)
        faiss_index = build_faiss_index(np.array(embeddings))
        
        # Generate session ID and store the data
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "chunks": chunks,
            "index": faiss_index,
            "filename": file.filename
        }
        
        # Set session cookie and return
        response_data = {"message": "Document processed successfully", "session_id": session_id}
        if save_to_kb:
            response_data["saved_to_kb"] = True
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return response_data
        
    except Exception as e:
        # Clean up if file was created
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: Question):
    """Ask a question about the processed document (non-streaming)"""
    global model
    
    # For simplicity, using the latest session
    # In production, you'd use the session ID from a cookie or request
    if not sessions:
        raise HTTPException(
            status_code=400,
            detail="No document has been processed. Please upload a document first."
        )
    
    session_id = list(sessions.keys())[-1]
    session = sessions[session_id]
    
    try:
        # Embed the question
        query_embedding = model.encode([question.question])[0]
        
        # Search for relevant chunks
        top_chunks = search_index(session["index"], query_embedding, session["chunks"])
        
        # Create prompt for Ollama
        context = "\n\n".join(top_chunks)
        full_prompt = f"""Use the following context to answer the question. 
If the answer is not contained within the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question.question}

Answer:"""
        
        # Query Ollama
        answer = query_ollama("mistral", full_prompt)
        
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask/stream")
async def ask_question_stream(question: Question):
    """Ask a question about the processed document with streaming response"""
    global model
    
    # For simplicity, using the latest session
    # In production, you'd use the session ID from a cookie or request
    if not sessions:
        raise HTTPException(
            status_code=400,
            detail="No document has been processed. Please upload a document first."
        )
    
    session_id = list(sessions.keys())[-1]
    session = sessions[session_id]
    
    try:
        # Embed the question
        query_embedding = model.encode([question.question])[0]
        
        # Search for relevant chunks
        top_chunks = search_index(session["index"], query_embedding, session["chunks"])
        
        # Create prompt for Ollama
        context = "\n\n".join(top_chunks)
        full_prompt = f"""Use the following context to answer the question. 
If the answer is not contained within the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question.question}

Answer:"""
        
        # Stream the response
        return StreamingResponse(
            query_ollama_stream("mistral", full_prompt),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_session():
    """Clear the current session data"""
    global sessions
    sessions = {}
    return {"message": "Session cleared successfully"}

@app.get("/kb/{filename}")
async def get_kb_file(filename: str):
    """Get a specific file from the knowledge base folder"""
    file_path = os.path.join(KB_FOLDER, filename)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in knowledge base")
    
    try:
        text = load_text_from_file(file_path)
        return {"filename": filename, "content": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

@app.post("/search/kb")
async def search_kb(search_request: SearchRequest):
    """Search across all knowledge base files"""
    global model
    
    if not model:
        raise HTTPException(
            status_code=500, 
            detail="Model not initialized. Please try again later."
        )
    
    try:
        # Get all files from KB
        kb_path = Path(KB_FOLDER)
        if not kb_path.exists() or not kb_path.is_dir():
            return {"results": [], "message": "Knowledge base folder not found"}
            
        supported_extensions = ['.txt']
        file_paths = [
            file for file in kb_path.iterdir()
            if file.is_file() and file.suffix.lower() in supported_extensions
        ]
        
        if not file_paths:
            return {"results": [], "message": "No files found in knowledge base"}
            
        # Search across all files
        search_results = []
        
        # Embed the search query
        query_embedding = model.encode([search_request.query])[0]
        
        for file_path in file_paths:
            try:
                # Load and process the file
                text = load_text_from_file(str(file_path))
                chunks = split_text(text)
                
                if not chunks:
                    continue
                    
                # Embed chunks
                embeddings = embed_chunks(chunks, model)
                
                # Build index
                index = build_faiss_index(np.array(embeddings))
                
                # Search for relevant chunks
                top_chunks = search_index(index, query_embedding, chunks, top_k=1)
                
                if top_chunks:
                    search_results.append({
                        "filename": file_path.name,
                        "content_snippet": top_chunks[0][:200] + "..." if len(top_chunks[0]) > 200 else top_chunks[0]
                    })
            except Exception as e:
                # Skip files that cause errors
                print(f"Error processing file {file_path}: {str(e)}")
                continue
        
        return {
            "results": search_results,
            "count": len(search_results),
            "query": search_request.query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching knowledge base: {str(e)}")

# Run the API with Uvicorn when executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)