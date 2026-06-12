import torch
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn

# Core Pipeline Processing Tools
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Initialize FastAPI Application Instance
app = FastAPI(
    title="AI Philosophy & CogSci Tutor Local Server",
    description="Enterprise-grade local GenAI endpoint running QLoRA fine-tuned weights augmented with a Cross-Encoder Re-ranked FAISS RAG pipeline.",
    version="1.0.0"
)

# Global variables to hold models in memory across API requests
tokenizer = None
model = None
db = None
ranker = None
chat_history = [] # Server-side memory cache block

# Define strict Pydantic Schemas for API Input/Output Validation
class StudentQuery(BaseModel):
    question: str
    temperature: Optional[float] = 0.1  # Set to strict 0.1 default for historical precision

class TutorResponse(BaseModel):
    query: str
    search_query_used: str
    response: str
    status: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes modern structural memory configurations upon local server startup."""
    global tokenizer, model, db, ranker
    
    # FIXED: Re-added all four core environment configurations for perfect consistency
    BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
    LORA_ADAPTERS = "./qwen_lora_results"
    PDF_FOLDER = "knowledge_base"
    FAISS_INDEX_DIR = "faiss_index"

    print("\n[FastAPI Server Startup] Initializing QLoRA Cognitive Matrix...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTERS)
    model.eval()

    print("[FastAPI Server Startup] Linking Semantic FAISS Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # FIXED: Uses the explicitly named variable now
    if os.path.exists(FAISS_INDEX_DIR):
        db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        print(f"[FastAPI Server Startup] FAISS Vector Database Loaded from '{FAISS_INDEX_DIR}'.")
    else:
        print(f"[FastAPI Server Startup] Warning: '{FAISS_INDEX_DIR}' missing. Please ensure database is built.")

    print("[FastAPI Server Startup] Synchronizing CPU Cross-Encoder Re-Ranker...")
    ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    print("[FastAPI Server Startup] All Engines Successfully Online!\n")
    
    yield  # Server runs here
    print("[FastAPI Server Shutdown] Releasing allocations...")

# Pass the lifespan function directly inside the app definition
app = FastAPI(
    title="AI Philosophy & CogSci Tutor Local Server",
    lifespan=lifespan  
)

@app.post("/api/v1/tutor/chat", response_model=TutorResponse)
async def chat_endpoint(payload: StudentQuery):
    """Asynchronous endpoint executing Context-Aware Query Rewriting, RAG, and Text Generation."""
    global chat_history, tokenizer, model, db, ranker
    
    user_prompt = payload.question.strip()
    if not user_prompt:
        raise HTTPException(status_code=400, detail="Student question string cannot be empty.")

    # --- A. CONTEXT-AWARE QUERY REWRITING ---
    search_query = user_prompt
    if len(chat_history) > 0:
        history_context = "\n".join([f"{h['role']}: {h['content']}" for h in chat_history])
        rewrite_prompt = f"<|im_start|>system\nGiven the chat history and a new student question, rewrite it into a single standalone search query for a textbook search. Output ONLY the rewritten string.<|im_end|>\n<|im_start|>user\nHistory:\n{history_context}\nNew Question: {user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer([rewrite_prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=64, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        decoded_query = tokenizer.decode(ids[len(inputs.input_ids):], skip_special_tokens=True).strip()
        if len(decoded_query) > 2:
            search_query = decoded_query

    # --- B. SEMANTIC SEARCH + NATIVE RE-RANKING LOOP ---
    context_text = ""
    if db is not None:
        raw_results = db.similarity_search(search_query, k=8)
        if len(raw_results) > 0:
            pairs = [[search_query, doc.page_content] for doc in raw_results]
            scores = ranker.predict(pairs)
            scored_results = sorted(zip(scores, raw_results), key=lambda x: x[0], reverse=True)
            top_passages = [doc for score, doc in scored_results[:2]]
            context_text = "\n\n".join([doc.page_content for doc in top_passages])

    # --- C. CONTEXT-AWARE SYSTEM GENERATION ---
    system_instruction = (
        "You are an expert Socratic Philosophy and Cognitive Science Tutor. "
        "Formulate an educational response using the textbook context and conversation history."
    )
    
    memory_block = ""
    if len(chat_history) > 0:
        memory_block = "\nPrevious Exchange:\n" + "\n".join([f"{h['role'].upper()}: {h['content']}" for h in chat_history])

    full_prompt = (
        f"<|im_start|>system\n{system_instruction}\n"
        f"Grounded Textbook Context:\n{context_text}\n{memory_block}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=payload.temperature, 
            repetition_penalty=1.1, # Enforces strict factual constraints against adjacent noise
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    trimmed_ids = generated_ids[0][input_length:]
    response_text = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

    # --- D. MEMORY APPEND BUFFER ---
    chat_history.append({"role": "user", "content": user_prompt})
    chat_history.append({"role": "assistant", "content": response_text})
    if len(chat_history) > 6:
        chat_history = chat_history[-6:]

    return TutorResponse(
        query=user_prompt,
        search_query_used=search_query,
        response=response_text,
        status="success"
    )

@app.post("/api/v1/tutor/reset")
async def reset_memory():
    """Wipes active session history completely to refresh cognitive framework boundaries."""
    global chat_history
    chat_history = []
    return {"message": "Conversational session memory completely reset."}

if __name__ == "__main__":
    # Boots server on localhost port 8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
