# AI Philosophy & Cognitive Science Tutor Platform 🧠🤖 
### Local End-to-End GenAI Ecosystem: QLoRA Fine-Tuned Socratic Tutor with Advanced Cross-Encoder RAG & FastAPI Deployment

A production-grade, highly specialized Generative AI tutoring ecosystem fine-tuned locally using **QLoRA (4-bit Quantized Low-Rank Adaptation)** and integrated into an advanced **Conversational RAG (Retrieval-Augmented Generation)** pipeline. 

The entire framework is explicitly engineered, optimized, and containerised to execute complete training, indexing, and deployment lifecycles on **constraint-heavy local consumer hardware** (such as an NVIDIA RTX 3050 4GB Laptop GPU) without experiencing Out-of-Memory (OOM) memory faults or gradient math disruptions.

---

## 📁 Repository Directory Architecture

The platform maintains a highly decoupled, modular directory layout to cleanly isolate core computational and deployment scripts from heavy local binary runtime environments or model storage checkpoint layers:

```text
ai_philosophy_tutor/
│
├── train.py                    # Supervised Fine-Tuning execution script via TRL
├── chat.py                     # Standalone local terminal chat interface loop
├── advanced_rag_chat.py        # Local terminal client featuring FAISS + Cross-Encoder re-ranking
├── app.py                      # Production Asynchronous FastAPI web server backend script
├── test_api.py                 # Pure Python REST API client test integration script
├── tutor_dataset_sharegpt.json # High-density 140-sample multi-turn curriculum dataset
├── knowledge_base/             # Ingestion directory for ungrounded PDF textbooks
│   ├── Functionalism.pdf
│   ├── The Chinese Room Argument.pdf
│   └── Externalism About the Mind.pdf
├── requirements.txt            # Pinned stable dependency tracking sheet
└── README.md                   # Complete system documentation profile (This file)
```

---

## 🛠️ Phase 1: QLoRA Fine-Tuning & Memory Engineering (`train.py`)

To transform a generic base model into a highly precise academic tutor within a strict 4GB VRAM ceiling, the project implements a **Quantized Low-Rank Adaptation (QLoRA)** pipeline. This stage optimizes behavior, structural dialogue templates, and instructional strategies.

### 1. 4-Bit Quantum Compression Math
Using `bitsandbytes`, the 1.8 Billion parameters of the base model are loaded into a specialized **NF4 (Normal Float 4)** data type with **double quantization** enabled. This compresses the model's footprint on the GPU from **~4.5GB down to ~1.5GB**, leaving 2.5GB of VRAM entirely free to handle active training gradients and token tensors.

### 2. Deep Matrix Target Expansion
Instead of only modifying basic query/value vectors, this pipeline injects adapter tracks into **both the Multi-Head Attention blocks and the internal MLP (Multi-Layer Perceptron) factual layers** to maximize concept retention:
*   **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
*   **Hyperparameters**: Rank (r=16) for expanded learning capacity, Alpha (α=32) for stable scaling, and a Dropout rate of `0.05`.
*   **Total Trainable Weights**: **14,991,360 parameters** active.

### 3. Training Constraints & Convergence Metrics
*   **Optimization State**: `paged_adamw_8bit` (Offloads heavy memory tracking parameters safely to System RAM).
*   **Virtual Batching**: `per_device_train_batch_size=1` combined with `gradient_accumulation_steps=4` to simulate a stable batch size of 4 without VRAM spikes.
*   **Precision Type**: Native, hardware-stabilized `BFloat16` (Brain Float) to eliminate mixed-precision math clashes.
*   **Convergence Performance**: Curated a custom 140-sample conversational curriculum in ShareGPT format. By executing a 4-epoch SFT window using boundary policing and strict negative constraints, training loss successfully dropped to a **final step loss of ~1.07**, and neutralizing cross-concept hallucinations.

---

## 🔍 Phase 2: Advanced Conversational RAG Architecture (`advanced_rag_chat.py`)

Fine-tuning adapts the model’s behavior and teaching style, but it cannot memorize entire libraries. Phase 2 introduces an advanced, local **Conversational RAG Pipeline** to serve as an "open-book" reference sheet, grounding the model's responses in factual data.

```text
Student Question ➔ Query Rewriter ➔ Semantic Search (FAISS) ➔ Cross-Encoder Re-Ranking ➔ Prompt Injected Context ➔ QLoRA Generation
```

### 1. Multi-PDF Document Ingestion & Chunking
The system scans a dedicated subfolder (`/knowledge_base`), automatically identifies all raw PDF files, and parses their raw text strings using `pypdf`. To prevent cutting off key arguments, a `RecursiveCharacterTextSplitter` divides the text into **600-character chunks with a 100-character overlap**.

### 2. Semantic Retrieval via FAISS
Chunks are transformed into high-dimensional mathematical vectors using the local embedding model `all-MiniLM-L6-v2` and saved as a local **FAISS (Facebook AI Similarity Search)** vector store index on your drive. When a question is submitted, FAISS executes a dense mathematical similarity scan to pull the top 8 most relevant text candidates.

### 3. Deep CPU Cross-Encoder Re-Ranking
Standard vector searches can be noisy. The pipeline passes the top 8 raw FAISS candidates through an explicit **Hugging Face Cross-Encoder Re-Ranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) on the CPU. The model re-scores every paragraph based on true semantic relevance and isolates only the **top 2 absolute best chunks**, keeping the prompt clean and protecting the model's 256 VRAM token limit.

### 4. Context-Aware Query Rewriting & Memory Loops
To support natural conversation, a **sliding-window memory buffer** tracks the last 3 back-and-forth exchanges. If a student asks a follow-up question using pronouns (e.g., *"Who created it?"*), the fine-tuned engine automatically reads the memory buffer and **rewrites the query** (e.g., *"Who created the Chinese Room Argument?"*) before running the FAISS search, keeping the conversation contextually locked.

---
## Workflow

User Query
      ↓
Memory Retrieval
      ↓
FAISS Search
      ↓
FlashRank Reranking
      ↓
Context Assembly
      ↓
Qwen + QLoRA
      ↓
Response

## 🌐 Phase 3: Production FastAPI Web Server Deployment (`app.py`)

To scale the architecture into a commercial, enterprise-ready application framework, the entire hybrid QLoRA-RAG cognitive engine is wrapped inside a production-grade **FastAPI web server**.

### 1. Modern Lifespan Event Operations
The server utilizes an asynchronous `@asynccontextmanager` lifespan configuration loop. Upon running `python app.py`, the server executes all heavy background resource allocations **before** opening the network port: loading 4-bit weights into VRAM, loading the FAISS vector index from disk, and pinning the Cross-Encoder layers to the CPU.

### 2. Strict Input/Output Schema Validation
The endpoints enforce type-safe data transactions by validating incoming network requests and outgoing payloads using **Pydantic Data Schemas**:
*   `StudentQuery`: Validates input strings and enforces a strict `temperature=0.1` and `repetition_penalty=1.1` default configuration to guarantee high factual precision.
*   `TutorResponse`: Formats the outgoing response into an explicit decoupled payload returning the original query, the rewritten search query used, the generated response, and a success status code.

### 3. Asynchronous Routing Endpoints
*   `POST /api/v1/tutor/chat`: Core asynchronous pathway executing query rewriting, document re-ranking, and streaming QLoRA text generation.
*   `POST /api/v1/tutor/reset`: Instantly clears out the server-side memory buffer array to let the student refresh boundaries and change academic study topics cleanly.

---

## 🧪 Phase 4: REST API Integration Client Testing (`test_api.py`)

To verify the health and connectivity of the server without running into text-paste limits or broken escaping bugs in the Windows Command Prompt, a native Python client test script (`test_api.py`) handles transactions cleanly.

The script formats a structured dictionary object, executes an official HTTP `POST` network transaction to the local address gateway (`http://127.0.0`), captures the JSON packet, and outputs a formatted, clean API payload response log directly to your console:

```json
{
  "query": "According to the documents, what is the core argument of Functionalism?",
  "search_query_used": "According to the documents, what is the core argument of Functionalism?",
  "response": "The core argument of functionalism is that mental states are defined entirely by their functional role within a system. It argues that 'thinking' or 'beliefs' do not have intrinsic properties; they are merely data structures that map inputs to outputs.",
  "status": "success"
}
```

---

## 🚀 Step-by-Step Installation & Deployment Guide

### 1. Initialize and Activate Environment
```bash
cd C:\Users\divya\ai_philosophy_tutor
python -m venv ai_philosophy_tutor
ai_philosophy_tutor\Scripts\activate
```

### 2. Install Pinned Stable Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Initial Training Model (Optional)
```cmd
set PYTHONUTF8=1
python train.py
```

### 4. Deploy the Local Production ASGI Web Server
Drop your Cognitive Science/Philosophy PDFs into the `/knowledge_base` folder and launch the Uvicorn engine:
```cmd
set PYTHONUTF8=1
python app.py
```

### 5. Ping the Server with Client Integration Tests
Open a **secondary, separate terminal window** and run the Python script client to verify operations:
```cmd
cd C:\Users\divya\ai_philosophy_tutor
python test_api.py
```
