import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Stable Core text processing tools
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

def main():
    BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
    LORA_ADAPTERS = "./qwen_lora_results"
    PDF_FOLDER = "knowledge_base"
    FAISS_INDEX_DIR = "faiss_index"

    # --- 1. INITIALIZE FINE-TUNED 4-BIT COGNITIVE ENGINE ---
    print("Initializing local fine-tuned QLoRA weights...")
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

    # --- 2. CONFIGURING SEMANTIC RETRIEVAL & FAISS ENGINE ---
    print("Setting up semantic embedding engine...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)

    db = None
    if not os.path.exists(FAISS_INDEX_DIR):
        print(f"Scanning folder '{PDF_FOLDER}' for PDF documents...")
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
        
        if len(pdf_files) == 0:
            print(f"\n[Warning] No PDF files found inside '{PDF_FOLDER}'. Place PDFs there for RAG.")
        else:
            all_chunks = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            
            for pdf in pdf_files:
                full_pdf_path = os.path.join(PDF_FOLDER, pdf)
                print(f"Parsing document text: '{full_pdf_path}'...")
                
                reader = PdfReader(full_pdf_path)
                full_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                docs = text_splitter.create_documents(texts=[full_text], metadatas=[{"source": pdf}])
                all_chunks.extend(docs)
            
            if len(all_chunks) > 0:
                db = FAISS.from_documents(all_chunks, embeddings)
                db.save_local(FAISS_INDEX_DIR)
                print("Successfully compiled PDFs into FAISS index!")
    else:
        print("Loading local semantic FAISS vector engine...")
        db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    # Initialize standard HuggingFace CrossEncoder directly for stable re-ranking on CPU
    print("Loading local CPU Cross-Encoder Re-Ranker...")
    ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    # --- 3. CONVERSATIONAL SLIDING-WINDOW MEMORY STORE ---
    chat_history = [] 

    print("\n--- Advanced AI Tutor Initialized! ---")
    print("Type 'exit' to end your study session.")

    while True:
        user_prompt = input("\nYou (Student): ")
        if user_prompt.lower() == "exit":
            break
        if not user_prompt.strip():
            continue

        # --- A. CONTEXT-AWARE QUERY REWRITING ---
        search_query = user_prompt.strip()
        
        # FIXED: Only trigger the model rewrite if chat_history actually contains data
        if len(chat_history) > 0:
            history_context = "\n".join([f"{h['role']}: {h['content']}" for h in chat_history])
            rewrite_prompt = f"<|im_start|>system\nGiven the chat history and a new student question, rewrite it into a single standalone search query for a textbook search. Output ONLY the rewritten string.<|im_end|>\n<|im_start|>user\nHistory:\n{history_context}\nNew Question: {user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer([rewrite_prompt], return_tensors="pt").to("cuda")
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=64, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
            
            decoded_query = tokenizer.decode(ids[len(inputs.input_ids):], skip_special_tokens=True).strip()
            # Safety fallback: If the model returns blank or garbage, use the original user prompt
            if len(decoded_query) > 2:
                search_query = decoded_query

        # --- B. SEMANTIC SEARCH + NATIVE RE-RANKING LOOP ---
        context_text = ""
        if db is not None:
            print(f"Scanning textbook database for: '{search_query}'...")
            raw_results = db.similarity_search(search_query, k=8) # Pull top 8 candidates
            
            if len(raw_results) > 0:
                # Build sentence pairs for CrossEncoder evaluation: [(query, document_text), ...]
                pairs = [[search_query, doc.page_content] for doc in raw_results]
                
                # Compute deep relevance scores
                scores = ranker.predict(pairs)
                
                # Pair results with scores and sort from highest to lowest relevance
                scored_results = sorted(zip(scores, raw_results), key=lambda x: x[0], reverse=True)
                
                # Isolate the top 2 best entries safely
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
        input_length = inputs.input_ids.shape[1] # Track the exact size of the input prompt

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                repetition_penalty=1.1, 
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        trimmed_ids = generated_ids[0][input_length:]
        response = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()
        print(f"\nTutor: {response}")

        # --- D. MEMORY APPEND ---
        chat_history.append({"role": "user", "content": user_prompt})
        chat_history.append({"role": "assistant", "content": response})
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

if __name__ == "__main__":
    main()
