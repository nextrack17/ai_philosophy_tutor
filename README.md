# AI Philosophy & Cognitive Science Tutor Platform 🧠🤖

An elite, highly specialized Large Language Model (LLM) tutoring platform fine-tuned locally using **QLoRA (4-bit Quantized Low-Rank Adaptation)**. This platform turns an open-source 1.8 Billion parameter model into a high-fidelity academic tutor capable of deep philosophical dialogue, architectural analysis, and Socratic instruction.

The platform is explicitly optimized to train and execute on constraint-heavy local consumer hardware (such as an NVIDIA RTX 3050 4GB Laptop GPU) without experiencing Out-of-Memory (OOM) failures.

---

## 📁 Project Structure

The project maintains a lightweight, decoupled directory architecture to keep tracking files isolated from heavy operational runtime code and model layers:

```text
ai_philosophy_tutor/
│
├── train.py                    # Supervised Fine-Tuning execution engine via TRL
├── chat.py                     # Local interactive Socratic user interface loop
├── tutor_dataset_sharegpt.json # Fleshed-out 140-sample multi-turn curriculum file
├── .gitignore                  # Active filtering schema to block deployment uploads
├── qwen_lora_results/          # trained LoRA adapter and related files
│ ├── adapter_config.json
│ ├── adapter_model.safetensors
│
└── README.md                   # System documentation profile (This file)
```

### Operational Architecture Breakdown:
*   **`train.py`**: Freezes the base model, sets up 4-bit Normal Float configurations, injects LoRA adapters into multi-headed attention and MLP projection layers, and triggers the SFT training iterations.
*   **`chat.py`**: Merges your saved custom adapter vectors on top of the base model dynamically in VRAM, enabling private local multi-turn text generation.
*   **`tutor_dataset_sharegpt.json`**: Contains a curated multi-turn dialogue curriculum covering critical conceptual intersections (e.g., *Searle's Chinese Room, Fodor's Modularity, Harnad's Symbol Grounding, and Friston's Predictive Processing*).

---

## 🛠️ Technical Specifications & Hyperparameters

The model has been optimized with strict computational precision vectors to secure an elite **84% token accuracy rate** while maintaining structural hardware stability:

*   **Base Engine**: Qwen 1.5 (1.8 Billion Parameter Chat-Aligned Edition)
*   **Quantization**: 4-bit Normal Float (NF4) with double quantization enabled
*   **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (Attacks both attention tracking and internal MLP factual concept matrices)
*   **LoRA Hyperparameters**: Rank ($r=16$), Alpha ($\alpha=32$), Dropout ($0.05$)
*   **Precision Type**: Native `BFloat16` processing (Brain Float)
*   **Optimization State**: `paged_adamw_8bit` (Offloads heavy memory tracking parameters safely to System RAM)
*   **Batching Optimization**: `per_device_train_batch_size=1` with `gradient_accumulation_steps=4` (Simulates a stable batch size of 4 via mathematical step accumulation)

---

## 🚀 Step-by-Step Local Deployment

### 1. Initialize and Activate Environment
```bash
python -m venv ai_philosophy_tutor
ai_philosophy_tutor\Scripts\activate
```

### 2. Clean-Install Core Hardware Dependencies
```bash
pip install torch --index-url https://pytorch.org
pip install transformers peft bitsandbytes accelerate datasets trl numpy
```

### 3. Initialize Training Run
To bypass Windows Jinja text template file-decoding conflicts and unlock high-speed downloading lanes via your Hugging Face access token, execute:
```powershell
set PYTHONUTF8=1  & set HF_TOKEN= your_huggingface_read_token
python train.py
```

### 4. Talk with Your Trained Tutor
```bash
python chat.py
```
*Note: Toggle the temperature flag in `chat.py` to `0.1` for strict factual accuracy audits, or `0.7` for open-ended, human-like fluid Socratic reasoning.*
