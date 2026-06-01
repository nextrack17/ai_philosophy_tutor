import torch
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"  
    OUTPUT_DIR = "./qwen_lora_results"
    DATASET_PATH = "C:\\Users\\divya\\ai_philosophy_tutor\\tutor_dataset.json"

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Please confirm your dataset file is saved at: {DATASET_PATH}")
    
    print("Loading local ShareGPT JSON dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    print("Loading tokenizer and quantized base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token  

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def format_sharegpt_to_chat(example):
        text = ""
        # Access the list of message turns directly for this specific row
        for turn in example['conversations']:
            role = "user" if turn['from'] == "human" else "assistant"
            text += f"<|im_start|>{role}\n{turn['value']}<|im_end|>\n"
        return text

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,       
        gradient_accumulation_steps=4,       
        learning_rate=2e-4,
        fp16=False,                          
        bf16=True,                           
        logging_steps=1,
        num_train_epochs=4,                  
        optim="paged_adamw_8bit",            
        max_length=256,  
        packing=False,                    
        report_to="none"                     
    )

    print("Initializing TRL SFTTrainer with ShareGPT formatting...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,             
        formatting_func=format_sharegpt_to_chat,
    )

    print("Starting training on your GPU...")
    trainer.train()

    print(f"Saving trained tutor LoRA adapters to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    print("Training finished successfully!")

if __name__ == "__main__":
    main()
