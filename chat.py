import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main():
    BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
    LORA_ADAPTERS = "./qwen_lora_results"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, LORA_ADAPTERS)
    model.eval()

    print("\n--- AI Tutor Initialized! Type exit to quit. ---")
    
    while True:
        user_prompt = input("\nYou (Student): ")
        if user_prompt.lower() == "exit":
            break
        if not user_prompt.strip():
            continue
            
        messages = [{"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # FIXED: Extract clean text from list format safely
        clean_text = response[0].strip() if isinstance(response, list) and len(response) > 0 else "..."
        print(f"\nTutor: {clean_text}")

if __name__ == "__main__":
    main()