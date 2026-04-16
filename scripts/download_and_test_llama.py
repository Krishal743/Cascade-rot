#!/usr/bin/env python3
"""
Download and verify Llama-3.2-3B-Instruct on 6GB VRAM system.
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ✅ CHANGED MODEL
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def check_gpu():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Exiting.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_memory:.2f} GB")
    
    if total_memory < 5.5:
        print("❌ Less than 5.5GB VRAM detected. Model may not fit.")
        sys.exit(1)
    
    print("✅ GPU check passed")
    return True

def download_and_load_model():
    print(f"\nDownloading {MODEL_NAME}...")
    print("Files are cached at:", CACHE_DIR)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        print("✅ Model loaded successfully")
        
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM Allocated: {memory_allocated:.2f} GB")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def test_json_output(model, tokenizer):
    print("\nTesting JSON output capability...")
    
    system_msg = """You are a helpful assistant that always responds with valid JSON.
Your response must be a single JSON object with no markdown formatting.

Required format: {"city": string, "temperature": number, "unit": string}
Respond only with the JSON object:"""
    
    test_prompt = "What is the weather in London?"
    
    # ✅ FIXED CHAT FORMAT FOR LLAMA
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": test_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # ✅ PAD TOKEN FIX
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"\nRaw response:\n{response}")
    
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        
        json_str = response[start:end]
        parsed = json.loads(json_str)
        
        print(f"\nParsed JSON: {parsed}")
        print("✅ JSON validation passed")
        return True
        
    except Exception as e:
        print(f"❌ JSON validation failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Llama-3.2-3B-Instruct Download and Verification")
    print("=" * 60)
    
    check_gpu()
    model, tokenizer = download_and_load_model()
    success = test_json_output(model, tokenizer)
    
    print("\nCleaning up GPU memory...")
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Llama-3.2-3B ready for use")
        sys.exit(0)
    else:
        print("❌ JSON test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()