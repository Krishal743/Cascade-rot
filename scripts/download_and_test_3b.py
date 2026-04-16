#!/usr/bin/env python3
"""
Download and verify Qwen2.5-3B-Instruct on 6GB VRAM system.
Run this after cleaning up the 7B model.
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration for 6GB VRAM
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def check_gpu():
    """Verify GPU is available and has sufficient memory."""
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
    """Download model with 4-bit quantization."""
    print(f"\nDownloading {MODEL_NAME}...")
    print("This will take 5-15 minutes depending on internet speed.")
    print("Files are cached at:", CACHE_DIR)
    
    # 4-bit quantization config - critical for 6GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # Saves additional memory
    )
    
    try:
        # Download tokenizer first (small, fast)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        # Download and load model with quantization
        print("Loading model (this downloads weights)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically places layers on GPU/CPU as needed
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        print("✅ Model loaded successfully")
        
        # Check memory usage
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM Allocated: {memory_allocated:.2f} GB")
        print(f"VRAM Reserved: {memory_reserved:.2f} GB")
        
        if memory_allocated > 5.0:
            print("⚠️  Warning: Using more than 5GB. Generation may OOM.")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def test_json_output(model, tokenizer):
    """Test if model can produce valid JSON."""
    print("\nTesting JSON output capability...")
    
    # JSON forcing prompt
    system_msg = """You are a helpful assistant that always responds with valid JSON.
Your response must be a single JSON object with no markdown formatting, no explanations, and no code blocks.
    
Required format: {"city": string, "temperature": number, "unit": string}
    
Respond only with the JSON object:"""
    
    test_prompt = "What is the weather in London?"
    
    # Construct full prompt in Qwen chat format
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": test_prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Generate with deterministic settings
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
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nRaw response:\n{response}")
    
    # Extract JSON
    try:
        # Find JSON boundaries
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = response[start:end]
        parsed = json.loads(json_str)
        
        print(f"\nParsed JSON: {parsed}")
        
        # Validate structure
        required_keys = {
            "city": str,
            "temperature": (int, float),
            "unit": str
        }
        
        for key, expected_type in required_keys.items():
            if key not in parsed:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(parsed[key], expected_type):
                raise ValueError(f"Wrong type for {key}: got {type(parsed[key])}, expected {expected_type}")
        
        print("✅ JSON validation passed")
        print(f"City: {parsed['city']}")
        print(f"Temperature: {parsed['temperature']}")
        print(f"Unit: {parsed['unit']}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON validation failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Qwen2.5-3B-Instruct Download and Verification")
    print("=" * 60)
    
    # Step 1: Check GPU
    check_gpu()
    
    # Step 2: Download and load model
    model, tokenizer = download_and_load_model()
    
    # Step 3: Test JSON output
    success = test_json_output(model, tokenizer)
    
    # Cleanup
    print("\nCleaning up GPU memory...")
    del model
    torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("✅ PHASE 1 COMPLETE: Qwen2.5-3B ready for use")
        print("VRAM usage is within 6GB limit")
        print("JSON output is functional")
        sys.exit(0)
    else:
        print("❌ JSON test failed - model loads but needs prompt tuning")
        sys.exit(1)

if __name__ == "__main__":
    main()