#!/usr/bin/env python3
"""
Download and verify Phi-3-mini-4k-instruct on 6GB VRAM system.
Phi-3 uses a different chat format than Qwen - simpler text-based template.
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configuration for 6GB VRAM
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # No HF token needed
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
        print("❌ Less than 5.5GB VRAM detected.")
        sys.exit(1)
    
    print("✅ GPU check passed")
    return True

def download_and_load_model():
    """Download model with 4-bit quantization."""
    print(f"\nDownloading {MODEL_NAME}...")
    print("This will take 5-15 minutes depending on internet speed.")
    
    # Same 4-bit config as Qwen
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Set pad token if missing (Phi-3 specific fix)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("Loading model (this downloads weights)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            # No trust_remote_code needed for Phi-3
        )
        
        print("✅ Model loaded successfully")
        
        # Check memory
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM Allocated: {memory_allocated:.2f} GB")
        
        if memory_allocated > 5.0:
            print("⚠️  Warning: Using more than 5GB.")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def test_json_output(model, tokenizer):
    """Test if Phi-3 can produce valid JSON."""
    print("\nTesting JSON output capability...")
    
    # Phi-3 uses simpler text format (not chat template like Qwen)
    # Phi-3 expects: <|user|>\nPrompt<|end|>\n<|assistant|>\n
    system_msg = """You must respond with ONLY valid JSON. No explanations.
Format: {"city": string, "temperature": number, "unit": string}"""
    
    test_prompt = "What is the weather in London?"
    
    # Phi-3 format
    full_prompt = f"<|user|>\n{system_msg}\n{test_prompt}<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nRaw response:\n{response}")
    
    # Extract JSON
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON object found")
        
        json_str = response[start:end]
        parsed = json.loads(json_str)
        
        print(f"\nParsed JSON: {parsed}")
        
        # Validate
        assert isinstance(parsed.get("city"), str), "city must be string"
        assert isinstance(parsed.get("temperature"), (int, float)), "temp must be number"
        assert isinstance(parsed.get("unit"), str), "unit must be string"
        
        print("✅ JSON validation passed")
        return True
        
    except Exception as e:
        print(f"❌ JSON validation failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Phi-3-mini-4k-instruct Download and Verification")
    print("=" * 60)
    
    check_gpu()
    model, tokenizer = download_and_load_model()
    success = test_json_output(model, tokenizer)
    
    print("\nCleaning up GPU memory...")
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ PHASE 1 COMPLETE: Phi-3 ready for use")
        print("VRAM usage is within 6GB limit")
        sys.exit(0)
    else:
        print("❌ JSON test failed - model loads but needs prompt tuning")
        sys.exit(1)

if __name__ == "__main__":
    main()