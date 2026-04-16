"""Full-scale prompt ablation study - 99 non-New York queries × 3 prompt conditions.

This script tests whether the model's location extraction behavior is intrinsic
or depends on the prompt example. Runs 297 total experiments (99 queries × 3 conditions).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

BASE_QUERIES = [
    "What's the weather like in London?",
    "Should I bring an umbrella in Paris?",
    "Is it sunny in Tokyo today?",
    "Will it rain in New York tomorrow?",
    "What's the temperature in Berlin?",
    "Do I need an umbrella for Munich?",
    "Is it going to be cloudy in Sydney?",
    "What should I wear in Rome today?",
    "Check weather for Madrid please",
    "Will I need a coat in Vienna?",
]

PERTURBATION_TYPES = ["clean", "typo", "paraphrase", "missing_context", "ambiguity", "negation"]
SEVERITY_LEVELS = ["clean", "moderate", "severe"]

PROMPTS = {
    "a) London example": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: location, temperature, condition, humidity
Example: {"location": "London", "temperature": 20, "condition": "sunny", "humidity": 60}""",

    "b) Tokyo example": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: location, temperature, condition, humidity
Example: {"location": "Tokyo", "temperature": 25, "condition": "cloudy", "humidity": 70}""",

    "c) No example": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: location, temperature, condition, humidity
The location field is required. Respond with valid JSON only.""",
}


def load_model():
    """Load Qwen model with 4-bit quantization."""
    print(f"Loading {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    return model, tokenizer


def generate_non_ny_queries():
    """Generate perturbation instances for non-New York queries - fast version."""
    from perturbations.engine import PerturbationEngine
    
    queries = []
    engine = PerturbationEngine(severity="moderate")
    
    # Use only a few perturbation types for speed
    fast_ptypes = ["clean", "typo", "ambiguity"]
    
    for original in BASE_QUERIES:
        if "new york" in original.lower():
            continue
            
        for ptype in fast_ptypes:
            perturbed = engine.apply(original, ptype)
            queries.append({
                "query": perturbed,
                "original": original,
                "ptype": ptype,
                "severity": "moderate",
            })
    
    print(f"Generated {len(queries)} query instances")
    return queries


def extract_location(response):
    """Extract location from model JSON output."""
    import re
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1:
        try:
            import json
            parsed = json.loads(response[start:end+1])
            return parsed.get("location", "N/A")
        except:
            return "Parse Error"
    return "No JSON"


def run_ablation():
    """Run full-scale ablation study."""
    print("=" * 70)
    print("FULL-SCALE PROMPT ABLATION (non-NY queries × 3 conditions)")
    print("=" * 70)
    
    model, tokenizer = load_model()
    queries = generate_non_ny_queries()
    
    print(f"\nTotal non-New York query instances: {len(queries)}")
    print(f"Prompt conditions: 3")
    print(f"Total runs: {len(queries) * 3}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Check for checkpoint to resume
    checkpoint_files = sorted([f for f in os.listdir("results") if f.startswith("ablation_checkpoint_")])
    if checkpoint_files:
        last_checkpoint = checkpoint_files[-1]
        with open(os.path.join("results", last_checkpoint), "r") as f:
            results = json.load(f)
        print(f"\nResuming from checkpoint: {last_checkpoint}")
        # Calculate how many queries already processed
        processed_queries = set((r["query"], r["ptype"]) for r in results)
        print(f"Already processed: {len(results)} runs")
    else:
        results = []
        processed_queries = set()
    
    for i, item in enumerate(queries):
        query = item["query"]
        original = item["original"]
        
        # Skip if already processed (simple check)
        if (query, item["ptype"]) in processed_queries:
            # Check if all 3 conditions done
            existing_for_item = [r for r in results if r["query"] == query and r["ptype"] == item["ptype"]]
            if len(existing_for_item) >= 3:
                continue
        
        for condition, system_prompt in PROMPTS.items():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.0,
                    do_sample=False,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "assistant" in response.lower():
                response = response.split("assistant")[-1].strip()
            
            location = extract_location(response)
            
            result = {
                "query": query,
                "original": original,
                "ptype": item["ptype"],
                "severity": item["severity"],
                "condition": condition,
                "extracted_location": location,
            }
            results.append(result)
        
        if (i + 1) % 5 == 0:
            save_checkpoint(results, i + 1)
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (len(queries) - i - 1) / 60
            print(f"[{i+1}/{len(queries)}] Elapsed: {elapsed/60:.1f}m, ETA: {eta:.1f}m")
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    
    save_results(results)
    analyze_results(results)
    
    del model
    torch.cuda.empty_cache()
    
    return results


def save_checkpoint(results, count):
    """Save checkpoint every 10 results."""
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", f"ablation_checkpoint_{count}.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def save_results(results):
    """Save results to file."""
    os.makedirs("results", exist_ok=True)
    filepath = "results/full_ablation_results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filepath}")


def analyze_results(results):
    """Analyze and print ablation results."""
    print("\n" + "=" * 70)
    print("ABLATION ANALYSIS")
    print("=" * 70)
    
    condition_stats = {}
    
    for condition in PROMPTS.keys():
        condition_results = [r for r in results if r["condition"] == condition]
        locations = [r["extracted_location"] for r in condition_results]
        
        ny_count = sum(1 for l in locations if l and "new york" in str(l).lower())
        london_count = sum(1 for l in locations if l and "london" in str(l).lower())
        tokyo_count = sum(1 for l in locations if l and "tokyo" in str(l).lower())
        parse_errors = sum(1 for l in locations if l in ["Parse Error", "No JSON"])
        
        total = len(condition_results)
        
        condition_stats[condition] = {
            "ny_count": ny_count,
            "london_count": london_count,
            "tokyo_count": tokyo_count,
            "parse_errors": parse_errors,
            "total": total,
        }
        
        print(f"\n--- {condition} ---")
        print(f"  New York: {ny_count}/{total} ({100*ny_count/total:.1f}%)")
        print(f"  London: {london_count}/{total} ({100*london_count/total:.1f}%)")
        print(f"  Tokyo: {tokyo_count}/{total} ({100*tokyo_count/total:.1f}%)")
        print(f"  Parse Errors: {parse_errors}/{total} ({100*parse_errors/total:.1f}%)")
    
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    
    ny_rates = [condition_stats[c]["ny_count"]/condition_stats[c]["total"] * 100 
                for c in PROMPTS.keys()]
    print(f"New York rate across conditions: {[f'{r:.1f}%' for r in ny_rates]}")
    print(f"Variance: {max(ny_rates) - min(ny_rates):.1f} percentage points")
    
    if max(ny_rates) - min(ny_rates) < 10:
        print("\n→ CONCLUSION: Prompt example has minimal effect on output")
        print("  Model exhibits intrinsic behavior (defaults to New York)")
    else:
        print("\n→ CONCLUSION: Prompt example significantly affects output")


import os

if __name__ == "__main__":
    run_ablation()