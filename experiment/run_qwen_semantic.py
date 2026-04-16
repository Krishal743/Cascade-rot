"""Experiment runner for Chain A with Qwen model - with semantic validation."""

import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from experiment.chain_runner import ChainRunner, print_trace
from tools.schemas import CHAIN_A
from perturbations.engine import PerturbationEngine
from perturbations.dataset import BASE_QUERIES, PERTURBATION_TYPES, SEVERITY_LEVELS


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_model():
    """Load Llama model with 4-bit quantization."""
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


def generate_full_dataset():
    """Generate all perturbed query instances."""
    engine = PerturbationEngine(severity="moderate")
    dataset = []
    
    for original in BASE_QUERIES:
        for ptype in PERTURBATION_TYPES:
            for severity in SEVERITY_LEVELS:
                # Skip invalid combinations
                if ptype == "clean" and severity != "clean":
                    continue
                if severity == "clean" and ptype != "clean":
                    continue
                
                # Apply perturbation
                perturbed = engine.apply(original, ptype)
                dataset.append({
                    "query": perturbed,
                    "original": original,
                    "ptype": ptype,
                    "severity": severity,
                })
    
    return dataset


def save_checkpoint(results, count, checkpoint_dir="results"):
    """Save checkpoint every 10 results."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, f"llama_checkpoint_{count}.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Checkpoint saved: {count} results -> {filepath}")


def run_full_dataset():
    """Run full dataset with checkpointing."""
    import time
    
    print("=" * 70)
    print("LLAMA EXPERIMENT RUN")
    print("=" * 70)
    
    # Load model once
    model, tokenizer = load_model()
    runner = ChainRunner(model, tokenizer, CHAIN_A)
    
    # Generate dataset
    dataset = generate_full_dataset()
    total = len(dataset)
    print(f"Total instances: {total}")
    print("=" * 70)
    
    results = []
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        query = item["query"]
        ptype = item["ptype"]
        severity = item["severity"]
        
        # Print progress
        print(f"\n[{i+1}/{total}] {ptype}/{severity}: {query[:50]}...")
        
        # Run through chain
        trace = runner.run(query)
        
        # Add metadata - include semantic validation
        result = {
            "query": query,
            "original": item["original"],
            "ptype": ptype,
            "severity": severity,
            "step1_success": trace["step1"]["success"],
            "step2_success": trace["step2"]["success"],
            "cascade_failure": trace["cascade_failure"],
            "step1_syntactic_success": trace["step1"].get("syntactic_success", False),
            "step1_semantic_success": trace["step1"].get("semantic_success", False),
            "step1_extracted_location": trace["step1"].get("extracted_location", None),
            "ground_truth_location": trace.get("ground_truth_location"),
            "semantic_cascade": trace.get("semantic_cascade", False),
        }
        results.append(result)
        
        # Checkpoint every 10
        if (i + 1) % 10 == 0:
            save_checkpoint(results, i + 1)
    
    # Save final results
    os.makedirs("results", exist_ok=True)
    final_path = "results/qwen_semantic_results.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("QWEN RESULTS SUMMARY (With Semantic Validation)")
    print("=" * 70)
    
    step1_success = sum(1 for r in results if r["step1_success"])
    step2_success = sum(1 for r in results if r["step2_success"])
    cascade_failures = sum(1 for r in results if r["cascade_failure"])
    
    # Semantic metrics
    syntactic_success = sum(1 for r in results if r.get("step1_syntactic_success", False))
    semantic_success = sum(1 for r in results if r.get("step1_semantic_success", False))
    silent_failures = syntactic_success - semantic_success  # Valid JSON but wrong location
    
    # Semantic cascade metrics
    semantic_cascades = sum(1 for r in results if r.get("semantic_cascade", False))
    
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    print(f"Step 1 success: {step1_success}/{total} ({100*step1_success/total:.1f}%)")
    print(f"Step 2 success: {step2_success}/{total} ({100*step2_success/total:.1f}%)")
    print(f"Cascade failures: {cascade_failures}/{total} ({100*cascade_failures/total:.1f}%)")
    print(f"\n--- SEMANTIC VALIDATION ---")
    print(f"Syntactic Success (valid JSON): {syntactic_success}/{total} ({100*syntactic_success/total:.1f}%)")
    print(f"Semantic Success (correct location): {semantic_success}/{total} ({100*semantic_success/total:.1f}%)")
    print(f"Silent Failures (valid JSON + wrong location): {silent_failures}/{total} ({100*silent_failures/total:.1f}%)")
    print(f"\n--- SEMANTIC CASCADE (DANGEROUS) ---")
    print(f"Semantic Cascades: {semantic_cascades}/{total} ({100*semantic_cascades/total:.1f}%)")
    print(f"(Hallucinated location propagated to Step 2, user receives wrong but confident advice)")
    print(f"\nCascade Failure Rate (CFR): {100*cascade_failures/total:.2f}%")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    run_full_dataset()