"""Experiment runner for Chain B (search -> extract -> summarize)."""

import json
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from experiment.chain_runner import ChainRunner
from tools.schemas import CHAIN_B
from perturbations.engine import PerturbationEngine
from perturbations.dataset import CHAIN_B_QUERIES, PERTURBATION_TYPES, SEVERITY_LEVELS


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_model():
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


def generate_test_dataset(num_queries: int = 5) -> list:
    """Generate test dataset - 50 instances (5 queries × 10 perturbation combos)."""
    engine = PerturbationEngine(severity="moderate")
    dataset = []
    
    selected_queries = CHAIN_B_QUERIES[:num_queries]
    
    for original in selected_queries:
        for ptype in PERTURBATION_TYPES:
            for severity in SEVERITY_LEVELS:
                if ptype == "clean" and severity != "clean":
                    continue
                if severity == "clean" and ptype != "clean":
                    continue
                
                perturbed = engine.apply(original, ptype)
                dataset.append({
                    "query": perturbed,
                    "original": original,
                    "ptype": ptype,
                    "severity": severity,
                })
    
    return dataset


def run_chain_b_experiment():
    print("=" * 70)
    print("CHAIN B EXPERIMENT (Search -> Extract -> Summarize)")
    print("=" * 70)
    
    model, tokenizer = load_model()
    runner = ChainRunner(model, tokenizer, CHAIN_B)
    
    dataset = generate_test_dataset(num_queries=5)
    total = len(dataset)
    print(f"Total instances: {total}")
    print("=" * 70)
    
    results = []
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        query = item["query"]
        ptype = item["ptype"]
        severity = item["severity"]
        
        print(f"\n[{i+1}/{total}] {ptype}/{severity}: {query[:50]}...")
        
        trace = runner.run(query, chain_name="chain_b")
        
        # Calculate metrics
        step_successes = [s["success"] for s in trace["steps"]]
        syntactic_successes = [s["syntactic_success"] for s in trace["steps"]]
        
        result = {
            "query": query,
            "original": item["original"],
            "ptype": ptype,
            "severity": severity,
            "chain": "chain_b",
            "total_steps": len(trace["steps"]),
            "all_steps_successful": all(step_successes),
            "syntactic_success_rate": sum(syntactic_successes) / len(syntactic_successes) if syntactic_successes else 0,
            "cascade_failure": trace["cascade_failure"],
            "steps": [
                {
                    "step": s["step"],
                    "tool": s["tool"],
                    "success": s["success"],
                    "syntactic": s["syntactic_success"]
                }
                for s in trace["steps"]
            ]
        }
        results.append(result)
    
    elapsed = time.time() - start_time
    
    # Summary
    all_successful = sum(1 for r in results if r["all_steps_successful"])
    any_failure = sum(1 for r in results if r["cascade_failure"])
    
    print("\n" + "=" * 70)
    print("CHAIN B RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {elapsed/60:.1f} minutes")
    print(f"Total instances: {total}")
    print(f"All steps successful: {all_successful}/{total} ({100*all_successful/total:.1f}%)")
    print(f"Any cascade failure: {any_failure}/{total} ({100*any_failure/total:.1f}%)")
    
    # Step-by-step analysis
    print("\n--- Step-by-Step Success Rates ---")
    for step_idx in range(3):
        step_successes = [r["steps"][step_idx]["success"] for r in results if len(r["steps"]) > step_idx]
        step_syntactic = [r["steps"][step_idx]["syntactic"] for r in results if len(r["steps"]) > step_idx]
        print(f"Step {step_idx+1}: Syntactic={sum(step_syntactic)}/{len(step_syntactic)} ({100*sum(step_syntactic)/len(step_syntactic):.1f}%), Success={sum(step_successes)}/{len(step_successes)} ({100*sum(step_successes)/len(step_successes):.1f}%)")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/chain_b_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/chain_b_results.json")
    
    del model
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    run_chain_b_experiment()