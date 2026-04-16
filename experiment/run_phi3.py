"""Experiment runner for Chain A with Phi-3 model."""

import json
import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from experiment.chain_runner import ChainRunner
from tools.schemas import CHAIN_A
from perturbations.engine import PerturbationEngine
from perturbations.dataset import BASE_QUERIES, PERTURBATION_TYPES, SEVERITY_LEVELS


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


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


def generate_full_dataset():
    engine = PerturbationEngine(severity="moderate")
    dataset = []
    
    for original in BASE_QUERIES:
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


def run_phi3_experiment():
    print("=" * 70)
    print("PHI-3 EXPERIMENT (Chain A - Weather -> Umbrella)")
    print("=" * 70)
    
    model, tokenizer = load_model()
    runner = ChainRunner(model, tokenizer, CHAIN_A)
    
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
        
        print(f"\n[{i+1}/{total}] {ptype}/{severity}: {query[:50]}...")
        
        trace = runner.run(query, chain_name="chain_a")
        
        result = {
            "query": query,
            "original": item["original"],
            "ptype": ptype,
            "severity": severity,
            "step1_success": trace["steps"][0]["success"] if trace["steps"] else False,
            "step2_success": trace["steps"][1]["success"] if len(trace["steps"]) > 1 else False,
            "cascade_failure": trace["cascade_failure"],
            "step1_syntactic_success": trace["steps"][0]["syntactic_success"] if trace["steps"] else False,
            "step1_semantic_success": trace["steps"][0].get("semantic_success", False),
            "step1_extracted_location": trace["steps"][0].get("extracted_location", None),
            "ground_truth_location": trace.get("ground_truth_location"),
            "semantic_cascade": trace.get("semantic_cascade", False),
        }
        results.append(result)
        
        if (i + 1) % 10 == 0:
            os.makedirs("results", exist_ok=True)
            with open(f"results/phi3_checkpoint_{i+1}.json", "w") as f:
                json.dump(results, f, indent=2)
    
    elapsed = time.time() - start_time
    
    # Save final results
    os.makedirs("results", exist_ok=True)
    final_path = "results/phi3_results.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Calculate metrics
    step1_success = sum(1 for r in results if r["step1_success"])
    step2_success = sum(1 for r in results if r["step2_success"])
    cascade_failures = sum(1 for r in results if r["cascade_failure"])
    
    syntactic_success = sum(1 for r in results if r.get("step1_syntactic_success", False))
    semantic_success = sum(1 for r in results if r.get("step1_semantic_success", False))
    silent_failures = syntactic_success - semantic_success
    
    semantic_cascades = sum(1 for r in results if r.get("semantic_cascade", False))
    
    print("\n" + "=" * 70)
    print("PHI-3 RESULTS SUMMARY")
    print("=" * 70)
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
    print(f"\nCascade Failure Rate (CFR): {100*cascade_failures/total:.2f}%")
    
    del model
    torch.cuda.empty_cache()
    
    return results


if __name__ == "__main__":
    run_phi3_experiment()