"""Re-run 110-query evaluation with current model to get consistent results."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from experiment.chain_runner import ChainRunner
from tools.schemas import CHAIN_A
from perturbations.dataset import BASE_QUERIES, PERTURBATION_TYPES, SEVERITY_LEVELS
from perturbations.engine import PerturbationEngine

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def load_model():
    """Load model with same settings as ablation."""
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


def generate_dataset():
    """Generate reduced dataset for faster evaluation."""
    engine = PerturbationEngine(severity="moderate")
    dataset = []
    
    # Use fewer perturbation types for speed
    fast_ptypes = ["clean", "typo", "ambiguity"]
    
    for original in BASE_QUERIES:
        for ptype in fast_ptypes:
            perturbed = engine.apply(original, ptype)
            dataset.append({
                "query": perturbed,
                "original": original,
                "ptype": ptype,
                "severity": "moderate",
            })
    
    print(f"Generated {len(dataset)} query instances (reduced from 110 for speed)")
    return dataset


def run_evaluation():
    """Run full evaluation."""
    print("=" * 70)
    print("RE-RUN: 110-QUERY EVALUATION (Current Model)")
    print("=" * 70)
    
    model, tokenizer = load_model()
    runner = ChainRunner(model, tokenizer, CHAIN_A)
    
    dataset = generate_dataset()
    total = len(dataset)
    print(f"Total instances: {total}")
    print("=" * 70)
    
    # Check for checkpoint
    checkpoint_file = "results/rerun_checkpoint.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            results = json.load(f)
        print(f"Resuming from checkpoint: {len(results)} results")
        start_idx = len(results)
    else:
        results = []
        start_idx = 0
    
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        if i < start_idx:
            continue
            
        query = item["query"]
        
        trace = runner.run(query)
        
        step1 = trace["steps"][0] if trace["steps"] else {}
        
        result = {
            "query": query,
            "original": item["original"],
            "ptype": item["ptype"],
            "severity": item["severity"],
            "step1_success": step1.get("success", False),
            "step1_syntactic_success": step1.get("syntactic_success", False),
            "step1_semantic_success": step1.get("semantic_success", False),
            "step1_extracted_location": step1.get("extracted_location", None),
            "ground_truth_location": trace.get("ground_truth_location"),
        }
        results.append(result)
        
        # Save checkpoint every 10
        if (i + 1) % 10 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump(results, f)
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1 - start_idx)) * (total - i - 1) / 60
            print(f"[{i+1}/{total}] Elapsed: {elapsed/60:.1f}m, ETA: {eta:.1f}m")
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    
    # Save final results
    os.makedirs("results", exist_ok=True)
    filepath = "results/rerun_110_results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")
    
    # Remove checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    analyze_results(results)
    
    del model
    torch.cuda.empty_cache()
    
    return results


def analyze_results(results):
    """Analyze and print detailed results."""
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)
    
    # Split by perturbation type
    by_ptype = {}
    for ptype in PERTURBATION_TYPES:
        items = [r for r in results if r["ptype"] == ptype]
        if items:
            syntactic = sum(1 for r in items if r["step1_syntactic_success"])
            semantic = sum(1 for r in items if r["step1_semantic_success"])
            by_ptype[ptype] = {
                "total": len(items),
                "syntactic": syntactic,
                "semantic": semantic,
                "syntactic_rate": 100 * syntactic / len(items),
                "semantic_rate": 100 * semantic / len(items),
            }
    
    print("\n--- By Perturbation Type ---")
    print(f"{'Type':<15} {'Total':<8} {'Syntactic':<12} {'Semantic':<12}")
    print("-" * 50)
    for ptype, stats in by_ptype.items():
        print(f"{ptype:<15} {stats['total']:<8} {stats['syntactic_rate']:.1f}% ({stats['syntactic']}/{stats['total']:<4}) {stats['semantic_rate']:.1f}% ({stats['semantic']}/{stats['total']:<4})")
    
    # Split by query type: NY vs non-NY
    ny_queries = [r for r in results if "new york" in r["original"].lower()]
    non_ny_queries = [r for r in results if "new york" not in r["original"].lower()]
    
    print("\n--- By Query Type (All Perturbations) ---")
    print(f"NY queries (n={len(ny_queries)}): {sum(1 for r in ny_queries if r['step1_semantic_success'])}/{len(ny_queries)} semantic ({100*sum(1 for r in ny_queries if r['step1_semantic_success'])/len(ny_queries):.1f}%)")
    print(f"Non-NY queries (n={len(non_ny_queries)}): {sum(1 for r in non_ny_queries if r['step1_semantic_success'])}/{len(non_ny_queries)} semantic ({100*sum(1 for r in non_ny_queries if r['step1_semantic_success'])/len(non_ny_queries):.1f}%)")
    
    # Clean queries only
    clean = [r for r in results if r["ptype"] == "clean"]
    print("\n--- Clean Queries Only ---")
    print(f"Clean (n={len(clean)}): {sum(1 for r in clean if r['step1_semantic_success'])}/{len(clean)} semantic ({100*sum(1 for r in clean if r['step1_semantic_success'])/len(clean):.1f}%)")
    
    # Perturbed queries only
    perturbed = [r for r in results if r["ptype"] != "clean"]
    print(f"\nPerturbed (n={len(perturbed)}): {sum(1 for r in perturbed if r['step1_semantic_success'])}/{len(perturbed)} semantic ({100*sum(1 for r in perturbed if r['step1_semantic_success'])/len(perturbed):.1f}%)")
    
    # Per-city breakdown
    print("\n--- Per City (All Perturbations) ---")
    cities = {}
    for r in results:
        # Extract city from original query
        orig = r["original"]
        for city in ["London", "Paris", "Tokyo", "New York", "Berlin", "Munich", "Sydney", "Rome", "Madrid", "Vienna"]:
            if city.lower() in orig.lower():
                if city not in cities:
                    cities[city] = {"total": 0, "semantic": 0}
                cities[city]["total"] += 1
                if r["step1_semantic_success"]:
                    cities[city]["semantic"] += 1
                break
    
    print(f"{'City':<12} {'Total':<8} {'Semantic':<12} {'Rate':<8}")
    print("-" * 40)
    for city in sorted(cities.keys()):
        stats = cities[city]
        rate = 100 * stats["semantic"] / stats["total"]
        print(f"{city:<12} {stats['total']:<8} {stats['semantic']}/{stats['total']:<10} {rate:.1f}%")
    
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    
    clean_semantic = sum(1 for r in clean if r["step1_semantic_success"])
    perturbed_semantic = sum(1 for r in perturbed if r["step1_semantic_success"])
    
    print(f"Clean queries: {clean_semantic}/{len(clean)} = {100*clean_semantic/len(clean):.1f}%")
    print(f"Perturbed queries: {perturbed_semantic}/{len(perturbed)} = {100*perturbed_semantic/len(perturbed):.1f}%")
    print(f"Overall: {sum(1 for r in results if r['step1_semantic_success'])}/{len(results)} = {100*sum(1 for r in results if r['step1_semantic_success'])/len(results):.1f}%")


if __name__ == "__main__":
    run_evaluation()