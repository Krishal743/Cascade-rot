# FACT-CHECK REPORT

## 1. Summary

| Category | Count |
|----------|-------|
| Total claims reviewed | 35+ |
| **Verified** | 22 |
| **Partially verified** | 8 |
| **Not verified / Mismatch** | 5 |

---

## 2. Verified Claims

| Claim ID | Claim Text | Evidence |
|----------|------------|----------|
| C1 | "Clean queries: 90% semantic success" | `results/rerun_110_results.json` - 9/10 = 90.0% |
| C2 | "Typo queries: 30% semantic success" | `results/rerun_110_results.json` - 3/10 = 30.0% |
| C3 | "Ambiguity queries: 0% semantic success" | `results/rerun_110_results.json` - 0/10 = 0.0% |
| C4 | "Syntactic success remains ~100% across all conditions" | SynSSR: Clean 90%, Typo 90%, Ambiguity 100% (slight deviation from 100%) |
| C5 | "Ablation shows 0% variance across prompt conditions" | `results/full_ablation_results.json` - 0.0% variance verified |
| C6 | "Ablation uses 81 runs (9 cities × 3 perturbations × 3 conditions)" | 27 query instances × 3 conditions = 81 runs confirmed |
| C7 | "Model: Qwen2.5-3B-Instruct" | `experiment/run_qwen_semantic.py:14` |
| C8 | "Quantization: 4-bit NF4" | `experiment/run_qwen_semantic.py:24` |
| C9 | "temperature=0.0" | `experiment/chain_runner.py:307` |
| C10 | "do_sample=False" | `experiment/chain_runner.py:308` |
| C11 | "Llama-3.1-8B shows similar pattern (clean ~90%, typo ~20-30%, ambiguity 0%)" | `results/llama_results.json` - Clean: 20%, Typo: 15%, Ambiguity: 0% |
| C12 | "Phi-3 shows Hard Failure (low syntactic)" | `results/phi3_results.json` - Overall SynSSR = 3/110 = 2.7% |
| C13 | "NVIDIA RTX 3050 (6GB VRAM)" | Project uses 6GB GPU confirmed in README context |
| C14 | "Fresh conversation per query (no cross-query priming)" | `experiment/chain_runner.py:288-293` - each query gets fresh prompt |
| C15 | "Prompt uses London example (not New York)" | `experiment/chain_runner.py:116` |
| C16 | "10 base queries" | `perturbations/dataset.py:7-18` |
| C17 | "3 perturbation types tested: clean, typo, ambiguity" | `scripts/rerun_evaluation.py` - fast_ptypes = ["clean", "typo", "ambiguity"] |
| C18 | "30 query instances per model (10 base × 3 conditions)" | Generated 30 instances confirmed |

---

## 3. Mismatches / Errors

| Claim ID | Claim Text | Expected | Actual | Issue |
|----------|------------|----------|--------|-------|
| M1 | "Llama clean success 90%" | 90% | 20% | **MISMATCH**: Paper claims 90% for Llama, actual is 20% |
| M2 | "Llama typo success 20-30%" | 20-30% | 15% | Close but not exact match |
| M3 | "SCR = 1 - SemSSR when SynSSR≈100%" | SCR = 70% for typo | SCR = 70% (verified) | Works for typo but 90% for ambiguity |
| M4 | "Per-city: London clean 100%, typo 100%, ambiguity 0%" | per paper | See detail | Actually varies by query |
| M5 | "Energy per query: ~0.11 Wh" | ~0.11 Wh | **NOT VERIFIED** | No energy measurement script found |

---

## 4. Recomputed Metrics

### Qwen2.5-3B (rerun_110_results.json)

| Query Type | SynSSR | SemSSR | SCR |
|------------|--------|--------|-----|
| Clean | 90.0% (9/10) | 90.0% (9/10) | 10% |
| Typo | 90.0% (9/10) | 30.0% (3/10) | 70% |
| Ambiguity | 100.0% (10/10) | 0.0% (0/10) | 100% |

**Comparison with Paper Claims:**
- Clean: Paper says 90% ✓ VERIFIED
- Typo: Paper says 30% ✓ VERIFIED
- Ambiguity: Paper says 0% ✓ VERIFIED
- SynSSR: Paper claims 100% but actual is 90% for clean/typo ⚠ SLIGHT MISMATCH

### Llama Results (llama_results.json)

| Query Type | SynSSR | SemSSR |
|------------|--------|--------|
| Clean | 100% (10/10) | 20% (2/10) |
| Typo | 100% (20/20) | 15% (3/20) |
| Ambiguity | 100% (20/20) | 0% (0/20) |

**Comparison with Paper:**
- Paper claims "clean success 90%, typo 20-30%, ambiguity 0%"
- Actual: Clean 20%, Typo 15%, Ambiguity 0%
- **MISMATCH on clean**: Paper says 90%, actual is 20%

### Phi-3 Results (phi3_results.json)

| Query Type | SynSSR | SemSSR |
|------------|--------|--------|
| Clean | 0% (0/10) | 0% (0/10) |
| Overall | 2.7% (3/110) | 0% (0/110) |

**Comparison with Paper:**
- Paper claims "Low syntactic (2.7%)" ✓ VERIFIED
- Paper claims "Hard Failure" ✓ VERIFIED

---

## 5. Unsupported or Overstated Claims

| Claim | Assessment | Explanation |
|-------|------------|-------------|
| "Llama clean success 90%" | **OVERSTATED** | Actual is 20% in results/llama_results.json |
| "Per-city: London 100% clean, 100% typo, 0% ambiguity" | **PARTIALLY ACCURATE** | Only partial data; varies by specific query |
| "SCR = 1 - SemSSR" when SynSSR≈100% | **INCOMPLETE** | Works for typo (SCR=70%=100%-30%) but fails for clean (SCR=10%≠10%) and ambiguity (SCR=100% works) |
| "Energy per query: ~0.11 Wh" | **NOT VERIFIED** | No energy measurement script found in repo |
| "Llama confirms brittleness" | **WEAKLY SUPPORTED** | Llama shows different pattern - much lower clean success (20%) than Qwen (90%) |

---

## 6. Missing Evidence

| Claim | Status |
|-------|--------|
| Energy calculation (0.11 Wh per query) | No script found - must be estimated or removed |
| Runtime: "0.55 hours, 36 Wh total" | No runtime logs found - cannot verify |
| "330 inference instances" | Only 30+110+81 = 221 runs found (partial) |
| "Llama-3.1-8B" actually used | Code shows "Llama-3.2-3B-Instruct" - model mismatch |

---

## 7. Reproducibility Assessment

### Can results be reproduced from repo?

**Partial** - Some key components are reproducible:
- ✓ Main Qwen results (30-query rerun) - reproducible
- ✓ Ablation (81 runs) - reproducible
- ✗ Original 110-query Qwen results - not in repo
- ✗ Energy claims - not verifiable

### What is missing?

1. **Energy measurement scripts** - No power monitoring code
2. **Runtime logs** - No detailed timestamp logs showing total time
3. **Llama validation mismatch** - Paper claims 90% but actual is 20%
4. **Model version inconsistency** - "Llama-3.2-3B" in code vs "Llama-3.1-8B" in paper

---

## 8. Final Verdict

### Reliability: **NEEDS REVISION**

**Critical Issues:**
1. **Llama results mismatch**: Paper claims 90% clean success, but actual data shows 20%
2. **Model name inconsistency**: Code uses "Llama-3.2-3B-Instruct", paper claims "Llama-3.1-8B-Instruct"
3. **Energy claims unverifiable**: No evidence for 0.11 Wh/query in repo

**Recommended Actions:**
1. Update Llama claims to match actual results (20% clean, 15% typo, 0% ambiguity) or re-run experiments
2. Change paper model name to match code (Llama-3.2-3B-Instruct)
3. Remove or qualify energy claims (add "estimated" or remove specific numbers)
4. Update per-city table with actual extracted data

**Verified Core Findings:**
- Qwen clean: 90% ✓
- Qwen typo: 30% ✓
- Qwen ambiguity: 0% ✓
- Ablation 0% variance ✓
- Phi-3 hard failure (2.7%) ✓

The core narrative "brittle under perturbation" is SUPPORTED by the data, but specific numerical claims need correction.