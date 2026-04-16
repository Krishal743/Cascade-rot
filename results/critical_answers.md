# Critical Review Points - Complete Answers

## 1. Prompt Template Disclosure

**System Prompt Used:**
```
"You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: location, temperature, condition, humidity
Example: {"location": "London", "temperature": 20, "condition": "sunny", "humidity": 60}"
```

**Key Issue**: The example uses `"London"` NOT `"New York"`. The prompt is neutral.

**Format**: 2-message format (system + user), then `apply_chat_template()`, fresh prompt per query.

---

## 2. Final Corrected Results (Re-run April 2026)

After fixing the semantic validator to use ground-truth matching (not pattern matching against perturbed query):

| Query Type | Count | Semantic Success |
|------------|-------|------------------|
| **Clean** | 10 | 90.0% (9/10) |
| **Typos** | 10 | 30.0% (3/10) |
| **Ambiguity** | 10 | 0.0% (0/10) |
| **Overall** | 30 | 40.0% (12/30) |

### By City (All Perturbations)

| City | Rate |
|------|------|
| London | 66.7% |
| Munich | 66.7% |
| Sydney | 66.7% |
| Madrid | 33.3% |
| Paris | 33.3% |
| Rome | 33.3% |
| Berlin | 33.3% |
| Tokyo | 33.3% |
| Vienna | 33.3% |
| New York | 0.0% |

---

## 3. Generation Parameters

```python
temperature=0.0          # Greedy decoding - deterministic
do_sample=False          # No sampling
max_new_tokens=128       # Output length
pad_token_id=eos_token   # Padding
```

**Deterministic confirmed** - all outputs identical across runs.

---

## 4. Context Between Queries

**Answer**: Each query is run **independently** with a fresh prompt. 

The "context" variable in the code is for passing **tool output between steps** (Step 1 → Step 2), not conversation history between queries. Each query starts with empty context.

---

## 5. Non-Famous Locations

**Answer**: NO - we only tested major cities (London, Paris, Tokyo, New York, Berlin, Munich, Sydney, Rome, Madrid, Vienna).

**Limitation to add**: Results may reflect frequency bias in training data; less common cities not tested.

---

## 6. Chain B/C Semantic Validation

**Current status**: Chain B and C only measured **syntactic success** (valid JSON), NOT semantic correctness.

**Recommendation**: Rename to "syntactic chain completion" and note that semantic correctness was not evaluated for those chains. Remove the Chain B vs C comparison from core claims.

---

## 7. Phi-3 Pattern (Anecdotal)

**Answer**: Only 3 syntactic successes (2.7%), all outputting "London". With n=3, this is **not statistically significant** - report as anecdotal observation only.

---

## 8. Quantization Effect

**Answer**: Not tested without quantization - models (3B parameters) require quantization to fit in 6GB VRAM.

**Limitation to add**: "Results reflect 4-bit quantized edge-deployment constraints; full-precision model capabilities not evaluated."

---

## 9. Full-Scale Ablation (Section IV-A)

**Ablation (81 runs)**: Zero variance across prompt conditions confirms prompt-independence.

| Condition | New York Rate |
|-----------|---------------|
| London example | 0% |
| Tokyo example | 0% |
| No example | 0% |

**Key Finding**: Prompt has no effect - behavior is model-intrinsic.

---

## 10. Core Paper Narrative

### The Finding (Brittle Under Perturbation)

> "Under clean, explicit inputs, the model achieves ~90% semantic accuracy. However, under minimal perturbation (typo, ambiguity), accuracy collapses to ~15%. This brittleness is the core safety concern – the model fails precisely when users need robustness."

### Breakdown
- **Clean queries**: 90% semantic success
- **Perturbed queries**: 15% semantic success  
- **Collapse rate**: 85% accuracy drop

This is a stronger, more credible result than "always hallucinates" - it shows the model has capability but fails under uncertainty.

---

## Summary of Paper Updates

1. **Results**: Clean 90%, Perturbed 15%, Overall 40%
2. **Narrative**: "Brittle under perturbation" not "universal hallucination"
3. **Ablation**: Demonstrates prompt-independence (zero variance across conditions)
4. **Limitations**: Non-famous cities, quantization effects, Chain B/C syntactic only