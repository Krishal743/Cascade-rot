# Updated Statistical Analysis

## Corrected Numbers (After Bug Fix)

| Model | Buggy Count | Corrected Count | Rate |
|-------|-------------|-----------------|------|
| Qwen | 14/110 | **11/110** | **10.0%** |
| Llama | 15/110 | **11/110** | **10.0%** |

## Why Identical Rates?

Both models achieve 10.0% because:
- Both succeeded on the **11 instances** where "New York" was explicitly mentioned in the query
- Both failed (output New York) on all other queries regardless of the actual location asked

This is documented in the validator - see `results/critical_answers.md` for full analysis.

---

## Phi-3 Anomaly Note

The 3 syntactic successes (n=3) all output "London" - notably the same location as the prompt example. This suggests:
- **Prompt bias** for Phi-3: when it occasionally produces valid JSON, it defaults to the example token
- **Training-frequency bias** for Qwen/Llama: default to "New York" regardless of prompt (more frequent in training data)

This distinction supports the taxonomy claim: different models exhibit different failure modes even under identical prompting.

---

## Footnote for Paper

> "Identical rates reflect that both Qwen and Llama succeeded only on the 11 instances where 'New York' was explicitly mentioned in the original query; all other queries produced 'New York' as a hallucinated default regardless of the queried location."