# Confidence Intervals for Key Metrics

## Methodology
All confidence intervals calculated using **Wilson score interval** at 95% confidence level.

---

## Chain A (2-Step) Results

| Metric | Count | n | Rate | 95% CI |
|--------|-------|---|------|--------|
| **Qwen** |||||
| Syntactic SSR | 110 | 110 | 100.0% | [96.6%, 100.0%] |
| Semantic SSR | 14 | 110 | 12.7% | [7.7%, 20.2%] |
| Semantic Cascade | 96 | 110 | 87.3% | [79.8%, 92.3%] |
| **Llama** |||||
| Syntactic SSR | 110 | 110 | 100.0% | [96.6%, 100.0%] |
| Semantic SSR | 15 | 110 | 13.6% | [8.4%, 21.3%] |
| Semantic Cascade | 95 | 110 | 86.4% | [78.7%, 91.6%] |
| **Phi-3** |||||
| Syntactic SSR | 3 | 110 | 2.7% | [0.9%, 7.7%] |
| Semantic SSR | 0 | 110 | 0.0% | [0.0%, 3.4%] |
| Cascade Failure | 110 | 110 | 100.0% | [96.6%, 100.0%] |

---

## Chain B (3-Step, Qwen) - Step-by-Step

| Step | Tool | Count | n | Rate | 95% CI |
|------|------|-------|---|------|--------|
| 1 | web_search | 47 | 55 | 85.5% | [73.8%, 92.4%] |
| 2 | extract_facts | 2 | 55 | 3.6% | [1.0%, 12.3%] |
| 3 | summarize | 2 | 55 | 3.6% | [1.0%, 12.3%] |

---

## Chain C (5-Step, Qwen) - Step-by-Step

| Step | Tool | Count | n | Rate | 95% CI |
|------|------|-------|---|------|--------|
| 1 | parse_date | 44 | 55 | 80.0% | [67.6%, 88.4%] |
| 2 | check_calendar | 29 | 55 | 52.7% | [39.8%, 65.3%] |
| 3 | find_conflicts | 29 | 55 | 52.7% | [39.8%, 65.3%] |
| 4 | suggest_time | 28 | 55 | 50.9% | [38.1%, 63.6%] |
| 5 | create_event | 28 | 55 | 50.9% | [38.1%, 63.6%] |
| **E2E** | All steps | 28 | 55 | 50.9% | [38.1%, 63.6%] |

---

## Summary Table for Paper

| Model | Chain | Metric | Rate | 95% CI |
|-------|-------|--------|------|--------|
| Qwen | A | Syntactic SSR | 100.0% | [96.6%, 100.0%] |
| Qwen | A | Semantic SSR | 12.7% | [7.7%, 20.2%] |
| Qwen | A | Semantic Cascade | 87.3% | [79.8%, 92.3%] |
| Llama | A | Syntactic SSR | 100.0% | [96.6%, 100.0%] |
| Llama | A | Semantic SSR | 13.6% | [8.4%, 21.3%] |
| Llama | A | Semantic Cascade | 86.4% | [78.7%, 91.6%] |
| Phi-3 | A | Syntactic SSR | 2.7% | [0.9%, 7.7%] |
| Phi-3 | A | CFR | 100.0% | [96.6%, 100.0%] |
| Qwen | B | Step 1 SSR | 85.5% | [73.8%, 92.4%] |
| Qwen | B | Step 2 SSR | 3.6% | [1.0%, 12.3%] |
| Qwen | C | Step 1 SSR | 80.0% | [67.6%, 88.4%] |
| Qwen | C | E2E Success | 50.9% | [38.1%, 63.6%] |