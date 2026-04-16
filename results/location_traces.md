# Location Distribution & Trace Examples

## Finding: Universal "New York" Default

**Critical Discovery**: Both Qwen and Llama produce "New York" for **100%** of queries (110/110), regardless of what the user asks.

This is not "frequently defaults" - it is an **absolute default** that constitutes the primary finding of this paper.

---

## Location Frequency Distribution

| Model | Location | Count | Percentage |
|-------|----------|-------|------------|
| Qwen | New York | 110 | 100% |
| Llama | New York | 110 | 100% |
| Phi-3 | (none) | - | - |

**Conclusion**: Models consistently hallucinate "New York" regardless of input.

---

## Full Chain Trace Examples

### Example 1: Clean Query → Hallucination (Most Common)

```
QUERY: "What's the weather like in London?"
GROUND TRUTH: London
MODEL OUTPUT (Step 1): {"location": "New York", "temperature": 75, "condition": "cloudy", "humidity": 45}
STEP 1 SUCCESS: True (valid JSON)
STEP 2 INPUT: Weather data for New York
MODEL OUTPUT (Step 2): {"bring_umbrella": false, "reason": "It is not raining outside"}
STEP 2 SUCCESS: True (valid JSON)
SEMANTIC VALID: False (hallucinated wrong city)
RESULT: Semantic Cascade - User receives confident advice for WRONG city
```

### Example 2: Ambiguity Perturbation → Cascade

```
ORIGINAL QUERY: "What's the weather like in London?"
PERTURBED QUERY: "What's the weather like in the city?"
GROUND TRUTH: None (no location - query is ambiguous)
MODEL OUTPUT (Step 1): {"location": "New York", "temperature": 75, "condition": "cloudy", "humidity": 45}
SEMANTIC CASCADE: True
RESULT: Hallucinated location despite ambiguous input → proceeded confidently
```

### Example 3: Accidental Match (False Positive)

```
QUERY: "Will it rain in New York tomorrow?"
GROUND TRUTH: New York
MODEL OUTPUT: {"location": "New York", ...}
SEMANTIC VALID: True (but ACCIDENTAL - model never understood query, just defaults to NYC)
```

---

## Interpretation for Paper

The 20% "semantic success" on clean queries is NOT a sign of partial understanding - it is purely statistical luck where the user's query happened to mention New York. The model never extracts the actual location from the query; it always outputs "New York" and the 20% success rate reflects how many base queries happened to mention that city.

**Key Statement for Paper:**
> Both Qwen and Llama exhibit a systematic default behavior, producing "New York" for 100% of weather queries regardless of user input. This hallucination is not a random error but a deterministic output pattern. The 12-13% semantic success rate on clean queries reflects the proportion of base queries mentioning New York, not model comprehension.

---

## Phi-3 Failure Mode

Phi-3 fails syntactically (produces invalid JSON) rather than semantically:

```
QUERY: "What's the weather like in London?"
MODEL OUTPUT: "I cannot provide weather information..."
STEP 1 SUCCESS: False (invalid JSON)
RESULT: Hard failure - system crashes rather than hallucinates
```

This is fundamentally different from Qwen/Llama's silent corruption behavior.