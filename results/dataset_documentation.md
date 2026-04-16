# Dataset Composition Documentation

## Actual Instance Counts

| Experiment | Base Queries | Conditions | Instances |
|------------|--------------|------------|-----------|
| Chain A (Qwen) | 10 | 11 | **110** |
| Chain A (Llama) | 10 | 11 | **110** |
| Chain A (Phi-3) | 10 | 11 | **110** |
| Chain B (Qwen) | 5 | 11 | **55** |
| Chain C (Qwen) | 5 | 11 | **55** |

## Dataset Calculation

**Formula:** 10 base queries × (1 clean + 5 perturbation types × 2 severity levels) = 110

- **clean**: 1 condition per query (baseline)
- **perturbation types**: 5 types (typo, paraphrase, missing_context, ambiguity, negation)
- **severity levels**: 2 levels (moderate, severe) for non-clean types

## Chain B/C Note

Chain B and C experiments were run with **5 base queries** (half the full dataset) for time efficiency:
- 5 queries × 11 conditions = 55 instances per chain

## Perturbation Breakdown (Per Chain)

| Perturbation | Instances | Severity |
|--------------|-----------|----------|
| clean | 10 | baseline |
| typo | 20 | 10 moderate + 10 severe |
| paraphrase | 20 | 10 moderate + 10 severe |
| missing_context | 20 | 10 moderate + 10 severe |
| ambiguity | 20 | 10 moderate + 10 severe |
| negation | 20 | 10 moderate + 10 severe |
| **TOTAL** | **110** | - |

## Base Queries (Chain A)

1. What's the weather like in London?
2. Should I bring an umbrella in Paris?
3. Is it sunny in Tokyo today?
4. Will it rain in New York tomorrow?
5. What's the temperature in Berlin?
6. Do I need an umbrella for Munich?
7. Is it going to be cloudy in Sydney?
8. What should I wear in Rome today?
9. Check weather for Madrid please
10. Will I need a coat in Vienna?

## Perturbation Details

- **typo**: Keyboard augmentation (5% moderate, 15% severe character error rate)
- **paraphrase**: T5-small generated rephrasing
- **missing_context**: Removed location entity from query
- **ambiguity**: Replaced specific location with generic reference ("the city", "the capital")
- **negation**: Flipped query intent with "unless", "except", "don't"