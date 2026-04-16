"""Base query dataset for Chains A, B, C."""

from typing import List, Dict
import itertools

# ============ Chain A: Weather Queries (10 queries) ============
CHAIN_A_QUERIES = [
    "What's the weather like in London?",
    "Should I bring an umbrella in Paris?",
    "Is it sunny in Tokyo today?",
    "Will it rain in New York tomorrow?",
    "What's the temperature in Berlin?",
    "Do I need an umbrella for Munich?",
    "Is it going to be cloudy in Sydney?",
    "What should I wear in Rome today?",
    "Check weather for Madrid please",
    "Will I need a coat in Vienna?",
]

# For backward compatibility
BASE_QUERIES = CHAIN_A_QUERIES


# ============ Chain B: Search -> Extract -> Summarize (10 queries) ============
CHAIN_B_QUERIES = [
    "Search for information about machine learning and summarize it",
    "Find recent news about climate change and give me the key points",
    "Look up what is quantum computing and summarize",
    "Search for best practices in software development",
    "Find information about renewable energy trends",
    "Search for history of artificial intelligence",
    "Look up tips for healthy eating and summarize",
    "Find news about space exploration recent developments",
    "Search for information about blockchain technology",
    "Look up best travel destinations for 2026 and summarize",
]


# ============ Chain C: Calendar (10 queries) ============
CHAIN_C_QUERIES = [
    "Schedule a meeting for next Monday at 3pm",
    "Check my calendar for tomorrow and find a free time slot",
    "Create an event for next Friday at 10am",
    "Find conflicts if I schedule a 1-hour meeting tomorrow",
    "Suggest a time for a 30-minute call next week",
    "Schedule a team meeting on Wednesday afternoon",
    "Check what's on my calendar for next Monday",
    "Find available time slots for a 2-hour workshop",
    "Create a reminder for next Tuesday morning",
    "Schedule a lunch meeting for Thursday at noon",
]


PERTURBATION_TYPES = ["clean", "typo", "paraphrase", "missing_context", "ambiguity", "negation"]
SEVERITY_LEVELS = ["clean", "moderate", "severe"]


def generate_dataset(chain_queries: List[str] = None, ptypes: List[str] = None, severities: List[str] = None) -> List[Dict]:
    """Generate dataset of perturbed queries."""
    if chain_queries is None:
        chain_queries = BASE_QUERIES
    if ptypes is None:
        ptypes = PERTURBATION_TYPES
    if severities is None:
        severities = SEVERITY_LEVELS
    
    dataset = []
    
    for original in chain_queries:
        for ptype in ptypes:
            for severity in severities:
                # Skip invalid combinations
                if ptype == "clean" and severity != "clean":
                    continue
                if severity == "clean" and ptype != "clean":
                    continue
                
                # Apply perturbation
                from perturbations.engine import PerturbationEngine
                engine = PerturbationEngine(severity="moderate" if severity != "clean" else "clean")
                perturbed = engine.apply(original, ptype)
                
                dataset.append({
                    "query": perturbed,
                    "original": original,
                    "ptype": ptype,
                    "severity": severity,
                })
    
    return dataset


def get_base_queries(chain: str = "chain_a") -> List[str]:
    """Return list of base queries for a chain."""
    if chain == "chain_a":
        return CHAIN_A_QUERIES.copy()
    elif chain == "chain_b":
        return CHAIN_B_QUERIES.copy()
    elif chain == "chain_c":
        return CHAIN_C_QUERIES.copy()
    else:
        return CHAIN_A_QUERIES.copy()
