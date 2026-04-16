"""Perturbation engine for generating noisy queries."""

from typing import Optional
import random
import re

try:
    from nlpaug.augmenter.char import KeyboardAug
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False

try:
    from transformers import pipeline
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False


class PerturbationEngine:
    """Engine for applying perturbations to user queries."""
    
    def __init__(self, severity: str = "moderate"):
        """
        Initialize perturbation engine.
        
        Args:
            severity: 'clean', 'moderate', or 'severe'
        """
        self.severity = severity
        
        # Configure typo augmentation
        self.typo_aug = None
        if NLPAUG_AVAILABLE:
            aug_p = 0.05 if severity == "moderate" else 0.15
            self.typo_aug = KeyboardAug(aug_char_p=aug_p)
        
        # Paraphraser (load lazily)
        self._paraphraser = None
    
    @property
    def paraphraser(self):
        """Lazy-load T5 paraphraser."""
        if self._paraphraser is None and T5_AVAILABLE:
            self._paraphraser = pipeline(
                "text2text-generation",
                model="t5-small",
                device=-1  # CPU
            )
        return self._paraphraser
    
    def apply(self, query: str, ptype: str) -> str:
        """
        Apply perturbation to query.
        
        Args:
            query: Original user query
            ptype: Perturbation type ('typo', 'paraphrase', 'missing_context', 'ambiguity', 'negation')
        
        Returns:
            Perturbed query
        """
        if ptype == "clean":
            return query
        
        if ptype == "typo":
            return self._apply_typo(query)
        
        if ptype == "paraphrase":
            return self._apply_paraphrase(query)
        
        if ptype == "missing_context":
            return self._apply_missing_context(query)
        
        if ptype == "ambiguity":
            return self._apply_ambiguity(query)
        
        if ptype == "negation":
            return self._apply_negation(query)
        
        raise ValueError(f"Unknown perturbation type: {ptype}")
    
    def _apply_typo(self, query: str) -> str:
        """Apply keyboard-based typos."""
        if self.typo_aug is None:
            # Fallback: simple random character swap
            chars = list(query)
            if len(chars) > 3:
                idx = random.randint(1, len(chars) - 2)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return "".join(chars)
        
        result = self.typo_aug.augment(query)
        return result[0] if isinstance(result, list) else result
    
    def _apply_paraphrase(self, query: str) -> str:
        """Apply T5-based paraphrasing."""
        if self.paraphraser is None:
            # Fallback: simple word substitution
            return query.replace("weather", "conditions").replace("umbrella", "rain gear")
        
        try:
            input_text = f"paraphrase: {query}"
            output = self.paraphraser(input_text, max_length=50, do_sample=False)
            return output[0]["generated_text"]
        except Exception:
            return query
    
    def _apply_missing_context(self, query: str) -> str:
        """Remove proper nouns and specific references."""
        # Remove capitalized words (likely proper nouns) except first word
        words = query.split()
        if len(words) <= 2:
            return query
        
        # Remove words that are capitalized (proper nouns)
        result = [w for w in words if not w[0].isupper() or w == words[0]]
        
        # If removed too much, keep some context
        if len(result) < 2:
            return " ".join(words[:-1])
        
        return " ".join(result) + "?"
    
    def _apply_ambiguity(self, query: str) -> str:
        """Replace specific references with vague ones."""
        replacements = {
            "London": "the city",
            "Paris": "the capital",
            "Tokyo": "the metropolitan area",
            "New York": "the big city",
            "Berlin": "the german city",
            "Munich": "the bavarian city",
            "Sydney": "the australian city",
            "Rome": "the italian city",
            "Madrid": "the spanish city",
            "Vienna": "the austrian city",
            "umbrella": "rain gear",
            "sunny": "clear",
            "rainy": "wet",
            "cloudy": "overcast"
        }
        
        result = query
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        return result
    
    def _apply_negation(self, query: str) -> str:
        """Add negation to make intent ambiguous."""
        negation_patterns = [
            ("Don't ", ""),
            ("Do not ", ""),
            ("Should I ", "Should I not "),
            ("Will it ", "Will it not "),
            ("Is it ", "Is it not "),
            ("Can you ", "Can you not "),
        ]
        
        result = query
        
        # Randomly decide to add or modify negation
        if random.random() < 0.5:
            # Add negation at start
            if not result.lower().startswith("don't"):
                result = "Don't " + result[0].lower() + result[1:]
        else:
            # Modify existing question
            for pattern, replacement in negation_patterns:
                if pattern in result:
                    result = result.replace(pattern, replacement)
                    break
            else:
                # Add "unless..." clause
                result = result.rstrip("?") + " unless...?"
        
        return result
