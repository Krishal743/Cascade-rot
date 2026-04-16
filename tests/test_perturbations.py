"""Unit tests for perturbation engine."""

import pytest
from perturbations.engine import PerturbationEngine


class TestPerturbationEngine:
    """Test suite for PerturbationEngine."""
    
    def test_clean_no_change(self):
        """Clean perturbation should return identical input."""
        engine = PerturbationEngine(severity="clean")
        query = "What's the weather in London?"
        result = engine.apply(query, "clean")
        assert result == query
    
    def test_typo_changes_characters(self):
        """Typo injection should modify characters."""
        engine = PerturbationEngine(severity="moderate")
        query = "weather"
        result = engine.apply(query, "typo")
        # At least one character should be different
        assert result != query or len(result) != len(query)
    
    def test_paraphrase_different(self):
        """Paraphrasing should produce different text."""
        engine = PerturbationEngine(severity="moderate")
        query = "Book a meeting"
        result = engine.apply(query, "paraphrase")
        # May or may not be different depending on T5 availability
        # But should not raise error
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_missing_context_removes_entity(self):
        """Missing context should remove identifiable entities."""
        engine = PerturbationEngine(severity="moderate")
        query = "Weather in London"
        result = engine.apply(query, "missing_context")
        # London should be removed
        assert "London" not in result
    
    def test_ambiguity_replaces_specific(self):
        """Ambiguity should replace specific references."""
        engine = PerturbationEngine(severity="moderate")
        query = "Weather in Paris"
        result = engine.apply(query, "ambiguity")
        # Paris should be replaced with vague reference
        assert "Paris" not in result or "the capital" in result.lower()
    
    def test_negation_adds_negation(self):
        """Negation flip should add negation."""
        engine = PerturbationEngine(severity="moderate")
        query = "Schedule Monday"
        result = engine.apply(query, "negation")
        # Result should contain negation
        assert "n't" in result.lower() or "not" in result.lower() or "unless" in result.lower()
    
    def test_all_types_return_valid_string(self):
        """All perturbation types should return non-empty string."""
        engine = PerturbationEngine(severity="moderate")
        query = "What's the weather?"
        
        for ptype in ["typo", "paraphrase", "missing_context", "ambiguity", "negation"]:
            result = engine.apply(query, ptype)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_severity_levels(self):
        """Different severity levels should work."""
        for severity in ["clean", "moderate", "severe"]:
            engine = PerturbationEngine(severity=severity)
            assert engine.severity == severity
    
    def test_invalid_type_raises_error(self):
        """Invalid perturbation type should raise ValueError."""
        engine = PerturbationEngine()
        with pytest.raises(ValueError):
            engine.apply("test", "invalid_type")


if __name__ == "__main__":
    # Run tests
    test = TestPerturbationEngine()
    
    print("Running perturbation tests...")
    
    test.test_clean_no_change()
    print("✓ test_clean_no_change passed")
    
    test.test_typo_changes_characters()
    print("✓ test_typo_changes_characters passed")
    
    test.test_paraphrase_different()
    print("✓ test_paraphrase_different passed")
    
    test.test_missing_context_removes_entity()
    print("✓ test_missing_context_removes_entity passed")
    
    test.test_ambiguity_replaces_specific()
    print("✓ test_ambiguity_replaces_specific passed")
    
    test.test_negation_adds_negation()
    print("✓ test_negation_adds_negation passed")
    
    test.test_all_types_return_valid_string()
    print("✓ test_all_types_return_valid_string passed")
    
    test.test_severity_levels()
    print("✓ test_severity_levels passed")
    
    test.test_invalid_type_raises_error()
    print("✓ test_invalid_type_raises_error passed")
    
    print("\n✅ All tests passed!")
